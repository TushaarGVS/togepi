import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from torch import autograd
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm, trange

from togepi.data_processors.utils import get_torch_dataset, get_dataloader
from togepi.models.modules.attention.togepi.sparse import TogepiSparse
from togepi.models.modules.attention.togepi.utils import prune_sparse
from togepi.models.transformer import Transformer
from togepi.schedulers.noam import NoamLrScheduler


class Trainer(nn.Module):
    def __init__(self, model: Transformer, optim, tok_train_data, tok_val_data, loss_fn=nn.CrossEntropyLoss,
                 noam_num_warmup_steps=4000, noam_factor=1.0, rop_patience=5, rop_factor=0.1, sparse_dens=0.3,
                 sparse_penalty_lambda=0.05, labels_ignore_idx=0, gradient_clip_norm=None, num_workers=0, use_amp=True,
                 use_dp=True, tracker=None, detect_anomaly=False, device=torch.device('cpu')):
        super().__init__()

        self.detect_anomaly = detect_anomaly
        if self.detect_anomaly:
            logging.warning(f'detect_anomaly is set to {detect_anomaly}; this will significantly slow down the speed')

        self.device = device
        self.model = model
        self._use_dp = False
        if use_dp and torch.cuda.device_count() > 1:
            logging.info(f'using {torch.cuda.device_count()} gpus for training ...')
            self._use_dp = True
            self.model = nn.DataParallel(self.model)
        elif use_dp:
            logging.info(f'ignoring use_dp={use_dp}: only {torch.cuda.device_count()} gpu available')
        self.model = self.model.to(self.device)
        _enc = self.model.enc if not self._use_dp else self.model.module.enc  # note the use of data-parallelism
        self.tracker = tracker

        self.use_amp = use_amp
        if self.device.type != 'cuda':
            # https://discuss.pytorch.org/t/error-while-using-16-bit-floats-half/139465/2
            logging.info(f'ignoring use_amp={use_amp}: device not supported')
            self.use_amp = False
        self.grad_scaler = GradScaler(enabled=self.use_amp)

        self.optim = optim
        self.gradient_clip_norm = gradient_clip_norm
        # https://www.reddit.com/r/MachineLearning/comments/oy3co1/comment/h7qzshz
        self.noam_scheduler = NoamLrScheduler(self.optim, warmup_steps=noam_num_warmup_steps,
                                              d_model=_enc.emb.embedding_dim, factor=noam_factor)
        self.rop_scheduler = ReduceLROnPlateau(self.optim, mode='min', patience=rop_patience, factor=rop_factor)

        self.num_workers = num_workers
        self.tok_train_data = get_torch_dataset(tok_train_data)
        self.tok_val_data = tok_val_data
        if tok_val_data is not None:
            self.tok_val_data = get_torch_dataset(tok_val_data)

        self.use_togepi_mha = _enc.enc_layers[0].use_togepi_mha
        self.use_spectral_norm = False
        self.sparse_penalty_lambda = 0.0
        self.model_max_length = None
        self.sparse_dens = 1.0
        if self.use_togepi_mha:
            self.use_spectral_norm = _enc.enc_layers[0].mha.use_spectral_norm
            self.sparse_penalty_lambda = sparse_penalty_lambda
            self.model_max_length = _enc.emb.max_length
            if self.sparse_dens is not None:
                self.sparse_dens = sparse_dens

        self.labels_ignore_idx = labels_ignore_idx
        self.loss_fn = loss_fn(ignore_index=labels_ignore_idx)

        # variables stored to resume training
        self._epoch = 0
        self._step = 0

    def save_checkpoint(self, checkpoint_path):
        torch.save({'epoch': self._epoch,
                    'step': self._step,
                    'model_state_dict': self.model.state_dict() if not self._use_dp else self.model.module.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'noam_scheduler_state_dict': self.noam_scheduler.state_dict(),
                    'rop_scheduler_state_dict': self.rop_scheduler.state_dict()}, checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        self.noam_scheduler.load_state_dict(checkpoint['noam_scheduler_state_dict'])
        self.rop_scheduler.load_state_dict(checkpoint['rop_scheduler_state_dict'])

        self._epoch = checkpoint['epoch'] + 1
        self._step = checkpoint['step'] + 1

    def _compute_loss(self, predictions, labels):
        return self.loss_fn(input=predictions.cpu().float(), target=labels.cpu())

    def _compute_entropy(self, predictions, labels):
        return F.cross_entropy(input=predictions.cpu().float(), target=labels.cpu(),
                               ignore_index=self.labels_ignore_idx).detach()

    @staticmethod
    def _compute_ppl_from_entropy(entropy):
        return np.exp(entropy)

    def _compute_ppl(self, predictions, labels):
        return torch.exp(F.cross_entropy(input=predictions.cpu().float(), target=labels.cpu(),
                                         ignore_index=self.labels_ignore_idx)).detach()

    def __count_nonzero_sparse_vals(self, layer):
        if self.use_spectral_norm:
            return np.count_nonzero(layer.sparse_mat_orig.detach().cpu().numpy())
        else:
            return np.count_nonzero(layer.sparse_mat.detach().cpu().numpy())

    def __compute_sparse_densities(self, sparse_layers):
        total_vals = self.model_max_length * self.model_max_length
        return [self.__count_nonzero_sparse_vals(layer) / total_vals for layer in sparse_layers]

    def _compute_sparsity_penalty(self):
        togepi_sparse_layers = filter(lambda layer: isinstance(layer, TogepiSparse), self.model.modules())
        sparse_densities = self.__compute_sparse_densities(togepi_sparse_layers)
        # try: penalize beyond 1.5 * sparse_dens: 1.5x to account for "boolean or" between gradient and weight masking
        sparse_nonzero_weight_dens = [np.maximum(0, dens - self.sparse_dens) for dens in sparse_densities]
        del togepi_sparse_layers  # clear out cuda memory

        return np.average(sparse_nonzero_weight_dens), sparse_densities

    def _train_epoch(self, dataloader, grad_update_every=1):
        all_batches_metrics = {'loss': [], 'ce': [], 'avg_sparse_dens_across_layers': []}
        epoch_metrics = {}

        # handle last few batches of gradient updates separately
        start_step = self._step
        num_full_steps_per_epoch = len(dataloader) // grad_update_every
        last_grad_update_every = len(dataloader) % grad_update_every

        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        self.model.train()
        self.optim.zero_grad(set_to_none=True)  # https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide
        accumulated_loss, accumulated_entropies, accumulated_avg_sparse_dens_across_layers = 0.0, [], []
        for batch_idx, data_batch in enumerate(tqdm(dataloader)):
            # https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                input_ids = data_batch['input_ids'].to(self.device)
                token_type_ids = data_batch['token_type_ids'].to(self.device)
                padding_mask = data_batch['attention_mask'].to(self.device)
                lm_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids, padding_mask=padding_mask,
                                       return_attn_filters_or_psfs=False)
                del input_ids, token_type_ids, padding_mask  # clear out cuda memory

                # predictions: (batch_size * (max_length - 1), vocab_size)
                # labels: (batch_size * (max_length - 1))
                predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1]).cpu()
                labels = data_batch['input_ids'][:, 1:].contiguous().view(-1).cpu()
                del lm_output  # clear out memory

                batch_loss = self._compute_loss(predictions=predictions, labels=labels)
                if self.use_togepi_mha:
                    # sparse weights penalty
                    sparse_penalty, sparse_densities = self._compute_sparsity_penalty()
                    accumulated_avg_sparse_dens_across_layers.append(np.average(sparse_densities))
                    batch_loss = \
                        (1 - self.sparse_penalty_lambda) * batch_loss + self.sparse_penalty_lambda * sparse_penalty
                batch_loss = batch_loss / grad_update_every if self._step < start_step + num_full_steps_per_epoch else \
                    batch_loss / last_grad_update_every
            self.grad_scaler.scale(batch_loss.to(self.device)).backward()  # accumulate gradients
            batch_loss.detach_()  # drop immediate buffers

            # loss accumulation: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3#gistcomment-4173824
            accumulated_loss = accumulated_loss + batch_loss.item()
            accumulated_entropies.append(self._compute_entropy(predictions=predictions, labels=labels).item())
            del predictions, labels, batch_loss  # clear out memory

            #  grad accumulation: https://gist.github.com/thomwolf/ac7a7da6b1888c2eeac8ac8b9b05d3d3
            if (batch_idx + 1) % grad_update_every == 0 or (batch_idx + 1) == len(dataloader):
                self.grad_scaler.unscale_(self.optim)  # unscale for clipping
                # https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/trainer.py
                if self.gradient_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                self.grad_scaler.step(self.optim)
                self.noam_scheduler.step()  # update lr based on Noam lr scheduling
                self.grad_scaler.update()

                # prune sparse mask based on gradient and weight
                if self.use_togepi_mha:
                    self.model = prune_sparse(self.model, sparse_dens=self.sparse_dens,
                                              use_spectral_norm=self.use_spectral_norm)

                self.optim.zero_grad(set_to_none=True)  # reset gradients post accumulation

                all_batches_metrics['loss'].append(accumulated_loss)
                all_batches_metrics['ce'].append(np.average(accumulated_entropies))
                step_metrics = {'loss': accumulated_loss, 'ce': all_batches_metrics['ce'][-1],
                                'ppl': self._compute_ppl_from_entropy(all_batches_metrics['ce'][-1])}
                if self.use_togepi_mha:
                    avg_sparse_dens_across_layers = np.average(accumulated_avg_sparse_dens_across_layers)
                    all_batches_metrics['avg_sparse_dens_across_layers'].append(avg_sparse_dens_across_layers)
                    step_metrics['avg_sparse_dens_across_layers'] = avg_sparse_dens_across_layers
                self.tracker.log_metrics(epoch_or_step=self._step, split_name='train', metrics=step_metrics,
                                         epoch_or_step_id='step', log_to_console=False)

                # reset accumulators and update step number
                accumulated_loss, accumulated_entropies, accumulated_avg_sparse_dens_across_layers = 0.0, [], []
                self._step = self._step + 1

        for metric in all_batches_metrics.keys():
            if len(all_batches_metrics[metric]) > 0:
                epoch_metrics[metric] = np.average(all_batches_metrics[metric])
        # batched ppl: https://github.com/pytorch/examples/blob/main/word_language_model/main.py
        epoch_metrics['ppl'] = self._compute_ppl_from_entropy(epoch_metrics['ce'])
        return epoch_metrics

    def _eval_epoch(self, dataloader):
        all_batches_metrics = {'loss': [], 'ce': []}
        epoch_metrics = {}

        self.model.eval()
        with torch.no_grad():
            for data_batch in tqdm(dataloader):
                # https://discuss.pytorch.org/t/mixed-precision-for-validation/92319/2
                with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp):
                    input_ids = data_batch['input_ids'].to(self.device)
                    token_type_ids = data_batch['token_type_ids'].to(self.device)
                    padding_mask = data_batch['attention_mask'].to(self.device)
                    lm_output = self.model(input_ids=input_ids, token_type_ids=token_type_ids,
                                           padding_mask=padding_mask, return_attn_filters_or_psfs=False)
                    del input_ids, token_type_ids, padding_mask  # clear out cuda memory

                    predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1]).cpu()
                    labels = data_batch['input_ids'][:, 1:].contiguous().view(-1).cpu()
                    del lm_output  # clear out memory

                    batch_loss = self._compute_loss(predictions=predictions, labels=labels).item()
                self.rop_scheduler.step(batch_loss)  # update lr based on reduce-on-plateau lr scheduling

                all_batches_metrics['loss'].append(batch_loss)
                all_batches_metrics['ce'].append(self._compute_entropy(predictions=predictions, labels=labels).item())
                del predictions, labels, batch_loss  # clear out memory

        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = np.average(all_batches_metrics[metric])
        # batched ppl: https://github.com/pytorch/examples/blob/main/word_language_model/main.py
        epoch_metrics['ppl'] = self._compute_ppl_from_entropy(epoch_metrics['ce'])
        return epoch_metrics

    def train_and_eval(self, batch_size=64, grad_update_every=1, num_steps_per_epoch=None, num_epochs=8,
                       checkpoint_every=10):
        val_dataloader = None
        if self.tok_val_data is not None:
            val_dataloader = get_dataloader(self.tok_val_data, batch_size=batch_size, shuffle=False,
                                            num_samples=num_steps_per_epoch, num_workers=self.num_workers)
        for epoch in trange(num_epochs):
            train_dataloader = get_dataloader(self.tok_train_data, batch_size=batch_size, shuffle=True,
                                              num_samples=num_steps_per_epoch, num_workers=self.num_workers)
            with autograd.set_detect_anomaly(self.detect_anomaly, check_nan=True):
                train_metrics = self._train_epoch(train_dataloader, grad_update_every=grad_update_every)
                val_metrics = self._eval_epoch(val_dataloader) if val_dataloader is not None else None
            del train_dataloader  # clear out memory

            if self.tracker is not None:
                self.tracker.log_metrics(epoch_or_step=self._epoch, split_name='train', metrics=train_metrics,
                                         epoch_or_step_id='epoch')
                if val_metrics is not None:
                    self.tracker.log_metrics(epoch_or_step=self._epoch, split_name='val', metrics=val_metrics,
                                             epoch_or_step_id='epoch')
                if (epoch + 1) % checkpoint_every == 0:
                    self.tracker.save_checkpoint(self, epoch=self._epoch)
            self._epoch = self._epoch + 1

        # save model trained for specified num_epochs
        self.tracker.save_model(self.model) if not self._use_dp else self.tracker.save_model(self.model.module)

    @staticmethod
    @torch.no_grad()
    def test(model, tok_test_data, batch_size=128, labels_ignore_idx=0, num_workers=0, use_amp=True, tracker=None,
             device=torch.device('cpu')):
        test_dataloader = get_dataloader(get_torch_dataset(tok_test_data), batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
        all_predictions, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for data_batch in tqdm(test_dataloader):
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    input_ids = data_batch['input_ids'].to(device)
                    token_type_ids = data_batch['token_type_ids'].to(device)
                    padding_mask = data_batch['attention_mask'].to(device)
                    # lm_output = (batch_size, max_length, vocab_size)
                    lm_output = model(input_ids=input_ids, token_type_ids=token_type_ids, padding_mask=padding_mask)
                    del input_ids, token_type_ids, padding_mask  # clear out cuda memory

                    predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1]).cpu()
                    labels = data_batch['input_ids'][:, 1:].contiguous().view(-1).cpu()
                    all_predictions.append(predictions)
                    all_labels.append(labels)
                    del lm_output, predictions, labels  # clear out memory

        test_ppl = F.cross_entropy(input=torch.vstack(all_predictions).cpu(), target=torch.hstack(all_labels).cpu(),
                                   ignore_index=labels_ignore_idx).item()
        if tracker is not None:
            tracker.log_metrics(epoch_or_step=0, split_name='test', metrics={'ppl': test_ppl})
        return test_ppl
