import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
                 sparse_penalty_lambda=0.05, labels_ignore_idx=0, gradient_clip_value=None, num_workers=0, tracker=None,
                 device=torch.device('cpu')):
        super().__init__()

        self.device = device
        self.model = model.to(device)
        self.tracker = tracker

        self.optim = optim
        self.gradient_clip_value = gradient_clip_value
        # https://www.reddit.com/r/MachineLearning/comments/oy3co1/comment/h7qzshz
        self.noam_scheduler = NoamLrScheduler(self.optim, warmup_steps=noam_num_warmup_steps,
                                              d_model=self.model.enc.emb.embedding_dim, factor=noam_factor)
        self.rop_scheduler = ReduceLROnPlateau(self.optim, mode='min', patience=rop_patience, factor=rop_factor)

        self.num_workers = num_workers
        self.tok_train_data = get_torch_dataset(tok_train_data)
        self.tok_val_data = tok_val_data
        if tok_val_data is not None:
            self.tok_val_data = get_torch_dataset(tok_val_data)

        self.use_togepi_mha = self.model.enc.enc_layers[0].use_togepi_mha
        self.use_spectral_norm = False
        self.sparse_penalty_lambda = 0.0
        self.model_max_length = None
        self.sparse_dens = 1.0
        if self.use_togepi_mha:
            self.use_spectral_norm = self.model.enc.enc_layers[0].mha.use_spectral_norm
            self.sparse_penalty_lambda = sparse_penalty_lambda
            self.model_max_length = self.model.enc.emb.max_length
            if self.sparse_dens is not None:
                self.sparse_dens = sparse_dens

        self.labels_ignore_idx = labels_ignore_idx
        self.loss_fn = loss_fn(ignore_index=labels_ignore_idx)

        self._start_epoch = 0  # variable stored to resume training

    def save_checkpoint(self, epoch, checkpoint_path):
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optim_state_dict': self.optim.state_dict(),
                    'noam_scheduler_state_dict': self.noam_scheduler.state_dict(),
                    'rop_scheduler_state_dict': self.rop_scheduler.state_dict()}, checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        self.noam_scheduler.load_state_dict(checkpoint['noam_scheduler_state_dict'])
        self.rop_scheduler.load_state_dict(checkpoint['rop_scheduler_state_dict'])

        self._start_epoch = checkpoint['epoch'] + 1

    def _compute_loss(self, predictions, labels):
        return self.loss_fn(input=predictions.cpu(), target=labels.cpu())

    def _compute_ppl(self, predictions, labels):
        return torch.exp(F.cross_entropy(input=predictions.cpu(), target=labels.cpu(),
                                         ignore_index=self.labels_ignore_idx))

    def _compute_sparsity_penalty(self):
        max_nonzero = min(1.5 * self.sparse_dens * self.model_max_length * self.model_max_length,
                          self.model_max_length * self.model_max_length)

        togepi_sparse_layers = filter(lambda layer: isinstance(layer, TogepiSparse), self.model.modules())
        sparse_nonzero_weight_counts = \
            [np.maximum(0, np.count_nonzero(layer.sparse_mat.clone().cpu().detach().numpy()) - max_nonzero)
             for layer in togepi_sparse_layers]
        del togepi_sparse_layers  # clear out cuda memory
        return sum(sparse_nonzero_weight_counts)

    def _train_epoch(self, dataloader):
        all_batches_metrics = {'loss': [], 'ppl': []}
        epoch_metrics = {}

        # https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
        # https://pytorch.org/docs/stable/notes/amp_examples.html
        self.model.train()
        for data_batch in tqdm(dataloader):
            # https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide
            self.optim.zero_grad(set_to_none=True)

            lm_output = self.model(input_ids=data_batch['input_ids'].to(self.device),
                                   token_type_ids=data_batch['token_type_ids'].to(self.device),
                                   padding_mask=data_batch['attention_mask'].to(self.device))

            # predictions: (batch_size * (max_length - 1), vocab_size)
            # labels: (batch_size * (max_length - 1))
            predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1])
            labels = data_batch['input_ids'][:, 1:].contiguous().view(-1)

            batch_loss = self._compute_loss(predictions=predictions, labels=labels)
            if self.use_togepi_mha:
                # sparse weights penalty
                sparse_penalty = self._compute_sparsity_penalty()
                batch_loss = (1 - self.sparse_penalty_lambda) * batch_loss + self.sparse_penalty_lambda * sparse_penalty

            batch_loss.backward()
            batch_loss = batch_loss.item()
            if self.gradient_clip_value is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
            self.optim.step()
            self.noam_scheduler.step()  # update lr based on Noam lr scheduling

            if self.use_togepi_mha:
                # prune sparse mask based on gradient and weight
                prune_sparse(self.model, sparse_dens=self.sparse_dens, use_spectral_norm=self.use_spectral_norm)

            all_batches_metrics['loss'].append(batch_loss)
            all_batches_metrics['ppl'].append(self._compute_ppl(predictions=predictions, labels=labels))

        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = sum(all_batches_metrics[metric]) / len(dataloader)
        return epoch_metrics

    def _eval_epoch(self, dataloader):
        all_batches_metrics = {'loss': [], 'ppl': []}
        epoch_metrics = {}

        self.model.eval()
        with torch.no_grad():
            for data_batch in tqdm(dataloader):
                lm_output = self.model(input_ids=data_batch['input_ids'].to(self.device),
                                       token_type_ids=data_batch['token_type_ids'].to(self.device),
                                       padding_mask=data_batch['attention_mask'].to(self.device))
                predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1])
                labels = data_batch['input_ids'][:, 1:].contiguous().view(-1)

                batch_loss = self._compute_loss(predictions=predictions, labels=labels).item()

                # update lr based on reduce-on-plateau lr scheduling
                self.rop_scheduler.step(batch_loss)

                all_batches_metrics['loss'].append(batch_loss)
                all_batches_metrics['ppl'].append(self._compute_ppl(predictions=predictions, labels=labels))

        for metric in all_batches_metrics.keys():
            epoch_metrics[metric] = sum(all_batches_metrics[metric]) / len(dataloader)
        return epoch_metrics

    def train_and_eval(self, batch_size=64, num_steps_per_epoch=None, num_epochs=8, checkpoint_every=10):
        val_dataloader = None
        if self.tok_val_data is not None:
            val_dataloader = get_dataloader(self.tok_val_data, batch_size=batch_size, shuffle=False,
                                            num_samples=num_steps_per_epoch, num_workers=self.num_workers)
        for epoch in trange(self._start_epoch, self._start_epoch + num_epochs, 1):
            train_dataloader = get_dataloader(self.tok_train_data, batch_size=batch_size, shuffle=True,
                                              num_samples=num_steps_per_epoch, num_workers=self.num_workers)
            train_metrics = self._train_epoch(train_dataloader)
            val_metrics = self._eval_epoch(val_dataloader) if val_dataloader is not None else None

            if self.tracker is not None:
                self.tracker.log_metrics(epoch=epoch, split_name='train', metrics=train_metrics)
                if val_metrics is not None:
                    self.tracker.log_metrics(epoch=epoch, split_name='val', metrics=val_metrics)
                if (epoch + 1) % checkpoint_every == 0:
                    self.tracker.save_checkpoint(self, epoch=epoch)

            self.tracker.save_model(self.model)

    @staticmethod
    @torch.no_grad()
    def test(model, tok_test_data, batch_size=128, labels_ignore_idx=0, num_workers=0, tracker=None,
             device=torch.device('cpu')):
        test_dataloader = get_dataloader(get_torch_dataset(tok_test_data), batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
        all_predictions, all_labels = [], []

        model.eval()
        with torch.no_grad():
            for data_batch in tqdm(test_dataloader):
                # lm_output = (batch_size, max_length, vocab_size)
                lm_output = model(input_ids=data_batch['input_ids'].to(device),
                                  token_type_ids=data_batch['token_type_ids'].to(device),
                                  padding_mask=data_batch['attention_mask'].to(device))
                predictions = lm_output[:, :-1, :].contiguous().view(-1, lm_output.shape[-1])
                labels = data_batch['input_ids'][:, 1:].contiguous().view(-1)
                all_predictions.append(predictions)
                all_labels.append(labels)

        test_ppl = F.cross_entropy(input=torch.vstack(all_predictions).to(device),
                                   target=torch.hstack(all_labels).to(device), ignore_index=labels_ignore_idx)
        if tracker is not None:
            tracker.log_metrics(epoch=0, split_name='test', metrics={'ppl': test_ppl})
        return test_ppl
