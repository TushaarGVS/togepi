import torch
import torch.nn as nn
import torch.nn.functional as F
from prettytable import PrettyTable
from torchinfo import summary

from togepi.models.modules.encoder import Encoder
from togepi.models.modules.lm_heads.linear import LinearLmHead


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, num_token_types=2, hidden_dim=1024, num_enc_layers=12,
                 num_attn_heads=12, max_position_embeddings=1025, use_togepi_mha=True, softmax_psf_weights=True,
                 attn_actn='gelu', use_spectral_norm=True, sparse_init_dens=None, num_power_iters=1,
                 attn_dropout_proba=0.1, embedding_dropout_proba=0.1, keep_prob=1.0, causal_attn=True,
                 use_explicit_lm_head=False, padding_idx=0, pad_position=0, pad_token_type=0):
        super().__init__()

        self.enc = Encoder(vocab_size=vocab_size, embedding_dim=embedding_dim, num_token_types=num_token_types,
                           hidden_dim=hidden_dim, num_enc_layers=num_enc_layers, num_attn_heads=num_attn_heads,
                           max_position_embeddings=max_position_embeddings, use_togepi_mha=use_togepi_mha,
                           softmax_psf_weights=softmax_psf_weights, attn_actn=attn_actn,
                           use_spectral_norm=use_spectral_norm, sparse_init_dens=sparse_init_dens,
                           num_power_iters=num_power_iters, attn_dropout_proba=attn_dropout_proba,
                           embedding_dropout_proba=embedding_dropout_proba, keep_prob=keep_prob,
                           causal_attn=causal_attn, padding_idx=padding_idx, pad_position=pad_position,
                           pad_token_type=pad_token_type)
        self._use_lm_head = use_explicit_lm_head
        if use_explicit_lm_head:
            self.lm_head = LinearLmHead(vocab_size=vocab_size, embedding_dim=embedding_dim)

    def summary(self):
        return summary(self)

    def get_trainable_params(self):
        return (param for param in self.parameters() if param.requires_grad)

    def get_params(self, print_params=False):
        params_table = PrettyTable(['module', 'num_params', 'requires_grad'])
        total_trainable_params = 0
        for name, param in self.named_parameters():
            params_table.add_row([name, param.numel(), param.requires_grad])
            if param.requires_grad:
                total_trainable_params = total_trainable_params + param.numel()

        if print_params:
            print(params_table)
            print(f'total trainable params: {(total_trainable_params / 1e6):0.2f}M')
        return params_table, total_trainable_params

    def save_pretrained(self, model_save_path):
        torch.save(self.state_dict(), model_save_path)

    def from_pretrained(self, model_save_path):
        self.load_state_dict(torch.load(model_save_path))

    def forward(self, input_ids, token_type_ids=None, padding_mask=None, return_attn_filters_or_psfs=False):
        emb_all_layers, attn_filters_or_psfs_all_layers = self.enc(input_ids=input_ids, token_type_ids=token_type_ids,
                                                                   padding_mask=padding_mask)
        last_emb = emb_all_layers[-1]

        # clear out cuda memory
        if not return_attn_filters_or_psfs:
            del attn_filters_or_psfs_all_layers
        del emb_all_layers

        if self._use_lm_head:
            lm_output = self.lm_head(last_emb)
        else:
            # weight tying: https://github.com/openai/gpt-2/blob/master/src/model.py#L169
            # last_emb: (batch_size, max_length, embedding_dim)
            # tok_emb_weight: (vocab_size, embedding_dim)
            lm_output = torch.matmul(last_emb, self.enc.emb.tok_emb.weight.data.transpose(0, 1))
        return lm_output if not return_attn_filters_or_psfs else (lm_output, attn_filters_or_psfs_all_layers)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, num_samples=1, top_k=None):
        num_samples = max(num_samples, 1)
        model_max_length = self.enc.emb.max_length

        for _ in range(max_new_tokens):
            # input_ids: (num_samples, prompt_length)
            input_ids = input_ids if input_ids.shape[1] <= model_max_length else input_ids[:, -model_max_length:]
            lm_output = self(input_ids=input_ids.to(next(self.parameters()).device))
            lm_output = lm_output[:, -1, :] / temperature  # (num_samples, vocab_size)

            if top_k is not None:
                top_k_values, top_k_idxs = torch.topk(lm_output, k=top_k, dim=-1, largest=True, sorted=True)
                lm_output[lm_output < top_k_values[:, [-1]]] = -torch.inf
            probs = F.softmax(lm_output, dim=-1)
            if num_samples > 1:
                next_input_id = torch.multinomial(probs, num_samples=1)
            else:
                _, next_input_id = torch.topk(probs, k=1, dim=-1)
            input_ids = torch.cat((input_ids, next_input_id), dim=1)

        return input_ids
