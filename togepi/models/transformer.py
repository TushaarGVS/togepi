import torch
import torch.nn as nn
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

    def forward(self, input_ids, token_type_ids=None, padding_mask=None):
        emb_all_layers, attn_filters_all_layers = self.enc(input_ids=input_ids, token_type_ids=token_type_ids,
                                                           padding_mask=padding_mask)
        last_emb = emb_all_layers[-1]
        if self._use_lm_head:
            lm_output = self.lm_head(last_emb)
        else:
            # weight tying: https://github.com/openai/gpt-2/blob/master/src/model.py#L169
            # last_emb: (batch_size, max_length, embedding_dim)
            # tok_emb_weight: (vocab_size, embedding_dim)
            lm_output = torch.matmul(last_emb, self.enc.emb.tok_emb.weight.data.transpose(0, 1))
        return lm_output

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, num_samples=1, top_k=None):
        pass
