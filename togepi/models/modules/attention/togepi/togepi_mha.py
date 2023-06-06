import torch.nn as nn

from togepi.models.modules.attention.togepi.sparse import TogepiSparse
from togepi.models.modules.attention.togepi.toeplitz import TogepiToeplitz


class TogepiMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_attn_heads=12, max_position_embeddings=1025, softmax_psf_weights=True,
                 attn_actn='gelu', use_spectral_norm=True, sparse_init_dens=None, num_power_iters=1,
                 attn_dropout_proba=0.1, keep_prob=1.0, causal_attn=True):
        super().__init__()

        assert (embedding_dim % num_attn_heads == 0)

        # out_features: (num_heads * per_head_dim)
        self.pre_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.pre_proj.weight.data)

        self.toeplitz = TogepiToeplitz(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads,
                                       max_position_embeddings=max_position_embeddings,
                                       softmax_psf_weights=softmax_psf_weights, attn_actn=attn_actn,
                                       causal_attn=causal_attn)
        self.sparse = TogepiSparse(embedding_dim=embedding_dim, max_position_embeddings=max_position_embeddings,
                                   sparse_init_dens=sparse_init_dens, causal_attn=causal_attn)
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            # spectral normalization for numeric stability: https://arxiv.org/abs/1802.05957
            # https://jonathan-hui.medium.com/gan-spectral-normalization-893b6a4e8f53
            nn.utils.spectral_norm(self.sparse, name='sparse_mat', n_power_iterations=num_power_iters)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=attn_dropout_proba)

        # https://github.com/IntelLabs/academic-budget-bert/blob/main/pretraining/modeling.py#L431
        self._keep_prob = keep_prob

    def forward(self, embeddings, padding_mask=None):
        # embeddings: (batch_size, max_length, embedding_dim)
        # padding_mask: (batch_size, max_length)
        # pre_proj_emb: (batch_size, max_length, num_heads * per_head_dim)
        pre_proj_emb = self.pre_proj(embeddings)
        if padding_mask is not None:
            # 1: no pad, 0: pad
            pre_proj_emb.masked_fill_(padding_mask.unsqueeze(2) == 0, 0)

        # conv_emb: (batch_size, max_length, embedding_dim)
        # psfs_weights: (num_heads, 2 * max_length - 1, per_head_dim)
        conv_emb, psfs_weights = self.toeplitz(pre_proj_emb)
        # sparse_emb: (batch_size, max_length, embedding_dim)
        # sparse_mat: (mqx_length, max_length)
        sparse_emb, sparse_mat = self.sparse(pre_proj_emb, padding_mask=padding_mask)
        togepi_emb = self.dropout(conv_emb + sparse_emb) * 1 / self._keep_prob

        return self.layer_norm(togepi_emb + embeddings), (psfs_weights, sparse_mat)
