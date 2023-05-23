import torch
import torch.nn as nn
import torch.nn.functional as F

from togepi.models.modules.attention.togepi.sparse import TogepiSparse


class TogepiMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_attn_heads=12, max_position_embeddings=1025, softmax_psf_weights=True,
                 attn_actn='gelu', use_spectral_norm=True, sparse_init_dens=None, num_power_iters=1,
                 attn_dropout_proba=0.1, keep_prob=1.0, causal_attn=True):
        super().__init__()

        assert (embedding_dim % num_attn_heads == 0)

        self._num_heads = num_attn_heads
        self._per_head_dim = embedding_dim // num_attn_heads
        max_length = max_position_embeddings - 1  # one position reserved for pad position
        self._training_max_length = max_length

        # out_features: (num_heads * per_head_dim)
        self.pre_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.pre_proj.weight.data)

        # randomly initialize point-spread functions, one per head
        # psf: [tok_weight, [tok_-1_weights, tok_-2_weight, ...], [..., tok_+2_weight, tok_+1_weight]]
        self.toeplitz_psfs = nn.Parameter(torch.randn(self._num_heads, 2 * max_length - 1, self._per_head_dim))
        self.attn_actn = F.gelu if attn_actn == 'gelu' else F.relu
        self._softmax_psf_weights = softmax_psf_weights
        self.post_conv_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.toeplitz_psfs.data)
        nn.init.xavier_normal_(self.post_conv_proj.weight.data)

        self.sparse = TogepiSparse(embedding_dim=embedding_dim, max_position_embeddings=max_position_embeddings,
                                   sparse_init_dens=sparse_init_dens, causal_attn=causal_attn)
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            # spectral normalization for numeric stability: https://arxiv.org/abs/1802.05957
            # https://jonathan-hui.medium.com/gan-spectral-normalization-893b6a4e8f53
            nn.utils.spectral_norm(self.sparse, name='sparse_mat', n_power_iterations=num_power_iters)

        self._causal = causal_attn
        if causal_attn:
            # causal_psf_mask: ignore the tokens appearing ahead of the current token.
            self.register_buffer('causal_psf_mask',
                                 torch.tensor([1] * max_length + [0] * (max_length - 1)).unsqueeze(0).unsqueeze(2))

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=attn_dropout_proba)

        # https://github.com/IntelLabs/academic-budget-bert/blob/main/pretraining/modeling.py#L431
        self._keep_prob = keep_prob

    def forward(self, embeddings, padding_mask=None):
        # embeddings: (batch_size, max_length, embedding_dim)
        # padding_mask: (batch_size, max_length)
        batch_size = embeddings.shape[0]
        max_length = embeddings.shape[1]

        # pre_proj_emb: (batch_size, max_length, num_heads * per_head_dim)
        pre_proj_emb = self.pre_proj(embeddings)
        if padding_mask is not None:
            # 1: no pad, 0: pad
            pre_proj_emb.masked_fill_(padding_mask.unsqueeze(2) == 0, 0)

        # padded_embeddings: (batch_size, 2 * max_length - 1, embedding_dim)
        # F.pad: pad=(padding_left, padding_right, padding_top, padding_bottom)
        pre_proj_padded_embeddings = F.pad(pre_proj_emb, pad=(0, 0, 0, max_length - 1), mode='constant')
        # pre_proj_padded_embeddings: (batch_size, num_heads, 2 * max_length - 1, per_head_dim)
        pre_proj_padded_embeddings = \
            pre_proj_padded_embeddings.view(batch_size, 2 * max_length - 1, self._num_heads,
                                            self._per_head_dim).permute(0, 2, 1, 3)

        psfs_weights = self.toeplitz_psfs
        if self._causal:
            if self._training_max_length == max_length:
                psfs_weights = psfs_weights.masked_fill(self.causal_psf_mask == 0, 0)
            else:
                # at inference time, the max_length changes per prompt
                causal_psf_mask = torch.tensor([1] * max_length + [0] * (max_length - 1)).unsqueeze(0).unsqueeze(2)
                psfs_weights = psfs_weights.masked_fill(causal_psf_mask == 0, 0)
        if self._softmax_psf_weights:
            psfs_weights = F.softmax(psfs_weights, dim=1)

        # fft: convolution in time domain is point-wise multiplication in frequency domain
        psfs_fft = torch.fft.fftn(psfs_weights, dim=(1, 2))
        emb_fft = torch.fft.fftn(pre_proj_padded_embeddings, dim=(2, 3))
        # conv_output: (batch_size, num_heads, max_length, per_head_dim)
        conv_output = torch.real(torch.fft.ifftn(psfs_fft * emb_fft, dim=(2, 3))[:, :, :max_length, :])
        # conv_output: (batch_size, max_length, num_heads * per_head_dim)
        conv_output = self.attn_actn(conv_output).permute(0, 2, 1, 3).reshape(batch_size, max_length, -1)
        conv_emb = self.post_conv_proj(conv_output)

        # sparse_emb: (batch_size, max_length, embedding_dim)
        sparse_emb = self.sparse(pre_proj_emb, padding_mask=padding_mask)
        togepi_emb = self.dropout(conv_emb + sparse_emb) * 1 / self._keep_prob

        return self.layer_norm(togepi_emb + embeddings), None  # None added to keep consistent with bert output format
