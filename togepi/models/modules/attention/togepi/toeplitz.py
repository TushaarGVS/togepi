import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast


class TogepiToeplitz(nn.Module):
    def __init__(self, embedding_dim=768, num_attn_heads=12, max_position_embeddings=1025, softmax_psf_weights=True,
                 attn_actn='gelu', causal_attn=True):
        super().__init__()

        self._num_heads = num_attn_heads
        self._per_head_dim = embedding_dim // num_attn_heads
        max_length = max_position_embeddings - 1  # one position reserved for pad position
        self._training_max_length = max_length

        # randomly initialize point-spread functions, one per head
        # psf: [tok_weight, [tok_-1_weights, tok_-2_weight, ...], [..., tok_+2_weight, tok_+1_weight]]
        self.toeplitz_psfs = nn.Parameter(torch.randn(self._num_heads, 2 * max_length - 1, self._per_head_dim))
        self.attn_actn = F.gelu if attn_actn == 'gelu' else F.relu
        self._softmax_psf_weights = softmax_psf_weights
        self.post_conv_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.trunc_normal_(self.toeplitz_psfs.data, mean=0.0, std=1.0, a=0.0, b=1.0)

        self._causal = causal_attn
        if causal_attn:
            # causal_psf_mask: ignore the tokens appearing ahead of the current token.
            self.register_buffer('causal_psf_mask',
                                 torch.tensor([1] * max_length + [0] * (max_length - 1)).unsqueeze(0).unsqueeze(2))

    def forward(self, pre_proj_emb):
        # pre_proj_emb: (batch_size, max_length, num_heads * per_head_dim)
        batch_size = pre_proj_emb.shape[0]
        max_length = pre_proj_emb.shape[1]

        # padded_embeddings: (batch_size, 2 * max_length - 1, embedding_dim)
        # F.pad: pad=(padding_left, padding_right, padding_top, padding_bottom)
        pre_proj_padded_embeddings = F.pad(pre_proj_emb, pad=(0, 0, 0, max_length - 1), mode='constant')
        # pre_proj_padded_embeddings: (batch_size, num_heads, 2 * max_length - 1, per_head_dim)
        pre_proj_padded_embeddings = pre_proj_padded_embeddings.view(batch_size, 2 * max_length - 1, self._num_heads,
                                                                     self._per_head_dim).permute(0, 2, 1, 3)

        psfs_weights = self.toeplitz_psfs
        if self._training_max_length == max_length:
            causal_psf_mask = self.causal_psf_mask
        else:
            # at inference time, the max_length changes per prompt
            curr_prev_toks_psfs_weights = psfs_weights[:, : max_length, :]
            next_toks_psfs_weights = \
                psfs_weights[:, self._training_max_length: self._training_max_length + max_length - 1, :]
            psfs_weights = torch.hstack((curr_prev_toks_psfs_weights, next_toks_psfs_weights))
            causal_psf_mask = torch.tensor([1] * max_length + [0] * (max_length - 1)).unsqueeze(0).unsqueeze(2)
        if self._softmax_psf_weights:
            psfs_weights = F.softmax(psfs_weights, dim=1)
        if self._causal:
            psfs_weights = psfs_weights.masked_fill(causal_psf_mask == 0, 0)

        # fft: convolution in time domain is point-wise multiplication in frequency domain
        # note: cuFFT doesn't support half-precision for dimensions that aren't powers of two, which is the case
        # with (2 * max_length - 1)
        with autocast(enabled=False):
            psfs_fft = torch.fft.fftn(psfs_weights.float(), dim=(1, 2))
            emb_fft = torch.fft.fftn(pre_proj_padded_embeddings.float(), dim=(2, 3))
            # conv_output: (batch_size, num_heads, max_length, per_head_dim)
            conv_output = torch.real(torch.fft.ifftn(psfs_fft * emb_fft, dim=(2, 3))[:, :, :max_length, :])
        # conv_output: (batch_size, max_length, num_heads * per_head_dim)
        conv_output = self.attn_actn(conv_output).permute(0, 2, 1, 3).reshape(batch_size, max_length, -1)
        conv_emb = self.post_conv_proj(conv_output)

        return conv_emb, psfs_weights
