import numpy as np
import torch
import torch.nn as nn


class BertMultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim=768, num_attn_heads=12, max_position_embeddings=1025, attn_dropout_proba=0.1,
                 keep_prob=1.0, causal_attn=True):
        super().__init__()

        assert (embedding_dim % num_attn_heads == 0)

        self._num_heads = num_attn_heads
        self._per_head_dim = embedding_dim // num_attn_heads
        max_length = max_position_embeddings - 1

        self.wq = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.wk = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.wv = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.wq.weight.data)
        nn.init.xavier_normal_(self.wk.weight.data)
        nn.init.xavier_normal_(self.wv.weight.data)

        self._causal = causal_attn
        if causal_attn:
            self.register_buffer('causal_attn_mask',
                                 torch.tril(torch.ones(max_length, max_length)).view(1, 1, max_length, max_length))

        self.wo = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.wo.weight.data)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=attn_dropout_proba)
        self.softmax = nn.Softmax(dim=-1)

        # https://github.com/IntelLabs/academic-budget-bert/blob/main/pretraining/modeling.py#L431
        self._keep_prob = keep_prob

    @staticmethod
    def _extend_padding_mask(padding_mask, embeddings):
        # padding_mask: (batch_size, max_length)
        if padding_mask is None:
            padding_mask = torch.ones(embeddings.shape[0], embeddings.shape[1])

        extended_padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        extended_padding_mask = extended_padding_mask.to(dtype=embeddings.dtype)  # amp/fp16 compatibility
        extended_padding_mask = (1 - extended_padding_mask) * -1e4
        return extended_padding_mask

    def forward(self, embeddings, padding_mask=None):
        batch_size = embeddings.shape[0]
        max_length = embeddings.shape[1]
        embedding_dim = embeddings.shape[2]

        # embeddings: (batch_size, max_length, embedding_dim)
        # attn_mask: 1 = non-pad, 0 = pad
        # projected_*: (batch_size, max_length, num_heads * per_head_dim)
        projected_query = self.wq(embeddings)
        projected_key = self.wk(embeddings)
        projected_value = self.wv(embeddings)

        sliced_projected_query = projected_query.view(batch_size, max_length, self._num_heads,
                                                      self._per_head_dim).permute(0, 2, 1, 3)
        sliced_projected_key_tr = projected_key.view(batch_size, max_length, self._num_heads,
                                                     self._per_head_dim).permute(0, 2, 3, 1)
        sliced_projected_value = projected_value.view(batch_size, max_length, self._num_heads,
                                                      self._per_head_dim).permute(0, 2, 1, 3)

        # attn_mat: (batch_size, num_heads, max_length, max_length)
        # attn_mat: QK' / sqrt(d)
        # attn_mask: set [pad] tok attn values to -inf
        attn_mat = torch.matmul(sliced_projected_query, sliced_projected_key_tr) / np.power(embedding_dim, 0.5)
        attn_mat = attn_mat + self._extend_padding_mask(padding_mask=padding_mask, embeddings=embeddings)
        if self._causal:
            attn_mat.masked_fill_(self.causal_attn_mask[:, :, :max_length, :max_length] == 0, -1e4)
        # attn_probs: (batch_size, num_heads, max_length, max_length)
        attn_probs = self.softmax(attn_mat)
        attn_probs = self.dropout(attn_probs)

        # ctx_vectors: (batch_size, num_heads, max_length, per_head_dim)
        #    .permute: (batch_size, max_length, num_heads, per_head_dim)
        #    .view   : (batch_size, max_length, num_heads * per_head_dim)
        ctx_vectors = torch.matmul(attn_probs, sliced_projected_value).permute(0, 2, 1, 3).contiguous()
        ctx_vectors = ctx_vectors.view(batch_size, max_length, -1)

        attn_output = self.wo(ctx_vectors)
        attn_output = self.dropout(attn_output) * 1 / self._keep_prob

        return self.layer_norm(attn_output + embeddings), attn_probs
