import torch
import torch.nn as nn


class TogepiSparse(nn.Module):
    def __init__(self, embedding_dim=768, max_position_embeddings=1025, sparse_init_dens=None, causal_attn=True):
        super().__init__()

        max_length = max_position_embeddings - 1  # one position reserved for pad position
        self._training_max_length = max_length

        if sparse_init_dens is not None:
            num_nonzero = int(max_length * max_length * sparse_init_dens)
            sparse_idxs = torch.randint(0, max_length, (num_nonzero, 2))
            sparse_vals = torch.randn(num_nonzero)
            self.sparse_mat = nn.Parameter(
                torch.sparse_coo_tensor(sparse_idxs.t(), sparse_vals.abs(), size=(max_length, max_length)).to_dense())
        else:
            self.sparse_mat = nn.Parameter(torch.randn(max_length, max_length))
            nn.init.xavier_normal_(self.sparse_mat.data)

        self._causal = causal_attn
        if causal_attn:
            self.register_buffer('causal_sparse_mask', torch.tril(torch.ones(max_length, max_length)))

        self.pre_sparse_proj = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.pre_sparse_proj.weight.data)

    def forward(self, pre_proj_emb, padding_mask=None):
        # pre_proj_emb: (batch_size, max_length, embedding_dim)
        # padding_mask: (batch_size, max_length)
        max_length = pre_proj_emb.shape[1]

        sparse_data = self.sparse_mat
        if self._training_max_length != max_length:
            sparse_data = sparse_data[:max_length, :max_length]
        if self._causal:
            sparse_data = sparse_data.masked_fill(self.causal_sparse_mask[:max_length, :max_length] == 0, 0)
        pre_sparse_emb = self.pre_sparse_proj(pre_proj_emb)
        if padding_mask is not None:
            pre_sparse_emb.masked_fill_(padding_mask.unsqueeze(2) == 0, 0)
        return torch.matmul(sparse_data, pre_sparse_emb), sparse_data
