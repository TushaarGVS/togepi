import torch
from torch import nn

from togepi.utils.utils import device_mapper


class SegmentEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_token_types=2, pad_token_type=0, device=torch.device('cpu')):
        super().__init__()

        self._device = device
        self._num_token_types = num_token_types  # excludes [PAD] token

        # Initialize interval segment (token_type) embeddings, and initialize weights using the
        # Xavier distribution.
        num_token_types = num_token_types + 1  # [PAD] token
        self._segment_embedding = nn.Embedding(num_embeddings=num_token_types, embedding_dim=embedding_dim,
                                               padding_idx=self._num_token_types, device=device)
        nn.init.xavier_normal_(self._segment_embedding.weight.data)  # note: resets padding_idx weights
        self._segment_embedding.weight.data[pad_token_type] = torch.zeros(embedding_dim)

    @property
    def type(self):
        return 'segment_embedding'

    def forward(self, token_type_ids):
        # token_type_ids: (batch_size, max_length)
        token_type_ids = device_mapper(token_type_ids, self._device)

        # segment_embeddings: (batch_size, max_length, embedding_dim)
        segment_embeddings = self._segment_embedding(token_type_ids)

        return segment_embeddings
