import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, max_position_embeddings=1025, num_token_types=2, padding_idx=0,
                 pad_position=0, pad_token_type=0, embedding_dropout_proba=0.1):
        super().__init__()

        self._padding_idx = padding_idx
        self._pad_position = pad_position
        self._pad_token_type = pad_token_type

        self.embedding_dim = embedding_dim
        self.max_length = max_position_embeddings - 1  # one position is reserved for pad token

        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding(num_embeddings=max_position_embeddings, embedding_dim=embedding_dim,
                                    padding_idx=pad_position)
        self.type_emb = nn.Embedding(num_embeddings=num_token_types, embedding_dim=embedding_dim,
                                     padding_idx=pad_token_type)

        nn.init.xavier_uniform_(self.tok_emb.weight.data)
        self.tok_emb.weight.data[self._padding_idx] = torch.zeros(embedding_dim)
        nn.init.xavier_uniform_(self.pos_emb.weight.data)
        self.tok_emb.weight.data[self._pad_position] = torch.zeros(embedding_dim)
        nn.init.xavier_uniform_(self.type_emb.weight.data)
        self.tok_emb.weight.data[self._pad_token_type] = torch.zeros(embedding_dim)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, eps=1e-12)
        self.dropout = nn.Dropout(p=embedding_dropout_proba)

    def forward(self, input_ids, token_type_ids=None, padding_mask=None):
        # input_ids: (batch_size, max_length)
        # padding_mask: (batch_size, max_length)
        max_length = input_ids.shape[1]
        if padding_mask is None:
            # 1: no pad, 0: pad
            padding_mask = torch.where(input_ids == self._padding_idx, 0, 1)

        # position_ids: (batch_size, max_length)
        position_ids = torch.arange(max_length, dtype=torch.long,
                                    device=input_ids.device) + 1  # assuming zero is reserved for pad position
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_ids = position_ids.masked_fill(padding_mask == 0, self._pad_position)

        # token_type_ids: (batch_size, max_length)
        if token_type_ids is None:
            token_type_ids = torch.ones_like(input_ids, device=input_ids.device)  # zero is reserved for pad position
        token_type_ids = token_type_ids.masked_fill(padding_mask == 0, self._pad_token_type)

        token_embeddings = self.tok_emb(input_ids)
        position_embeddings = self.pos_emb(position_ids)
        token_type_embeddings = self.type_emb(token_type_ids)

        return self.dropout(self.layer_norm(token_embeddings + position_embeddings + token_type_embeddings))
