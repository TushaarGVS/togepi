import torch.nn as nn


class PositionalFeedForward(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=1024, embedding_dropout_proba=0.1, keep_prob=1.0):
        super().__init__()

        self.dense_intermediate = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.dense_final = nn.Linear(in_features=hidden_dim, out_features=embedding_dim)
        nn.init.xavier_normal_(self.dense_intermediate.weight.data)
        nn.init.xavier_normal_(self.dense_final.weight.data)

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.dropout = nn.Dropout(p=embedding_dropout_proba)
        self.gelu = nn.GELU()

        # # https://github.com/IntelLabs/academic-budget-bert/blob/main/pretraining/modeling.py#L431
        self._keep_prob = keep_prob

    def forward(self, embeddings):
        # embeddings: (batch_size, max_length, embedding_dim)
        # emb: (batch_size, max_length, hidden_dim)
        emb = self.gelu(self.dense_intermediate(embeddings))
        # emb: (batch_size, max_length, embedding_dim)
        emb = self.dropout(self.dense_final(emb)) * 1 / self._keep_prob

        return self.layer_norm(emb + embeddings)
