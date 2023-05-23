import torch.nn as nn


class LinearLmHead(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768):
        super().__init__()

        self.lm_head = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        nn.init.xavier_normal_(self.lm_head.weight)
        self.lm_head.bias.data.fill_(0.0)

    def forward(self, embeddings):
        # embeddings: (batch_size, max_length, embedding_dim)
        return self.lm_head(embeddings)
