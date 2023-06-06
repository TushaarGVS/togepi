import torch.nn as nn

from togepi.models.modules.attention.bert_mha import BertMultiHeadAttention
from togepi.models.modules.attention.togepi.togepi_mha import TogepiMultiHeadAttention
from togepi.models.modules.embedding import Embedding
from togepi.models.modules.positional_networks.ffnn import PositionalFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=1024, num_attn_heads=12, max_position_embeddings=1025,
                 use_togepi_mha=True, softmax_psf_weights=True, attn_actn='gelu', use_spectral_norm=True,
                 sparse_init_dens=None, num_power_iters=1, attn_dropout_proba=0.1, embedding_dropout_proba=0.1,
                 keep_prob=1.0, causal_attn=True):
        super().__init__()

        self.use_togepi_mha = use_togepi_mha
        if use_togepi_mha:
            self.mha = TogepiMultiHeadAttention(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads,
                                                max_position_embeddings=max_position_embeddings,
                                                softmax_psf_weights=softmax_psf_weights, attn_actn=attn_actn,
                                                use_spectral_norm=use_spectral_norm, sparse_init_dens=sparse_init_dens,
                                                num_power_iters=num_power_iters, attn_dropout_proba=attn_dropout_proba,
                                                keep_prob=keep_prob, causal_attn=causal_attn)
        else:
            self.mha = BertMultiHeadAttention(embedding_dim=embedding_dim, num_attn_heads=num_attn_heads,
                                              max_position_embeddings=max_position_embeddings,
                                              attn_dropout_proba=attn_dropout_proba, keep_prob=keep_prob,
                                              causal_attn=causal_attn)
        self.pos_net = PositionalFeedForward(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                             embedding_dropout_proba=embedding_dropout_proba, keep_prob=keep_prob)

    def forward(self, embeddings, padding_mask=None):
        # embeddings: (batch_size, max_length, embedding_dim)
        mha_emb, attn_filters = self.mha(embeddings, padding_mask=padding_mask)
        return self.pos_net(mha_emb), attn_filters


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, num_token_types=2, hidden_dim=1024, num_enc_layers=12,
                 num_attn_heads=12, max_position_embeddings=1025, use_togepi_mha=True, softmax_psf_weights=True,
                 attn_actn='gelu', use_spectral_norm=True, sparse_init_dens=None, num_power_iters=1,
                 attn_dropout_proba=0.1, embedding_dropout_proba=0.1, keep_prob=1.0, causal_attn=True, padding_idx=0,
                 pad_position=0, pad_token_type=0):
        super().__init__()

        self.emb = Embedding(vocab_size=vocab_size, embedding_dim=embedding_dim,
                             max_position_embeddings=max_position_embeddings, num_token_types=num_token_types,
                             padding_idx=padding_idx, pad_position=pad_position, pad_token_type=pad_token_type,
                             embedding_dropout_proba=embedding_dropout_proba)
        enc_layers = []
        for _ in range(num_enc_layers):
            enc_layers.append(
                EncoderLayer(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_attn_heads=num_attn_heads,
                             max_position_embeddings=max_position_embeddings, use_togepi_mha=use_togepi_mha,
                             softmax_psf_weights=softmax_psf_weights, attn_actn=attn_actn,
                             use_spectral_norm=use_spectral_norm, sparse_init_dens=sparse_init_dens,
                             num_power_iters=num_power_iters, attn_dropout_proba=attn_dropout_proba,
                             embedding_dropout_proba=embedding_dropout_proba, keep_prob=keep_prob,
                             causal_attn=causal_attn))
        self.enc_layers = nn.ModuleList(enc_layers)

    def forward(self, input_ids, token_type_ids=None, padding_mask=None):
        embeddings = self.emb(input_ids=input_ids, token_type_ids=token_type_ids, padding_mask=padding_mask)
        emb_all_layers, attn_filters_or_psfs_all_layers = [], []
        for enc_layer in self.enc_layers:
            embeddings, attn_filters = enc_layer(embeddings, padding_mask=padding_mask)
            emb_all_layers.append(embeddings)
            if attn_filters is not None:
                attn_filters_or_psfs_all_layers.append(attn_filters)

        if len(attn_filters_or_psfs_all_layers) == 0:
            attn_filters_or_psfs_all_layers = None
        return emb_all_layers, attn_filters_or_psfs_all_layers
