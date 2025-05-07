import torch
import torch.nn as nn
from torch.distributions import Categorical

from llm.abstract_language_model import LanguageModelInterface


nn.TransformerDecoder
nn.TransformerDecoderLayer


class CharGenerativeTransformer(LanguageModelInterface):
    def __init__(self, vocab_size, seq_length, embedding_dim, latent_dim, n_heads, n_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_length, embedding_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, latent_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def _encode(self, input_seq):
        positions = torch.arange(input_seq.size(0), device=input_seq.device)  # [seq_len,]
        if input_seq.ndim == 1:
            pos_emb = self.position_embedding(positions)  # [seq_len, embedding_dim]
            tok_emb = self.token_embedding(input_seq)  # [seq_len, embedding_dim]
        else:
            positions = positions.unsqueeze(1)  # [seq_len, 1]
            pos_emb = self.position_embedding(positions)  # [seq_len, 1, embedding_dim]
            tok_emb = self.token_embedding(input_seq)  # [seq_len, batch_size, embedding_dim]
        embedded_tokens = tok_emb + pos_emb

        mask = nn.Transformer.generate_square_subsequent_mask(input_seq.size(0))
        return self.encoder(embedded_tokens, mask=mask)  # [seq_len, ..., embedding_dim]

    def forward(self, input_seq):
        """Apply the forward pass on input_seq.

        Arguments
        ---------
        input_seq
            [seq_len,] or [seq_len, batch_size]

        Returns
        -------
        logits
            [seq_len, vocab_size] or [seq_len, batch_size, vocab_size]
        """
        encoded_tokens = self._encode(input_seq)
        return self.linear(encoded_tokens)

    def predict_argmax(self, context, n_predictions):
        context_len = context.size(0)
        tokens = torch.empty((context_len + n_predictions, *context.size()[1:]), dtype=torch.int64)
        tokens[:context_len] = context

        for p in range(context_len, context_len + n_predictions):
            input_seq = tokens[:p]
            encoded_tokens = self._encode(input_seq)
            logits = self.linear(encoded_tokens[-1])
            tokens[p] = logits.argmax(dim=-1)

        return tokens[context_len:]

    def predict_proba(self, context, n_predictions):
        context_len = context.size(0)
        tokens = torch.empty((context_len + n_predictions, *context.size()[1:]), dtype=torch.int64)
        tokens[:context_len] = context

        for p in range(context_len, context_len + n_predictions):
            input_seq = tokens[:p]
            encoded_tokens = self._encode(input_seq)
            logits = self.linear(encoded_tokens[-1])
            tokens[p] = Categorical(logits=logits).sample()

        return tokens[context_len:]
