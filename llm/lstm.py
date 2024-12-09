import torch
import torch.nn as nn
from torch.distributions import Categorical

from llm.abstract_language_model import LanguageModelInterface


class CharLSTM(LanguageModelInterface):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def initial_hidden_states(self, device, batch_size=None):
        """Returns the default hidden states of the model to use when predicting
        without any context.
        """
        if batch_size is None:
            hidden_states = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size)
            cell_states = torch.zeros(self.lstm.num_layers, self.lstm.hidden_size)
        else:
            hidden_states = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
            cell_states = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size)
        return hidden_states.to(device), cell_states.to(device)

    @staticmethod
    def detach_hidden_states(hidden_states):
        """Detach the hidden states from their computational graph."""
        hidden_states, cell_states = hidden_states
        return hidden_states.detach(), cell_states.detach()

    def forward(self, input_seq, initial_hidden_states):
        """Apply the forward pass on input_seq.

        Arguments
        ---------
        input_seq
            [seq_len,] or [seq_len, batch_size]
        initial_hidden_states
            pair of [num_layers, hidden_dim] or [num_layers, batch_size, hidden_dim]

        Returns
        -------
        logits
            [seq_len, vocab_size] or [seq_len, batch_size, vocab_size]
        hidden_states
            pair of [num_layers, hidden_dim], [num_layers, batch_size, hidden_dim]
        """
        embedded_tokens = self.embedding(input_seq)
        outputs, hidden_states = self.lstm(embedded_tokens, initial_hidden_states)
        logits = self.linear(outputs)
        return logits, hidden_states

    def predict_argmax(self, context, n_predictions, device=torch.device('cpu')):
        """Apply one the forward pass on the given context tensor, then apply
        n_predictions times the forward pass starting from the last predicted
        token and the last hidden states. Tokens are generated deterministically
        by taking those with the largest logits.

        Arguments
        ---------
        context
            [seq_len,] or [seq_len, batch_size]
        n_predictions
            int

        Returns
        -------
        predictions
            [n_predictions,] or [n_predictions, batch_size]
        """
        batch_size = context.size(1) if context.dim() == 2 else None
        hidden_states = self.initial_hidden_states(device, batch_size)

        logits, hidden_states = self(context, hidden_states)
        last_predicted_token = logits[-1].unsqueeze(0).argmax(dim=-1)  # [1,] or [1, batch_size]

        predicted_tokens = torch.empty(n_predictions, *context.size()[1:])
        predicted_tokens[0] = last_predicted_token
        for p in range(1, n_predictions):
            logits, hidden_states = self(last_predicted_token, hidden_states)
            last_predicted_token = logits.argmax(dim=-1)
            predicted_tokens[p] = last_predicted_token

        return predicted_tokens

    def predict_proba(self, context, n_predictions, device=torch.device('cpu')):
        """Apply one the forward pass on the given context tensor, then apply
        n_predictions times the forward pass starting from the last predicted
        token and the last hidden states. Tokens are generated randomly
        according to the predicted logits.

        Arguments
        ---------
        context
            [seq_len,] or [seq_len, batch_size]
        n_predictions
            int

        Returns
        -------
        predictions
            [n_predictions,] or [n_predictions, batch_size]
        """
        batch_size = context.size(1) if context.dim() == 2 else None
        hidden_states = self.initial_hidden_states(device, batch_size)

        logits, hidden_states = self(context, hidden_states)
        last_predicted_token = Categorical(logits=logits[-1].unsqueeze(0)).sample()  # [1,] or [1, batch_size]

        predicted_tokens = torch.empty(n_predictions, *context.size()[1:])
        predicted_tokens[0] = last_predicted_token
        for p in range(1, n_predictions):
            logits, hidden_states = self(last_predicted_token, hidden_states)
            last_predicted_token = Categorical(logits=logits).sample()
            predicted_tokens[p] = last_predicted_token

        return predicted_tokens
