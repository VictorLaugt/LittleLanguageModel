import torch.nn as nn
import abc


class LanguageModelInterface(nn.Module, abc.ABC):
    @abc.abstractmethod
    def initial_hidden_states(self, device, batch_size=None):
        """Returns the default hidden states of the model to use when predicting
        without any context.

        Returns
        -------
        hidden_states
            [num_layers, hidden_dim] or [num_layers, batch_size, hidden_dim]
        """

    @staticmethod
    @abc.abstractmethod
    def detach_hidden_states(hidden_states):
        """Detach the hidden states from their computational graph."""

    @abc.abstractmethod
    def forward(self, input_seq, initial_hidden_states):
        """Apply the forward pass on input_seq."""

    @abc.abstractmethod
    def predict_argmax(self, context, n_predictions):
        """Apply one the forward pass on the given context tensor, then apply
        n_predictions times the forward pass starting from the last predicted
        token and the last hidden states. Tokens are generated deterministically
        by taking those with the largest logits.
        """

    @abc.abstractmethod
    def predict_proba(self, context, n_predictions):
        """Apply one the forward pass on the given context tensor, then apply
        n_predictions times the forward pass starting from the last predicted
        token and the last hidden states. Tokens are generated randomly
        according to the predicted logits.
        """
