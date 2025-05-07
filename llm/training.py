from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import abc

from llm.data import iter_sub_sequences

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
    from llm.abstract_language_model import LanguageModelInterface, ReccurentLanguageModelInterface


class TrainingLogs:
    def __init__(self):
        self._start_time = 0.

        self.train_loss = []
        self.valid_loss = []
        self.lr = []
        self.ellapsed_time = 0.

    def __enter__(self):
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.ellapsed_time = time.perf_counter() - self._start_time


class AbstractLMTrainingLoop(abc.ABC):
    def __init__(
        self,
        device: torch.device,
        n_epochs: int,
        sequence_length: int,
        batch_size: Optional[int],
        train_dataset: torch.LongTensor,
        valid_dataset: torch.LongTensor,
        model: LanguageModelInterface,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler]=None
    ) -> None:
        self.device = device
        self.n_epochs = n_epochs
        self.seq_length = sequence_length
        self.batch_size = batch_size

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if self.batch_size is None:
            self.criterion = self._criterion_no_batch
        else:
            self.criterion = self._criterion_batch

        if self.scheduler is None:
            self.update_lr = self._update_lr_constant
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.update_lr = self._update_lr_reduce_on_plateau
        else:
            self.update_lr = self._update_lr

    def _criterion_no_batch(self, logits, target_seq):
        return self.cross_entropy(logits, target_seq)

    def _criterion_batch(self, logits, target_seq):
        return self.cross_entropy(logits.permute(0, 2, 1), target_seq)

    def _update_lr(self, _: float) -> float:
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]

    def _update_lr_reduce_on_plateau(self, valid_loss: float) -> float:
        self.scheduler.step(valid_loss)
        return self.scheduler.get_last_lr()[0]

    def _update_lr_constant(self, _: float) -> float:
        return self.optimizer.param_groups[0]['lr']

    def run(self, logs: Optional[TrainingLogs]=None) -> TrainingLogs:
        logs = logs or TrainingLogs()
        self.train_dataset = self.train_dataset.to(self.device)
        self.valid_dataset = self.valid_dataset.to(self.device)
        self.model.to(self.device)

        print(f"Training on device {self.device}")
        for epoch in range(self.n_epochs):
            train_loss = self.train()
            valid_loss = self.evaluate()
            lr = self.update_lr(valid_loss)

            logs.train_loss.append(train_loss)
            logs.valid_loss.append(valid_loss)
            logs.lr.append(lr)
            # if epoch % 100 == 0 or epoch == self.n_epochs-1:
            if epoch % 20 == 0 or epoch == self.n_epochs-1:
                print(
                    f"epoch {epoch:5d}/{self.n_epochs-1}: "
                    f"train loss = {train_loss:.6f}, "
                    f"valid loss = {valid_loss:.6f}, "
                    f"learning rate = {lr}"
                )

        return logs

    @abc.abstractmethod
    def train(self) -> float:
        pass

    @abc.abstractmethod
    def evaluate(self) -> float:
        pass


class ReccurentLMTrainingLoop(AbstractLMTrainingLoop):
    model: ReccurentLanguageModelInterface

    def train(self) -> float:
        self.model.train()

        epoch_loss = .0
        token_count = 0
        hidden_states = self.model.initial_hidden_states(self.device, self.batch_size)
        for input_seq, target_seq in iter_sub_sequences(self.train_dataset, self.seq_length, self.batch_size):
            self.optimizer.zero_grad()
            hidden_states = self.model.detach_hidden_states(hidden_states)

            logits, hidden_states = self.model(input_seq, hidden_states)
            loss = self.criterion(logits, target_seq)

            loss.backward()
            self.optimizer.step()

            n_tokens = input_seq.numel()
            epoch_loss += loss.item() * n_tokens
            token_count += n_tokens

        return epoch_loss / token_count

    def evaluate(self) -> float:
        self.model.eval()

        epoch_loss = .0
        token_count = 0
        with torch.no_grad():
            hidden_states = self.model.initial_hidden_states(self.device, self.batch_size)
            for input_seq, target_seq in iter_sub_sequences(self.valid_dataset, self.seq_length, self.batch_size):
                logits, hidden_states = self.model(input_seq, hidden_states)
                n_tokens = input_seq.numel()
                epoch_loss += self.criterion(logits, target_seq).item() * n_tokens
                token_count += n_tokens

        return epoch_loss / token_count

class TransformerLMTrainingLoop(AbstractLMTrainingLoop):
    def train(self) -> float:
        self.model.train()

        epoch_loss = .0
        token_count = 0
        for input_seq, target_seq in iter_sub_sequences(self.train_dataset, self.seq_length, self.batch_size):
            self.optimizer.zero_grad()

            logits = self.model(input_seq)
            loss = self.criterion(logits, target_seq)

            loss.backward()
            self.optimizer.step()

            n_tokens = input_seq.numel()
            epoch_loss += loss.item() * n_tokens
            token_count += n_tokens

        return epoch_loss / token_count

    def evaluate(self) -> float:
        self.model.eval()

        epoch_loss = .0
        token_count = 0
        with torch.no_grad():
            for input_seq, target_seq in iter_sub_sequences(self.valid_dataset, self.seq_length, self.batch_size):
                logits = self.model(input_seq)
                n_tokens = input_seq.numel()
                epoch_loss += self.criterion(logits, target_seq).item() * n_tokens
                token_count += n_tokens

        return epoch_loss / token_count
