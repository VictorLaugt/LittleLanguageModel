from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataclasses import dataclass

from llm.data import iter_sub_sequences


@dataclass
class TrainingConfig:
    device: torch.device
    n_epochs: int
    seq_len: int
    batch_size: int | None
    lr: float
    lr_sched_factor: float
    lr_sched_patience: int


def train(model, dataset, criterion, config, optimizer):
    epoch_loss = .0
    model.train()

    hidden_states = model.initial_hidden_states(config.device, config.batch_size)
    for input_seq, target_seq in iter_sub_sequences(dataset, config.seq_len, config.batch_size):
        optimizer.zero_grad()
        hidden_states = model.detach_hidden_states(hidden_states)

        logits, hidden_states = model(input_seq, hidden_states)
        loss = criterion(logits, target_seq)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(input_seq)

    return epoch_loss / len(dataset)


def evaluate(model, dataset, criterion, config):
    epoch_loss = .0
    model.eval()

    with torch.no_grad():
        hidden_states = model.initial_hidden_states(config.device, config.batch_size)
        for input_seq, target_seq in iter_sub_sequences(dataset, config.seq_len, config.batch_size):
            logits, hidden_states = model(input_seq, hidden_states)
            epoch_loss += criterion(logits, target_seq).item() * len(input_seq)

    return epoch_loss / len(dataset)


def training_loop(model, config, train_dataset, valid_dataset):
    print(f"Training loop configuration: {config}")
    train_dataset = train_dataset.to(config.device)
    valid_dataset = valid_dataset.to(config.device)

    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    if config.batch_size is None:
        criterion = cross_entropy
    else:
        criterion = lambda logits, target_seq: cross_entropy(logits.permute(0, 2, 1), target_seq)

    optimizer = Adam(model.parameters(), lr=config.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=config.lr_sched_factor, patience=config.lr_sched_patience)

    train_loss_values = []
    valid_loss_values = []
    learning_rate_values = []
    for epoch in range(config.n_epochs):
        train_loss = train(model, train_dataset, criterion, config, optimizer)
        valid_loss = evaluate(model, valid_dataset, criterion, config)
        lr_scheduler.step(valid_loss)
        lr = lr_scheduler.get_last_lr()[0]

        train_loss_values.append(train_loss)
        valid_loss_values.append(valid_loss)
        learning_rate_values.append(lr)
        if epoch % 100 == 0 or epoch == config.n_epochs-1:
            print(
                f"epoch {epoch:5d}/{config.n_epochs-1}: "
                f"train loss = {train_loss:.6f}, "
                f"valid loss = {valid_loss:.6f}, "
                f"learning rate = {lr}"
            )

    return train_loss_values, valid_loss_values, learning_rate_values
