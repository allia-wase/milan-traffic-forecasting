"""LSTM and TCN one-step-ahead forecasters with a shared early-stopping trainer."""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMForecaster(nn.Module):
    """Stacked LSTM over the input window; the last hidden state is mapped to the next value."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x.unsqueeze(-1))
        return self.head(output[:, -1]).squeeze(-1)


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection (Bai et al., 2018)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Padding only on the left keeps each output from seeing later time steps.
        h = self.dropout(F.relu(self.conv1(F.pad(x, (self.left_padding, 0)))))
        h = self.dropout(F.relu(self.conv2(F.pad(h, (self.left_padding, 0)))))
        return F.relu(h + self.residual(x))


class TCNForecaster(nn.Module):
    """Stack of temporal blocks with dilations 1, 2, 4, ...; the last time step feeds a linear head."""

    def __init__(self, channels: int = 32, levels: int = 6, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dilations = [2 ** i for i in range(levels)]
        self.kernel_size = kernel_size
        blocks = [
            TemporalBlock(1 if i == 0 else channels, channels, kernel_size, dilation, dropout)
            for i, dilation in enumerate(self.dilations)
        ]
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x.unsqueeze(1))[:, :, -1]).squeeze(-1)


ARCHITECTURES = {"lstm": LSTMForecaster, "tcn": TCNForecaster}


@dataclass(frozen=True)
class TrainConfig:
    lookback: int = 144
    batch_size: int = 128
    learning_rate: float = 1e-3
    max_epochs: int = 40
    patience: int = 6
    seed: int = 42
    model: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainResult:
    best_epoch: int
    epochs_run: int
    best_val_loss: float
    seconds: float
    train_losses: list[float]
    val_losses: list[float]


def build_model(architecture: str, cfg: TrainConfig) -> nn.Module:
    set_seed(cfg.seed)
    model = ARCHITECTURES[architecture](**cfg.model)
    if architecture == "tcn" and model.receptive_field < cfg.lookback:
        raise ValueError(f"TCN receptive field {model.receptive_field} is shorter than lookback {cfg.lookback}")
    return model


def train_model(
    model: nn.Module,
    train_data: tuple[np.ndarray, np.ndarray],
    val_data: tuple[np.ndarray, np.ndarray],
    cfg: TrainConfig,
) -> TrainResult:
    """Adam on MSE of the scaled series, keeping the weights from the epoch with the lowest validation loss."""
    x_train, y_train = (torch.from_numpy(a) for a in train_data)
    x_val, y_val = (torch.from_numpy(a) for a in val_data)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(cfg.seed)

    best_state, best_loss, best_epoch = None, float("inf"), 0
    train_losses, val_losses = [], []
    started = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        running = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start:start + cfg.batch_size]
            optimiser.zero_grad()
            loss = loss_fn(model(x_train[batch]), y_train[batch])
            loss.backward()
            optimiser.step()
            running += loss.item() * len(batch)
        train_losses.append(running / len(order))

        val_loss = float(loss_fn(torch.from_numpy(predict(model, x_val.numpy())), y_val))
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_state, best_loss, best_epoch = copy.deepcopy(model.state_dict()), val_loss, epoch
        elif epoch - best_epoch >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(best_epoch, len(val_losses), best_loss, time.perf_counter() - started, train_losses, val_losses)


@torch.no_grad()
def predict(model: nn.Module, inputs: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(inputs)
    return torch.cat([model(tensor[i:i + batch_size]) for i in range(0, len(tensor), batch_size)]).numpy()
