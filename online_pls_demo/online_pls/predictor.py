from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class GradientGRU(nn.Module):
    def __init__(self, feature_dim: int, output_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(seq)
        return self.head(h[-1])


@dataclass
class Prediction:
    mean: torch.Tensor
    rel_uncertainty: float
    angle_cosine: float


class GradientEnsemble:
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        members: int = 3,
        hidden_dim: int = 96,
        lr: float = 2.0e-3,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.models = [
            GradientGRU(feature_dim, output_dim, hidden_dim).to(self.device) for _ in range(members)
        ]
        self.opts = [torch.optim.Adam(model.parameters(), lr=lr) for model in self.models]

    def train_steps(self, x: torch.Tensor, y: torch.Tensor, steps: int = 60) -> None:
        if x.numel() == 0:
            return
        x = x.to(self.device)
        y = y.to(self.device)
        for model, opt in zip(self.models, self.opts):
            model.train()
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

    @torch.no_grad()
    def predict(self, seq: torch.Tensor, last_grad: torch.Tensor) -> Prediction:
        preds = []
        for model in self.models:
            model.eval()
            preds.append(model(seq[None, :, :].to(self.device)).squeeze(0))
        stacked = torch.stack(preds, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0).norm()
        rel_unc = (std / (mean.norm() + 1.0e-8)).item()
        angle = (
            torch.dot(mean, last_grad.to(self.device))
            / ((mean.norm() + 1.0e-8) * (last_grad.to(self.device).norm() + 1.0e-8))
        ).item()
        return Prediction(mean=mean.detach().cpu(), rel_uncertainty=rel_unc, angle_cosine=angle)

