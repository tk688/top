from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from .predictor import GradientEnsemble
from .rbf_level_set import RBFLevelSet


@dataclass
class HistoryRow:
    iteration: int
    mode: str
    loss: float
    compliance: float
    volume: float
    grad_norm: float
    exact_calls: int
    predicted_steps: int
    accepted_step: float
    step_size: float
    gradient_source: str
    prediction_uncertainty: float
    angle_cosine: float
    correction_error: float


@dataclass
class DesignSnapshot:
    iteration: int
    mode: str
    alpha: torch.Tensor
    compliance: float
    volume: float


class ExactPLSOptimizer:
    """Deterministic RBF/PLSM baseline using exact FEM sensitivities."""

    def __init__(
        self,
        level_set: RBFLevelSet,
        solver,
        max_iter: int = 80,
        step_size: float = 0.035,
        move_limit: float = 0.025,
        backtracking_steps: int = 8,
        initial_project: bool = True,
        project_each_step: bool = True,
        projection_refine_steps: int = 12,
        volume_start: float | None = None,
        volume_relax: int = 0,
        snapshot_callback: Callable[[DesignSnapshot, list[HistoryRow]], None] | None = None,
        device: str = "cpu",
    ) -> None:
        self.level_set = level_set
        self.solver = solver
        self.max_iter = max_iter
        self.step_size = step_size
        self.move_limit = move_limit
        self.backtracking_steps = backtracking_steps
        self.initial_project = initial_project
        self.project_each_step = project_each_step
        self.projection_refine_steps = projection_refine_steps
        self.final_volume_target = float(self.solver.volume_target)
        self.volume_start = volume_start
        self.volume_relax = volume_relax
        self.current_projection_target = self._projection_target_for_iteration(0)
        self.snapshot_callback = snapshot_callback
        self.device = torch.device(device)
        self.rows: list[HistoryRow] = []
        self.snapshots: list[DesignSnapshot] = []

    def exact_oracle(self, alpha: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        return self.solver.evaluate_alpha(self.level_set, alpha)

    def _projection_target_for_iteration(self, iteration: int) -> float:
        if self.volume_start is None or self.volume_relax <= 0:
            return self.final_volume_target
        ratio = min(max(iteration / self.volume_relax, 0.0), 1.0)
        return (1.0 - ratio) * self.volume_start + ratio * self.final_volume_target

    def set_projection_target(self, iteration: int) -> None:
        self.current_projection_target = self._projection_target_for_iteration(iteration)

    def has_volume_continuation(self) -> bool:
        return self.volume_start is not None and self.volume_relax > 0

    def project_volume(self, alpha: torch.Tensor) -> torch.Tensor:
        """Shift the level-set bias to the current continuation volume target."""
        out = alpha.detach().clone()
        target = self.current_projection_target
        lo = out[-1] - 8.0
        hi = out[-1] + 8.0
        for _ in range(35):
            mid = 0.5 * (lo + hi)
            trial = out.clone()
            trial[-1] = mid
            if hasattr(self.solver, "projection_volume_fraction"):
                vol = self.solver.projection_volume_fraction(self.level_set, trial)
            else:
                vol = self.solver.volume_fraction(self.level_set, trial)
            if vol > target:
                hi = mid
            else:
                lo = mid
        out[-1] = 0.5 * (lo + hi)
        if self.projection_refine_steps > 0:
            lo = out[-1] - 0.2
            hi = out[-1] + 0.2
            for _ in range(self.projection_refine_steps):
                mid = 0.5 * (lo + hi)
                trial = out.clone()
                trial[-1] = mid
                vol = self.solver.volume_fraction(self.level_set, trial)
                if vol > target:
                    hi = mid
                else:
                    lo = mid
            out[-1] = 0.5 * (lo + hi)
        return out

    def normalized_descent(self, grad: torch.Tensor) -> torch.Tensor:
        direction = -grad.detach().cpu().to(torch.float64)
        return direction / (direction.norm() + 1.0e-12)

    def apply_limited_step(self, alpha: torch.Tensor, direction: torch.Tensor, step: float) -> torch.Tensor:
        delta = (step * direction).clamp(-self.move_limit, self.move_limit)
        trial = (alpha + delta).detach()
        if self.project_each_step:
            trial = self.project_volume(trial)
        return trial

    def exact_step(
        self,
        alpha: torch.Tensor,
        grad: torch.Tensor,
        metrics: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], float, int]:
        direction = self.normalized_descent(grad)
        best_alpha = alpha
        best_grad = grad
        best_metrics = metrics
        calls = 0
        step = self.step_size

        for _ in range(self.backtracking_steps):
            trial = self.apply_limited_step(alpha, direction, step)
            trial_grad, trial_metrics = self.exact_oracle(trial)
            calls += 1
            if trial_metrics["loss"] < best_metrics["loss"]:
                best_alpha = trial
                best_grad = trial_grad.detach().cpu()
                best_metrics = trial_metrics
            if trial_metrics["loss"] <= metrics["loss"]:
                return trial, trial_grad.detach().cpu(), trial_metrics, step, calls
            step *= 0.5

        if best_metrics["loss"] < metrics["loss"]:
            return best_alpha, best_grad, best_metrics, step, calls
        return alpha, grad, metrics, 0.0, calls

    def make_row(
        self,
        iteration: int,
        mode: str,
        metrics: dict[str, float],
        grad: torch.Tensor,
        exact_calls: int,
        predicted_steps: int = 0,
        accepted_step: float = 0.0,
        gradient_source: str = "exact",
        prediction_uncertainty: float = float("nan"),
        angle_cosine: float = float("nan"),
        correction_error: float = float("nan"),
    ) -> HistoryRow:
        return HistoryRow(
            iteration=iteration,
            mode=mode,
            loss=metrics["loss"],
            compliance=metrics["compliance"],
            volume=metrics["volume"],
            grad_norm=grad.norm().item(),
            exact_calls=exact_calls,
            predicted_steps=predicted_steps,
            accepted_step=accepted_step,
            step_size=self.step_size,
            gradient_source=gradient_source,
            prediction_uncertainty=prediction_uncertainty,
            angle_cosine=angle_cosine,
            correction_error=correction_error,
        )

    def record_snapshot(
        self,
        iteration: int,
        mode: str,
        alpha: torch.Tensor,
        metrics: dict[str, float],
    ) -> None:
        snapshot = DesignSnapshot(
            iteration,
            mode,
            alpha.clone(),
            metrics["compliance"],
            metrics["volume"],
        )
        self.snapshots.append(snapshot)
        if self.snapshot_callback is not None:
            self.snapshot_callback(snapshot, self.rows)

    def run(
        self, seed: int = 7, initial_alpha: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[HistoryRow]]:
        alpha = initial_alpha.detach().cpu() if initial_alpha is not None else self.level_set.initial_parameters(seed=seed).detach().cpu()
        self.set_projection_target(0)
        if self.initial_project:
            alpha = self.project_volume(alpha)

        grad, metrics = self.exact_oracle(alpha)
        grad = grad.detach().cpu()
        exact_calls = 1
        self.rows = [self.make_row(0, "exact", metrics, grad, exact_calls)]
        self.snapshots = []
        self.record_snapshot(0, "exact", alpha, metrics)

        for it in range(1, self.max_iter + 1):
            self.set_projection_target(it)
            if self.has_volume_continuation():
                alpha = self.project_volume(alpha)
                grad, metrics = self.exact_oracle(alpha)
                grad = grad.detach().cpu()
                exact_calls += 1
            alpha, grad, metrics, accepted_step, calls = self.exact_step(alpha, grad, metrics)
            exact_calls += calls
            self.rows.append(
                self.make_row(
                    it,
                    "exact",
                    metrics,
                    grad,
                    exact_calls,
                    accepted_step=accepted_step,
                    gradient_source="exact",
                )
            )
            self.record_snapshot(it, "exact", alpha, metrics)
        return alpha, self.rows

    def write_history(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(HistoryRow.__dataclass_fields__.keys()))
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row.__dict__)


class OnlinePLSOptimizer(ExactPLSOptimizer):
    """Confidence-aware online gradient prediction on top of the exact baseline."""

    def __init__(
        self,
        level_set: RBFLevelSet,
        solver,
        seq_len: int = 5,
        warmup: int = 10,
        exact_every: int = 5,
        uncertainty_threshold: float = 0.40,
        angle_threshold: float = -0.05,
        correction_tolerance: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(level_set=level_set, solver=solver, **kwargs)
        self.seq_len = seq_len
        self.warmup = warmup
        self.exact_every = exact_every
        self.uncertainty_threshold = uncertainty_threshold
        self.angle_threshold = angle_threshold
        self.correction_tolerance = correction_tolerance

        n = self.level_set.num_params
        self.feature_dim = 3 * n + 3
        self.predictor = GradientEnsemble(self.feature_dim, n, device=str(self.device))
        self.features: list[torch.Tensor] = []
        self.gradients: list[torch.Tensor] = []

    def make_feature(
        self,
        alpha: torch.Tensor,
        prev_alpha: torch.Tensor,
        prev_grad: torch.Tensor,
        metrics: dict[str, float],
    ) -> torch.Tensor:
        delta = alpha - prev_alpha
        scalars = torch.tensor([metrics["loss"], metrics["compliance"], metrics["volume"]], dtype=alpha.dtype)
        feature = torch.cat([alpha.detach().cpu(), delta.detach().cpu(), prev_grad.detach().cpu(), scalars])
        return feature / feature.abs().mean().clamp_min(1.0)

    def remember_exact(
        self,
        alpha: torch.Tensor,
        prev_alpha: torch.Tensor,
        prev_grad: torch.Tensor,
        grad: torch.Tensor,
        metrics: dict[str, float],
    ) -> None:
        self.features.append(self.make_feature(alpha, prev_alpha, prev_grad, metrics))
        self.gradients.append(grad.detach().cpu() / (grad.norm() + 1.0e-12))
        x_train, y_train = self.training_tensors()
        self.predictor.train_steps(x_train, y_train, steps=35)

    def training_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.features) <= self.seq_len:
            return torch.empty(0), torch.empty(0)
        xs = []
        ys = []
        for end in range(self.seq_len, len(self.features)):
            xs.append(torch.stack(self.features[end - self.seq_len : end], dim=0))
            ys.append(self.gradients[end])
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)

    def run(
        self, seed: int = 7, initial_alpha: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[HistoryRow]]:
        alpha = initial_alpha.detach().cpu() if initial_alpha is not None else self.level_set.initial_parameters(seed=seed).detach().cpu()
        self.set_projection_target(0)
        if self.initial_project:
            alpha = self.project_volume(alpha)

        grad, metrics = self.exact_oracle(alpha)
        grad = grad.detach().cpu()
        exact_calls = 1
        predicted_steps = 0
        prev_alpha = alpha.clone()
        prev_grad = torch.zeros_like(alpha)
        last_exact_alpha = alpha.clone()
        last_exact_grad = grad.clone()
        last_exact_metrics = metrics
        has_unverified_prediction = False
        self.rows = [self.make_row(0, "exact", metrics, grad, exact_calls, gradient_source="exact")]
        self.snapshots = []
        self.record_snapshot(0, "exact", alpha, metrics)
        self.remember_exact(alpha, prev_alpha, prev_grad, grad, metrics)

        for it in range(1, self.max_iter + 1):
            self.set_projection_target(it)
            must_check = it <= self.warmup or it % self.exact_every == 0 or len(self.features) <= self.seq_len
            correction_error = float("nan")

            if must_check:
                if has_unverified_prediction:
                    # If predicted steps have moved the design, first verify the true objective.
                    check_grad, check_metrics = self.exact_oracle(alpha)
                    exact_calls += 1
                    if last_exact_metrics["loss"] > 0:
                        correction_error = (check_metrics["loss"] - last_exact_metrics["loss"]) / last_exact_metrics["loss"]
                    if correction_error > self.correction_tolerance:
                        alpha = last_exact_alpha.clone()
                        grad = last_exact_grad.clone()
                        metrics = last_exact_metrics
                        has_unverified_prediction = False
                        self.rows.append(
                            self.make_row(
                                it,
                                "rollback",
                                metrics,
                                grad,
                                exact_calls,
                                predicted_steps,
                                gradient_source="rollback",
                                correction_error=correction_error,
                            )
                        )
                        self.record_snapshot(it, "rollback", alpha, metrics)
                        continue
                    grad = check_grad.detach().cpu()
                    metrics = check_metrics
                    has_unverified_prediction = False

                prev_alpha = alpha.clone()
                prev_grad = grad.clone()
                self.remember_exact(alpha, prev_alpha, prev_grad, grad, metrics)
                last_exact_alpha = alpha.clone()
                last_exact_grad = grad.clone()
                last_exact_metrics = metrics

                alpha, grad, metrics, accepted_step, calls = self.exact_step(alpha, grad, metrics)
                exact_calls += calls
                last_exact_alpha = alpha.clone()
                last_exact_grad = grad.clone()
                last_exact_metrics = metrics
                self.rows.append(
                    self.make_row(
                        it,
                        "exact",
                        metrics,
                        grad,
                        exact_calls,
                        predicted_steps,
                        accepted_step=accepted_step,
                        gradient_source="exact",
                        correction_error=correction_error,
                    )
                )
                self.record_snapshot(it, "exact", alpha, metrics)
                continue

            seq = torch.stack(self.features[-self.seq_len :], dim=0)
            pred = self.predictor.predict(seq, grad)
            if pred.rel_uncertainty > self.uncertainty_threshold or pred.angle_cosine < self.angle_threshold:
                alpha, grad, metrics, accepted_step, calls = self.exact_step(alpha, grad, metrics)
                exact_calls += calls
                last_exact_alpha = alpha.clone()
                last_exact_grad = grad.clone()
                last_exact_metrics = metrics
                source = "fallback_exact"
            else:
                direction = self.normalized_descent(pred.mean)
                alpha = self.apply_limited_step(alpha, direction, self.step_size)
                grad = pred.mean.detach().cpu()
                accepted_step = self.step_size
                predicted_steps += 1
                source = "pred"
                has_unverified_prediction = True

            self.rows.append(
                self.make_row(
                    it,
                    source,
                    metrics,
                    grad,
                    exact_calls,
                    predicted_steps,
                    accepted_step=accepted_step,
                    gradient_source=source,
                    prediction_uncertainty=pred.rel_uncertainty,
                    angle_cosine=pred.angle_cosine,
                )
            )
            self.record_snapshot(it, source, alpha, metrics)
        return alpha, self.rows
