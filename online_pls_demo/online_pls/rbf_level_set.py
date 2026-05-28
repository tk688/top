from __future__ import annotations

import numpy as np
import torch


class RBFLevelSet:
    """RBF-parametric level-set field on a regular 2D design grid."""

    def __init__(
        self,
        nx: int = 28,
        ny: int = 14,
        cx: int = 7,
        cy: int = 4,
        sigma: float = 0.18,
        transition_width: float = 0.18,
        rho_min: float = 1.0e-3,
        init_strategy: str = "holes",
        domain_aspect: float = 2.0,
        length: float | None = None,
        height: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.cx = cx
        self.cy = cy
        self.sigma = sigma
        self.transition_width = transition_width
        self.rho_min = rho_min
        self.init_strategy = init_strategy
        self.height = height
        self.length = float(length if length is not None else domain_aspect * height)
        self.domain_aspect = self.length / self.height
        self.device = torch.device(device)

        xs = torch.linspace(0.0, 1.0, nx, device=self.device)
        ys = torch.linspace(0.0, 1.0, ny, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.points_unit = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        self.points = torch.stack([self.length * xx.reshape(-1), self.height * yy.reshape(-1)], dim=1)

        cxs = torch.linspace(0.0, 1.0, cx, device=self.device)
        cys = torch.linspace(0.0, 1.0, cy, device=self.device)
        cyy, cxx = torch.meshgrid(cys, cxs, indexing="ij")
        centers = torch.stack([self.length * cxx.reshape(-1), self.height * cyy.reshape(-1)], dim=1)
        self.centers = centers
        self.num_weights = centers.shape[0]
        self.num_params = self.num_weights + 1

        dist2 = torch.cdist(self.points, self.centers).pow(2)
        self.basis = torch.exp(-dist2 / (2.0 * sigma * sigma))
        self.points_unit_np = self.points_unit.detach().cpu().numpy()
        self.points_np = self.points.detach().cpu().numpy()
        self.centers_np = self.centers.detach().cpu().numpy()
        self.basis_np = self.basis.detach().cpu().numpy()

    def initial_parameters(self, seed: int = 7) -> torch.Tensor:
        if self.init_strategy == "holes":
            return self._hole_initial_parameters(seed)
        if self.init_strategy == "truss":
            return self._truss_initial_parameters(seed)
        raise ValueError(f"Unknown init_strategy: {self.init_strategy!r}")

    def _hole_initial_parameters(self, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        xy = self.points_np

        holes = np.array(
            [
                (0.16, 0.76, 0.095, 0.105, -0.18),
                (0.16, 0.24, 0.095, 0.105, 0.18),
                (0.34, 0.50, 0.080, 0.120, 0.00),
                (0.52, 0.76, 0.090, 0.105, -0.18),
                (0.52, 0.24, 0.090, 0.105, 0.18),
                (0.70, 0.50, 0.080, 0.120, 0.00),
                (0.86, 0.50, 0.065, 0.110, 0.00),
            ],
            dtype=np.float64,
        )
        distances = []
        for cx, cy, ax, ay, angle in holes:
            cx *= self.length
            cy *= self.height
            ax *= self.length
            ay *= self.height
            dx = xy[:, 0] - cx
            dy = xy[:, 1] - cy
            ca = np.cos(angle)
            sa = np.sin(angle)
            xr = ca * dx + sa * dy
            yr = -sa * dx + ca * dy
            # 椭圆孔的 signed-distance 近似：小于 0 表示孔洞内部。
            distances.append((np.sqrt((xr / ax) ** 2 + (yr / ay) ** 2) - 1.0) * min(ax, ay))
        distances = np.column_stack(distances)
        signed = np.min(distances, axis=1)

        # 归一化后做最小二乘拟合，避免初始 alpha 过大导致 Heaviside 全饱和。
        band = 0.12 * self.height
        target = np.clip(signed, -band, band) / band
        target += 0.010 * rng.standard_normal(target.shape[0])
        return self._fit_target_field(target)

    def _truss_initial_parameters(self, seed: int) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        xy = self.points_unit_np
        bars = [
            ((0.00, 0.82), (0.88, 0.62), 0.075),
            ((0.00, 0.18), (0.88, 0.38), 0.075),
            ((0.02, 0.82), (0.34, 0.52), 0.065),
            ((0.02, 0.18), (0.34, 0.48), 0.065),
            ((0.34, 0.52), (0.58, 0.76), 0.060),
            ((0.34, 0.48), (0.58, 0.24), 0.060),
            ((0.58, 0.76), (0.89, 0.52), 0.060),
            ((0.58, 0.24), (0.89, 0.48), 0.060),
            ((0.86, 0.38), (1.00, 0.50), 0.080),
            ((0.86, 0.62), (1.00, 0.50), 0.080),
        ]
        signed = np.full(xy.shape[0], -1.0, dtype=np.float64)
        for start, end, half_width in bars:
            distance = self._distance_to_segment(xy, np.asarray(start), np.asarray(end))
            signed = np.maximum(signed, half_width - distance)

        # Positive means material; negative means void. This truss-like initial
        # field places the optimizer in the same basin as classical cantilever
        # level-set examples with separated holes and diagonal webs.
        target = np.clip(signed, -0.12, 0.12) / 0.12
        target += 0.015 * rng.standard_normal(target.shape[0])
        return self._fit_target_field(target)

    def _fit_target_field(self, target: np.ndarray) -> torch.Tensor:
        design = np.column_stack([self.basis_np, np.ones(self.basis_np.shape[0])])
        coeffs, *_ = np.linalg.lstsq(design, target, rcond=1.0e-5)
        return torch.as_tensor(coeffs, dtype=torch.get_default_dtype(), device=self.device)

    @staticmethod
    def _distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        segment = end - start
        denom = float(np.dot(segment, segment))
        if denom == 0.0:
            return np.linalg.norm(points - start[None, :], axis=1)
        t = np.clip(((points - start[None, :]) @ segment) / denom, 0.0, 1.0)
        closest = start[None, :] + t[:, None] * segment[None, :]
        return np.linalg.norm(points - closest, axis=1)

    def phi(self, alpha: torch.Tensor) -> torch.Tensor:
        weights = alpha[:-1]
        bias = alpha[-1]
        values = self.basis @ weights + bias
        return values.reshape(self.ny, self.nx)

    def density(self, alpha: torch.Tensor) -> torch.Tensor:
        phi = self.phi(alpha)
        delta = self.transition_width
        rho_min = self.rho_min

        middle = (
            0.75
            * (1.0 - rho_min)
            * (phi / delta - phi.pow(3) / (3.0 * delta**3))
            + 0.5 * (1.0 + rho_min)
        )
        return torch.where(
            phi <= -delta,
            torch.full_like(phi, rho_min),
            torch.where(phi >= delta, torch.ones_like(phi), middle),
        )

    def basis_at_numpy(self, points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - self.centers_np[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        return np.exp(-dist2 / (2.0 * self.sigma * self.sigma))

    def phi_numpy_at(self, alpha: np.ndarray, points: np.ndarray, chunk_size: int = 20000) -> np.ndarray:
        values = np.empty(points.shape[0], dtype=np.float64)
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            basis = self.basis_at_numpy(points[start:end])
            values[start:end] = basis @ alpha[:-1] + alpha[-1]
        return values

    def basis_weighted_sum_numpy(
        self, points: np.ndarray, weights: np.ndarray, chunk_size: int = 20000
    ) -> np.ndarray:
        out = np.zeros(self.num_weights, dtype=np.float64)
        for start in range(0, points.shape[0], chunk_size):
            end = min(start + chunk_size, points.shape[0])
            basis = self.basis_at_numpy(points[start:end])
            out += basis.T @ weights[start:end]
        return out

    def phi_numpy(self, alpha: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
        active_basis = self.basis_np if basis is None else basis
        return active_basis @ alpha[:-1] + alpha[-1]

    def density_numpy(self, alpha: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
        phi = self.phi_numpy(alpha, basis)
        return self.heaviside_numpy(phi)

    def heaviside_numpy(self, phi: np.ndarray) -> np.ndarray:
        delta = self.transition_width
        rho_min = self.rho_min
        middle = (
            0.75
            * (1.0 - rho_min)
            * (phi / delta - phi**3 / (3.0 * delta**3))
            + 0.5 * (1.0 + rho_min)
        )
        return np.where(phi <= -delta, rho_min, np.where(phi >= delta, 1.0, middle))

    def heaviside_derivative_numpy(self, phi: np.ndarray) -> np.ndarray:
        delta = self.transition_width
        rho_min = self.rho_min
        middle = 0.75 * (1.0 - rho_min) / delta * (1.0 - (phi / delta) ** 2)
        return np.where(np.abs(phi) <= delta, middle, 0.0)
