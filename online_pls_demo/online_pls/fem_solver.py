from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .rbf_level_set import RBFLevelSet


@dataclass
class AnalysisResult:
    loss: torch.Tensor
    compliance: torch.Tensor
    volume: torch.Tensor
    displacement_norm: torch.Tensor


class Q4CantileverFEM:
    """Plane-stress Q4 FEM oracle with analytic compliance sensitivities."""

    def __init__(
        self,
        nx: int,
        ny: int,
        length: float = 2.0,
        height: float = 1.0,
        young: float = 1.0,
        poisson: float = 0.3,
        emin: float = 1.0e-6,
        penalty: float = 3.0,
        volume_target: float = 0.5,
        volume_penalty: float = 5000.0,
        force_magnitude: float = 1.0,
        load_position: str = "bottom",
        density_samples: int = 5,
        support_mode: str = "full_left",
        split_support: bool = False,
        support_fraction: float = 0.22,
    ) -> None:
        if nx < 2 or ny < 2:
            raise ValueError("Q4 FEM requires at least a 2 x 2 node grid.")
        self.nx = nx
        self.ny = ny
        self.length = length
        self.height = height
        self.young = young
        self.poisson = poisson
        self.emin = emin
        self.penalty = penalty
        self.volume_target = volume_target
        self.volume_penalty = volume_penalty
        self.force_magnitude = force_magnitude
        self.load_position = load_position
        self.density_samples = density_samples
        self.support_mode = support_mode
        self.split_support = split_support
        self.support_fraction = support_fraction

        self.num_nodes = nx * ny
        self.num_elements = (nx - 1) * (ny - 1)
        self.ndof = 2 * self.num_nodes
        self.coords = self._build_coords()
        self.edof = self._build_edof()
        self.element_centers = self._build_element_centers()
        self.sample_points, self.sample_element_ids = self._build_density_sample_points()
        self.ke0 = self._element_stiffness_unit_young()
        self.iK, self.jK = self._build_sparse_indices()
        self.force = self._build_force()
        self.fixed = self._build_fixed_dofs()
        all_dofs = np.arange(self.ndof)
        self.free = np.setdiff1d(all_dofs, self.fixed)

    def _node_id(self, ix: int, iy: int) -> int:
        return iy * self.nx + ix

    def _build_coords(self) -> np.ndarray:
        xs = np.linspace(0.0, self.length, self.nx)
        ys = np.linspace(0.0, self.height, self.ny)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        return np.column_stack([xx.reshape(-1), yy.reshape(-1)])

    def _build_edof(self) -> np.ndarray:
        rows: list[list[int]] = []
        for iy in range(self.ny - 1):
            for ix in range(self.nx - 1):
                n1 = self._node_id(ix, iy)
                n2 = self._node_id(ix + 1, iy)
                n3 = self._node_id(ix + 1, iy + 1)
                n4 = self._node_id(ix, iy + 1)
                rows.append(
                    [
                        2 * n1,
                        2 * n1 + 1,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n3,
                        2 * n3 + 1,
                        2 * n4,
                        2 * n4 + 1,
                    ]
                )
        return np.asarray(rows, dtype=np.int64)

    def _build_element_centers(self) -> np.ndarray:
        centers = []
        for dofs in self.edof:
            node_ids = dofs[0::2] // 2
            centers.append(self.coords[node_ids].mean(axis=0))
        return np.asarray(centers, dtype=np.float64)

    def _build_density_sample_points(self) -> tuple[np.ndarray, np.ndarray]:
        q = self.density_samples
        if q < 1:
            raise ValueError("density_samples must be at least 1.")
        hx = self.length / (self.nx - 1)
        hy = self.height / (self.ny - 1)
        local = (np.arange(q, dtype=np.float64) + 0.5) / q
        pts: list[tuple[float, float]] = []
        elem_ids: list[int] = []
        eid = 0
        for iy in range(self.ny - 1):
            for ix in range(self.nx - 1):
                x0 = ix * hx
                y0 = iy * hy
                for sy in local:
                    for sx in local:
                        pts.append((x0 + sx * hx, y0 + sy * hy))
                        elem_ids.append(eid)
                eid += 1
        return np.asarray(pts, dtype=np.float64), np.asarray(elem_ids, dtype=np.int64)

    def _build_sparse_indices(self) -> tuple[np.ndarray, np.ndarray]:
        iK = np.kron(self.edof, np.ones((8, 1), dtype=np.int64)).reshape(-1)
        jK = np.kron(self.edof, np.ones((1, 8), dtype=np.int64)).reshape(-1)
        return iK, jK

    def _build_force(self) -> np.ndarray:
        f = np.zeros(self.ndof, dtype=np.float64)
        if self.load_position == "bottom":
            load_y = 0
        elif self.load_position == "middle":
            load_y = self.ny // 2
        else:
            raise ValueError("load_position must be 'bottom' or 'middle'.")
        node = self._node_id(self.nx - 1, load_y)
        # 长悬臂梁 benchmark：右端指定节点施加竖向向下载荷。
        f[2 * node + 1] = -self.force_magnitude
        return f

    def _build_fixed_dofs(self) -> np.ndarray:
        fixed: list[int] = []
        if self.support_mode not in {"full_left", "split_left"}:
            raise ValueError("support_mode must be 'full_left' or 'split_left'.")
        for iy in range(self.ny):
            y = iy / (self.ny - 1)
            # full_left 对应整条左边界固支；split_left 仅用于保留旧实验入口。
            use_split = self.support_mode == "split_left" or self.split_support
            if use_split and self.support_fraction < y < 1.0 - self.support_fraction:
                continue
            node = self._node_id(0, iy)
            fixed.extend([2 * node, 2 * node + 1])
        return np.asarray(fixed, dtype=np.int64)

    def _element_stiffness_unit_young(self) -> np.ndarray:
        hx = self.length / (self.nx - 1)
        hy = self.height / (self.ny - 1)
        d = 1.0 / (1.0 - self.poisson**2) * np.array(
            [
                [1.0, self.poisson, 0.0],
                [self.poisson, 1.0, 0.0],
                [0.0, 0.0, (1.0 - self.poisson) / 2.0],
            ],
            dtype=np.float64,
        )
        ke = np.zeros((8, 8), dtype=np.float64)
        gp = 1.0 / np.sqrt(3.0)
        for xi in (-gp, gp):
            for eta in (-gp, gp):
                dN_dxi = 0.25 * np.array(
                    [
                        -(1.0 - eta),
                        1.0 - eta,
                        1.0 + eta,
                        -(1.0 + eta),
                    ]
                )
                dN_deta = 0.25 * np.array(
                    [
                        -(1.0 - xi),
                        -(1.0 + xi),
                        1.0 + xi,
                        1.0 - xi,
                    ]
                )
                dN_dx = dN_dxi * (2.0 / hx)
                dN_dy = dN_deta * (2.0 / hy)
                b = np.zeros((3, 8), dtype=np.float64)
                b[0, 0::2] = dN_dx
                b[1, 1::2] = dN_dy
                b[2, 0::2] = dN_dy
                b[2, 1::2] = dN_dx
                det_j = hx * hy / 4.0
                ke += b.T @ d @ b * det_j
        return 0.5 * (ke + ke.T)

    def _element_density_and_chain(
        self, level_set: RBFLevelSet, alpha_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        phi_s = level_set.phi_numpy_at(alpha_np, self.sample_points)
        rho_s = level_set.heaviside_numpy(phi_s)
        drho_dphi_s = level_set.heaviside_derivative_numpy(phi_s)
        rho_e = np.bincount(
            self.sample_element_ids,
            weights=rho_s,
            minlength=self.num_elements,
        ) / (self.density_samples * self.density_samples)
        return rho_e, rho_s, drho_dphi_s, self.sample_points

    def volume_fraction(self, level_set: RBFLevelSet, alpha: torch.Tensor | np.ndarray) -> float:
        alpha_np = self._as_numpy(alpha)
        rho_e, _, _, _ = self._element_density_and_chain(level_set, alpha_np)
        return float(np.mean(rho_e))

    def projection_volume_fraction(self, level_set: RBFLevelSet, alpha: torch.Tensor | np.ndarray) -> float:
        """Fast volume estimate used only inside scalar bias projection."""
        alpha_np = self._as_numpy(alpha)
        phi = level_set.phi_numpy_at(alpha_np, self.element_centers)
        return float(np.mean(level_set.heaviside_numpy(phi)))

    def evaluate_alpha(
        self, level_set: RBFLevelSet, alpha: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, dict[str, float]]:
        alpha_np = self._as_numpy(alpha)
        rho, rho_s, drho_dphi_s, sample_points = self._element_density_and_chain(level_set, alpha_np)
        # SIMP 材料插值：E(rho)=Emin+rho^p*(E0-Emin)。
        stiffness = self.emin + (self.young - self.emin) * rho**self.penalty

        # 稀疏组装全局刚度矩阵 K(rho)，随后只求解自由自由度 K_ff u_f=f_f。
        sK = (self.ke0.reshape(-1)[None, :] * stiffness[:, None]).reshape(-1)
        K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
        Kff = K[self.free[:, None], self.free]
        uf = spsolve(Kff, self.force[self.free])
        u = np.zeros(self.ndof, dtype=np.float64)
        u[self.free] = uf

        compliance = float(self.force @ u)
        volume = float(np.mean(rho))
        volume_error = volume - self.volume_target
        loss = compliance + self.volume_penalty * volume_error**2

        ue = u[self.edof]
        ce = np.einsum("ei,ij,ej->e", ue, self.ke0, ue)
        # 柔度伴随灵敏度：
        # dC/drho_e = -p*rho_e^(p-1)*(E0-Emin)*u_e^T*K0*u_e。
        dc_drho = (
            -self.penalty
            * (self.young - self.emin)
            * np.maximum(rho, 1.0e-9) ** (self.penalty - 1.0)
            * ce
        )
        dc_drho = dc_drho + self.volume_penalty * 2.0 * volume_error / self.num_elements

        sample_factor = 1.0 / (self.density_samples * self.density_samples)
        dloss_dphi_s = dc_drho[self.sample_element_ids] * sample_factor * drho_dphi_s
        grad_weights = level_set.basis_weighted_sum_numpy(sample_points, dloss_dphi_s)
        grad_bias = np.array([np.sum(dloss_dphi_s)], dtype=np.float64)
        grad = np.concatenate([grad_weights, grad_bias])

        metrics = {
            "loss": loss,
            "compliance": compliance,
            "volume": volume,
            "disp_norm": float(np.linalg.norm(u)),
        }
        return torch.as_tensor(grad, dtype=torch.float64), metrics

    def evaluate(self, density: torch.Tensor) -> AnalysisResult:
        raise NotImplementedError("Use evaluate_alpha(level_set, alpha) for the FEM oracle.")

    @staticmethod
    def _as_numpy(alpha: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(alpha, torch.Tensor):
            return alpha.detach().cpu().numpy().astype(np.float64, copy=False)
        return np.asarray(alpha, dtype=np.float64)
