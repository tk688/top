from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AnalysisResult:
    loss: torch.Tensor
    compliance: torch.Tensor
    volume: torch.Tensor
    displacement_norm: torch.Tensor


class LatticeCantilever:
    """Small differentiable spring-lattice analysis model.

    This is not a replacement for production FEM. It is a compact structural
    oracle with the same outer-loop behavior: density -> stiffness -> solve ->
    compliance -> gradient.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        penalty: float = 3.0,
        volume_target: float = 0.45,
        volume_penalty: float = 120.0,
        device: str = "cpu",
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.penalty = penalty
        self.volume_target = volume_target
        self.volume_penalty = volume_penalty
        self.device = torch.device(device)
        self.num_nodes = nx * ny
        self.ndof = 2 * self.num_nodes
        self.edges = self._build_edges()
        self.force = self._build_force()
        self.fixed = self._build_fixed_dofs()
        all_dofs = torch.arange(self.ndof, device=self.device)
        mask = torch.ones(self.ndof, dtype=torch.bool, device=self.device)
        mask[self.fixed] = False
        self.free = all_dofs[mask]

    def _node_id(self, ix: int, iy: int) -> int:
        return iy * self.nx + ix

    def _build_edges(self) -> list[tuple[int, int, float, float]]:
        edges: list[tuple[int, int, float, float]] = []
        for iy in range(self.ny):
            for ix in range(self.nx):
                i = self._node_id(ix, iy)
                if ix + 1 < self.nx:
                    edges.append((i, self._node_id(ix + 1, iy), 1.0, 0.0))
                if iy + 1 < self.ny:
                    edges.append((i, self._node_id(ix, iy + 1), 0.0, 1.0))
                if ix + 1 < self.nx and iy + 1 < self.ny:
                    edges.append((i, self._node_id(ix + 1, iy + 1), 1.0, 1.0))
                if ix + 1 < self.nx and iy - 1 >= 0:
                    edges.append((i, self._node_id(ix + 1, iy - 1), 1.0, -1.0))
        return edges

    def _build_force(self) -> torch.Tensor:
        f = torch.zeros(self.ndof, device=self.device)
        mid = self._node_id(self.nx - 1, self.ny // 2)
        f[2 * mid + 1] = -1.0
        return f

    def _build_fixed_dofs(self) -> torch.Tensor:
        fixed: list[int] = []
        for iy in range(self.ny):
            node = self._node_id(0, iy)
            fixed.extend([2 * node, 2 * node + 1])
        return torch.tensor(fixed, dtype=torch.long, device=self.device)

    def evaluate(self, density: torch.Tensor) -> AnalysisResult:
        rho_node = density.reshape(-1)
        stiffness = rho_node.clamp_min(1.0e-4).pow(self.penalty)
        K = torch.zeros((self.ndof, self.ndof), dtype=density.dtype, device=self.device)

        for ni, nj, dx, dy in self.edges:
            length = (dx * dx + dy * dy) ** 0.5
            c = dx / length
            s = dy / length
            k = 0.5 * (stiffness[ni] + stiffness[nj]) / length
            template = torch.tensor(
                [
                    [c * c, c * s, -c * c, -c * s],
                    [c * s, s * s, -c * s, -s * s],
                    [-c * c, -c * s, c * c, c * s],
                    [-c * s, -s * s, c * s, s * s],
                ],
                dtype=density.dtype,
                device=self.device,
            )
            dofs = torch.tensor([2 * ni, 2 * ni + 1, 2 * nj, 2 * nj + 1], device=self.device)
            K[dofs[:, None], dofs[None, :]] = K[dofs[:, None], dofs[None, :]] + k * template

        # Tiny diagonal spring prevents singular matrices during early designs.
        K = K + 1.0e-6 * torch.eye(self.ndof, dtype=density.dtype, device=self.device)
        Kff = K[self.free[:, None], self.free[None, :]]
        ff = self.force[self.free]
        uf = torch.linalg.solve(Kff, ff)
        u = torch.zeros_like(self.force)
        u[self.free] = uf

        compliance = torch.dot(self.force, u)
        volume = density.mean()
        violation = torch.relu(volume - self.volume_target)
        loss = compliance + self.volume_penalty * violation.pow(2)
        return AnalysisResult(loss=loss, compliance=compliance, volume=volume, displacement_norm=u.norm())

