from __future__ import annotations

import torch

from online_pls.fem_solver import Q4CantileverFEM
from online_pls.rbf_level_set import RBFLevelSet


def main() -> None:
    torch.set_default_dtype(torch.float64)
    level_set = RBFLevelSet(
        nx=41,
        ny=11,
        cx=15,
        cy=5,
        sigma=0.16,
        transition_width=0.08,
        init_strategy="holes",
        length=4.0,
        height=1.0,
    )
    solver = Q4CantileverFEM(
        nx=level_set.nx,
        ny=level_set.ny,
        length=4.0,
        height=1.0,
        volume_target=0.5,
        force_magnitude=1.0,
        load_position="bottom",
        density_samples=3,
        support_mode="full_left",
    )
    alpha = level_set.initial_parameters(seed=11).detach().cpu()

    # FEM benchmark sanity checks: Q4 单元为 8x8，左边界整边固支，右端中点受力。
    assert solver.ke0.shape == (8, 8)
    assert float(abs(solver.ke0 - solver.ke0.T).max()) < 1.0e-10
    assert len(solver.fixed) == 2 * level_set.ny
    load_dof = int(abs(solver.force).argmax())
    expected_load_dof = 2 * (level_set.nx - 1) + 1
    assert load_dof == expected_load_dof

    # 不做体积投影，直接检查原始链式法则梯度。
    grad, metrics = solver.evaluate_alpha(level_set, alpha)
    indices = [0, 3, 9, 17, level_set.num_params - 1]
    eps = 1.0e-6

    print(f"loss={metrics['loss']:.8e}, volume={metrics['volume']:.6f}")
    print("idx analytic finite-diff rel-error")
    worst = 0.0
    for idx in indices:
        plus = alpha.clone()
        minus = alpha.clone()
        plus[idx] += eps
        minus[idx] -= eps
        _, mp = solver.evaluate_alpha(level_set, plus)
        _, mm = solver.evaluate_alpha(level_set, minus)
        fd = (mp["loss"] - mm["loss"]) / (2.0 * eps)
        analytic = float(grad[idx])
        rel = abs(analytic - fd) / max(1.0, abs(fd), abs(analytic))
        worst = max(worst, rel)
        print(f"{idx:3d} {analytic: .8e} {fd: .8e} {rel: .3e}")

    if worst > 1.0e-2:
        raise SystemExit(f"gradient check failed: worst relative error {worst:.3e}")
    print(f"gradient check passed: worst relative error {worst:.3e}")


if __name__ == "__main__":
    main()
