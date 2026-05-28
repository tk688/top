from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from online_pls.fem_solver import Q4CantileverFEM
from online_pls.optimizer import ExactPLSOptimizer, HistoryRow, OnlinePLSOptimizer
from online_pls.rbf_level_set import RBFLevelSet


def evaluate_phi_image(level_set: RBFLevelSet, alpha: torch.Tensor, width: int, height: int) -> np.ndarray:
    xs = np.linspace(0.0, level_set.length, width)
    ys = np.linspace(0.0, level_set.height, height)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    alpha_np = alpha.detach().cpu().numpy().astype(np.float64, copy=False)
    return level_set.phi_numpy_at(alpha_np, points).reshape(height, width)


def save_topology_png(phi: np.ndarray, path: Path) -> None:
    solid = phi[::-1] >= 0.0
    img = np.where(solid[..., None], 0, 255).astype(np.uint8)
    img = np.repeat(img, 3, axis=2)
    Image.fromarray(img, mode="RGB").save(path)


def save_contour_png(phi: np.ndarray, path: Path) -> None:
    solid = phi[::-1] >= 0.0
    img = np.full((solid.shape[0], solid.shape[1], 3), 255, dtype=np.uint8)
    img[solid] = np.array([35, 190, 180], dtype=np.uint8)

    edge = np.zeros_like(solid, dtype=bool)
    edge[:, :-1] |= solid[:, :-1] != solid[:, 1:]
    edge[:-1, :] |= solid[:-1, :] != solid[1:, :]
    thick = edge.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            src_y = slice(max(0, -dy), solid.shape[0] - max(0, dy))
            src_x = slice(max(0, -dx), solid.shape[1] - max(0, dx))
            dst_y = slice(max(0, dy), solid.shape[0] - max(0, -dy))
            dst_x = slice(max(0, dx), solid.shape[1] - max(0, -dx))
            thick[dst_y, dst_x] |= edge[src_y, src_x]
    img[thick] = np.array([220, 20, 20], dtype=np.uint8)
    Image.fromarray(img, mode="RGB").save(path)


def save_convergence_png(histories: list[tuple[str, list[HistoryRow]]], path: Path) -> None:
    width, height = 860, 360
    margin = 48
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=1)
    draw.line((margin, margin, margin, height - margin), fill="black", width=1)

    all_comp = [row.compliance for _, rows in histories for row in rows if row.mode != "pred"]
    if not all_comp:
        img.save(path)
        return
    lo = min(all_comp)
    hi = max(all_comp)
    span = max(hi - lo, 1.0e-8)
    max_iter = max(row.iteration for _, rows in histories for row in rows)
    colors = [(30, 92, 180), (25, 145, 85), (190, 80, 40)]
    for idx, (label, rows) in enumerate(histories):
        pts = []
        for row in rows:
            if row.mode == "pred":
                continue
            x = margin + (width - 2 * margin) * row.iteration / max(max_iter, 1)
            y = height - margin - (height - 2 * margin) * (row.compliance - lo) / span
            pts.append((x, y))
        if len(pts) >= 2:
            color = colors[idx % len(colors)]
            draw.line(pts, fill=color, width=3)
            draw.text((margin + 150 + 120 * idx, 14), label, fill=color)
    draw.text((margin, height - margin + 12), "iteration", fill="black")
    draw.text((margin, 14), "compliance", fill="black")
    img.save(path)


def build_problem(args: argparse.Namespace) -> tuple[RBFLevelSet, Q4CantileverFEM]:
    level_set = RBFLevelSet(
        nx=args.nelx + 1,
        ny=args.nely + 1,
        cx=args.cx,
        cy=args.cy,
        sigma=args.sigma,
        transition_width=args.transition_width,
        init_strategy=args.init,
        length=args.length,
        height=args.height,
    )
    solver = Q4CantileverFEM(
        nx=level_set.nx,
        ny=level_set.ny,
        length=args.length,
        height=args.height,
        volume_target=args.volfrac,
        volume_penalty=args.volume_penalty,
        force_magnitude=args.force,
        load_position=args.load,
        density_samples=args.density_samples,
        support_mode="full_left",
    )
    return level_set, solver


def build_optimizer(
    mode: str, args: argparse.Namespace, level_set: RBFLevelSet, solver: Q4CantileverFEM
) -> ExactPLSOptimizer:
    common = dict(
        level_set=level_set,
        solver=solver,
        max_iter=args.max_iter,
        step_size=args.step_size,
        move_limit=args.move_limit,
        backtracking_steps=args.backtracking_steps,
    )
    if mode == "online":
        return OnlinePLSOptimizer(
            **common,
            warmup=args.warmup,
            exact_every=args.exact_every,
            uncertainty_threshold=args.uncertainty_threshold,
            angle_threshold=args.angle_threshold,
            correction_tolerance=args.correction_tolerance,
        )
    return ExactPLSOptimizer(**common)


def write_history(rows: list[HistoryRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HistoryRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def save_outputs(
    label: str,
    level_set: RBFLevelSet,
    alpha: torch.Tensor,
    rows: list[HistoryRow],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    phi = evaluate_phi_image(level_set, alpha, args.render_width, args.render_height)
    suffix = "" if label == "final" else f"_{label}"
    save_topology_png(phi, out_dir / f"final{suffix}_topology.png")
    save_contour_png(phi, out_dir / f"final{suffix}_contour.png")
    write_history(rows, out_dir / ("history.csv" if label == "final" else f"history_{label}.csv"))


def write_compare_csv(exact_rows: list[HistoryRow], online_rows: list[HistoryRow], path: Path) -> None:
    exact_final = exact_rows[-1]
    online_final = online_rows[-1]
    degradation = (online_final.compliance - exact_final.compliance) / max(abs(exact_final.compliance), 1.0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "exact", "online"])
        writer.writerow(["compliance", exact_final.compliance, online_final.compliance])
        writer.writerow(["volume", exact_final.volume, online_final.volume])
        writer.writerow(["exact_calls", exact_final.exact_calls, online_final.exact_calls])
        writer.writerow(["predicted_steps", exact_final.predicted_steps, online_final.predicted_steps])
        writer.writerow(["relative_compliance_degradation", 0.0, degradation])


def print_summary(label: str, rows: list[HistoryRow]) -> None:
    final = rows[-1]
    print(
        f"{label}: compliance={final.compliance:.6g}, volume={final.volume:.4f}, "
        f"exact_calls={final.exact_calls}, predicted_steps={final.predicted_steps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exact", "online", "compare"], default="exact")
    parser.add_argument("--enable-prediction", action="store_true", help="legacy alias for --mode online")
    parser.add_argument("--nelx", type=int, default=120)
    parser.add_argument("--nely", type=int, default=30)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--height", type=float, default=1.0)
    parser.add_argument("--cx", type=int, default=61)
    parser.add_argument("--cy", type=int, default=16)
    parser.add_argument("--sigma", type=float, default=0.12)
    parser.add_argument("--transition-width", type=float, default=0.08)
    parser.add_argument("--density-samples", type=int, default=5)
    parser.add_argument("--volfrac", type=float, default=0.5)
    parser.add_argument("--volume-penalty", type=float, default=5000.0)
    parser.add_argument("--force", type=float, default=1.0)
    parser.add_argument("--load", choices=["bottom", "middle"], default="bottom")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--step-size", type=float, default=0.04)
    parser.add_argument("--move-limit", type=float, default=0.03)
    parser.add_argument("--backtracking-steps", type=int, default=8)
    parser.add_argument("--init", choices=["holes", "truss"], default="holes")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--exact-every", type=int, default=5)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.40)
    parser.add_argument("--angle-threshold", type=float, default=-0.05)
    parser.add_argument("--correction-tolerance", type=float, default=0.03)
    parser.add_argument("--render-width", type=int, default=601)
    parser.add_argument("--render-height", type=int, default=151)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.enable_prediction:
        args.mode = "online"

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(4)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    if args.mode == "compare":
        level_set, solver = build_problem(args)
        initial_alpha = level_set.initial_parameters(seed=args.seed).detach().cpu()

        exact_opt = build_optimizer("exact", args, level_set, solver)
        exact_alpha, exact_rows = exact_opt.run(initial_alpha=initial_alpha)
        save_outputs("exact", level_set, exact_alpha, exact_rows, out_dir, args)

        online_opt = build_optimizer("online", args, level_set, solver)
        online_alpha, online_rows = online_opt.run(initial_alpha=initial_alpha)
        save_outputs("online", level_set, online_alpha, online_rows, out_dir, args)

        save_convergence_png([("exact", exact_rows), ("online", online_rows)], out_dir / "convergence_compare.png")
        write_compare_csv(exact_rows, online_rows, out_dir / "compare_exact_online.csv")
        print_summary("exact", exact_rows)
        print_summary("online", online_rows)
        print(f"Saved comparison outputs to {out_dir}")
        return

    level_set, solver = build_problem(args)
    opt = build_optimizer(args.mode, args, level_set, solver)
    alpha, rows = opt.run(seed=args.seed)
    save_outputs("final", level_set, alpha, rows, out_dir, args)
    save_convergence_png([(args.mode, rows)], out_dir / "convergence.png")
    print_summary(args.mode, rows)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
