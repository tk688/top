from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
from PIL import Image, ImageDraw


def configure_matplotlib_backend() -> str:
    """Use a GUI backend for --live, otherwise use a non-interactive backend."""
    if "--live" not in sys.argv:
        matplotlib.use("Agg", force=True)
        return "Agg"

    for backend in ("TkAgg", "QtAgg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except Exception:
            continue

    matplotlib.use("Agg", force=True)
    return "Agg"


MATPLOTLIB_BACKEND = configure_matplotlib_backend()
import matplotlib.pyplot as plt

from online_pls.fem_solver import Q4CantileverFEM
from online_pls.optimizer import DesignSnapshot, ExactPLSOptimizer, HistoryRow, OnlinePLSOptimizer
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


def save_topology_svg(phi: np.ndarray, path: Path, length: float, height: float) -> None:
    x = np.linspace(0.0, length, phi.shape[1])
    y = np.linspace(0.0, height, phi.shape[0])
    fig, ax = plt.subplots(figsize=(8.0, 8.0 * height / length))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    vmin = float(np.nanmin(phi))
    vmax = float(np.nanmax(phi))
    if vmin < 0.0 < vmax:
        eps = max(abs(vmin), abs(vmax), 1.0) * 1.0e-9
        ax.contourf(x, y, phi, levels=[vmin - eps, 0.0, vmax + eps], colors=["white", "black"])
    elif vmax <= 0.0:
        ax.add_patch(plt.Rectangle((0.0, 0.0), length, height, facecolor="white", edgecolor="none"))
    else:
        ax.add_patch(plt.Rectangle((0.0, 0.0), length, height, facecolor="black", edgecolor="none"))
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


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


def save_contour_svg(phi: np.ndarray, path: Path, length: float, height: float) -> None:
    x = np.linspace(0.0, length, phi.shape[1])
    y = np.linspace(0.0, height, phi.shape[0])
    fig, ax = plt.subplots(figsize=(8.0, 8.0 * height / length))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    vmin = float(np.nanmin(phi))
    vmax = float(np.nanmax(phi))
    if vmin < 0.0 < vmax:
        eps = max(abs(vmin), abs(vmax), 1.0) * 1.0e-9
        ax.contourf(x, y, phi, levels=[vmin - eps, 0.0, vmax + eps], colors=["white", "#23beb4"])
        ax.contour(x, y, phi, levels=[0.0], colors=["#dc1414"], linewidths=1.2)
    elif vmax <= 0.0:
        ax.add_patch(plt.Rectangle((0.0, 0.0), length, height, facecolor="white", edgecolor="none"))
    else:
        ax.add_patch(plt.Rectangle((0.0, 0.0), length, height, facecolor="#23beb4", edgecolor="#dc1414", linewidth=1.2))
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def plot_contour_on_axis(ax, phi: np.ndarray, length: float, height: float, title: str | None = None) -> None:
    x = np.linspace(0.0, length, phi.shape[1])
    y = np.linspace(0.0, height, phi.shape[0])
    vmin = float(np.nanmin(phi))
    vmax = float(np.nanmax(phi))
    ax.set_facecolor("white")
    if vmin < 0.0 < vmax:
        eps = max(abs(vmin), abs(vmax), 1.0) * 1.0e-9
        ax.contourf(x, y, phi, levels=[vmin - eps, 0.0, vmax + eps], colors=["white", "#23beb4"])
        ax.contour(x, y, phi, levels=[0.0], colors=["#dc1414"], linewidths=0.9)
    elif vmax <= 0.0:
        ax.add_patch(plt.Rectangle((0.0, 0.0), length, height, facecolor="white", edgecolor="none"))
    else:
        ax.add_patch(plt.Rectangle((0.0, 0.0), length, height, facecolor="#23beb4", edgecolor="#dc1414", linewidth=0.9))
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontname="Times New Roman", fontsize=9, pad=2.0)


def select_process_snapshots(
    snapshots: list[DesignSnapshot],
    every: int,
    max_frames: int,
) -> list[DesignSnapshot]:
    if not snapshots:
        return []
    selected = [
        snapshot
        for snapshot in snapshots
        if snapshot.iteration == 0 or snapshot.iteration == snapshots[-1].iteration or snapshot.iteration % every == 0
    ]
    if len(selected) <= max_frames:
        return selected
    indices = np.linspace(0, len(selected) - 1, max_frames).round().astype(int)
    return [selected[int(i)] for i in indices]


def save_process_frames_svg(
    label: str,
    level_set: RBFLevelSet,
    snapshots: list[DesignSnapshot],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    frame_dir = out_dir / f"process_{label}_frames"
    frame_dir.mkdir(exist_ok=True)
    for snapshot in snapshots:
        phi = evaluate_phi_image(level_set, snapshot.alpha, args.render_width, args.render_height)
        path = frame_dir / f"{label}_iter_{snapshot.iteration:04d}.svg"
        save_contour_svg(phi, path, level_set.length, level_set.height)


def save_process_montage_svg(
    label: str,
    level_set: RBFLevelSet,
    snapshots: list[DesignSnapshot],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    if not snapshots:
        return
    cols = max(1, args.process_cols)
    rows = int(np.ceil(len(snapshots) / cols))
    fig_w = 3.4 * cols
    fig_h = 1.05 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, snapshot in zip(axes.ravel(), snapshots):
        phi = evaluate_phi_image(level_set, snapshot.alpha, args.render_width, args.render_height)
        title = f"it={snapshot.iteration}, C={snapshot.compliance:.3g}, V={snapshot.volume:.3f}"
        plot_contour_on_axis(ax, phi, level_set.length, level_set.height, title=title)
    fig.tight_layout(pad=0.25)
    fig.savefig(out_dir / f"process_{label}.svg", format="svg")
    plt.close(fig)


def save_process_visualization(
    label: str,
    level_set: RBFLevelSet,
    snapshots: list[DesignSnapshot],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    selected = select_process_snapshots(snapshots, args.snapshot_every, args.process_max_frames)
    save_process_frames_svg(label, level_set, selected, out_dir, args)
    save_process_montage_svg(label, level_set, selected, out_dir, args)


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


def save_convergence_svg(histories: list[tuple[str, list[HistoryRow]]], path: Path) -> None:
    colors = ["#1e5cb4", "#199155", "#be5028"]
    fig, ax_comp = plt.subplots(figsize=(7.2, 3.2))
    ax_vol = ax_comp.twinx()
    plotted = False
    for idx, (label, rows) in enumerate(histories):
        visible = [row for row in rows if row.mode != "pred"]
        if not visible:
            continue
        color = colors[idx % len(colors)]
        xs = [row.iteration for row in visible]
        comp = [row.compliance for row in visible]
        vol = [row.volume for row in visible]
        ax_comp.plot(xs, comp, color=color, linewidth=1.8, label=f"{label} compliance")
        ax_vol.plot(xs, vol, color=color, linewidth=1.2, linestyle="--", label=f"{label} volume")
        plotted = True
    ax_comp.set_xlabel("Iteration", fontname="Times New Roman", fontsize=11)
    ax_comp.set_ylabel("Compliance", fontname="Times New Roman", fontsize=11)
    ax_vol.set_ylabel("Volume fraction", fontname="Times New Roman", fontsize=11)
    ax_comp.grid(True, linewidth=0.35, alpha=0.35)
    ax_vol.set_ylim(0.0, 1.0)
    lines_1, labels_1 = ax_comp.get_legend_handles_labels()
    lines_2, labels_2 = ax_vol.get_legend_handles_labels()
    if plotted:
        ax_comp.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best", frameon=False, prop={"family": "Times New Roman", "size": 9})
    for tick in ax_comp.get_xticklabels() + ax_comp.get_yticklabels() + ax_vol.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


class LiveTopologyWindow:
    """Matplotlib window that updates the topology during optimization."""

    def __init__(self, level_set: RBFLevelSet, args: argparse.Namespace, title: str) -> None:
        self.level_set = level_set
        self.args = args
        self.title = title
        self.compliances: list[float] = []
        self.volumes: list[float] = []
        self.iters: list[int] = []
        self.fig = None
        self.ax_topology = None
        self.ax_compliance = None
        self.ax_volume = None
        self._last_update = -1
        self._init_window()

    def _init_window(self) -> None:
        plt.ion()
        self.fig, (self.ax_topology, self.ax_compliance, self.ax_volume) = plt.subplots(
            3,
            1,
            figsize=(9.8, 6.2),
            gridspec_kw={"height_ratios": [1.45, 0.80, 0.70]},
            constrained_layout=True,
        )
        self.fig.canvas.manager.set_window_title(self.title)
        self.fig.show()
        plt.pause(0.1)

    def __call__(self, snapshot: DesignSnapshot, rows: list[HistoryRow]) -> None:
        if snapshot.iteration != rows[-1].iteration:
            return
        if snapshot.iteration == self._last_update:
            return
        self._last_update = snapshot.iteration
        self.iters.append(snapshot.iteration)
        self.compliances.append(snapshot.compliance)
        self.volumes.append(snapshot.volume)
        self.update(snapshot)

    def update(self, snapshot: DesignSnapshot) -> None:
        phi = evaluate_phi_image(
            self.level_set,
            snapshot.alpha,
            self.args.live_render_width,
            self.args.live_render_height,
        )
        self.ax_topology.clear()
        plot_contour_on_axis(
            self.ax_topology,
            phi,
            self.level_set.length,
            self.level_set.height,
            title=f"Iteration {snapshot.iteration} | {snapshot.mode} | C={snapshot.compliance:.4g} | V={snapshot.volume:.4f}",
        )

        self.ax_compliance.clear()
        self.ax_volume.clear()
        self.ax_compliance.plot(self.iters, self.compliances, color="#1e5cb4", linewidth=1.8, label="Compliance")
        self.ax_volume.plot(self.iters, self.volumes, color="#199155", linewidth=1.8, label="Volume fraction")
        self.ax_volume.axhline(self.args.volfrac, color="#199155", linewidth=1.0, linestyle="--", alpha=0.55)
        self.ax_compliance.set_xlabel("Iteration", fontname="Times New Roman")
        self.ax_compliance.set_ylabel("Compliance", fontname="Times New Roman")
        self.ax_volume.set_xlabel("Iteration", fontname="Times New Roman")
        self.ax_volume.set_ylabel("Volume fraction", fontname="Times New Roman")
        self.ax_volume.set_ylim(0.0, 1.0)
        self.ax_compliance.grid(True, linewidth=0.35, alpha=0.35)
        self.ax_volume.grid(True, linewidth=0.35, alpha=0.35)
        self.ax_compliance.legend(loc="best", frameon=False, prop={"family": "Times New Roman", "size": 9})
        self.ax_volume.legend(loc="best", frameon=False, prop={"family": "Times New Roman", "size": 9})
        for ax in (self.ax_compliance, self.ax_volume):
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontname("Times New Roman")
                tick.set_fontsize(9)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(self.args.live_pause)

    def finish(self) -> None:
        if self.fig is not None:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            if self.args.live_hold:
                plt.ioff()
                plt.show()


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
    live_window = LiveTopologyWindow(level_set, args, f"{mode} topology optimization") if args.live else None
    common = dict(
        level_set=level_set,
        solver=solver,
        max_iter=args.max_iter,
        step_size=args.step_size,
        move_limit=args.move_limit,
        backtracking_steps=args.backtracking_steps,
        volume_start=args.volume_start,
        volume_relax=args.volume_relax,
        snapshot_callback=live_window,
    )
    if mode == "online":
        opt = OnlinePLSOptimizer(
            **common,
            warmup=args.warmup,
            exact_every=args.exact_every,
            uncertainty_threshold=args.uncertainty_threshold,
            angle_threshold=args.angle_threshold,
            correction_tolerance=args.correction_tolerance,
        )
        opt.live_window = live_window
        return opt
    opt = ExactPLSOptimizer(**common)
    opt.live_window = live_window
    return opt


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
    if not args.svg_only:
        save_topology_png(phi, out_dir / f"final{suffix}_topology.png")
    save_topology_svg(phi, out_dir / f"final{suffix}_topology.svg", level_set.length, level_set.height)
    if not args.svg_only:
        save_contour_png(phi, out_dir / f"final{suffix}_contour.png")
    save_contour_svg(phi, out_dir / f"final{suffix}_contour.svg", level_set.length, level_set.height)
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
    parser.add_argument("--volume-start", type=float, default=None, help="initial volume fraction for continuation")
    parser.add_argument("--volume-relax", type=int, default=0, help="iterations used to reduce volume-start to volfrac")
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
    parser.add_argument("--svg-only", action="store_true", help="write only SVG image files; CSV history is still written")
    parser.add_argument("--save-process", action="store_true", help="save SVG snapshots of the optimization process")
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--process-max-frames", type=int, default=18)
    parser.add_argument("--process-cols", type=int, default=3)
    parser.add_argument("--live", action="store_true", help="open a live window and update topology during optimization")
    parser.add_argument("--live-render-width", type=int, default=401)
    parser.add_argument("--live-render-height", type=int, default=101)
    parser.add_argument("--live-pause", type=float, default=0.05)
    parser.add_argument("--live-hold", action="store_true", help="keep the live window open after optimization finishes")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.enable_prediction:
        args.mode = "online"
    if (args.live or args.save_process) and args.volume_start is None:
        args.volume_start = 0.95
    if args.volume_start is not None and args.volume_relax <= 0:
        args.volume_relax = max(args.max_iter, 1)
    if args.live:
        print(f"Matplotlib backend: {MATPLOTLIB_BACKEND}")
        if MATPLOTLIB_BACKEND == "Agg":
            print("Warning: current Matplotlib backend is non-interactive, so no live window can pop up.")
        print(f"Live volume continuation: start={args.volume_start}, target={args.volfrac}, relax={args.volume_relax}")

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(4)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    if args.mode == "compare":
        level_set, solver = build_problem(args)
        initial_alpha = level_set.initial_parameters(seed=args.seed).detach().cpu()

        exact_opt = build_optimizer("exact", args, level_set, solver)
        exact_alpha, exact_rows = exact_opt.run(initial_alpha=initial_alpha)
        if getattr(exact_opt, "live_window", None) is not None:
            exact_opt.live_window.finish()
        save_outputs("exact", level_set, exact_alpha, exact_rows, out_dir, args)
        if args.save_process:
            save_process_visualization("exact", level_set, exact_opt.snapshots, out_dir, args)

        online_opt = build_optimizer("online", args, level_set, solver)
        online_alpha, online_rows = online_opt.run(initial_alpha=initial_alpha)
        if getattr(online_opt, "live_window", None) is not None:
            online_opt.live_window.finish()
        save_outputs("online", level_set, online_alpha, online_rows, out_dir, args)
        if args.save_process:
            save_process_visualization("online", level_set, online_opt.snapshots, out_dir, args)

        if not args.svg_only:
            save_convergence_png([("exact", exact_rows), ("online", online_rows)], out_dir / "convergence_compare.png")
        save_convergence_svg([("exact", exact_rows), ("online", online_rows)], out_dir / "convergence_compare.svg")
        write_compare_csv(exact_rows, online_rows, out_dir / "compare_exact_online.csv")
        print_summary("exact", exact_rows)
        print_summary("online", online_rows)
        print(f"Saved comparison outputs to {out_dir}")
        return

    level_set, solver = build_problem(args)
    opt = build_optimizer(args.mode, args, level_set, solver)
    alpha, rows = opt.run(seed=args.seed)
    if getattr(opt, "live_window", None) is not None:
        opt.live_window.finish()
    save_outputs("final", level_set, alpha, rows, out_dir, args)
    if args.save_process:
        save_process_visualization(args.mode, level_set, opt.snapshots, out_dir, args)
    if not args.svg_only:
        save_convergence_png([(args.mode, rows)], out_dir / "convergence.png")
    save_convergence_svg([(args.mode, rows)], out_dir / "convergence.svg")
    print_summary(args.mode, rows)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
