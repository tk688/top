# Confidence-Aware Online Sensitivity Prediction for RBF/PLSM

This project is a research prototype for testing one specific idea:
can an online recurrent model predict parameter sensitivities during
RBF-parametric level-set topology optimization, so that fewer exact FEM
reanalyses are needed?

It is not a direct reproduction of TOINR. TOINR is used here only as a useful
reference for benchmark selection, smooth implicit-boundary rendering, and
comparison discipline.

## Method

1. Represent the topology with an RBF-parametric level-set field.
2. Map the level set to material density using a smooth Heaviside function.
3. Evaluate compliance with sparse Q4 plane-stress FEM.
4. Compute exact analytic adjoint sensitivities as the expensive oracle.
5. Run a deterministic exact baseline first.
6. Optionally train an online GRU ensemble to predict later sensitivities.
7. Periodically correct predicted steps with exact FEM and roll back bad steps.

The default benchmark follows the long cantilever setting used in TOINR-style
validation:

- `nelx = 120`, `nely = 30`
- `length = 4.0`, `height = 1.0`
- `volfrac = 0.5`
- left boundary fully fixed
- downward point load at the right-bottom node
- element density estimated by sub-sampling each Q4 element
- final topology rendered from a high-resolution implicit level-set field

## Run

```powershell
cd D:\codex\codex-skills\online_pls_demo
python run_demo.py
```

Useful variants:

```powershell
# Fast smoke test
python run_demo.py --nelx 60 --nely 15 --cx 31 --cy 8 --density-samples 3 --max-iter 10 --render-width 401 --render-height 101

# Online prediction only
python run_demo.py --mode online --max-iter 40

# Exact-vs-online comparison
python run_demo.py --mode compare --max-iter 40

# Legacy load location comparison
python run_demo.py --load middle
```

Gradient verification:

```powershell
python check_fem_gradient.py
```

## Outputs

Single-run mode writes:

- `outputs/history.csv`
- `outputs/final_topology.png`
- `outputs/final_contour.png`
- `outputs/convergence.png`

Comparison mode writes:

- `outputs/history_exact.csv`
- `outputs/history_online.csv`
- `outputs/final_exact_topology.png`
- `outputs/final_exact_contour.png`
- `outputs/final_online_topology.png`
- `outputs/final_online_contour.png`
- `outputs/convergence_compare.png`
- `outputs/compare_exact_online.csv`
