"""
plot_gm_derivatives.py -- visualize the MEASURED data's Vgs-derivatives (the gm family).

gm_n = d^n Ids/dVgs^n. Unlike gds (the Ids-Vds analog computed ALONG a single measurement
trace, since each TN is already a Vds sweep at fixed Vgs -- see plot_ids_vds_derivatives.py),
gm needs a slice ACROSS traces at a fixed Vds (a transfer curve: Ids vs Vgs at one Vds), then
a derivative along Vgs.

This reuses create_gms_for_train (per_neuron_simple_angelov_nn_test.py) -- the EXACT
ground-truth computation the training/shape-analysis pipeline itself uses (same Step_Index
grouping, same smooth_derivative smoothing) -- so what you see here is what analyze_shape.py
/ compute_region_metrics.py are actually comparing model predictions against, not a
separately re-derived approximation.

Usage
-----
  python plot_gm_derivatives.py --csv ..\csvs\<meas>.csv
  python plot_gm_derivatives.py --root ..\runs\<exp>          # auto-detect csv from manifest
  python plot_gm_derivatives.py --csv <..> --vds "0.14,5,10,15,20,28" --open
  # overlay a trained model's exact (autograd) gm on top of the measured curves:
  python plot_gm_derivatives.py --csv <..> --ranked_csv <ranked.csv> --id 147 --open
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch
from plot_saved_state import _load_T_train
from per_neuron_simple_angelov_nn_test import create_gms_for_train


def _load_gm_truth(csv, min_vgs, device):
    """T (DataFrame, one row per measured point) with Step_Index + gm1/gm2/gm3 columns added,
    computed the SAME way the training pipeline does (create_gms_for_train)."""
    with contextlib.redirect_stdout(io.StringIO()):
        T = _load_T_train(csv, min_vgs, device)
    T = T.copy()
    T["Step_Index"] = T.groupby("TN").cumcount()
    T = T.sort_values(by=["Step_Index", "TN"]).reset_index(drop=True)
    gm1, gm2, gm3, X, y = create_gms_for_train(T)   # row order matches T (same sort, internally redone)
    T["gm1"], T["gm2"], T["gm3"] = gm1, gm2, gm3
    return T


def _model_gm_derivs(model, vds_val, vgs_grid, device):
    """(d0,d1,d2,d3) = model Ids and its 1st/2nd/3rd Vgs-derivatives (exact, autograd) at fixed Vds."""
    vg = torch.tensor(vgs_grid, dtype=torch.float64, device=device)
    vd = torch.full_like(vg, float(vds_val))
    X = torch.stack([vg, vd], dim=1).requires_grad_(True)
    ids = model(X).reshape(-1)
    g1 = torch.autograd.grad(ids.sum(), X, create_graph=True)[0][:, 0]
    g2 = torch.autograd.grad(g1.sum(), X, create_graph=True)[0][:, 0]
    g3 = torch.autograd.grad(g2.sum(), X)[0][:, 0]
    d = lambda t: t.detach().cpu().numpy()
    return d(ids), d(g1), d(g2), d(g3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Measurement CSV. If omitted, auto-detected from --root or --ranked_csv.")
    ap.add_argument("--root", help="Results folder; auto-detect the measurement CSV from its manifest.")
    # optional model overlay: a trained run's Vgs-derivatives (exact, autograd) on top of data
    ap.add_argument("--ranked_csv", help="Ranked CSV with a file_path column (to pick a model by --id).")
    ap.add_argument("--id", help="Row id in --ranked_csv -> overlay that model's gm.")
    ap.add_argument("--dir", help="A run directory directly (alternative to --ranked_csv/--id).")
    ap.add_argument("--vds", default=None,
                    help="Comma-separated Vds setpoints to plot (default: ~6 spread over the data range).")
    ap.add_argument("--vgs_min", type=float, default=None, help="Zoom: only plot Vgs >= this.")
    ap.add_argument("--min_vgs", type=float, default=-4.0,
                    help="min_vgs passed to the data loader (extrapolation floor, not a plot zoom).")
    ap.add_argument("--n_grid", type=int, default=200, help="Uniform Vgs grid points for the model overlay.")
    ap.add_argument("--out", default=None, help="Output PNG (default: next to the CSV).")
    ap.add_argument("--open", action="store_true", help="Open the PNG when done.")
    args = ap.parse_args()

    # --- resolve an optional model to overlay (id in a ranked csv, or a run dir) ---
    run_dir = args.dir
    ranked_folder = None
    if args.id and args.ranked_csv:
        import csv as _csv
        rows = {r["id"]: r for r in _csv.DictReader(open(args.ranked_csv, newline="", encoding="utf-8"))}
        if args.id not in rows:
            sys.exit(f"[error] id {args.id} not in {args.ranked_csv}")
        run_dir = os.path.dirname(rows[args.id]["file_path"])
        ranked_folder = os.path.dirname(os.path.abspath(args.ranked_csv))

    # --- resolve the measurement CSV (explicit, else from --root / the ranked csv's folder) ---
    csv = args.csv
    if not csv:
        import run_artifacts
        for base in (args.root, os.path.dirname(ranked_folder) if ranked_folder else None,
                     os.path.dirname(os.path.dirname(ranked_folder)) if ranked_folder else None):
            if base:
                csv = run_artifacts.run_meta(base).get("csv")
                if csv:
                    print(f"[auto] measurement CSV: {csv}")
                    break
    if not csv or not os.path.isfile(csv):
        sys.exit("[error] measurement CSV not found; pass --csv or --root.")

    device = torch.device("cpu")
    model = None
    if run_dir:
        if not os.path.isdir(run_dir):
            sys.exit(f"[error] run dir not found: {run_dir}")
        from plot_saved_state import _build_model_from_dir
        with contextlib.redirect_stdout(io.StringIO()):
            model = _build_model_from_dir(os.path.abspath(run_dir), device)
        print(f"[model] overlaying {os.path.basename(run_dir)}")

    T = _load_gm_truth(csv, args.min_vgs, device)

    # each Step_Index group = a transfer curve: Ids(Vgs) at ~fixed Vds (one point per TN)
    vds_of = T.groupby("Step_Index")["Vds"].mean()
    avail = sorted(vds_of.unique())
    if args.vds:
        want = [float(x) for x in args.vds.split(",")]
    else:
        want = list(np.linspace(avail[0], avail[-1], 6))
    step_sel = []
    for v in want:
        step = (vds_of - v).abs().idxmin()
        if step not in [s for s, _ in step_sel]:
            step_sel.append((step, float(vds_of[step])))
    step_sel.sort(key=lambda t: t[1])
    print("plotting Vds traces:", [round(v, 2) for _, v in step_sel])

    panels = [("Ids  (A)", 1), ("gm1 = dIds/dVgs", 2), ("gm2 = d²Ids/dVgs²", 3),
              ("gm3 = d³Ids/dVgs³", 4)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 11), sharex=True)
    cmap = plt.get_cmap("viridis")
    for k, (step, vd) in enumerate(step_sel):
        grp = T[T["Step_Index"] == step].sort_values("Vgs_meas")
        vg_pts = grp["Vgs_meas"].values.astype(float)
        if args.vgs_min is not None:
            m = vg_pts >= args.vgs_min
        else:
            m = np.ones_like(vg_pts, dtype=bool)
        vg_pts, ids_pts = vg_pts[m], grp["Ids"].values.astype(float)[m]
        gm1_pts = grp["gm1"].values.astype(float)[m]
        gm2_pts = grp["gm2"].values.astype(float)[m]
        gm3_pts = grp["gm3"].values.astype(float)[m]
        color = cmap(k / max(len(step_sel) - 1, 1))
        dlabel = f"Vds={vd:.2f}" + (" (data)" if model is not None and k == 0 else "")
        for ax, series in zip(axes, (ids_pts, gm1_pts, gm2_pts, gm3_pts)):
            ax.plot(vg_pts, series, color=color, lw=1.6, marker=".", ms=3, label=dlabel)
        if model is not None:                        # overlay this arch's exact derivatives (dashed)
            vg_grid = np.linspace(vg_pts.min(), vg_pts.max(), args.n_grid)
            md0, md1, md2, md3 = _model_gm_derivs(model, vd, vg_grid, device)
            mlabel = "model" if k == 0 else None
            for ax, series in zip(axes, (md0, md1, md2, md3)):
                ax.plot(vg_grid, series, color=color, lw=1.4, ls="--", label=mlabel)
    for ax, (title, _) in zip(axes, panels):
        ax.set_ylabel(title)
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.grid(alpha=0.25)
    _ttl = "gm (Vgs) derivatives"
    if model is not None:
        _ttl += f"  --  data (solid) vs model {os.path.basename(os.path.dirname(run_dir))}"
        if args.id:
            _ttl += f" id={args.id}"
        _ttl += " (dashed)"
    axes[0].set_title(f"{_ttl}\n({os.path.basename(csv)})", fontsize=10)
    axes[-1].set_xlabel("Vgs  (V)")
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()

    suffix = f"_id{args.id}" if args.id else ("_model" if model is not None else "")
    out = args.out or os.path.splitext(csv)[0] + f"_gm_derivs{suffix}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    if args.open:
        try:
            os.startfile(out)
        except Exception as e:
            print(f"[open] {e}")


if __name__ == "__main__":
    main()
