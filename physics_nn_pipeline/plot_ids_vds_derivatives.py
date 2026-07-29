"""
plot_ids_vds_derivatives.py -- visualize the MEASURED data's Ids-Vds derivatives.

The Ids-Vds analog of gm (which is d^n Ids/dVgs^n) is the output-conductance family:
    gds1 = dIds/dVds,  gds2 = d^2Ids/dVds^2,  gds3 = d^3Ids/dVds^3
Each measurement trace (TN) is a Vds sweep at a fixed Vgs, so these are derivatives ALONG
each trace. Looking at how the REAL data behaves in these derivatives tells us what a
"good" model should reproduce -- and whether a 1st/2nd/3rd Vds-derivative metric (used the
same way as the gm-shape metrics) would flag the knee-region Ids-Vds bumps/dips.

Derivatives are computed on a uniform Vds grid (per trace) with a Savitzky-Golay filter, so
the higher orders are smooth and readable rather than pure-noise (real data + np.gradient^3
would be unusable).

Usage
-----
  python plot_ids_vds_derivatives.py --csv ..\csvs\<meas>.csv
  python plot_ids_vds_derivatives.py --root ..\runs\<exp>          # auto-detect csv from manifest
  python plot_ids_vds_derivatives.py --csv <..> --vgs " -3,-2,-1,-0.5,0" --vds_max 15 --open
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
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import torch
from plot_saved_state import _load_T_train


def _derivs_on_trace(vds, ids, n_grid, win, poly):
    """Return (vu, d0, d1, d2, d3): Ids and its 1st/2nd/3rd Vds-derivatives on a uniform Vds grid."""
    order = np.argsort(vds)
    vds, ids = vds[order], ids[order]
    vu = np.linspace(vds.min(), vds.max(), n_grid)
    iu = np.interp(vu, vds, ids)
    dx = vu[1] - vu[0]
    w = min(win, n_grid if n_grid % 2 else n_grid - 1)
    if w % 2 == 0:
        w += 1
    w = max(w, poly + 1 + (poly % 2 == 0))
    d0 = savgol_filter(iu, w, poly)
    d1 = savgol_filter(iu, w, poly, deriv=1, delta=dx)
    d2 = savgol_filter(iu, w, poly, deriv=2, delta=dx)
    d3 = savgol_filter(iu, w, poly, deriv=3, delta=dx)
    return vu, d0, d1, d2, d3


def _model_derivs(model, vgs_val, vds_grid, device):
    """(d0,d1,d2,d3) = model Ids and its 1st/2nd/3rd Vds-derivatives (exact, autograd) at fixed Vgs."""
    vd = torch.tensor(vds_grid, dtype=torch.float64, device=device)
    vg = torch.full_like(vd, float(vgs_val))
    X = torch.stack([vg, vd], dim=1).requires_grad_(True)
    ids = model(X).reshape(-1)
    g1 = torch.autograd.grad(ids.sum(), X, create_graph=True)[0][:, 1]
    g2 = torch.autograd.grad(g1.sum(), X, create_graph=True)[0][:, 1]
    g3 = torch.autograd.grad(g2.sum(), X)[0][:, 1]
    d = lambda t: t.detach().cpu().numpy()
    return d(ids), d(g1), d(g2), d(g3)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="Measurement CSV. If omitted, auto-detected from --root or --ranked_csv.")
    ap.add_argument("--root", help="Results folder; auto-detect the measurement CSV from its manifest.")
    # optional model overlay: a trained run's Ids-Vds derivatives (exact, autograd) on top of data
    ap.add_argument("--ranked_csv", help="Ranked CSV with a file_path column (to pick a model by --id).")
    ap.add_argument("--id", help="Row id in --ranked_csv -> overlay that model's Ids-Vds derivatives.")
    ap.add_argument("--dir", help="A run directory directly (alternative to --ranked_csv/--id).")
    ap.add_argument("--vgs", default=None,
                    help="Comma-separated Vgs setpoints to plot (default: ~7 spread over -3..0).")
    ap.add_argument("--vds_max", type=float, default=None, help="Zoom: only plot Vds up to this (knee).")
    ap.add_argument("--min_vgs", type=float, default=-4.0)
    ap.add_argument("--n_grid", type=int, default=200, help="Uniform Vds grid points per trace.")
    ap.add_argument("--win", type=int, default=21, help="Savitzky-Golay window (odd).")
    ap.add_argument("--poly", type=int, default=4, help="Savitzky-Golay polynomial order (>=3).")
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

    model = None
    if run_dir:
        if not os.path.isdir(run_dir):
            sys.exit(f"[error] run dir not found: {run_dir}")
        from plot_saved_state import _build_model_from_dir
        with contextlib.redirect_stdout(io.StringIO()):
            model = _build_model_from_dir(os.path.abspath(run_dir), torch.device("cpu"))
        print(f"[model] overlaying {os.path.basename(run_dir)}")

    with contextlib.redirect_stdout(io.StringIO()):
        T = _load_T_train(csv, args.min_vgs, torch.device("cpu"))

    # each TN = a Vds sweep at a fixed Vgs setpoint
    vgs_of = T.groupby("TN")["Vgs"].first()
    avail = sorted(vgs_of.unique())
    if args.vgs:
        want = [float(x) for x in args.vgs.split(",")]
    else:
        want = list(np.linspace(-3, 0, 7))
    tn_sel = []
    for v in want:
        tn = (vgs_of - v).abs().idxmin()
        if tn not in [t for t, _ in tn_sel]:
            tn_sel.append((tn, float(vgs_of[tn])))
    tn_sel.sort(key=lambda t: t[1])
    print("plotting Vgs traces:", [round(v, 2) for _, v in tn_sel])

    panels = [("Ids  (A)", 1), ("gds1 = dIds/dVds", 2), ("gds2 = d²Ids/dVds²", 3),
              ("gds3 = d³Ids/dVds³", 4)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 11), sharex=True)
    cmap = plt.get_cmap("viridis")
    for k, (tn, vg) in enumerate(tn_sel):
        grp = T[T["TN"] == tn]
        vu, d0, d1, d2, d3 = _derivs_on_trace(grp["Vds"].values.astype(float),
                                              grp["Ids"].values.astype(float),
                                              args.n_grid, args.win, args.poly)
        if args.vds_max:
            m = vu <= args.vds_max
            vu, d0, d1, d2, d3 = vu[m], d0[m], d1[m], d2[m], d3[m]
        color = cmap(k / max(len(tn_sel) - 1, 1))
        dlabel = f"Vgs={vg:.2f}" + (" (data)" if model is not None and k == 0 else "")
        for ax, series in zip(axes, (d0, d1, d2, d3)):
            ax.plot(vu, series, color=color, lw=1.6, label=dlabel)
        if model is not None:                        # overlay this arch's exact derivatives (dashed)
            md0, md1, md2, md3 = _model_derivs(model, vg, vu, torch.device("cpu"))
            mlabel = "model" if k == 0 else None
            for ax, series in zip(axes, (md0, md1, md2, md3)):
                ax.plot(vu, series, color=color, lw=1.4, ls="--", label=mlabel)
    for ax, (title, _) in zip(axes, panels):
        ax.set_ylabel(title)
        ax.axhline(0, color="0.7", lw=0.8, zorder=0)
        ax.grid(alpha=0.25)
    _ttl = "Ids-Vds derivatives"
    if model is not None:
        _ttl += f"  --  data (solid) vs model {os.path.basename(os.path.dirname(run_dir))}"
        if args.id:
            _ttl += f" id={args.id}"
        _ttl += " (dashed)"
    axes[0].set_title(f"{_ttl}\n({os.path.basename(csv)})", fontsize=10)
    axes[-1].set_xlabel("Vds  (V)")
    axes[0].legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()

    suffix = f"_id{args.id}" if args.id else ("_model" if model is not None else "")
    out = args.out or os.path.splitext(csv)[0] + f"_ids_vds_derivs{suffix}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    if args.open:
        try:
            os.startfile(out)
        except Exception as e:
            print(f"[open] {e}")


if __name__ == "__main__":
    main()
