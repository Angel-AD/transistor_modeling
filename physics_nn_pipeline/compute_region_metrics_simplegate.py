"""
compute_region_metrics.py  --  Region-localized Ids/gm error metrics for trained runs.

Loads a trained run's optimized weights (same mechanism as plot_saved_state.py),
evaluates the model on the measurement set, and computes RMSE/MAE restricted to one
or more user-specified (Vgs, Vds) windows.  The metrics are written back into the
run's run_loss_*.json so compile_results_csv.py / compile_ranked.py can rank by them.

Motivation
----------
Global Ids RMSE can rank a wiggly-but-low-error curve above a smooth one.  A region
metric (especially the gm RMSEs, which spike on the "little bumps") lets you isolate
error where physical behavior matters -- e.g. the turn-on knee.

Region syntax  (--region, repeatable)
-------------------------------------
    "vgs=-3..0,vds=0..10"            auto-named r1, r2, ...
    "knee:vgs=-3..0,vds=0..10"       explicit name 'knee'
    range separator may be '..', 'to', or ':'   (e.g. vgs=-3 to 0)

Each region writes, into the JSON:
    region_metrics[<name>] = {vgs, vds, n, ids_rmse, ids_mae, gm1_rmse, gm2_rmse, gm3_rmse}
  plus FLAT keys so the compile pipeline picks them up as columns:
    region_<name>_ids_rmse, region_<name>_ids_mae,
    region_<name>_gm1_rmse, region_<name>_gm2_rmse, region_<name>_gm3_rmse, region_<name>_n

Usage
-----
    # one run
    python compute_region_metrics.py --dir <run_dir> --csv ../csvs/<m>.csv \
        --region "knee:vgs=-3..0,vds=0..10"

    # every run under a master root (batch), two regions
    python compute_region_metrics.py --root refine8 --csv ../csvs/<m>.csv \
        --region "knee:vgs=-3..0,vds=0..10" --region "sat:vgs=-1..0,vds=15..28"
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from plot_saved_state_simplegate import _build_model_from_dir, _load_T_train
from per_neuron_simple_angelov_nn_test_simplegate import create_gms_for_train, get_gm_rmse_metrics


# --------------------------------------------------------------------------- #
# Region parsing
# --------------------------------------------------------------------------- #
_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)"
_RANGE = rf"{_FLOAT}\s*(?:\.\.|to|:)\s*{_FLOAT}"


def parse_region(spec: str, fallback_name: str) -> dict:
    """Parse 'name:vgs=lo..hi,vds=lo..hi' -> {name, vgs:(lo,hi), vds:(lo,hi)}."""
    s = spec.strip()
    name = fallback_name
    m = re.match(r"^\s*([A-Za-z]\w*)\s*:\s*", s)   # leading 'name:' (letters first => not 'vgs=')
    if m:
        name = m.group(1)
        s = s[m.end():]

    def _axis(axis):
        mm = re.search(rf"{axis}\s*=\s*({_FLOAT})\s*(?:\.\.|to|:)\s*({_FLOAT})", s, re.I)
        if not mm:
            sys.exit(f"[error] --region {spec!r}: could not find '{axis}=lo..hi'.")
        lo, hi = float(mm.group(1)), float(mm.group(2))
        return (min(lo, hi), max(lo, hi))

    return {"name": name, "vgs": _axis("vgs"), "vds": _axis("vds")}


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2))) if len(a) else float("nan")


def _mae(a, b):
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


# --------------------------------------------------------------------------- #
# Measurement data + ground-truth gms (model-independent -> computed ONCE)
# --------------------------------------------------------------------------- #
def load_truth(csv, min_vgs, add_zero_vds, device):
    T_train = _load_T_train(csv, min_vgs, device, add_zero_vds=add_zero_vds)
    gm1_t, gm2_t, gm3_t, X, y = create_gms_for_train(T_train)
    X_t = torch.tensor(X, dtype=torch.float64, device=device)
    y = y.flatten()
    return {"X": X, "X_t": X_t, "y": y,
            "gm1": gm1_t, "gm2": gm2_t, "gm3": gm3_t,
            "vgs": X[:, 0], "vds": X[:, 1]}


def region_masks(truth, regions):
    """Boolean mask per region (Vgs in [lo,hi] AND Vds in [lo,hi])."""
    out = {}
    for r in regions:
        vgs_lo, vgs_hi = r["vgs"]; vds_lo, vds_hi = r["vds"]
        m = ((truth["vgs"] >= vgs_lo) & (truth["vgs"] <= vgs_hi) &
             (truth["vds"] >= vds_lo) & (truth["vds"] <= vds_hi))
        out[r["name"]] = m
    return out


# --------------------------------------------------------------------------- #
# Per-run computation
# --------------------------------------------------------------------------- #
def compute_for_run(run_dir, truth, masks, regions, do_gm, device):
    # _build_model_from_dir / DynamicNN print verbose construction debug — silence it.
    with contextlib.redirect_stdout(io.StringIO()):
        model = _build_model_from_dir(os.path.abspath(run_dir), device)
    with torch.no_grad():
        ids_pred = model(truth["X_t"]).cpu().numpy().flatten()
    if do_gm:
        _, _, _, gm1_p, gm2_p, gm3_p = get_gm_rmse_metrics(
            model, truth["X_t"], truth["gm1"], truth["gm2"], truth["gm3"])

    results = {}
    rmap = {r["name"]: r for r in regions}
    for name, m in masks.items():
        n = int(m.sum())
        rec = {"vgs": list(rmap[name]["vgs"]), "vds": list(rmap[name]["vds"]), "n": n,
               "ids_rmse": _rmse(truth["y"][m], ids_pred[m]),
               "ids_mae":  _mae(truth["y"][m], ids_pred[m])}
        if do_gm:
            rec["gm1_rmse"] = _rmse(truth["gm1"][m], gm1_p[m])
            rec["gm2_rmse"] = _rmse(truth["gm2"][m], gm2_p[m])
            rec["gm3_rmse"] = _rmse(truth["gm3"][m], gm3_p[m])
        results[name] = rec
    return results


def write_back(run_dir, results, dry_run):
    """Merge region_metrics (nested) + region_<name>_<metric> (flat) into run_loss_*.json."""
    jsons = glob.glob(os.path.join(run_dir, "run_loss_*.json"))
    if not jsons:
        print(f"  [skip] no run_loss_*.json in {run_dir}")
        return False
    jpath = max(jsons, key=os.path.getmtime)
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    data.setdefault("region_metrics", {})
    for name, rec in results.items():
        data["region_metrics"][name] = rec
        for metric, val in rec.items():
            if metric in ("vgs", "vds"):
                continue
            data[f"region_{name}_{metric}"] = val

    if dry_run:
        return True
    tmp = jpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, jpath)
    return True


# --------------------------------------------------------------------------- #
# Run-dir discovery (batch)
# --------------------------------------------------------------------------- #
def find_run_dirs(root):
    out = []
    for dp, dirs, files in os.walk(root):
        parts = set(os.path.normpath(dp).split(os.sep))
        if parts & {"best_n_configs", "plotted_configs", "base_files"}:
            continue
        if any(f.startswith("weights_loss_") for f in files) and \
           any(f.startswith("run_loss_") for f in files):
            out.append(dp)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(
        description="Compute region-localized Ids/gm metrics and write them into run JSONs.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="A single run directory (contains run_loss_*.json + weights_loss_*.pt).")
    src.add_argument("--root", help="Master root; recurse into every run dir under it.")
    ap.add_argument("--csv", default=None, help="Measurement CSV (ground truth). If omitted, "
                    "auto-detected from the run folder (_run_meta.json / run records).")
    ap.add_argument("--region", action="append", required=True, metavar="SPEC",
                    help="Region 'name:vgs=lo..hi,vds=lo..hi' (repeatable). Range sep: .. / to / :")
    ap.add_argument("--min_vgs", type=float, default=-4.0)
    ap.add_argument("--add_zero_vds", action="store_true",
                    help="Match training: overwrite each TN group's first row to Vds=0, Ids=0.")
    ap.add_argument("--no_gm", action="store_true",
                    help="Skip gm RMSEs (faster). Only Ids RMSE/MAE per region.")
    ap.add_argument("--dry_run", action="store_true", help="Compute and print; do not write JSONs.")
    args = ap.parse_args()

    if not args.csv:   # auto-detect the measurement CSV from the results folder
        import run_artifacts
        _root = args.root or (os.path.dirname(os.path.dirname(os.path.abspath(args.dir))) if args.dir else None)
        args.csv = run_artifacts.run_meta(_root).get("csv") if _root else None
        if args.csv:
            print(f"[auto] measurement CSV: {args.csv}")
    if not args.csv:
        sys.exit("[error] --csv not given and could not be auto-detected from the run folder.")

    regions = [parse_region(s, f"r{i+1}") for i, s in enumerate(args.region)]
    names = [r["name"] for r in regions]
    if len(names) != len(set(names)):
        sys.exit(f"[error] duplicate region names: {names}")
    print("Regions:")
    for r in regions:
        print(f"  {r['name']}: Vgs in [{r['vgs'][0]}, {r['vgs'][1]}]  Vds in [{r['vds'][0]}, {r['vds'][1]}]")

    device = torch.device("cpu")
    truth = load_truth(args.csv, args.min_vgs, args.add_zero_vds, device)
    masks = region_masks(truth, regions)
    for r in regions:
        n = int(masks[r["name"]].sum())
        print(f"  [{r['name']}] {n}/{len(truth['y'])} measured points in region")
        if n == 0:
            print(f"  [warn] region {r['name']!r} contains no measured points; metrics will be NaN.")

    run_dirs = [args.dir] if args.dir else find_run_dirs(args.root)
    print(f"\n{len(run_dirs)} run dir(s) to process (do_gm={not args.no_gm}, dry_run={args.dry_run})\n")

    ok = fail = 0
    for i, rd in enumerate(run_dirs, 1):
        try:
            results = compute_for_run(rd, truth, masks, regions, not args.no_gm, device)
            if write_back(rd, results, args.dry_run):
                ok += 1
            tag = " | ".join(
                f"{n}: ids={results[n]['ids_rmse']:.4e}"
                + ("" if args.no_gm else f" gm2={results[n]['gm2_rmse']:.3e}")
                for n in names)
            print(f"[{i}/{len(run_dirs)}] {os.path.basename(rd)}  {tag}", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(run_dirs)}] FAILED {rd}: {e}", flush=True)

    print(f"\ndone. ok={ok} fail={fail}"
          + ("  (dry run -- nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
