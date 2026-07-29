"""
compile_ranked.py -- rank ALL runs into best-first CSVs (one per metric).

Companion to compile_results_csv.py. Where compile_results_csv writes one CSV per
sub-experiment with raw columns, THIS script pools every run under a master root
and writes a handful of globally-sorted CSVs -- each containing all runs ordered
best-first by one metric -- and adds a normalized `combined_gm` all-gm score.

    combined_gm = mean( gm1_rmse/0.04661, gm2_rmse/0.20352, gm3_rmse/0.94910 )

(each gm divided by its single-gm baseline so the three are comparable; lower = a
better all-around gm fit.)

USAGE
    python compile_ranked.py --root refine2          # all sub-experiments under refine2/
    python compile_ranked.py --root refine2 --top 50 # also print the top-50 of each to console
    python compile_ranked.py --dir  refine2/pureNN_refine__gm1   # a single experiment folder
    python compile_ranked.py refine2                 # positional shorthand for --root

OUTPUT  (written into the root/dir, prefixed with its name)
    <name>_ranked_by_ids_rmse.csv
    <name>_ranked_by_gm1_rmse.csv   (and gm2_rmse, gm3_rmse)
    <name>_ranked_by_combined_gm.csv

Each row: category, combo, phase, eq, knee_combiner, best_loss, combined_gm,
ids_rmse, gm1/2/3_rmse, use_gm, gm1/2/3_weight, gm_surgery_mode,
output_activation, config_name, lr, architecture, file_path.
  - category/combo/phase are parsed from the sub-experiment folder name if it is
    of the form  <category>__<combo>__<phase>  (e.g. physicsNN_refinePC__gm2__amp);
    otherwise `category` is the whole folder name and combo/phase are blank.
  - eq / knee_combiner are set to "NA" for pure-NN runs (no physics/knee).
  - gm1/2/3_weight are read from the run-dir name (the JSON stores gm2/gm3 as null).
"""
import argparse
import csv
import glob
import json
import os
import sys
import hashlib

# gm baselines for the normalized combined score (best single-gm RMSEs from the
# base architecture search). Override with --baselines g1,g2,g3 if desired.
DEFAULT_BASELINES = (0.04661, 0.20352, 0.94910)

# metrics we produce a sorted CSV for (all "lower is better")
METRICS = ["ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse", "combined_gm"]

# Region-metric suffixes (mirror compute_region_metrics.py) used to peel off the region
# name for folder grouping, e.g. region_knee_gm2_rmse -> region name "knee".
_REGION_SUFFIXES = ("_ids_rmse", "_gm1_rmse", "_gm2_rmse", "_gm3_rmse",
                    "_combined_gm", "_ids_mae", "_n")


def _metric_subfolder(metric):
    """Folder a ranked CSV is grouped into: whole-curve metrics -> 'ranked_global',
    region metrics -> 'ranked_region_<name>' (one folder per region)."""
    if metric.startswith("region_"):
        body = metric[len("region_"):]
        for suf in _REGION_SUFFIXES:
            if body.endswith(suf):
                return f"ranked_region_{body[:-len(suf)]}"
        return "ranked_region"
    return "ranked_global"


def _weights_from_dirname(json_path):
    """gm1/gm2/gm3 weights, parsed from the run-dir name (..._W<g1>-<g2>-<g3>_s<seed>)."""
    name = os.path.basename(os.path.dirname(json_path))
    if "_W" in name and "_s" in name:
        parts = name.split("_W", 1)[1].split("_s", 1)[0].split("-")
        if len(parts) == 3:
            return parts
    return "", "", ""


def _row_from_json(json_path, baselines):
    """Build one output row from a run_loss_*.json, or None if it's not a valid run."""
    try:
        d = json.load(open(json_path))
    except Exception:
        return None
    if not all(k in d for k in ("ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse")):
        return None

    # sub-experiment folder = the path component right under the root we globbed
    sub = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
    bits = sub.split("__")
    category = bits[0]
    combo = bits[1] if len(bits) > 1 else ""
    phase = bits[2] if len(bits) > 2 else ""

    is_pure = d.get("equation_type") == "pure"
    b1, b2, b3 = baselines
    combined = (d["gm1_rmse"] / b1 + d["gm2_rmse"] / b2 + d["gm3_rmse"] / b3) / 3.0
    g1, g2, g3 = _weights_from_dirname(json_path)

    row = {
        "id": None,   # 1-based rank in this file (1 = best); assigned at write time
        "category": category, "combo": combo, "phase": phase,
        "eq": "NA" if is_pure else d.get("equation_type"),
        "knee_combiner": "NA" if is_pure else d.get("knee_combiner"),
        "best_loss": round(d.get("best_loss", d.get("mse_loss", float("nan"))), 6),
        "combined_gm": round(combined, 4),
        "ids_rmse": round(d["ids_rmse"], 5),
        "gm1_rmse": round(d["gm1_rmse"], 5),
        "gm2_rmse": round(d["gm2_rmse"], 5),
        "gm3_rmse": round(d["gm3_rmse"], 5),
        "use_gm": d.get("use_gm"),
        "gm1_weight": g1, "gm2_weight": g2, "gm3_weight": g3,
        "gm_surgery_mode": d.get("gm_surgery_mode"),
        "output_activation": d.get("output_activation"),
        "config_name": d.get("config_name", ""),
        "lr": d.get("lr", d.get("learning_rate", "")),
        # --- full physics / knee-window / Ids-guard config (added for inspection) ---
        "freeze_physics": d.get("freeze_physics", ""),
        "use_opt_params": d.get("use_opt_params", ""),
        "knee_alpha_scale": d.get("knee_alpha_scale", ""),
        "knee_vgs_thr": d.get("knee_vgs_thr", ""),
        "knee_vgs_tau": d.get("knee_vgs_tau", ""),
        "knee_max_correction": d.get("knee_max_correction", ""),
        "ids_constraint": d.get("ids_constraint", ""),
        "ids_target": d.get("ids_target", ""),
        "ids_lambda": d.get("ids_lambda", ""),
        "gm_warmup_epochs": d.get("gm_warmup_epochs", ""),
        "gm_warmup_lr": d.get("gm_warmup_lr", ""),
        "gm_vds_min": d.get("gm_vds_min", ""),
        "gm_vgs_min": d.get("gm_vgs_min", ""),
        "ids_region_weight": d.get("ids_region_weight", ""),
        "epochs": d.get("epochs", ""),
        "seed": d.get("seed", ""),
        "architecture": d["architecture"],
        "file_path": os.path.abspath(json_path),
    }
    # Region-localized metrics (compute_region_metrics.py) -> flat region_<name>_<metric>.
    row.update({k: v for k, v in d.items()
                if k.startswith("region_") and k != "region_metrics"})
    # Region combined_gm: same baseline-normalized mean of the three region gm RMSEs as the
    # global combined_gm, derived per region so a region can be ranked by overall gm fidelity.
    for gk in [k for k in row if k.startswith("region_") and k.endswith("_gm1_rmse")]:
        rname = gk[len("region_"):-len("_gm1_rmse")]
        try:
            rg1 = float(row[f"region_{rname}_gm1_rmse"])
            rg2 = float(row[f"region_{rname}_gm2_rmse"])
            rg3 = float(row[f"region_{rname}_gm3_rmse"])
            row[f"region_{rname}_combined_gm"] = round((rg1 / b1 + rg2 / b2 + rg3 / b3) / 3.0, 4)
        except (KeyError, ValueError, TypeError):
            pass
    return row


def collect_rows(target, is_root, baselines):
    """Read all runs under `target`. is_root=True -> target/<exp>/<run>/; else target/<run>/."""
    pattern = (os.path.join(target, "*", "*", "run_loss_*.json") if is_root
               else os.path.join(target, "*", "run_loss_*.json"))
    rows = []
    for jp in glob.glob(pattern):
        # Skip copies that plot_csv_row.py / plot_best_configs.py drop into these dirs — they have a
        # run_loss_*.json (matches the glob) but no weights, and would double-count / shift ids.
        parts = set(os.path.normpath(jp).split(os.sep))
        if "plotted_configs" in parts or "best_n_configs" in parts:
            continue
        row = _row_from_json(jp, baselines)
        if row is not None:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Rank all runs into best-first CSVs (one per metric), with a combined_gm score.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--root", help="Master root dir; ranks all runs under <root>/*/*/.")
    src.add_argument("--dir", help="A single experiment folder; ranks runs under <dir>/*/.")
    ap.add_argument("root_pos", nargs="?", help="Positional shorthand for --root.")
    ap.add_argument("--out", help="Output directory (default: the root/dir itself).")
    ap.add_argument("--top", type=int, default=0,
                    help="Also print the top-N rows of each ranking to the console.")
    ap.add_argument("--baselines", default=None,
                    help="Comma-separated gm1,gm2,gm3 baselines for combined_gm "
                         f"(default {DEFAULT_BASELINES}).")
    ap.add_argument("--metrics", default=None,
                    help="Comma-separated metric column(s) to rank by (one ranked CSV each). "
                         "Any column works, including region metrics, e.g. "
                         "--metrics region_knee_gm2_rmse,ids_rmse. "
                         f"Default: the standard set {METRICS} plus every region_*_*_rmse found.")
    args = ap.parse_args()

    target = args.root or args.dir or args.root_pos
    if not target:
        ap.error("provide a directory: --root <master_root>  (or --dir <experiment>, or positional).")
    if not os.path.isdir(target):
        ap.error(f"not a directory: {target}")
    is_root = args.dir is None  # --dir is the only single-experiment mode

    baselines = DEFAULT_BASELINES
    if args.baselines:
        baselines = tuple(float(x) for x in args.baselines.split(","))

    rows = collect_rows(target, is_root, baselines)
    if not rows:
        sub = "<root>/*/*/" if is_root else "<dir>/*/"
        print(f"No run_loss_*.json found under {target} ({sub} run_loss_*.json). "
              f"If you pointed at a single experiment, use --dir instead of --root.")
        sys.exit(1)

    out_dir = args.out or target
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(target))
    # Robust header: region_* columns vary across runs (some processed, some not), so use
    # the UNION of all rows' keys (not just rows[0]) and slot region cols before arch/file_path.
    _base = [k for k in rows[0].keys()
             if not k.startswith("region_") and k not in ("architecture", "file_path")]
    _region = sorted({k for r in rows for k in r if k.startswith("region_")})
    _tail = [k for k in ("architecture", "file_path") if k in rows[0]]
    fields = _base + _region + _tail

    # Auto-detect region ranking metrics (compute_region_metrics.py): emit a ranked CSV per
    # region RMSE column, e.g. ranked_by_region_knee_gm2_rmse.csv for the smoothest-knee runs.
    # Region name is whatever the user passed to --region, so this adapts to any naming.
    region_rank = sorted({k for r in rows for k in r
                          if k.startswith("region_")
                          and k.endswith(("_ids_rmse", "_gm1_rmse", "_gm2_rmse",
                                          "_gm3_rmse", "_combined_gm"))})

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")    # missing/blank (un-annotated runs) sorts last

    # --metrics selects exactly which columns to rank by; default = standard set + every
    # region RMSE found. Validate user-named columns exist in the data (else warn + skip).
    if args.metrics:
        wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
        present = set().union(*(r.keys() for r in rows))
        rank_metrics = []
        for m in wanted:
            if m in present:
                rank_metrics.append(m)
            else:
                print(f"  [WARN] --metrics {m!r} not a column in the data; skipping. "
                      f"(region metrics need compute_region_metrics.py run first.)")
        if not rank_metrics:
            sys.exit("[error] --metrics matched no columns; nothing to rank.")
        auto_skip = set()      # explicit selection: don't auto-skip
    else:
        rank_metrics = METRICS + region_rank
        auto_skip = set(region_rank)

    for metric in rank_metrics:
        # Region metrics are absent on runs not yet post-processed; skip if none have a value
        # (only for the auto-detected defaults; an explicit --metrics request is always honored).
        if metric in auto_skip and not any(_num(r.get(metric)) != float("inf") for r in rows):
            continue
        ranked = sorted(rows, key=lambda r: _num(r.get(metric)))
        for _i, _r in enumerate(ranked, 1):      # 1-based id in this file (1 = best by metric)
            _r["id"] = _i
        metric_dir = os.path.join(out_dir, _metric_subfolder(metric))
        os.makedirs(metric_dir, exist_ok=True)
        out_path = os.path.join(metric_dir, f"{name}_ranked_by_{metric}.csv")
        try:
            with open(out_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
                w.writeheader()
                w.writerows(ranked)
        except PermissionError:
            print(f"  [SKIP] {out_path} is open/locked (e.g. in Excel) -- close it and re-run.")
            continue
        best = ranked[0]
        print(f"wrote {out_path}  ({len(ranked)} runs)  best {metric}={best.get(metric)}  "
              f"(ids={best.get('ids_rmse')}, {best['category']}/{best['combo']}/{best['gm_surgery_mode']})")
        if args.top:
            print(f"  --- top {args.top} by {metric} ---")
            for r in ranked[:args.top]:
                print(f"    {str(r.get(metric)):<10} ids={r['ids_rmse']:<8} gm1={r['gm1_rmse']:<8} "
                      f"gm2={r['gm2_rmse']:<8} gm3={r['gm3_rmse']:<8} "
                      f"{r['category']}/{r['combo']} {r['gm_surgery_mode']} {r['architecture'][:32]}")


if __name__ == "__main__":
    main()
