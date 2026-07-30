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
import run_artifacts

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


def _region_name_of(metric):
    """region_<name>_<suffix> -> <name> (or None if not a region metric)."""
    if not metric.startswith("region_"):
        return None
    body = metric[len("region_"):]
    for suf in _REGION_SUFFIXES:
        if body.endswith(suf):
            return body[:-len(suf)]
    return None


def _fmt_num(x):
    try:
        f = float(x)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except (TypeError, ValueError):
        return str(x)


def _range_tag(vgs, vds):
    """Folder-safe tag for a region metric window, e.g. (-2.5,0),(0,6) -> vgs-2.5to0_vds0to6."""
    return (f"vgs{_fmt_num(vgs[0])}to{_fmt_num(vgs[1])}"
            f"_vds{_fmt_num(vds[0])}to{_fmt_num(vds[1])}")


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

    run_dir = os.path.dirname(json_path)
    is_pure = d.get("equation_type") == "pure"
    b1, b2, b3 = baselines
    combined = (d["gm1_rmse"] / b1 + d["gm2_rmse"] / b2 + d["gm3_rmse"] / b3) / 3.0
    # gm weights: prefer the run JSON (records all three since the gmN_weight fix); fall back
    # to the run-dir name for older runs whose JSON stored only gm1_weight. Coerce to float.
    _dn = _weights_from_dirname(json_path)
    def _gw(key, i):
        v = d.get(key)
        if v is None:
            v = _dn[i]
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    g1, g2, g3 = _gw("gm1_weight", 0), _gw("gm2_weight", 1), _gw("gm3_weight", 2)

    row = {
        "id": None,   # 1-based rank in this file (1 = best by this file's metric); reassigned per file
        "run_id": None,    # readable 1..N per distinct run; SAME in every ranked file (assigned after collection)
        "run_hash": run_artifacts.run_hash(json_path),   # globally-unique stable id for this run
        "category": category, "combo": combo, "phase": phase,
        "arch_id": None,   # small readable 1..N per distinct arch; assigned after collection
        "arch_hash": run_artifacts.arch_id(d.get("architecture", "")),   # stable id (same arch -> same hash)
        "base_exp": "NA", "base_id": "NA", "base_csv": "NA",   # lineage; filled after collection
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
        # ids-region box shape (present in the JSON; surfaced here for completeness)
        "ids_region_center": d.get("ids_region_center", ""),
        "ids_region_width": d.get("ids_region_width", ""),
        "ids_region_lo": d.get("ids_region_lo", ""),
        "ids_region_hi": d.get("ids_region_hi", ""),
        "deterministic": d.get("deterministic", ""),
        "mixed_init": d.get("mixed_init", ""),
        "vds_loss": d.get("vds_loss", 0),   # absent (pre-feature runs) = penalty off = 0
        # Config/DEFAULTS-level knobs: newer JSONs record them; for older runs fall back
        # to the exact invocation preserved in run_log.txt.gz (CMD line).
        "gm_max_ratio": run_artifacts.field(d, run_dir, "gm_max_ratio"),
        "lbfgs_epochs": run_artifacts.field(d, run_dir, "lbfgs_epochs"),
        "lbfgs_max_iter": run_artifacts.field(d, run_dir, "lbfgs_max_iter"),
        "lbfgs_gm_aware": run_artifacts.field(d, run_dir, "lbfgs_gm_aware"),
        "loss_norm": run_artifacts.field(d, run_dir, "loss_norm"),
        "adamw_avoid_localmin": run_artifacts.field(d, run_dir, "adamw_avoid_localmin"),
        "csv": run_artifacts.field(d, run_dir, "csv"),
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
        if parts & {"plotted_configs", "best_n_configs", "base_files"}:
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
    ap.add_argument("--base_csv", default=None,
                    help="Base ranked CSV this sweep was generated from (Option B): fills "
                         "base_exp/base_id/base_csv by matching arch_hash. Not needed when the "
                         "runs carry a _provenance.json (written from a gen_config_from_rows config).")
    ap.add_argument("--base_exp", default=None,
                    help="Override the base experiment name (default: parsed from --base_csv path).")
    ap.add_argument("--region_range_only", action="store_true",
                    help="Only write the range-tagged region folder(s) "
                         "(ranked_region_<name>_vgs..to.._vds..to../), for combined_gm and ids_rmse "
                         "only. Skips ranked_global/, the plain ranked_region_<name>/ folder, and "
                         "the per-region gm1/gm2/gm3-only rankings (which only ever lived in the "
                         "now-skipped plain folder) -- cuts a large sweep's ranked-CSV count from "
                         "~12 to 2, which cascades through xlsx conversion and the filter steps too. "
                         "Ignored for metrics named explicitly via --metrics (those are always honored "
                         "and written to their normal folder).")
    ap.add_argument("--short_name", action="store_true",
                    help="Use a short hash of the sweep name instead of the full name in "
                         "output filenames (<hash>_ranked_by_<metric>.csv instead of "
                         "<name>_ranked_by_<metric>.csv). Off by default -- OPT IN, since some "
                         "tooling (e.g. plot_run_id_all_csvs.ps1) reconstructs the full-name "
                         "filename directly and would break if this became the default. Useful "
                         "for sweeps whose downstream filenames get long (filter -> shape -> "
                         "gmshape_ok -> slim can otherwise push paths near Excel's/Windows' "
                         "path-length limits).")
    args = ap.parse_args()

    target = args.root or args.dir or args.root_pos
    if not target:
        ap.error("provide a directory: --root <master_root>  (or --dir <experiment>, or positional).")
    if not os.path.isdir(target):
        ap.error(f"not a directory: {target}")
    is_root = args.dir is None  # --dir is the only single-experiment mode

    if not args.base_csv:   # auto-detect the base ranked CSV from the results folder
        _meta = run_artifacts.run_meta(target)
        _bc = _meta.get("base_csv")
        if _bc and os.path.isfile(_bc):
            args.base_csv = _bc
            # the base CSV is copied into base_files/, so the folder name isn't the base
            # experiment -- take the true base_exp from the manifest.
            args.base_exp = args.base_exp or _meta.get("base_exp")
            print(f"[auto] base CSV: {_bc}"
                  + (f"  (base_exp={args.base_exp})" if args.base_exp else ""))

    baselines = DEFAULT_BASELINES
    if args.baselines:
        baselines = tuple(float(x) for x in args.baselines.split(","))

    rows = collect_rows(target, is_root, baselines)
    if not rows:
        sub = "<root>/*/*/" if is_root else "<dir>/*/"
        print(f"No run_loss_*.json found under {target} ({sub} run_loss_*.json). "
              f"If you pointed at a single experiment, use --dir instead of --root.")
        sys.exit(1)

    # Assign a small readable arch_id (1..N) per distinct architecture (keyed by arch_hash),
    # in a stable order, so a sweep's runs group by a simple number. arch_hash stays the id.
    _num = {h: i for i, h in enumerate(sorted({r["arch_hash"] for r in rows}), 1)}
    for r in rows:
        r["arch_id"] = _num[r["arch_hash"]]
    print(f"{len(_num)} distinct architecture(s) -> arch_id 1..{len(_num)}")

    # Global run_id: 1..N per distinct run (run_hash), stable order, SAME across every ranked file.
    _rnum = {h: i for i, h in enumerate(sorted({r["run_hash"] for r in rows}), 1)}
    for r in rows:
        r["run_id"] = _rnum[r["run_hash"]]
    print(f"{len(_rnum)} run(s) -> run_id 1..{len(_rnum)}")

    # Region metric windows (vgs/vds bounds) for the range-tagged archive folders. Read from
    # one run's region_metrics (uniform within a compile run).
    region_windows = {}
    try:
        _d0 = json.load(open(rows[0]["file_path"], encoding="utf-8"))
        for _nm, _rec in (_d0.get("region_metrics") or {}).items():
            if isinstance(_rec, dict) and _rec.get("vgs") and _rec.get("vds"):
                region_windows[_nm] = (_rec["vgs"], _rec["vds"])
    except Exception:
        pass

    # Lineage: base_exp / base_id / base_csv, from the _provenance.json sidecar (A) or --base_csv (B).
    _base_index = run_artifacts.base_csv_index(args.base_csv, args.base_exp) if args.base_csv else None
    _linked = 0
    for r in rows:
        be, bid, bc = run_artifacts.base_lineage(r["file_path"], r["arch_hash"], _base_index)
        r["base_exp"], r["base_id"], r["base_csv"] = be, bid, bc
        if be != "NA":
            _linked += 1
    if _linked:
        print(f"linked {_linked}/{len(rows)} runs to a base experiment")

    out_dir = args.out or target
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.normpath(target))
    # Output-filename prefix: the full name, or (--short_name) a short hash of it -- see
    # --short_name help. Only affects the FILENAME prefix; `name` itself (used for category/
    # combo parsing etc. elsewhere) is untouched.
    file_prefix = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6] if args.short_name else name
    if args.short_name:
        print(f"--short_name: filenames use {file_prefix!r} instead of {name!r}")
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
    elif args.region_range_only:
        # Only the region metrics that get archived into a range-tagged folder (combined_gm,
        # ids_rmse) -- skips ranked_global entirely and region gm1/gm2/gm3 (plain-folder-only).
        rank_metrics = [m for m in region_rank if m.endswith(("_combined_gm", "_ids_rmse"))]
        auto_skip = set(rank_metrics)
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
        best = ranked[0]

        # Whether this metric has a range-tagged region-archive destination (combined_gm / ids_rmse
        # of a region with a known vgs/vds window). With --region_range_only, that archive is the
        # ONLY copy written for such metrics -- the plain/global metric_dir write below is skipped.
        _rname = _region_name_of(metric)
        _has_range_archive = bool(_rname and (metric.endswith("_combined_gm") or metric.endswith("_ids_rmse"))
                                   and _rname in region_windows)

        if not (args.region_range_only and _has_range_archive):
            metric_dir = os.path.join(out_dir, _metric_subfolder(metric))
            os.makedirs(metric_dir, exist_ok=True)
            out_path = os.path.join(metric_dir, f"{file_prefix}_ranked_by_{metric}.csv")
            try:
                with open(out_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
                    w.writeheader()
                    w.writerows(ranked)
                print(f"wrote {out_path}  ({len(ranked)} runs)  best {metric}={best.get(metric)}  "
                      f"(ids={best.get('ids_rmse')}, {best['category']}/{best['combo']}/{best['gm_surgery_mode']})")
            except PermissionError:
                print(f"  [SKIP] {out_path} is open/locked (e.g. in Excel) -- close it and re-run.")
                continue

        # Also archive the combined_gm / ids_rmse region rankings into a range-tagged folder
        # (ranked_region_<name>_vgs..to.._vds..to..) so rankings from DIFFERENT metric windows
        # don't overwrite each other. With --region_range_only this is the ONLY copy written.
        if _has_range_archive:
            _vgs, _vds = region_windows[_rname]
            _rdir = os.path.join(out_dir, f"ranked_region_{_rname}_{_range_tag(_vgs, _vds)}")
            os.makedirs(_rdir, exist_ok=True)
            _rpath = os.path.join(_rdir, f"{file_prefix}_ranked_by_{metric}.csv")
            try:
                with open(_rpath, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=fields, restval="", extrasaction="ignore")
                    w.writeheader()
                    w.writerows(ranked)
                print(f"  archived -> {_rpath}")
            except PermissionError:
                print(f"  [SKIP] {_rpath} is open/locked (e.g. in Excel) -- close it and re-run.")
        if args.top:
            print(f"  --- top {args.top} by {metric} ---")
            for r in ranked[:args.top]:
                print(f"    {str(r.get(metric)):<10} ids={r['ids_rmse']:<8} gm1={r['gm1_rmse']:<8} "
                      f"gm2={r['gm2_rmse']:<8} gm3={r['gm3_rmse']:<8} "
                      f"{r['category']}/{r['combo']} {r['gm_surgery_mode']} {r['architecture'][:32]}")


if __name__ == "__main__":
    main()
