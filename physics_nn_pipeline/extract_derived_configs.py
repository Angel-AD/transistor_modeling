"""
Derive best200-style and bothshapeok-style configs from a pure_combined9069-style sweep (one
folder per equation-type/output-activation/gm combo, single measurement csv, ~9069 architectures
each) -- the same derivation that originally turned runs/pure_combined9069/ into
runs/best200ids_of_9069_byloss/ and runs/bothshapeok_of_9069_byshape/, applied to a NEW sweep
(e.g. runs/pure_combined9069_rw0_2.5_20/).

Three independent derivations per folder:

  bothshapeok (fully mechanical): full-population analyze_shape.py -> bothshape_ok id list ->
  gen_config_from_rows.py on those ids. Same shape-compliance formula used throughout this
  session's analysis (important_mds/shape_analysis_rules.md). If --bothshapeok_top_n is set and
  the bothshape_ok pool exceeds it, applies the SAME gate-then-rank selection as best200/
  best100_gmshapeok (Approach 1: escalating combined_gm ceiling, then top-N by ids_rmse) rather
  than a plain ids_rmse sort -- keeps all three derivations' "top N" selection consistent.

  best200 (needs a per-folder combined_gm ceiling, auto-searched here since the ORIGINAL
  per-folder thresholds in important_mds/pure_combined9069.md were tuned against the 2.4_5/rw4
  population and aren't guaranteed to land near --top_n rows against a DIFFERENT csv/region_weight):
  start at --gm_ceiling_start, escalate x1.5 until at least --top_n rows pass, then
  filter_results.py --top_n <N> --sort_by region_knee_ids_rmse to pick the best N of those,
  then gen_config_from_rows.py on that filtered csv.

  best100_gmshapeok: SAME principle as best200 (auto-escalating combined_gm ceiling, top-N by
  ids_rmse) but restricted to the subset that's gmshape_ok=True (the weaker of the two shape
  checks -- gm-smoothness only, doesn't require gds_residual_bad_frac==0 too). Needs the
  full-population shape.csv (reuses the one bothshapeok already computed/reused, if run in the
  same invocation), then does the combined_gm-ceiling search + top-N selection entirely in
  Python against that gmshape_ok-filtered row set (no filter_results.py step, since it doesn't
  support filtering by an arbitrary precomputed id subset -- gen_config_from_rows.py's own
  --ids list is used directly instead, same approach as bothshapeok).

Usage (run AFTER the source sweep's compile_overall.ps1 has finished for every folder):
    python extract_derived_configs.py --source_root ../runs/pure_combined9069_rw0_2.5_20
    python extract_derived_configs.py --source_root ../runs/pure_combined9069_rw0_2.5_20 \\
        --folders tanh_margin10,softplus --dry_run
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_csv_row import _read_rows  # noqa: E402

ANALYZE_SHAPE = HERE / "analyze_shape.py"
KEEP_COLUMNS = HERE / "keep_columns.py"
FILTER_RESULTS = HERE / "filter_results.py"
GEN_CONFIG = HERE / "gen_config_from_rows.py"

FOLDERS = ["sigmoid_margin10", "sigmoid_margin10_nogm", "softplus", "softplus_nogm",
           "tanh_margin10", "tanh_margin10_nogm", "vdsgate_aeff_quad_tanhm", "vdsgate_aeff_quad_v3"]

SHAPE_KEEP = [
    "id", "run_id", "arch_id", "arch_hash", "vds_loss", "region_knee_combined_gm",
    "region_knee_ids_rmse", "neg_gds_energy", "gds_hump", "gds_hump_max",
    "gm3_maxabs", "gm3_bad_frac", "gm3_start_diff", "gm2_start_diff",
    "gm3_tail_maxabs", "gm3_tail_bad_frac", "gm3_max_missing_frac", "gm3_min_missing_frac",
    "gm3_global_excess_max", "gm3_global_bad_frac",
    "gds_residual_bad_frac", "gds_residual_neg_max", "gds_residual_pos_max",
    "gds_residual_worst_max", "shape_rank",
    "gm3_n_min", "gm3_n_max", "gm3_start_bad_frac", "gm2_start_bad_frac",
]


def _f(row, key):
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def gmshape_ok(row) -> bool:
    return (
        (_f(row, "gm3_n_min") or 0) >= 1 and (_f(row, "gm3_n_max") or 0) >= 1 and
        _f(row, "gm3_bad_frac") == 0 and _f(row, "gm3_min_missing_frac") == 0 and
        _f(row, "gm3_max_missing_frac") == 0 and _f(row, "gm3_start_bad_frac") == 0 and
        _f(row, "gm2_start_bad_frac") == 0 and _f(row, "gm3_tail_bad_frac") == 0 and
        _f(row, "gm3_global_bad_frac") == 0
    )


def bothshape_ok(row) -> bool:
    return gmshape_ok(row) and _f(row, "gds_residual_bad_frac") == 0


def _base_ranked_csv(folder_root: Path) -> Path | None:
    ranked_dir = folder_root / "ranked_region_knee_vgs-3to0_vds0to15"
    matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
    return matches[0] if matches else None


def _run(cmd: list[str], dry_run: bool) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _escalate_and_select(rows: list[dict], top_n: int, gm_ceiling_start: float) -> tuple[list[dict], float]:
    """Approach 1 (gate-then-rank): escalate a region_knee_combined_gm ceiling x1.5 from
    gm_ceiling_start until >=top_n of `rows` pass it, then take the best top_n of THOSE by
    region_knee_ids_rmse. Same logic used by best200 and best100_gmshapeok, shared here so
    bothshapeok_top_n can use the identical selection criterion instead of a plain ids_rmse
    sort with no combined_gm gate at all."""
    gm_ceiling = gm_ceiling_start
    def _pool(ceiling):
        return [r for r in rows if (_f(r, "region_knee_combined_gm") or 1e9) <= ceiling]
    pool = _pool(gm_ceiling)
    while len(pool) < top_n and gm_ceiling < 20.0:
        gm_ceiling = round(gm_ceiling * 1.5, 4)
        pool = _pool(gm_ceiling)
    pool.sort(key=lambda r: _f(r, "region_knee_ids_rmse") if _f(r, "region_knee_ids_rmse") is not None else 1e9)
    return pool[:top_n], gm_ceiling


def _ensure_shape_csv(folder: str, base_csv: Path, python_exe: str, dry_run: bool) -> Path:
    """Full-population shape.csv for base_csv -- reused if already present, else computed
    (analyze_shape.py + keep_columns.py cleanup, same convention as run_full_pop_shape.py)."""
    shape_csv = base_csv.with_name(base_csv.stem + "_shape.csv")
    if shape_csv.is_file():
        print(f"  {folder}: reusing existing {shape_csv.name}")
        return shape_csv

    rc = _run([python_exe, str(ANALYZE_SHAPE), "--ranked_csv", str(base_csv)], dry_run)
    if rc != 0:
        print(f"  FAILED analyze_shape.py for {folder}")
        return shape_csv
    if not dry_run:
        shape_outputs = sorted(base_csv.parent.glob(f"{base_csv.stem}_shape*.csv"))
        if shape_outputs:
            _run([python_exe, str(KEEP_COLUMNS), "--csv", *[str(p) for p in shape_outputs],
                  "--keep", *SHAPE_KEEP], dry_run)
            for p in shape_outputs:
                if p.name != shape_csv.name:
                    p.unlink()
    return shape_csv


def do_bothshapeok(folder: str, folder_root: Path, out_configs_dir: Path, python_exe: str, dry_run: bool,
                    top_n: int | None = None, gm_ceiling_start: float = 0.9):
    base_csv = _base_ranked_csv(folder_root)
    if base_csv is None:
        print(f"  SKIP {folder}: no base ranked csv found under {folder_root}")
        return
    shape_csv = _ensure_shape_csv(folder, base_csv, python_exe, dry_run)

    if dry_run:
        cap_note = f" (capped to best {top_n} via gate-then-rank on combined_gm/ids_rmse)" if top_n else ""
        print(f"  (dry run) would compute bothshape_ok ids from {shape_csv.name}{cap_note} and gen_config")
        return

    rows = _read_rows(shape_csv)
    ok_rows = [r for r in rows if r.get("id") and bothshape_ok(r)]
    print(f"  {folder}: {len(ok_rows)} / {len(rows)} bothshape_ok")
    if not ok_rows:
        print(f"  SKIP gen_config for {folder}: 0 bothshape_ok rows")
        return

    if top_n is not None and len(ok_rows) > top_n:
        ok_rows, gm_ceiling = _escalate_and_select(ok_rows, top_n, gm_ceiling_start)
        print(f"  {folder}: capped to best {len(ok_rows)} bothshape_ok rows "
              f"(combined_gm<={gm_ceiling}, ranked by ids_rmse) -- same Approach 1 as best200")

    ok_ids = [r["id"] for r in ok_rows]

    out_configs_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_configs_dir / f"{folder}_bothshapeok.json"
    _run([python_exe, str(GEN_CONFIG),
          "--csv", str(base_csv),
          "--ids", ",".join(ok_ids),
          "--out", str(out_path),
          "--prefix", f"{folder}_bothshapeok"], dry_run)


def do_best100_gmshapeok(folder: str, folder_root: Path, out_configs_dir: Path, python_exe: str,
                          top_n: int, gm_ceiling_start: float, dry_run: bool):
    base_csv = _base_ranked_csv(folder_root)
    if base_csv is None:
        print(f"  SKIP {folder}: no base ranked csv found under {folder_root}")
        return
    shape_csv = _ensure_shape_csv(folder, base_csv, python_exe, dry_run)

    if dry_run:
        print(f"  (dry run) would select top {top_n} gmshape_ok rows from {shape_csv.name} and gen_config")
        return

    rows = _read_rows(shape_csv)
    gm_ok_rows = [r for r in rows if r.get("id") and gmshape_ok(r)]
    print(f"  {folder}: {len(gm_ok_rows)} / {len(rows)} gmshape_ok")
    if not gm_ok_rows:
        print(f"  SKIP gen_config for {folder}: 0 gmshape_ok rows")
        return

    chosen, gm_ceiling = _escalate_and_select(gm_ok_rows, top_n, gm_ceiling_start)
    print(f"  {folder}: gmshape_ok & combined_gm<={gm_ceiling} gives best {len(chosen)} by ids_rmse (need >={top_n})")
    chosen_ids = [r["id"] for r in chosen]

    out_configs_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_configs_dir / f"{folder}_best100_gmshapeok.json"
    _run([python_exe, str(GEN_CONFIG),
          "--csv", str(base_csv),
          "--ids", ",".join(chosen_ids),
          "--out", str(out_path),
          "--prefix", f"{folder}_best100_gmshapeok"], dry_run)


def do_best200(folder: str, folder_root: Path, out_configs_dir: Path, python_exe: str,
                top_n: int, gm_ceiling_start: float, filter_number: int, dry_run: bool):
    base_csv = _base_ranked_csv(folder_root)
    if base_csv is None:
        print(f"  SKIP {folder}: no base ranked csv found under {folder_root}")
        return

    rows = _read_rows(base_csv)
    gm_ceiling = gm_ceiling_start
    n_pass = sum(1 for r in rows if (_f(r, "region_knee_combined_gm") or 1e9) <= gm_ceiling)
    while n_pass < top_n and gm_ceiling < 20.0:
        gm_ceiling = round(gm_ceiling * 1.5, 4)
        n_pass = sum(1 for r in rows if (_f(r, "region_knee_combined_gm") or 1e9) <= gm_ceiling)
    print(f"  {folder}: combined_gm<={gm_ceiling} gives {n_pass} rows (need >={top_n}), "
          f"total population={len(rows)}")

    rc = _run([python_exe, str(FILTER_RESULTS),
               "--root", str(folder_root),
               "--filter", f"region_knee_combined_gm<={gm_ceiling}",
               "--top_n", str(top_n),
               "--sort_by", "region_knee_ids_rmse",
               "--filter_number", str(filter_number)], dry_run)
    if rc != 0:
        print(f"  FAILED filter_results.py for {folder}")
        return

    if dry_run:
        print(f"  (dry run) would gen_config from filter #{filter_number} output")
        return

    ranked_dir = folder_root / "ranked_region_knee_vgs-3to0_vds0to15"
    filtered = sorted(ranked_dir.glob(f"*_ranked_by_region_knee_combined_gm_{filter_number}.csv"))
    if not filtered:
        print(f"  FAILED: no filter #{filter_number} output found for {folder}")
        return

    out_configs_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_configs_dir / f"{folder}_best200.json"
    _run([python_exe, str(GEN_CONFIG),
          "--csv", str(filtered[0]),
          "--ids", "all",
          "--out", str(out_path),
          "--prefix", f"{folder}_best200"], dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_root", required=True,
                    help="pure_combined9069-style sweep root, e.g. ../runs/pure_combined9069_rw0_2.5_20")
    ap.add_argument("--folders", default=",".join(FOLDERS))
    ap.add_argument("--best200_configs_out", default=None,
                    help="default: <repo>/runs/best200ids_of_9069_byloss_<source_root_suffix>/_configs")
    ap.add_argument("--bothshapeok_configs_out", default=None,
                    help="default: <repo>/runs/bothshapeok_of_9069_byshape_<source_root_suffix>/_configs")
    ap.add_argument("--best100_gmshapeok_configs_out", default=None,
                    help="default: <repo>/runs/best100_gmshapeok_of_9069_byloss_<source_root_suffix>/_configs")
    ap.add_argument("--top_n", type=int, default=200, help="best200's row count")
    ap.add_argument("--gmshapeok_top_n", type=int, default=100, help="best100_gmshapeok's row count")
    ap.add_argument("--bothshapeok_top_n", type=int, default=None,
                    help="cap bothshapeok to the best N (by ids_rmse) instead of ALL bothshape_ok "
                         "rows (default: no cap, matches the original bothshapeok_of_9069_byshape "
                         "convention). Only truncates if the pool exceeds N.")
    ap.add_argument("--gm_ceiling_start", type=float, default=0.9)
    ap.add_argument("--filter_number", type=int, default=5, help="matches pure_combined9069's own filter #5 convention")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--skip_bothshapeok", action="store_true")
    ap.add_argument("--skip_best200", action="store_true")
    ap.add_argument("--skip_best100_gmshapeok", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    suffix = source_root.name.replace("pure_combined9069_", "").replace("pure_combined9069", "").strip("_")
    repo_root = HERE.parent

    best200_out = Path(args.best200_configs_out).resolve() if args.best200_configs_out else \
        repo_root / "runs" / f"best200ids_of_9069_byloss_{suffix}" / "_configs"
    bothshapeok_out = Path(args.bothshapeok_configs_out).resolve() if args.bothshapeok_configs_out else \
        repo_root / "runs" / f"bothshapeok_of_9069_byshape_{suffix}" / "_configs"
    best100_gmshapeok_out = Path(args.best100_gmshapeok_configs_out).resolve() if args.best100_gmshapeok_configs_out else \
        repo_root / "runs" / f"best100_gmshapeok_of_9069_byloss_{suffix}" / "_configs"

    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    print(f"source_root={source_root}")
    print(f"best200_out={best200_out}")
    print(f"bothshapeok_out={bothshapeok_out}")
    print(f"best100_gmshapeok_out={best100_gmshapeok_out}\n")

    for folder in folders:
        folder_root = source_root / folder
        if not folder_root.is_dir():
            print(f"SKIP {folder}: {folder_root} does not exist")
            continue
        print(f"=== {folder} ===")
        if not args.skip_bothshapeok:
            do_bothshapeok(folder, folder_root, bothshapeok_out, args.python_exe, args.dry_run,
                           top_n=args.bothshapeok_top_n, gm_ceiling_start=args.gm_ceiling_start)
        if not args.skip_best200:
            do_best200(folder, folder_root, best200_out, args.python_exe, args.top_n,
                        args.gm_ceiling_start, args.filter_number, args.dry_run)
        if not args.skip_best100_gmshapeok:
            do_best100_gmshapeok(folder, folder_root, best100_gmshapeok_out, args.python_exe,
                                  args.gmshapeok_top_n, args.gm_ceiling_start, args.dry_run)
        print()

    print("done.")


if __name__ == "__main__":
    main()
