"""
Plot the top-N best configs from a compiled_results*.csv.

Reads `<experiment_dir>/compiled_results*.csv` (produced by
`compile_results_csv.py`), picks the top-N rows across all matching files
sorted by the requested metric(s), and invokes `plot_saved_state.py --dir <run_dir>`
for each one so the per-run 6-panel IV/GM plot is regenerated only for the winners.

Each winner's files (plot + run_loss json + weights + log) are also copied into
`<experiment_dir>/best_n_configs/<rank>_<run_dir>/` so the best configs are all in
one place. Disable with --no_best_copy.

Typical usage:

    python plot_best_configs.py \
        --dir <master_experiments_dirs>/pure_nn_no_gm \
        --csv path/to/measurements.csv \
        --top 5 \
        --sort_by best_loss

    # Rank by ids_rmse, then break ties by gm1_rmse:
    python plot_best_configs.py --dir ... --csv ... --top 10 \
        --sort_by ids_rmse,gm1_rmse

    # Regenerate the CSV first (re-scan all run_*.json):
    python plot_best_configs.py --dir ... --csv ... --top 5 --regen

Each row's `file_path` column is expected to point at the run's
`run_loss_*.json`; its parent directory is what `plot_saved_state.py`
needs as `--dir`.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOT_SCRIPT = HERE / "plot_saved_state.py"
COMPILE_SCRIPT = HERE / "compile_results_csv.py"
BEST_DIR_NAME = "best_n_configs"


def _copy_winner_to_best(run_dir: Path, best_dir: Path, rank: int) -> Path:
    """Copy a winner run dir's files (plot + run_loss json + weights + log) into
    <exp_dir>/best_n_configs/<rank>_<run_dir_name>/ for easy access."""
    dest = best_dir / f"{rank:02d}_{run_dir.name}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(run_dir.iterdir()):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    return dest
DEFAULT_CSV_GLOB = "compiled_results*.csv"


def _safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")


def _read_rows(csv_paths: list[Path]) -> list[dict]:
    """Reads and combines rows from all provided CSV paths."""
    all_rows = []
    for csv_path in csv_paths:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            all_rows.extend(list(csv.DictReader(f)))
    return all_rows


def _sort_rows(rows: list[dict], sort_keys: list[str]) -> list[dict]:
    return sorted(rows, key=lambda r: tuple(_safe_float(r.get(k, "")) for k in sort_keys))


# Base roots to search when a stored file_path no longer exists — e.g. the run
# dirs were moved out of the scripts folder into a sibling runs/ tree.
_REROOT_BASES = (HERE, HERE.parent, HERE.parent / "runs")


def _reroot(p: Path) -> Path:
    """If p is missing, return the longest tail of p that exists under a known
    base root; otherwise return p unchanged."""
    if p.exists():
        return p
    parts = p.parts
    for base in _REROOT_BASES:
        for i in range(len(parts)):
            cand = base.joinpath(*parts[i:])
            if cand.exists():
                return cand
    return p


def _resolve_run_dir(row: dict, exp_dir: Path) -> Path | None:
    """Derive the per-run directory from a CSV row's file_path column."""
    fp = row.get("file_path", "").strip()
    if not fp:
        return None
    p = Path(fp)
    if not p.is_absolute():
        # Older CSVs may store paths relative to the experiment dir.
        p = (exp_dir / p).resolve()
    run_dir = _reroot(p.parent)
    return run_dir if run_dir.is_dir() else None


def main():
    parser = argparse.ArgumentParser(
        description="Plot the top-N best configs from compiled_results*.csv "
                    "by re-running plot_saved_state.py per winner."
    )
    parser.add_argument("--dir", required=True,
                        help="Experiment directory (the one containing "
                             "compiled_results*.csv and the exp_*/ subfolders).")
    parser.add_argument("--csv", required=True,
                        help="Measurement CSV forwarded to plot_saved_state.py.")
    parser.add_argument("--top", type=int, default=5,
                        help="Number of best configs to plot (default: 5).")
    parser.add_argument("--sort_by", type=str, default="best_loss",
                        help="Comma-separated columns to rank by (ascending). "
                             "E.g. 'best_loss', 'ids_rmse', "
                             "'ids_rmse,gm1_rmse'. Default: 'best_loss'.")
    parser.add_argument("--results_csv", type=str, default=None,
                        help=f"Override path to the compiled CSV (default: "
                             f"<dir>/{DEFAULT_CSV_GLOB}).")
    parser.add_argument("--regen", action="store_true",
                        help="Re-run compile_results_csv.py on --dir before "
                             "reading the CSV (uses the same --sort_by).")
    parser.add_argument("--min_vgs", type=float, default=-4.0,
                        help="Forwarded to plot_saved_state.py.")
    parser.add_argument("--plot_vds_list", type=str, default="0,5,10,15,20,28",
                        help="Forwarded to plot_saved_state.py.")
    parser.add_argument("--plot_vgs_list", type=str, default="-3.5,-3,-2.5,-2,-1.5,-1,-.5,0",
                        help="Forwarded to plot_saved_state.py.")
    parser.add_argument("--python_exe", type=str, default=sys.executable,
                        help="Python interpreter to spawn the plot script with "
                             "(default: current interpreter).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands that would be run, then exit.")
    parser.add_argument("--no_best_copy", action="store_true",
                        help=f"Do NOT copy each winner's files (plot + json + weights "
                             f"+ log) into <dir>/{BEST_DIR_NAME}/. By default they are "
                             "copied there for easy access.")
    args = parser.parse_args()

    exp_dir = Path(args.dir).resolve()
    if not exp_dir.is_dir():
        raise SystemExit(f"--dir not a directory: {exp_dir}")
    if not PLOT_SCRIPT.is_file():
        raise SystemExit(f"plot_saved_state.py not found: {PLOT_SCRIPT}")

    sort_keys = [k.strip() for k in args.sort_by.split(",") if k.strip()]
    if not sort_keys:
        raise SystemExit("--sort_by produced an empty key list.")

    # Determine CSV paths to read from
    if args.results_csv:
        csv_paths = [Path(args.results_csv).resolve()]
    else:
        csv_paths = list(exp_dir.glob(DEFAULT_CSV_GLOB))

    if args.regen or not csv_paths:
        if not COMPILE_SCRIPT.is_file():
            raise SystemExit(f"compile_results_csv.py not found: {COMPILE_SCRIPT}")
        print(f"[plot_best] Regenerating CSV via {COMPILE_SCRIPT.name} ...")
        rc = subprocess.call([args.python_exe, str(COMPILE_SCRIPT),
                              "--dir", str(exp_dir),
                              "--sort_by", ",".join(sort_keys)])
        if rc != 0:
            raise SystemExit(f"compile_results_csv.py failed with rc={rc}")
        
        # Re-glob if using the default pattern so we pick up the newly generated files
        if not args.results_csv:
            csv_paths = list(exp_dir.glob(DEFAULT_CSV_GLOB))

    if not csv_paths:
        raise SystemExit(f"No results CSVs found matching: {DEFAULT_CSV_GLOB} in {exp_dir}")

    rows = _read_rows(csv_paths)
    if not rows:
        raise SystemExit(f"No rows found in {len(csv_paths)} CSV file(s).")

    # Re-sort in case the CSV was produced with a different --sort_by, or the
    # user wants to view-rank without regenerating, or we merged multiple CSVs.
    rows = _sort_rows(rows, sort_keys)
    winners = rows[: max(1, args.top)]

    print(f"[plot_best] {len(winners)} winners (sort_by={sort_keys}) "
          f"from {len(csv_paths)} CSV file(s)")
    for i, row in enumerate(winners, 1):
        ranked = " | ".join(f"{k}={row.get(k, '')}" for k in sort_keys)
        print(f"  #{i:>2d}  {ranked}  cfg={row.get('config_name', '?')}")

    # Collect winners' files in one place under the experiment dir for easy access.
    # Rebuilt fresh each run so it always reflects the current top-N selection.
    best_dir = exp_dir / BEST_DIR_NAME
    do_copy = not args.no_best_copy and not args.dry_run
    if do_copy and best_dir.exists():
        shutil.rmtree(best_dir, ignore_errors=True)

    n_ok = 0
    n_fail = 0
    n_skip = 0
    n_copied = 0
    for i, row in enumerate(winners, 1):
        run_dir = _resolve_run_dir(row, exp_dir)
        if run_dir is None:
            print(f"[#{i}] SKIP: could not resolve run dir from file_path="
                  f"{row.get('file_path','')!r}")
            n_skip += 1
            continue
        cmd = [args.python_exe, str(PLOT_SCRIPT),
               "--dir", str(run_dir),
               "--csv", args.csv]
        if args.min_vgs is not None:
            cmd += ["--min_vgs", str(args.min_vgs)]
        if args.plot_vds_list:
            cmd += [f"--plot_vds_list={args.plot_vds_list}"]
        if args.plot_vgs_list:
            cmd += [f"--plot_vgs_list={args.plot_vgs_list}"]

        print(f"\n[#{i}] {run_dir.name}\n    {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc == 0:
            n_ok += 1
            if do_copy:
                try:
                    dest = _copy_winner_to_best(run_dir, best_dir, i)
                    n_copied += 1
                    print(f"[#{i}] copied -> {dest}", flush=True)
                except Exception as e:
                    print(f"[#{i}] WARNING: failed to copy winner files: {e}", flush=True)
        else:
            n_fail += 1
            print(f"[#{i}] FAILED rc={rc}", flush=True)

    if args.dry_run:
        print(f"\n[dry_run] would have plotted {len(winners)} runs.")
    else:
        print(f"\n[plot_best] done. ok={n_ok} fail={n_fail} skip={n_skip}")
        if do_copy:
            print(f"[plot_best] copied {n_copied} winner(s) into {best_dir}")


if __name__ == "__main__":
    main()