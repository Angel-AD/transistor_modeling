"""
Plot a SPECIFIC config selected from a compiled / ranked CSV row.

Companion to plot_best_configs.py. Where plot_best plots the top-N by a metric, this
picks one (or several) explicit rows -- by their `id`, by physical row number, or by
config_name -- and regenerates each run's 6-panel IV/GM plot via plot_saved_state.py.

The `id` column (added by compile_ranked.py / compile_results_csv.py) is a simple 1-based
number written into the file (1 = best by that file's metric). Because it's a stored
column value, it SURVIVES re-sorting the CSV in Excel: sort/reorder the rows however you
like, copy the `id` of the row you want, and pass it with the same --ranked_csv.

Each row's `file_path` column points at the run's run_loss_*.json; its parent directory
is what plot_saved_state.py needs as --dir.

Usage:
    # by id (the 'id' column -- survives Excel re-sorting)
    python plot_csv_row.py --ranked_csv refine2/refine2_ranked_by_ids_rmse.csv \
        --csv ../csvs/<measurements>.csv --id 1

    # several at once
    python plot_csv_row.py --ranked_csv ... --csv ... --id 1,2,3

    # by current physical row position (1-based)
    python plot_csv_row.py --ranked_csv ... --csv ... --row 5

    # by config_name
    python plot_csv_row.py --ranked_csv ... --csv ... --config_name PhysNN_classic_W1.0-1.0-1.0_none
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOT_SCRIPT = HERE / "plot_saved_state.py"


def _run_dir_from_row(row: dict):
    """Per-run directory from a row's file_path column (parent of run_loss_*.json)."""
    fp = (row.get("file_path") or "").strip()
    if not fp:
        return None
    p = Path(fp)
    return p.parent if p.parent.is_dir() else None


def _select(rows: list[dict], args) -> list[dict]:
    if args.id:
        if "id" not in rows[0]:
            raise SystemExit("This CSV has no 'id' column -- regenerate it with the "
                             "updated compile_ranked.py / compile_results_csv.py.")
        by = {str(r["id"]): r for r in rows}
        out = []
        for u in [x.strip() for x in args.id.split(",") if x.strip()]:
            if u in by:
                out.append(by[u])
            else:
                print(f"[warn] id {u!r} not found in CSV")
        return out
    if args.row:
        out = []
        for x in [x.strip() for x in args.row.split(",") if x.strip()]:
            i = int(x)
            if 1 <= i <= len(rows):
                out.append(rows[i - 1])
            else:
                print(f"[warn] row {i} out of range (1..{len(rows)})")
        return out
    want = {c.strip() for c in args.config_name.split(",") if c.strip()}
    return [r for r in rows if (r.get("config_name") or "") in want]


def main():
    ap = argparse.ArgumentParser(
        description="Plot specific config(s) chosen from a CSV row (by uid / row / config_name).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ranked_csv", required=True,
                    help="CSV holding the rows (ranked_by_*.csv, compiled_results*.csv, or master_best_*.csv).")
    ap.add_argument("--csv", required=True,
                    help="Measurement CSV (forwarded to plot_saved_state.py).")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--id", help="Row id(s) from the 'id' column, comma-separated. Survives Excel re-sorting.")
    sel.add_argument("--row", help="Current physical row number(s), 1-based, comma-separated.")
    sel.add_argument("--config_name", help="Match by config_name (comma-separated).")
    ap.add_argument("--min_vgs", type=float, default=-4.0)
    ap.add_argument("--plot_vds_list", type=str, default="0,5,10,15,20,28")
    ap.add_argument("--plot_vgs_list", type=str, default="-3.5,-3,-2.9,-2.8,-2.7,-2.5,-2.2,-2,-1.8,-1.6,-1.5,-1.3,-1,-.5,0")
    ap.add_argument("--val", type=str, default=None,
                    help="Validation set option forwarded to plot_saved_state.py: "
                         "'interpolation', a float (e.g. '0.3'), or a path to a val CSV.")
    ap.add_argument("--add_zero_vds", action="store_true",
                    help="Forward --add_zero_vds to plot_saved_state.py.")
    ap.add_argument("--python_exe", type=str, default=sys.executable)
    ap.add_argument("--dry_run", action="store_true", help="Print the commands, then exit.")
    args = ap.parse_args()

    csv_path = Path(args.ranked_csv).resolve()
    if not csv_path.is_file():
        raise SystemExit(f"--ranked_csv not found: {csv_path}")
    if not PLOT_SCRIPT.is_file():
        raise SystemExit(f"plot_saved_state.py not found: {PLOT_SCRIPT}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    selected = _select(rows, args)
    if not selected:
        raise SystemExit("No matching rows selected.")

    ok = fail = 0
    for r in selected:
        label = f"id={r.get('id')}" if r.get("id") else (r.get("config_name") or "?")
        run_dir = _run_dir_from_row(r)
        if run_dir is None:
            print(f"[{label}] SKIP: cannot resolve run dir from file_path={r.get('file_path','')!r}")
            fail += 1
            continue
        cmd = [args.python_exe, str(PLOT_SCRIPT), "--dir", str(run_dir), "--csv", args.csv,
               "--min_vgs", str(args.min_vgs),
               f"--plot_vds_list={args.plot_vds_list}", f"--plot_vgs_list={args.plot_vgs_list}"]
        if args.val:
            cmd += ["--val", args.val]
        if args.add_zero_vds:
            cmd.append("--add_zero_vds")
        info = "  ".join(f"{k}={r.get(k)}" for k in
                         ("category", "ids_rmse", "combined_gm", "knee_combiner",
                          "knee_alpha_scale", "knee_vgs_thr") if r.get(k) not in (None, ""))
        print(f"\n[{label}] {run_dir.name}\n    {info}\n    {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc == 0:
            ok += 1
            id_val = str(r.get('id', r.get('config_name', label)))
            out_dir = csv_path.parent / "plotted_configs" / f"{csv_path.stem}_id{id_val}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for src_name in ("plot_saved_state_full.png",
                             "plot_saved_state_full_eq_comparison.png",
                             "plot_saved_state_full_val.png",
                             "plot_saved_state_full.md",
                             "plot_saved_state_full.va"):
                src = run_dir / src_name
                if src.is_file():
                    shutil.copy2(src, out_dir / src_name)
                    print(f"[{label}] copied -> {out_dir / src_name}", flush=True)
            run_jsons = sorted(run_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
            if run_jsons:
                src_json = run_jsons[-1]
                shutil.copy2(src_json, out_dir / src_json.name)
                print(f"[{label}] copied -> {out_dir / src_json.name}", flush=True)
        else:
            fail += 1
            print(f"[{label}] FAILED rc={rc}", flush=True)

    if not args.dry_run:
        print(f"\ndone. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
