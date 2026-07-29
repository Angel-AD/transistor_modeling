"""
Plot extrapolation curves for a config selected from a compiled / ranked CSV row.

Works identically to plot_csv_row.py but uses wider Vgs / Vds ranges by default
to visualise how well the model extrapolates beyond its training window.

Default ranges:
    Vds : 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50  V
    Vgs : -4, -3, -2, -1, 0, 1, 2                     V

For Vds / Vgs values outside the measured data the plot shows model predictions
only (no measured points to compare against) -- that is the extrapolation.

Output is saved as  <run_dir>/plot_extrapolate.png  (not plot_saved_state_full.png)
so it does not overwrite the standard training-range plot.

Usage:
    python plot_extrapolate_csv_row.py \\
        --ranked_csv refine5/refine5_ranked_by_ids_rmse.csv \\
        --csv ../csvs/cg2h40010_new_2.4_5_2_70W_center9.csv \\
        --id 1

    # several at once
    python plot_extrapolate_csv_row.py --ranked_csv ... --csv ... --id 1,2,3

    # custom ranges
    python plot_extrapolate_csv_row.py --ranked_csv ... --csv ... --id 1 \\
        --plot_vds_list 0,10,20,30,40,50 \\
        --plot_vgs_list -4,-2,0,2
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
    fp = (row.get("file_path") or "").strip()
    if not fp:
        return None
    p = Path(fp)
    return p.parent if p.parent.is_dir() else None


def _select(rows: list[dict], args) -> list[dict]:
    if args.id:
        if "id" not in rows[0]:
            raise SystemExit("This CSV has no 'id' column -- regenerate it with "
                             "compile_ranked.py / compile_results_csv.py.")
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
        description="Plot extrapolation curves for config(s) from a CSV row.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ranked_csv", required=True,
                    help="CSV holding the rows (ranked_by_*.csv, compiled_results*.csv, etc.).")
    ap.add_argument("--csv", required=True,
                    help="Measurement CSV (forwarded to plot_saved_state.py).")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--id",
                     help="Row id(s) from the 'id' column, comma-separated.")
    sel.add_argument("--row",
                     help="Physical row number(s), 1-based, comma-separated.")
    sel.add_argument("--config_name",
                     help="Match by config_name (comma-separated).")
    ap.add_argument("--min_vgs", type=float, default=-4.0,
                    help="Vgs floor for meas_load extrapolation (default -4.0).")
    ap.add_argument("--plot_vds_list", type=str,
                    default="0,5,10,15,20,25,30,35,40,45,50",
                    help="Comma-separated Vds targets in volts (default: 0 to 50 in steps of 5).")
    ap.add_argument("--plot_vgs_list", type=str,
                    default="-4,-3,-2,-1,0,1,2",
                    help="Comma-separated Vgs targets in volts (default: -4 to 2).")
    ap.add_argument("--out_suffix", type=str, default="extrapolate",
                    help="Suffix for the output PNG filename: plot_<suffix>.png (default: extrapolate).")
    ap.add_argument("--python_exe", type=str, default=sys.executable)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands without executing.")
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

        out_path = run_dir / f"plot_{args.out_suffix}.png"

        cmd = [
            args.python_exe, str(PLOT_SCRIPT),
            "--dir", str(run_dir),
            "--csv", args.csv,
            "--min_vgs", str(args.min_vgs),
            f"--plot_vds_list={args.plot_vds_list}",
            f"--plot_vgs_list={args.plot_vgs_list}",
            "--out", str(out_path),
            "--extrapolate",
        ]

        info = "  ".join(f"{k}={r.get(k)}" for k in
                         ("ids_rmse", "combined_gm", "gm_vds_min", "architecture")
                         if r.get(k) not in (None, ""))
        print(f"\n[{label}] {run_dir.name}")
        print(f"    {info}")
        print(f"    Vds: {args.plot_vds_list}")
        print(f"    Vgs: {args.plot_vgs_list}")
        print(f"    out: {out_path}", flush=True)

        if args.dry_run:
            print(f"    cmd: {' '.join(cmd)}")
            continue

        rc = subprocess.call(cmd)
        if rc == 0:
            ok += 1
            id_val = str(r.get('id', r.get('config_name', label)))
            out_dir = csv_path.parent / "plotted_configs" / f"{csv_path.stem}_extrap_id{id_val}"
            out_dir.mkdir(parents=True, exist_ok=True)
            if out_path.is_file():
                shutil.copy2(out_path, out_dir / out_path.name)
                print(f"[{label}] copied -> {out_dir / out_path.name}", flush=True)
            src_md = out_path.with_suffix('.md')
            if src_md.is_file():
                shutil.copy2(src_md, out_dir / src_md.name)
                print(f"[{label}] copied -> {out_dir / src_md.name}", flush=True)
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
