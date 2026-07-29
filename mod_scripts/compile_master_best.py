"""
Build a master CSV containing the best N rows from each experiment's
compiled_results*.csv, ranked by a chosen metric.

Usage:
    python compile_master_best.py --root "<MASTER_ROOT_PATH>" \
        --sort_by ids_rmse --top_n 5 \
        --output master_best.csv

Each immediate subdirectory of --root is treated as one experiment. It
must contain CSV files matching the --input_csv pattern (default: 
compiled_results*.csv). The script picks the N rows with the smallest 
value of --sort_by from each experiment, tags them with the experiment 
name, concatenates them, and writes a single CSV in --root (or wherever 
--output points).
"""

import os
import csv
import argparse
from pathlib import Path


VALID_METRICS = ["best_loss", "ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse", "combined_gm"]


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("inf")


def _row_key(row, sort_keys):
    """Tuple sort key (ascending) across multiple columns."""
    return tuple(_to_float(row.get(k)) for k in sort_keys)


def collect_best(root, sort_keys, top_n, input_csv_pattern):
    root_path = Path(root)
    if not root_path.is_dir():
        raise SystemExit(f"Root path does not exist: {root}")

    subdirs = sorted(d for d in root_path.iterdir() if d.is_dir())
    if not subdirs:
        raise SystemExit(f"No subdirectories found in {root}")

    all_rows = []
    fieldnames = None

    for subdir in subdirs:
        # Find all CSVs matching the pattern (e.g., "compiled_results*.csv")
        csv_paths = list(subdir.glob(input_csv_pattern))
        
        if not csv_paths:
            print(f"  [skip] {subdir.name}: no files matching {input_csv_pattern}")
            continue

        exp_rows = []
        for csv_path in csv_paths:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                exp_rows.extend(list(reader))
                if reader.fieldnames and fieldnames is None:
                    fieldnames = list(reader.fieldnames)

        if not exp_rows:
            print(f"  [skip] {subdir.name}: empty CSV(s)")
            continue
            
        missing = [k for k in sort_keys if k not in exp_rows[0]]
        if missing:
            print(f"  [skip] {subdir.name}: sort key(s) {missing} not in CSV columns")
            continue

        exp_rows.sort(key=lambda r: _row_key(r, sort_keys))
        best = exp_rows[:top_n]
        for r in best:
            r["experiment"] = subdir.name
        all_rows.extend(best)
        print(f"  [{subdir.name}] kept {len(best)} of {len(exp_rows)} rows from {len(csv_paths)} file(s)")

    if not all_rows:
        raise SystemExit("No rows collected; nothing to write.")

    # Final sort: experiment then sort_keys (primary, secondary, ...)
    all_rows.sort(key=lambda r: (r["experiment"], _row_key(r, sort_keys)))

    out_fields = ["experiment"] + (fieldnames or list(all_rows[0].keys()))
    # Deduplicate while preserving order
    seen = set()
    out_fields = [k for k in out_fields if not (k in seen or seen.add(k))]
    return all_rows, out_fields


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate best-N rows per experiment into a master CSV."
    )
    parser.add_argument("--root", required=True,
                        help="Master root path containing experiment subdirs.")
    parser.add_argument("--metric", default=None, choices=VALID_METRICS,
                        help="[Deprecated; use --sort_by] Single metric to rank by "
                             "(lower is better). If both are given, --sort_by wins.")
    parser.add_argument("--sort_by", type=str, default=None,
                        help="Comma-separated columns to rank by (ascending, lowest "
                             "first). First key is primary; subsequent keys break ties. "
                             "Valid keys: " + ", ".join(VALID_METRICS) + ". "
                             "Default: 'ids_rmse'. Example: --sort_by ids_rmse,gm1_rmse")
    parser.add_argument("--top_n", type=int, default=5,
                        help="How many best rows to keep per experiment.")
    parser.add_argument("--input_csv", default="compiled_results*.csv",
                        help="Per-experiment CSV file glob pattern to read from.")
    parser.add_argument("--output", default=None,
                        help="Output CSV path. Defaults to "
                             "<root>/master_best_<sort_tag>_top<N>.csv")
    args = parser.parse_args()

    # Resolve sort keys: --sort_by takes precedence over the legacy --metric.
    if args.sort_by:
        sort_keys = [k.strip() for k in args.sort_by.split(",") if k.strip()]
    elif args.metric:
        sort_keys = [args.metric]
    else:
        sort_keys = ["ids_rmse"]
    if not sort_keys:
        raise SystemExit("--sort_by produced an empty key list.")

    rows, fields = collect_best(args.root, sort_keys, args.top_n, args.input_csv)

    sort_tag = "-".join(sort_keys)
    parent_name = os.path.basename(os.path.normpath(args.root))
    output_path = args.output or os.path.join(
        args.root, f"master_best_{sort_tag}_top{args.top_n}_{parent_name}.csv"
    )
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {output_path} (sort_by={sort_keys})")


if __name__ == "__main__":
    main()