"""Find ids in another xlsx that match a reference row's "architecture".

Match = identical values across ALL columns except an exception list.
The exception list is DEFAULT_EXCLUDE (equation/identity + metric columns);
anything passed via --exclude is ADDED to it. NaN is treated as equal to NaN.
Floats can be matched with a tolerance via --tol.

Usage:
  python compare_architecture.py FILE1 ID FILE2 [--exclude col1 col2 ...] [--tol 0]

Example:
  python compare_architecture.py \
    physics_nn_pipeline/refine_vdsgate_gm_1/ranked_region_knee/refine_vdsgate_gm_1_ranked_by_region_knee_combined_gm.xlsx \
    70 \
    physics_nn_pipeline/refine_vdsk_gm_1/ranked_region_knee/refine_vdsk_gm_1_ranked_by_region_knee_combined_gm.xlsx \
    --exclude eq
"""
import argparse
import pandas as pd
import numpy as np

# Columns ignored by default: the equation/experiment-identity columns that
# differ by design between runs, plus all training-outcome metrics. Anything
# passed via --exclude is added on top of these.
DEFAULT_EXCLUDE = [
    "eq", "category", "combo", "phase", "config_name", "file_path",
    "best_loss", "combined_gm", "ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse",
    "region_knee_combined_gm", "region_knee_gm1_rmse", "region_knee_gm2_rmse",
    "region_knee_gm3_rmse", "region_knee_ids_mae", "region_knee_ids_rmse",
]


def values_match(a, b, tol):
    a_na = pd.isna(a)
    b_na = pd.isna(b)
    if a_na or b_na:
        return a_na and b_na          # NaN matches only NaN
    if tol and isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1")
    ap.add_argument("ref_id", type=int)
    ap.add_argument("file2")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra columns to ignore, ADDED to DEFAULT_EXCLUDE")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="absolute tolerance for float columns (default 0 = exact)")
    ap.add_argument("--id-col", default="id")
    args = ap.parse_args()

    d1 = pd.read_excel(args.file1)
    d2 = pd.read_excel(args.file2)

    ref_rows = d1[d1[args.id_col] == args.ref_id]
    if len(ref_rows) == 0:
        raise SystemExit(f"id {args.ref_id} not found in {args.file1}")
    ref = ref_rows.iloc[0]

    # Columns to compare: shared columns minus exclusions and the id column.
    shared = [c for c in d1.columns if c in d2.columns]
    exclude = set(DEFAULT_EXCLUDE) | set(args.exclude) | {args.id_col}
    compare_cols = [c for c in shared if c not in exclude]

    # Per-row match test.
    matches = []
    mismatch_counter = {c: 0 for c in compare_cols}
    for _, row in d2.iterrows():
        bad = [c for c in compare_cols if not values_match(ref[c], row[c], args.tol)]
        for c in bad:
            mismatch_counter[c] += 1
        if not bad:
            matches.append(row[args.id_col])

    print(f"Reference: id {args.ref_id} from {args.file1}")
    print(f"Comparing {len(compare_cols)} columns (excluded: {sorted(exclude)})")
    print(f"Rows scanned in file2: {len(d2)}")
    print(f"\n>>> Matching ids in file2: {matches if matches else '(none)'}")

    if not matches:
        print("\nColumns that block matches (mismatch count across all file2 rows):")
        blockers = sorted(((n, c) for c, n in mismatch_counter.items() if n), reverse=True)
        for n, c in blockers:
            print(f"  {c:28} mismatched in {n}/{len(d2)} rows  "
                  f"(ref={ref[c]!r})")
        print("\nAdd the always-mismatching columns above to --exclude to relax the match.")


if __name__ == "__main__":
    main()
