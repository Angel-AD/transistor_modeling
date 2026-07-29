"""Per-column value diff between two xlsx files.

For every shared column (minus the same exclusion list used by
compare_architecture.py), compare the SET of distinct values in each file and
report what file2 is MISSING (present in file1, absent in file2) and what is
EXTRA (present in file2, absent in file1).

Example output for this dataset:
  region_weight
    missing in file2 (in file1 only): [4]
    extra   in file2 (not in file1) : [<file2's value>]

Usage:
  python compare_column_values.py FILE1 FILE2 [--exclude col1 col2 ...]

--exclude is ADDED to DEFAULT_EXCLUDE (imported from compare_architecture.py).
"""
import argparse
import pandas as pd

from compare_architecture import DEFAULT_EXCLUDE


def distinct(series):
    """Distinct non-null values, with a single NaN sentinel if nulls exist."""
    vals = set()
    has_nan = series.isna().any()
    for v in series.dropna().tolist():
        vals.add(v)
    return vals, has_nan


def fmt(values):
    try:
        return sorted(values)
    except TypeError:                      # mixed/unsortable types
        return sorted(values, key=repr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1")
    ap.add_argument("file2")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="extra columns to ignore, ADDED to DEFAULT_EXCLUDE")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--show-all", action="store_true",
                    help="also list columns whose value sets are identical")
    args = ap.parse_args()

    d1 = pd.read_excel(args.file1)
    d2 = pd.read_excel(args.file2)

    exclude = set(DEFAULT_EXCLUDE) | set(args.exclude) | {args.id_col}
    compare_cols = [c for c in d1.columns if c in d2.columns and c not in exclude]

    print(f"file1: {args.file1}")
    print(f"file2: {args.file2}")
    print(f"Comparing value sets of {len(compare_cols)} columns "
          f"(excluded: {sorted(exclude)})\n")

    any_diff = False
    for c in compare_cols:
        s1, nan1 = distinct(d1[c])
        s2, nan2 = distinct(d2[c])
        missing = s1 - s2                  # in file1, not in file2
        extra = s2 - s1                    # in file2, not in file1
        missing_nan = nan1 and not nan2
        extra_nan = nan2 and not nan1

        if not (missing or extra or missing_nan or extra_nan):
            if args.show_all:
                print(f"{c}: identical ({len(s1)} values)")
            continue

        any_diff = True
        print(c)
        if missing or missing_nan:
            shown = fmt(missing) + (["NaN"] if missing_nan else [])
            print(f"    missing in file2 (in file1 only): {shown}")
        if extra or extra_nan:
            shown = fmt(extra) + (["NaN"] if extra_nan else [])
            print(f"    extra   in file2 (not in file1) : {shown}")

    if not any_diff:
        print("All compared columns have identical value sets.")


if __name__ == "__main__":
    main()
