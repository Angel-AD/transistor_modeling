"""
filter_results.py  --  Filter compiled ranked CSVs by metric thresholds.

Operates purely on the CSV files produced by compile_ranked.py / compile_master_best.py --
it reads them, keeps only the rows passing every --filter, and writes a *_filtered.csv next
to each (optionally a *_filtered.xlsx too). It NEVER reads or modifies any run_loss_*.json.

Because it works on the compiled CSVs, every column is already present -- global metrics
(combined_gm, ids_rmse, gm1/2/3_rmse) AND any region metric (region_<name>_combined_gm,
region_<name>_ids_rmse, ...). So you can filter on region metrics directly, no second pass.

Filter syntax  (--filter, repeatable, AND-combined)
---------------------------------------------------
    COL<op>VALUE      op in  =  <  >  <=  >=     (inequalities are numeric)
    e.g.  --filter combined_gm<0.8  --filter region_knee_ids_rmse<0.01

Usage
-----
    # filter every ranked CSV under a root (ranked_global/, ranked_region_*/, master_best_*)
    python filter_results.py --root base_arch \
        --filter combined_gm<0.8 --filter region_knee_combined_gm<0.8 --xlsx

    # filter a single CSV
    python filter_results.py --csv base_arch/ranked_region_knee/base_arch_ranked_by_region_knee_combined_gm.csv \
        --filter region_knee_ids_rmse<0.01
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
XLSX_SCRIPT = HERE / "compile_results_to_xlsx.py"


def parse_filter(spec: str):
    """'COL<op>VALUE' (op in =, <, >, <=, >=) -> (col, op, value).
    '=' keeps the raw value (text or number, e.g. output_activation=softplus); the
    inequalities require a numeric value."""
    op = next((o for o in ("<=", ">=", "<", ">", "=") if o in spec), None)
    if op is None:
        sys.exit(f"[error] --filter must be COL<op>VALUE (op in =,<,>,<=,>=), got: {spec!r}")
    col, val = (s.strip() for s in spec.split(op, 1))
    if op == "=":
        return col, op, val                       # text or numeric equality
    try:
        return col, op, float(val)                # <,>,<=,>= are numeric only
    except ValueError:
        sys.exit(f"[error] --filter {spec!r}: {val!r} is not numeric (only '=' allows text).")


import operator
_OPS = {"<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}


def _row_passes(row: dict, filters) -> bool:
    """True if every (col, op, val) holds for this CSV row. For '=', compare numerically
    when both sides parse as numbers, else case-insensitive text. For the inequalities, a
    blank/non-numeric cell fails."""
    for col, op, val in filters:
        cell = row.get(col)
        if op == "=":
            try:                                  # numeric equality (e.g. gm1_weight=0.1)
                if float(cell) == float(val):
                    continue
                return False
            except (TypeError, ValueError):       # text equality (e.g. output_activation=softplus)
                if str(cell).strip().lower() == str(val).strip().lower():
                    continue
                return False
        else:
            try:
                v = float(cell)
            except (TypeError, ValueError):
                return False
            if not _OPS[op](v, val):
                return False
    return True


def filter_csv(csv_path: Path, filters, drop_all_zero, suffix: str, want_xlsx: bool):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    # A CSV that lacks a filter column can't be filtered on it (e.g. a global ranked CSV
    # has no region_* column) -> skip it, don't abort the whole run.
    missing = [c for c, _, _ in filters if c not in fields]
    if missing:
        print(f"  [skip] {csv_path.name}: no column(s) {missing}")
        return None
    kept = [r for r in rows if _row_passes(r, filters)]
    # Drop rows where ALL the named columns are numeric 0 (e.g. the inert gm1=gm2=gm3=0 combo) —
    # an OR-style "keep if any nonzero" the plain AND filters can't express. Only applied when
    # every named column is present (else the "all zero" test is ambiguous), so it's safe to skip.
    if drop_all_zero:
        miss_dz = [c for c in drop_all_zero if c not in fields]
        if miss_dz:
            print(f"  [warn] {csv_path.name}: --drop-all-zero skipped, cols not present: {miss_dz}")
        else:
            def _all_zero(r):
                try:
                    return all(float(r[c]) == 0 for c in drop_all_zero)
                except (TypeError, ValueError):
                    return False
            _b = len(kept)
            kept = [r for r in kept if not _all_zero(r)]
            print(f"  [drop-all-zero {'+'.join(drop_all_zero)}] removed {_b - len(kept)} rows")
    out_path = csv_path.with_name(csv_path.stem + suffix + ".csv")
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(kept)
    except PermissionError:
        print(f"  [skip] {out_path.name} is open/locked (e.g. in Excel) -- close it and re-run.")
        return None
    print(f"  {csv_path.name}: {len(kept)}/{len(rows)} rows -> {out_path.name}")
    if want_xlsx and kept:
        try:
            subprocess.run([sys.executable, str(XLSX_SCRIPT), "--csv", str(out_path)],
                           check=True, capture_output=True, text=True)
            print(f"    + {out_path.with_suffix('.xlsx').name}")
        except Exception as e:
            print(f"    [warn] xlsx conversion failed: {e}")
    return len(kept), len(rows)


def find_csvs(root: Path):
    """Ranked / master-best CSVs under a root. Prefer compile_ranked's grouped subfolders
    (ranked_global/, ranked_region_*/); only fall back to root-level *_ranked_by_*.csv when
    no grouped folders exist (avoids picking up stale pre-grouping CSVs). Skips already-
    filtered outputs so re-runs don't filter the filtered."""
    grouped = sorted(root.glob("ranked_*/*_ranked_by_*.csv"))
    found = list(grouped)
    if not grouped:
        found += sorted(root.glob("*_ranked_by_*.csv"))    # legacy flat layout
    found += sorted(root.glob("master_best_*.csv"))
    return [p for p in found if "_filtered" not in p.stem]


def main():
    ap = argparse.ArgumentParser(
        description="Filter compiled ranked CSVs by metric thresholds (writes *_filtered.csv/.xlsx).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="Master root; filter every ranked/master_best CSV under it.")
    src.add_argument("--csv", help="A single CSV to filter.")
    ap.add_argument("--filter", action="append", default=[], metavar="COL<op>VAL",
                    help="Keep rows where COL<op>VAL (op in =,<,>,<=,>=). Repeatable (AND). "
                         "Any column: numeric (combined_gm, ids_rmse, region_*) with any op, or "
                         "TEXT columns with '=' (e.g. output_activation=softplus, gm_surgery_mode=none).")
    ap.add_argument("--drop-all-zero", nargs="+", default=[], metavar="COL",
                    help="Drop rows where ALL these columns are 0 (e.g. --drop-all-zero "
                         "gm1_weight gm2_weight gm3_weight removes the inert all-zero-gm rows).")
    ap.add_argument("--suffix", default="_filtered", help="Output filename suffix (default _filtered).")
    ap.add_argument("--xlsx", action="store_true", help="Also write a styled .xlsx of each filtered CSV.")
    args = ap.parse_args()

    if not args.filter and not args.drop_all_zero:
        sys.exit("[error] need at least one of --filter or --drop-all-zero.")
    filters = [parse_filter(f) for f in args.filter]

    if args.csv:
        targets = [Path(args.csv)]
    else:
        root = Path(args.root)
        if not root.is_dir():
            sys.exit(f"[error] --root not a directory: {root}")
        targets = find_csvs(root)
        if not targets:
            sys.exit(f"[error] no ranked/master_best CSVs under {root} "
                     "(run compile_ranked.py first).")

    _desc = " AND ".join(args.filter) if args.filter else "(no metric filter)"
    if args.drop_all_zero:
        _desc += f"  [drop rows where all of {args.drop_all_zero} are 0]"
    print(f"Filter: {_desc}\n{len(targets)} CSV(s):")
    tot_kept = tot_rows = filtered_files = skipped = 0
    for p in targets:
        if not p.is_file():
            print(f"  [skip] not found: {p}")
            skipped += 1
            continue
        res = filter_csv(p, filters, args.drop_all_zero, args.suffix, args.xlsx)
        if res is None:
            skipped += 1
            continue
        k, n = res
        tot_kept += k
        tot_rows += n
        filtered_files += 1
    print(f"\ndone. filtered {filtered_files} CSV(s) (kept {tot_kept}/{tot_rows} rows)"
          + (f"; skipped {skipped} (missing filter column)" if skipped else ""))


if __name__ == "__main__":
    main()
