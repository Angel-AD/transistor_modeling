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

Best-N-by-metric  (--top_n + --sort_by, applied AFTER any --filter)
---------------------------------------------------------------------
    --top_n 20 --sort_by region_knee_combined_gm   keeps the best 20 rows (lowest first;
                                                     pass --descending for highest-first)
    Combinable with --filter, e.g. threshold-filter first, then take the best 20 of survivors.

Usage
-----
    # filter every ranked CSV under a root (ranked_global/, ranked_region_*/, master_best_*)
    python filter_results.py --root base_arch \
        --filter combined_gm<0.8 --filter region_knee_combined_gm<0.8 --xlsx

    # filter a single CSV
    python filter_results.py --csv base_arch/ranked_region_knee/base_arch_ranked_by_region_knee_combined_gm.csv \
        --filter region_knee_ids_rmse<0.01

    # best 20 by region_knee_combined_gm, auto-numbered so different --top_n/--sort_by runs coexist
    python filter_results.py --root base_arch --top_n 20 --sort_by region_knee_combined_gm --auto_number
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from compile_results_to_xlsx import csv_to_xlsx


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


# --------------------------------------------------------------------------- #
# --auto_number: short "_<N>" suffix instead of a long descriptive one, with a
# filter_readme.md mapping N -> the exact criteria used, so re-running the same
# filter step with DIFFERENT thresholds doesn't overwrite a previous run's output --
# multiple filter criteria can coexist in the same folder.
# --------------------------------------------------------------------------- #
_README_NAME = "filter_readme.md"
_README_HEADER = (
    "# filter_readme.md -- maps filter-number suffixes (_<N>) to the exact criteria used.\n"
    "# Auto-maintained by filter_results.py --auto_number. Numbers are append-only; don't "
    "renumber by hand.\n\n"
)
_LINE_RE = re.compile(r"^(\d+):\s*(.+)$")
_APPLIED_RE = re.compile(r"^(.*?)\s*\[in: (.+)\]$")  # splits desc from a trailing dirs annotation
_NUMBERED_RE = re.compile(r"_\d+(?:_|$)")   # "_<N>" (trailing, or mid-name before a further
                                             # suffix like "_shape") from a prior --auto_number run


def filter_description(filters, drop_all_zero, top_spec=None):
    """Canonical, human-readable string for a filter combo -- used both as the manifest
    entry text and as the lookup key (so re-running the SAME criteria reuses its number)."""
    desc = " AND ".join(f"{c}{op}{v}" for c, op, v in filters) if filters else "(no metric filter)"
    if drop_all_zero:
        desc += f"  [drop rows where all of {list(drop_all_zero)} are 0]"
    if top_spec:
        sort_by, top_n, descending = top_spec
        desc += f"  [top {top_n} by {sort_by} ({'best-to-worst desc' if descending else 'best-to-worst asc'})]"
    return desc


def _readme_path(manifest_dir: Path) -> Path:
    return manifest_dir / _README_NAME


def _parse_manifest_lines(manifest_dir: Path):
    """[(number, desc, [dirs])] in file order, or [] if no manifest yet. `dirs` are the
    subfolders (relative to manifest_dir) this number's filter has actually been written
    into, parsed out of a trailing "  [in: dir1, dir2]" annotation."""
    p = _readme_path(manifest_dir)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        num, rest = int(m.group(1)), m.group(2)
        am = _APPLIED_RE.match(rest)
        if am:
            out.append((num, am.group(1), [d.strip() for d in am.group(2).split(",") if d.strip()]))
        else:
            out.append((num, rest, []))
    return out


def _read_manifest(manifest_dir: Path):
    """{description: number} parsed from filter_readme.md, or {} if it doesn't exist yet."""
    return {desc: num for num, desc, _ in _parse_manifest_lines(manifest_dir)}


def get_or_assign_number(manifest_dir: Path, desc: str) -> int:
    """Number for `desc` in manifest_dir's filter_readme.md -- reused if this exact criteria
    combo was used before, else the next integer (existing max + 1), appended to the file."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_manifest(manifest_dir)
    if desc in existing:
        return existing[desc]
    n = (max(existing.values()) + 1) if existing else 1
    p = _readme_path(manifest_dir)
    is_new = not p.is_file()
    with open(p, "a", encoding="utf-8") as f:
        if is_new:
            f.write(_README_HEADER)
        f.write(f"{n}: {desc}\n")
    return n


def register_filter_number(manifest_dir: Path, number: int, desc: str) -> None:
    """Explicit-number counterpart to get_or_assign_number -- for --filter_number, where the
    caller wants a SPECIFIC number (e.g. to keep "corresponding" filters aligned by number
    across different roots/folders, even when their concrete criteria differ per folder --
    unlike --auto_number, there's no requirement that the description text match). Registers
    (or, if `number` already exists with a DIFFERENT description in this folder, overwrites)
    the filter_readme.md entry for exactly `number`; a folder's [in: ...] applied-dirs
    annotation is preserved across the overwrite."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    entries = _parse_manifest_lines(manifest_dir)
    existing = {n: (d, dirs_) for n, d, dirs_ in entries}
    if number in existing and existing[number][0] != desc:
        print(f"[warn] filter #{number} already meant {existing[number][0]!r} in "
              f"{_readme_path(manifest_dir)} -- overwriting with {desc!r}.")
    new_entries, found = [], False
    for n, d, dirs_ in entries:
        if n == number:
            new_entries.append((n, desc, dirs_))
            found = True
        else:
            new_entries.append((n, d, dirs_))
    if not found:
        new_entries.append((number, desc, []))
    new_entries.sort(key=lambda t: t[0])
    with open(_readme_path(manifest_dir), "w", encoding="utf-8") as f:
        f.write(_README_HEADER)
        for n, d, ds in new_entries:
            line = f"{n}: {d}"
            if ds:
                line += f"  [in: {', '.join(ds)}]"
            f.write(line + "\n")


def record_applied_dirs(manifest_dir: Path, filter_num: int, dirs) -> None:
    """Merge `dirs` (relative subfolder paths a filter run actually wrote into) into
    filter_num's "  [in: ...]" annotation, rewriting the manifest in place. No-op if there's
    nothing new to add."""
    dirs = set(dirs)
    if not dirs:
        return
    entries = _parse_manifest_lines(manifest_dir)
    changed = False
    new_entries = []
    for num, desc, existing in entries:
        if num == filter_num:
            merged = sorted(set(existing) | dirs)
            changed = changed or merged != existing
            new_entries.append((num, desc, merged))
        else:
            new_entries.append((num, desc, existing))
    if not changed:
        return
    with open(_readme_path(manifest_dir), "w", encoding="utf-8") as f:
        f.write(_README_HEADER)
        for num, desc, ds in new_entries:
            line = f"{num}: {desc}"
            if ds:
                line += f"  [in: {', '.join(ds)}]"
            f.write(line + "\n")


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


def filter_csv(csv_path: Path, filters, drop_all_zero, top_spec, suffix: str, want_xlsx: bool):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    # A CSV that lacks a filter column can't be filtered on it (e.g. a global ranked CSV
    # has no region_* column) -> skip it, don't abort the whole run.
    missing = [c for c, _, _ in filters if c not in fields]
    if top_spec and top_spec[0] not in fields:
        missing = missing + [top_spec[0]]
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
    # Best-N-by-metric: applied AFTER the threshold filters above (so you can e.g. combine
    # region_knee_combined_gm<=0.9 with "top 20 by region_knee_ids_rmse among those"). Blank/
    # non-numeric values always sort last, regardless of --descending, so they never occupy a
    # "top N" slot ahead of a real value.
    if top_spec:
        sort_by, top_n, descending = top_spec
        def _key(r):
            try:
                return (0, float(r[sort_by]))
            except (TypeError, ValueError):
                return (1, 0.0)
        kept = sorted(kept, key=_key, reverse=descending)[:top_n]
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
            csv_to_xlsx(out_path)
            print(f"    + {out_path.with_suffix('.xlsx').name}")
        except Exception as e:
            print(f"    [warn] xlsx conversion failed: {e}")
    return len(kept), len(rows)


def find_csvs(root: Path):
    """Ranked / master-best CSVs under a root. Prefer compile_ranked's grouped subfolders
    (ranked_global/, ranked_region_*/); only fall back to root-level *_ranked_by_*.csv when
    no grouped folders exist (avoids picking up stale pre-grouping CSVs). Skips already-
    filtered outputs (both the "_filtered..." suffix and a bare "_<N>" --auto_number suffix)
    so re-runs don't filter the filtered."""
    grouped = sorted(root.glob("ranked_*/*_ranked_by_*.csv"))
    found = list(grouped)
    if not grouped:
        found += sorted(root.glob("*_ranked_by_*.csv"))    # legacy flat layout
    found += sorted(root.glob("master_best_*.csv"))
    return [p for p in found if "_filtered" not in p.stem and not _NUMBERED_RE.search(p.stem)]


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
    ap.add_argument("--top_n", type=int, default=None,
                    help="Keep only the best N rows by --sort_by (applied AFTER any --filter/"
                         "--drop-all-zero, so you can combine a threshold with a best-N cut). "
                         "Requires --sort_by. Rows with a blank/non-numeric --sort_by value "
                         "always sort last, never occupying a top-N slot.")
    ap.add_argument("--sort_by", default=None, metavar="COL",
                    help="Column to rank by for --top_n, e.g. region_knee_combined_gm. Lower is "
                         "better by default (matches every metric in this pipeline); pass "
                         "--descending for higher-is-better columns.")
    ap.add_argument("--descending", action="store_true",
                    help="With --top_n/--sort_by: rank highest-first instead of lowest-first.")
    ap.add_argument("--suffix", default="_filtered", help="Output filename suffix (default "
                    "_filtered). Ignored when --auto_number is set.")
    ap.add_argument("--auto_number", action="store_true",
                    help="Use a short '_<N>' suffix instead of --suffix, with a filter_readme.md "
                         "(next to --root, or alongside --csv's file) mapping N -> the exact "
                         "criteria used. Re-running with the SAME --filter/--drop-all-zero combo "
                         "reuses its number; a DIFFERENT combo gets a new one -- so multiple "
                         "filter results can coexist in the same folder instead of one "
                         "overwriting another. Mutually exclusive with --filter_number.")
    ap.add_argument("--filter_number", type=int, default=None,
                    help="Like --auto_number, but YOU pick the number instead of it being "
                         "derived from matching criteria text. Use this to keep 'corresponding' "
                         "filters aligned by number across DIFFERENT --root folders even when "
                         "their concrete criteria differ per folder (e.g. folder A needs "
                         "combined_gm<=0.9 to get enough rows, folder B needs <=1.35) -- with "
                         "--auto_number each folder's number would drift independently since "
                         "the description text wouldn't match. If this number already means "
                         "something different in this folder's filter_readme.md, it's "
                         "overwritten (with a warning) -- you're taking explicit responsibility "
                         "for what it means here. Mutually exclusive with --auto_number.")
    ap.add_argument("--xlsx", action="store_true", help="Also write a styled .xlsx of each filtered CSV.")
    args = ap.parse_args()

    if args.auto_number and args.filter_number is not None:
        sys.exit("[error] --auto_number and --filter_number are mutually exclusive.")
    if bool(args.top_n) != bool(args.sort_by):
        sys.exit("[error] --top_n and --sort_by must be given together.")
    if not args.filter and not args.drop_all_zero and not args.top_n:
        sys.exit("[error] need at least one of --filter, --drop-all-zero, or --top_n/--sort_by.")
    filters = [parse_filter(f) for f in args.filter]
    top_spec = (args.sort_by, args.top_n, args.descending) if args.top_n else None

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

    _desc = filter_description(filters, args.drop_all_zero, top_spec)
    suffix = args.suffix
    numbered = args.auto_number or args.filter_number is not None
    if numbered:
        manifest_dir = Path(args.root) if args.root else Path(args.csv).resolve().parent
        if args.filter_number is not None:
            filter_num = args.filter_number
            register_filter_number(manifest_dir, filter_num, _desc)
            print(f"--filter_number: using filter #{filter_num} ({_readme_path(manifest_dir)})")
        else:
            filter_num = get_or_assign_number(manifest_dir, _desc)
            print(f"--auto_number: filter #{filter_num} ({_readme_path(manifest_dir)})")
        suffix = f"_{filter_num}"
    print(f"Filter: {_desc}\n{len(targets)} CSV(s):")
    tot_kept = tot_rows = filtered_files = skipped = 0
    applied_dirs = set()
    for p in targets:
        if not p.is_file():
            print(f"  [skip] not found: {p}")
            skipped += 1
            continue
        res = filter_csv(p, filters, args.drop_all_zero, top_spec, suffix, args.xlsx)
        if res is None:
            skipped += 1
            continue
        k, n = res
        tot_kept += k
        tot_rows += n
        filtered_files += 1
        if numbered:
            rel = p.parent.resolve().relative_to(manifest_dir.resolve())
            if str(rel) != ".":
                applied_dirs.add(str(rel).replace("\\", "/"))
    if numbered and applied_dirs:
        record_applied_dirs(manifest_dir, filter_num, applied_dirs)
    print(f"\ndone. filtered {filtered_files} CSV(s) (kept {tot_kept}/{tot_rows} rows)"
          + (f"; skipped {skipped} (missing filter column)" if skipped else ""))


if __name__ == "__main__":
    main()
