"""
run_shape_analysis_csv_base.py -- step 1 of the consistency-analysis pipeline: ensure a
full-population <folder>_shape.csv exists for EVERY (base_config, csv, folder) leaf under a
csv_base_root (e.g. runs/csv_base_2.5_20), so find_consistent_archs.py's methods B/C
(gmshape_ok/bothshape_ok consistency) have real data to search -- see write_findings.py's
"Population: N csvs, full-population shape analysis available for 0/N" until this has run.

Reuses extract_derived_configs.py's _ensure_shape_csv (analyze_shape.py + keep_columns.py
cleanup, same convention used to build the best100_gmshapeok/bothshapeok derivations
themselves) -- resumable, skips any (base_config, csv, folder) that already has a _shape.csv.

Leaves are independent of each other (each is its own analyze_shape.py subprocess against one
folder's population for one csv), so they run in a worker pool (--workers, default 8) instead
of strictly sequentially -- interleaved progress output across workers is expected.

Usage:
    python run_shape_analysis_csv_base.py --csv_base_root ../runs/csv_base_2.5_20
    python run_shape_analysis_csv_base.py --csv_base_root ../runs/csv_base_2.5_20 --workers 16
    python run_shape_analysis_csv_base.py --csv_base_root ../runs/csv_base_2.5_20 \\
        --base_configs best200ids_of_9069_byloss_rw0_2.5_20 --folders tanh_margin10 --dry_run
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_derived_configs import FOLDERS, _base_ranked_csv, _ensure_shape_csv  # noqa: E402

DEFAULT_BASE_CONFIGS = [
    "best200ids_of_9069_byloss_rw0_2.5_20",
    "bothshapeok_of_9069_byshape_rw0_2.5_20",
    "best100_gmshapeok_of_9069_byloss_rw0_2.5_20",
]


def _do_leaf(label: str, base_csv: Path, python_exe: str, dry_run: bool) -> tuple[str, bool]:
    """Returns (label, already_existed)."""
    shape_csv = base_csv.with_name(base_csv.stem + "_shape.csv")
    already = shape_csv.is_file()
    print(f"=== {label} ===")
    _ensure_shape_csv(label, base_csv, python_exe, dry_run)
    return label, already


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_base_root", required=True, help="e.g. ../runs/csv_base_2.5_20")
    ap.add_argument("--base_configs", default=",".join(DEFAULT_BASE_CONFIGS))
    ap.add_argument("--folders", default=",".join(FOLDERS))
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel analyze_shape.py subprocesses (default: 8). Leaves are "
                         "fully independent (one csv/folder/base_config each), so this is safe "
                         "to raise -- CPU/IO bound, not GPU bound.")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    csv_base_root = Path(args.csv_base_root).resolve()
    base_configs = [b.strip() for b in args.base_configs.split(",") if b.strip()]
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]

    tasks = []  # (label, base_csv)
    n_missing = 0
    for base_config in base_configs:
        bc_root = csv_base_root / base_config
        if not bc_root.is_dir():
            print(f"SKIP base_config: {bc_root} does not exist")
            continue
        csv_dirs = sorted(p for p in bc_root.iterdir() if p.is_dir() and p.name != "_configs")
        for folder in folders:
            for csv_dir in csv_dirs:
                folder_root = csv_dir / folder
                if not folder_root.is_dir():
                    n_missing += 1
                    continue
                base_csv = _base_ranked_csv(folder_root)
                if base_csv is None:
                    print(f"SKIP {base_config}/{csv_dir.name}/{folder}: no base ranked csv found")
                    n_missing += 1
                    continue
                tasks.append((f"{base_config}/{csv_dir.name}/{folder}", base_csv))

    n_ok = n_skip = 0
    if args.dry_run:
        for label, base_csv in tasks:
            _do_leaf(label, base_csv, args.python_exe, True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_do_leaf, label, base_csv, args.python_exe, False): label
                    for label, base_csv in tasks}
            for fut in as_completed(futs):
                label, already = fut.result()
                if already:
                    n_skip += 1
                else:
                    n_ok += 1

    print(f"\ndone. computed={n_ok} reused_existing={n_skip} missing_leaf={n_missing} "
          f"(total leaves checked={len(tasks)})")


if __name__ == "__main__":
    main()
