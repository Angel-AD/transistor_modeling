"""
Run analyze_shape.py on the FULL-POPULATION base ranked csv (<hash>_ranked_by_region_knee_
combined_gm.csv -- every row, no fit-quality pre-filter) for one or more (csv, folder) pairs
under a <root>/<csv_name>/<folder>/... sweep, mirroring compile_overall.ps1's step 3c but on
the unfiltered base file instead of a quality-filtered subset (see this session's lesson:
cross-csv shape-consistency searches need the full population, not a narrow filter -- a
narrow filter answers "who ranks well in this smaller pool", not "who is shape-consistent").

For each (csv, folder):
    1. skip if <base>_shape.csv already exists (unless --force)
    2. python analyze_shape.py --ranked_csv <base>_ranked_by_region_knee_combined_gm.csv
    3. python keep_columns.py --csv <all *_shape*.csv outputs> --keep <shapeKeep>
       (same column list as compile_overall.ps1's $shapeKeep)
    4. delete the plain CSVs for the "_ok*"/"_removed" gmshape/gdsshape/bothshape compliance
       variants (slim-xlsx only, per compile_overall.ps1 step 3c) -- the base <hash>_..._shape.csv
       itself stays as CSV+slim (every row's metrics, read back by find_consistent_archs.py).

Usage:
    python run_full_pop_shape.py --root ../runs/bothshapeok_of_9069_byshape --folder tanh_margin10
    python run_full_pop_shape.py --root ../runs/best200ids_of_9069_byloss \\
        --folder tanh_margin10,tanh_margin10_nogm
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYZE_SHAPE = HERE / "analyze_shape.py"
KEEP_COLUMNS = HERE / "keep_columns.py"

SHAPE_KEEP = [
    "id", "run_id", "arch_id", "arch_hash", "vds_loss", "region_knee_combined_gm",
    "region_knee_ids_rmse", "neg_gds_energy", "gds_hump", "gds_hump_max",
    "gm3_maxabs", "gm3_bad_frac", "gm3_start_diff", "gm2_start_diff",
    "gm3_tail_maxabs", "gm3_tail_bad_frac", "gm3_max_missing_frac", "gm3_min_missing_frac",
    "gm3_global_excess_max", "gm3_global_bad_frac",
    "gds_residual_bad_frac", "gds_residual_neg_max", "gds_residual_pos_max",
    "gds_residual_worst_max", "shape_rank",
    "gm3_n_min", "gm3_n_max", "gm3_min_missing_frac", "gm3_max_missing_frac",
    "gm3_start_bad_frac", "gm2_start_bad_frac",
]


def _csv_subdirs(root: Path, folder: str) -> list[Path]:
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        ranked_dir = p / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        if ranked_dir.is_dir():
            out.append(p)
    return out


def run_one(root: Path, folder: str, csv_subdir: Path, python_exe: str, force: bool) -> str:
    ranked_dir = csv_subdir / folder / "ranked_region_knee_vgs-3to0_vds0to15"
    matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
    if not matches:
        return f"[{folder}] {csv_subdir.name}: SKIP no base ranked csv found in {ranked_dir}"
    base = matches[0]
    shape_path = base.with_name(base.stem + "_shape.csv")
    if shape_path.is_file() and not force:
        return f"[{folder}] {csv_subdir.name}: already have {shape_path.name} -- skip"

    n_lines = sum(1 for _ in base.open(encoding="utf-8"))
    if n_lines <= 1:
        return f"[{folder}] {csv_subdir.name}: SKIP (0 rows) {base}"

    rc = subprocess.call([python_exe, str(ANALYZE_SHAPE), "--ranked_csv", str(base)])
    if rc != 0:
        return f"[{folder}] {csv_subdir.name}: FAILED analyze_shape.py rc={rc}"

    shape_outputs = sorted(ranked_dir.glob(f"{base.stem}_shape*.csv"))
    if shape_outputs:
        subprocess.call([python_exe, str(KEEP_COLUMNS), "--csv", *[str(p) for p in shape_outputs],
                          "--keep", *SHAPE_KEEP])
        slim_only = [p for p in shape_outputs if p.name != shape_path.name]
        for p in slim_only:
            p.unlink()

    return f"[{folder}] {csv_subdir.name}: OK wrote {shape_path.name} ({n_lines - 1} rows)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--folder", required=True, help="comma-separated for several")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--force", action="store_true", help="re-run even if _shape.csv already exists")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    folders = [f.strip() for f in args.folder.split(",") if f.strip()]

    for folder in folders:
        for csv_subdir in _csv_subdirs(root, folder):
            print(run_one(root, folder, csv_subdir, args.python_exe, args.force), flush=True)


if __name__ == "__main__":
    main()
