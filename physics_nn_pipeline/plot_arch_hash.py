"""
Plot one architecture (by its stable arch_hash) across EVERY measurement-csv subfolder of a
<root>/<csv_name>/<folder>/... sweep (e.g. runs/best200ids_of_9069_byloss), consolidating all
resulting plots into one output directory, organized by shape-compliance category.

Companion to plot_csv_row.py -- reuses its row-reading/run-dir-resolution helpers. Where
plot_csv_row.py plots row(s) from ONE ranked csv, this plots the SAME architecture (by
arch_hash -- the content-stable id from run_artifacts.arch_id(); NOT the numeric arch_id
column, which is only a 1..N index local to each independently-compiled csv-named subfolder
and is NOT comparable across them) across every csv-named subfolder that shares the same
architecture pool, writing every resulting plot into one consolidated folder instead of
leaving them scattered in each run's own experiment directory.

Output is nested one level deeper than a plain arch_hash folder, by shape-compliance CATEGORY
(computed from each csv's full-population <hash>_..._shape.csv, if one exists -- see
analyze_shape.py / important_mds/shape_analysis_rules.md for gmshape_ok/bothshape_ok). Only
gmshape_ok/bothshape_ok drive categorization -- gdsshape_ok alone is not a category (per-csv
gdsshape_ok is still recorded in compliance.json, just not used to bucket the architecture):
    best_archs_plots/<folder>/both/<arch_hash>/        bothshape_ok on >=1 csv (best case)
    best_archs_plots/<folder>/gmshape/<arch_hash>/      gmshape_ok (never bothshape) on >=1 csv
    best_archs_plots/<folder>/none/<arch_hash>/         shape WAS checked somewhere, but never
                                                         passed gmshape/both anywhere
    best_archs_plots/<folder>/filter_only/<arch_hash>/  no _shape.csv exists for ANY csv this
                                                         arch was plotted on -- i.e. this arch
                                                         was only ever selected by the fit-
                                                         quality filter (ids_rmse/combined_gm),
                                                         shape was never analyzed at all
Category priority when an arch's compliance differs by csv: both > gmshape > none.
The category (plus the full per-csv breakdown) is also written to compliance.json in the
output dir and rendered on the PDF's summary page (see make_arch_pdf.py).

Usage:
    python plot_arch_hash.py --root ../runs/best200ids_of_9069_byloss --folder tanh_margin10 \\
        --arch_hash 93fd5b66
    # writes into
    # ../runs/best200ids_of_9069_byloss/best_archs_plots/tanh_margin10/<category>/93fd5b66/
    #   <csv_name>_plot_saved_state_full.png  (one per csv subfolder the arch_hash was found in)
    #   compliance.json  (per-csv gmshape_ok/gdsshape_ok/bothshape_ok + the overall category)
    #   all_csvs.pdf  (all 6 plot_saved_state_full.png combined, one page each, titled + a
    #                  trailing summary page with compliance.json's info if present -- see
    #                  make_arch_pdf.py; --format controls pdf/md/both, default pdf)

    # several architectures / folders at once
    python plot_arch_hash.py --root ... --folder tanh_margin10 --arch_hash 93fd5b66,86b46c94
    python plot_arch_hash.py --root ... --folder tanh_margin10,softplus --arch_hash 93fd5b66
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOT_SCRIPT = HERE / "plot_saved_state.py"

sys.path.insert(0, str(HERE))
from plot_csv_row import _read_rows, _run_dir_from_row  # noqa: E402  (reuse, don't duplicate)
from make_arch_pdf import build_pdf, build_md  # noqa: E402  (reuse, don't duplicate)

# Copied out of each run dir after plotting, same file list plot_csv_row.py copies.
_OUTPUT_FILES = (
    "plot_saved_state_full.png",
    "plot_saved_state_full_eq_comparison.png",
    "plot_saved_state_full_val.png",
    "plot_saved_state_full.md",
    "plot_saved_state_full.va",
)

COMPLIANCE_FILE = "compliance.json"


def _csv_subdirs(root: Path, folder: str) -> list[Path]:
    """Every immediate subdirectory of root that has a <name>/<folder>/ranked_region_knee_.../
    directory -- i.e. every measurement-csv subfolder this architecture could appear in.
    Skips non-csv dirs like _configs/ and the output dir itself (neither has a <folder>/
    subdirectory, so they're naturally excluded)."""
    out = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        ranked_dir = p / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        if ranked_dir.is_dir():
            out.append(p)
    return out


def _find_row_by_arch_hash(ranked_dir: Path, arch_hash: str) -> dict | None:
    """Row matching arch_hash from the BASE ranked csv in ranked_dir (the plain
    <hash>_ranked_by_region_knee_combined_gm.csv -- has file_path + csv columns; the _N /
    _filtered / _shape variants either subset it or drop those columns)."""
    matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
    if not matches:
        return None
    rows = _read_rows(matches[0])
    for r in rows:
        if r.get("arch_hash") == arch_hash:
            return r
    return None


def _shape_compliance(ranked_dir: Path, arch_hash: str) -> dict | None:
    """{"gmshape_ok": bool, "gdsshape_ok": bool, "bothshape_ok": bool} for arch_hash from the
    full-population <hash>_ranked_by_region_knee_combined_gm_shape.csv in ranked_dir (written
    by `analyze_shape.py --ranked_csv <hash>_..._combined_gm.csv`, no fit-quality pre-filter).
    None if that file doesn't exist (shape analysis was never run for this csv) or arch_hash
    isn't in it (this architecture wasn't part of whatever population WAS shape-analyzed --
    shouldn't normally happen for a full-population run, but handled defensively). Compliance
    formula matches important_mds/shape_analysis_rules.md / analyze_shape.py's own
    write_gmshape_outputs()."""
    matches = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
    if not matches:
        return None
    shape_path = matches[0].with_name(matches[0].stem + "_shape.csv")
    if not shape_path.is_file():
        return None
    row = next((r for r in _read_rows(shape_path) if r.get("arch_hash") == arch_hash), None)
    if row is None:
        return None

    def f(key):
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            return None

    gmshape_ok = (
        (f("gm3_n_min") or 0) >= 1 and (f("gm3_n_max") or 0) >= 1 and
        f("gm3_bad_frac") == 0 and f("gm3_min_missing_frac") == 0 and f("gm3_max_missing_frac") == 0 and
        f("gm3_start_bad_frac") == 0 and f("gm2_start_bad_frac") == 0 and
        f("gm3_tail_bad_frac") == 0 and f("gm3_global_bad_frac") == 0
    )
    gdsshape_ok = f("gds_residual_bad_frac") == 0
    return {"gmshape_ok": gmshape_ok, "gdsshape_ok": gdsshape_ok,
            "bothshape_ok": gmshape_ok and gdsshape_ok}


def _overall_category(per_csv: dict[str, dict | None]) -> str:
    """both > gmshape > none > filter_only, by priority across all csvs checked. Only
    gmshape_ok/bothshape_ok drive this -- gdsshape_ok alone never distinguishes a category
    (it's still recorded per-csv in compliance.json, just not used for bucketing)."""
    checked = [c for c in per_csv.values() if c is not None]
    if not checked:
        return "filter_only"
    if any(c["bothshape_ok"] for c in checked):
        return "both"
    if any(c["gmshape_ok"] and not c["bothshape_ok"] for c in checked):
        return "gmshape"
    return "none"


def plot_one(root: Path, folder: str, arch_hash: str, out_root: Path, args,
             goal: str | None = None) -> tuple[int, int]:
    csv_subdirs = _csv_subdirs(root, folder)
    if not csv_subdirs:
        print(f"[{folder}/{arch_hash}] no csv subfolders with a {folder}/ranked_region_knee_.../ "
              f"directory found under {root}")
        return 0, 0

    # Pass 1: find each csv's row + shape compliance (cheap -- just CSV reads, no plotting)
    # so the output CATEGORY (and therefore out_dir) is known before anything gets written.
    per_csv_row: dict[str, dict] = {}
    per_csv_compliance: dict[str, dict | None] = {}
    for csv_subdir in csv_subdirs:
        csv_name = csv_subdir.name
        ranked_dir = csv_subdir / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        row = _find_row_by_arch_hash(ranked_dir, arch_hash)
        if row is None:
            print(f"[{folder}/{arch_hash}] {csv_name}: arch_hash not found -- skip")
            continue
        per_csv_row[csv_name] = row
        per_csv_compliance[csv_name] = _shape_compliance(ranked_dir, arch_hash)

    if not per_csv_row:
        print(f"[{folder}/{arch_hash}] arch_hash not found in any csv subfolder -- nothing to plot")
        return 0, 0

    category = _overall_category(per_csv_compliance)
    out_dir = out_root / folder / category / (goal or "") / arch_hash
    print(f"[{folder}/{arch_hash}] category: {category}"
          f"{f' goal: {goal}' if goal else ''}  "
          f"({sum(1 for c in per_csv_compliance.values() if c)} of {len(per_csv_compliance)} "
          f"csvs had shape data)")

    # Pass 2: actually plot + copy into the now-known out_dir.
    ok = fail = 0
    for csv_name, row in per_csv_row.items():
        run_dir = _run_dir_from_row(row)
        if run_dir is None:
            print(f"[{folder}/{arch_hash}] {csv_name}: SKIP: cannot resolve run dir "
                  f"from file_path={row.get('file_path', '')!r}")
            fail += 1
            continue

        meas_csv = row.get("csv")
        if not meas_csv:
            print(f"[{folder}/{arch_hash}] {csv_name}: SKIP: row has no 'csv' column")
            fail += 1
            continue

        cmd = [args.python_exe, str(PLOT_SCRIPT), "--dir", str(run_dir), "--csv", meas_csv,
               "--min_vgs", str(args.min_vgs),
               f"--plot_vds_list={args.plot_vds_list}", f"--plot_vgs_list={args.plot_vgs_list}"]
        if args.val:
            cmd += ["--val", args.val]
        if args.add_zero_vds:
            cmd.append("--add_zero_vds")

        info = "  ".join(f"{k}={row.get(k)}" for k in
                         ("ids_rmse", "region_knee_combined_gm", "region_knee_ids_rmse")
                         if row.get(k) not in (None, ""))
        print(f"\n[{folder}/{arch_hash}] {csv_name} -> {run_dir.name}\n    {info}\n    {' '.join(cmd)}",
              flush=True)
        if args.dry_run:
            continue

        rc = subprocess.call(cmd)
        if rc != 0:
            fail += 1
            print(f"[{folder}/{arch_hash}] {csv_name}: FAILED rc={rc}", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        copied_any = False
        for src_name in _OUTPUT_FILES:
            src = run_dir / src_name
            if src.is_file():
                dst = out_dir / f"{csv_name}_{src_name}"
                shutil.copy2(src, dst)
                print(f"[{folder}/{arch_hash}] {csv_name}: copied -> {dst}", flush=True)
                copied_any = True
        ok += 1 if copied_any else 0
        if not copied_any:
            print(f"[{folder}/{arch_hash}] {csv_name}: plot ran (rc=0) but no output files found "
                  f"in {run_dir}", flush=True)

    if ok and not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        compliance_doc = {"arch_hash": arch_hash, "category": category, "per_csv": per_csv_compliance}
        if goal:
            compliance_doc["goal"] = goal
        (out_dir / COMPLIANCE_FILE).write_text(json.dumps(compliance_doc, indent=2), encoding="utf-8")
        if args.format in ("pdf", "both"):
            pdf_path = build_pdf(out_dir)
            if pdf_path is None:
                print(f"[{folder}/{arch_hash}] no PNGs found for PDF assembly (unexpected)", flush=True)
        if args.format in ("md", "both"):
            md_path = build_md(out_dir)
            if md_path is None:
                print(f"[{folder}/{arch_hash}] no PNGs found for markdown assembly (unexpected)", flush=True)
    return ok, fail


def main():
    ap = argparse.ArgumentParser(
        description="Plot one architecture (by arch_hash) across every measurement-csv "
                    "subfolder of a <root>/<csv_name>/<folder>/... sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--root", required=True,
                    help="Master root containing one subfolder per measurement csv "
                         "(e.g. ../runs/best200ids_of_9069_byloss).")
    ap.add_argument("--folder", required=True,
                    help="Equation-type folder name(s) under each csv subfolder "
                         "(e.g. tanh_margin10), comma-separated for several at once.")
    ap.add_argument("--arch_hash", required=True,
                    help="Stable arch_hash(es) to plot (the 'arch_hash' column in any ranked "
                         "csv -- NOT 'arch_id', which is only a 1..N index local to each "
                         "independently-compiled csv subfolder), comma-separated for several.")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for consolidated plots. Default: "
                         "<root>/best_archs_plots/<folder>/<category>/<arch_hash>/ -- one "
                         "subfolder per equation-type folder, then per shape-compliance "
                         "category (both/gmshape/gdsshape/none/filter_only -- see module "
                         "docstring), then per arch_hash.")
    ap.add_argument("--goal", default=None,
                    help="Optional extra subfolder inserted between <category> and "
                         "<arch_hash> describing WHY this arch_hash was picked (e.g. "
                         "'best_consistent_with_all_csvs' or "
                         "'best_consistent_with_2.5_20_and_2.9_28'), for when several "
                         "different searches land architectures in the same category and you "
                         "want to keep their results visually grouped by search goal. Also "
                         "recorded as a 'goal' key in compliance.json. Omit for the plain "
                         "<category>/<arch_hash> layout.")
    ap.add_argument("--min_vgs", type=float, default=-4.0)
    ap.add_argument("--plot_vds_list", type=str, default="0,5,10,15,20,28")
    ap.add_argument("--plot_vgs_list", type=str,
                    default="-3.5,-3,-2.9,-2.8,-2.7,-2.5,-2.2,-2,-1.8,-1.6,-1.5,-1.3,-1,-.5,0")
    ap.add_argument("--val", type=str, default=None,
                    help="Validation set option forwarded to plot_saved_state.py: "
                         "'interpolation', a float (e.g. '0.3'), or a path to a val CSV.")
    ap.add_argument("--add_zero_vds", action="store_true",
                    help="Forward --add_zero_vds to plot_saved_state.py.")
    ap.add_argument("--format", choices=("pdf", "md", "both"), default="pdf",
                    help="Per-arch_hash summary format assembled via make_arch_pdf.py (one "
                         "page/section per csv's plot_saved_state_full.png so the plots can be "
                         "flipped/scrolled through without opening each file): 'pdf' (default, "
                         "all_csvs.pdf), 'md' (all_csvs.md, markdown with embedded images), or "
                         "'both'.")
    ap.add_argument("--python_exe", type=str, default=sys.executable)
    ap.add_argument("--dry_run", action="store_true", help="Print the commands, then exit.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"--root not found: {root}")
    if not PLOT_SCRIPT.is_file():
        raise SystemExit(f"plot_saved_state.py not found: {PLOT_SCRIPT}")

    out_root = Path(args.out_dir).resolve() if args.out_dir else root / "best_archs_plots"

    folders = [f.strip() for f in args.folder.split(",") if f.strip()]
    arch_hashes = [h.strip() for h in args.arch_hash.split(",") if h.strip()]

    total_ok = total_fail = 0
    for folder in folders:
        for arch_hash in arch_hashes:
            ok, fail = plot_one(root, folder, arch_hash, out_root, args, goal=args.goal)
            total_ok += ok
            total_fail += fail

    if not args.dry_run:
        print(f"\ndone. ok={total_ok} fail={total_fail}")


if __name__ == "__main__":
    main()
