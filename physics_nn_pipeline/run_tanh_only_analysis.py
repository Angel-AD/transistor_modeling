"""
run_tanh_only_analysis.py -- for every folder, runs the tanh-only consistency analysis
(write_findings.py --only_tanh, once per base_config), plots whichever of those picks aren't
already plotted (reusing run_plot_consistent_archs.py's pick-collection/plotting, deduped
against the SHARED plots tree so arch_hashes the heterogeneous pass already covered aren't
replotted), then its overall cross-reference (write_overall_findings.py --subdir tanh_only) +
the tanh-only-vs-heterogeneous comparison (write_tanh_only_comparison.py). Companion to
run_plot_consistent_archs.py, which must have already produced the UNRESTRICTED
consistency_summary_<base_config>.json for every base_config (the comparison step reads those
alongside the tanh-only ones).

The search/overall/comparison steps are all cheap (CSV reads/writes only), so they run in one
shared worker pool (--workers) across every (folder, base_config) combination at once. Plotting
is the heavier step (loads each architecture's already-trained weights via plot_saved_state.py
and runs inference across every measurement csv to render plots + equation writeup -- no
training happens, the model was already trained when originally swept), so it gets its own,
smaller shared pool (--plot_workers) across ALL folders at once, same convention as
run_plot_consistent_archs.py's --workers/--search_workers split.

Usage:
    python run_tanh_only_analysis.py --csv_base_root ../runs/csv_base_2.5_20
    python run_tanh_only_analysis.py --csv_base_root ../runs/csv_base_2.5_20 --workers 24
    python run_tanh_only_analysis.py --csv_base_root ../runs/csv_base_2.5_20 \\
        --folders tanh_margin10 --dry_run
    python run_tanh_only_analysis.py --csv_base_root ../runs/csv_base_2.5_20 --skip_plotting
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_derived_configs import FOLDERS  # noqa: E402
from find_consistent_archs import DEFAULT_PAIR, find_arch_plot_dir  # noqa: E402
from run_plot_consistent_archs import _collect_picks, _plot_one  # noqa: E402

WRITE_FINDINGS = HERE / "write_findings.py"
WRITE_OVERALL = HERE / "write_overall_findings.py"
WRITE_COMPARISON = HERE / "write_tanh_only_comparison.py"
RENDER_HTML = HERE / "render_html_view.py"

DEFAULT_BASE_CONFIGS = [
    "best200ids_of_9069_byloss_rw0_2.5_20",
    "bothshapeok_of_9069_byshape_rw0_2.5_20",
    "best100_gmshapeok_of_9069_byloss_rw0_2.5_20",
]


def _run(cmd: list[str], dry_run: bool) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _write_findings_tanh_one(csv_base_root: Path, out_root: Path, base_config: str, folder: str,
                              pair: tuple[str, str], gm_ceiling: float | None, top: int,
                              python_exe: str, dry_run: bool) -> tuple[str, str, int]:
    """Returns (folder, base_config, returncode)."""
    root = csv_base_root / base_config
    if not root.is_dir():
        print(f"  SKIP {base_config}/{folder}: {root} does not exist")
        return folder, base_config, 1
    cmd = [python_exe, str(WRITE_FINDINGS), "--root", str(root), "--folder", folder,
           "--out_root", str(out_root), "--top", str(top), "--pair_csvs", ",".join(pair),
           "--only_tanh"]
    if gm_ceiling is not None:
        cmd += ["--gm_ceiling", str(gm_ceiling)]
    rc = _run(cmd, dry_run)
    if rc != 0:
        print(f"  FAILED write_findings.py --only_tanh for {base_config}/{folder}")
    return folder, base_config, rc


def _search_pass(csv_base_root: Path, out_root: Path, base_configs: list[str], folders: list[str],
                  pair: tuple[str, str], gm_ceiling: float | None, top: int, python_exe: str,
                  dry_run: bool, workers: int, label: str) -> dict[str, dict[str, Path]]:
    """Runs write_findings.py --only_tanh for every (folder, base_config) pair. Returns
    {folder: {base_config: json_path}} (only entries that succeeded)."""
    print(f"\n=== {label}: {len(folders)} folder(s) x {len(base_configs)} base_config(s) "
          f"(workers={workers}) ===")
    tasks = [(bc, f) for f in folders for bc in base_configs]
    results: dict[str, dict[str, Path]] = {f: {} for f in folders}
    if dry_run:
        for bc, f in tasks:
            _write_findings_tanh_one(csv_base_root, out_root, bc, f, pair, gm_ceiling, top,
                                      python_exe, True)
        return results
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_write_findings_tanh_one, csv_base_root, out_root, bc, f, pair,
                           gm_ceiling, top, python_exe, False) for bc, f in tasks]
        for fut in as_completed(futs):
            folder, base_config, rc = fut.result()
            if rc == 0:
                results[folder][base_config] = out_root / folder / "tanh_only" / \
                    f"consistency_summary_{base_config}.json"
    return results


def _unique_hash_picks(json_paths: dict[str, Path], pair: tuple[str, str]) -> list[tuple[str, str, str]]:
    """[(arch_hash, goal, base_config)], deduped down to ONE representative (goal, base_config)
    per unique arch_hash -- plot_arch_hash.py's output is equivalent across goal-folder copies
    of the same arch_hash (same architecture, same measurement csvs; only a cosmetic
    'Search goal:' label line differs -- see find_arch_plot_dir), so only one needs to exist for
    arch_link_cell to link every table cell showing this hash, regardless of which goal/method
    that cell is under."""
    seen_hashes = set()
    out = []
    for h, goal, bc in _collect_picks(json_paths, pair):
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        out.append((h, goal, bc))
    return out


def _overall_and_comparison_one(out_root: Path, folder: str, base_configs: list[str], top: int,
                                 python_exe: str, dry_run: bool):
    cmd = [python_exe, str(WRITE_OVERALL), "--out_root", str(out_root), "--folder", folder,
           "--top", str(max(top * 4, 20)), "--subdir", "tanh_only"]
    _run(cmd, dry_run)
    cmd = [python_exe, str(WRITE_COMPARISON), "--out_root", str(out_root), "--folder", folder,
           "--base_configs", ",".join(base_configs), "--top", str(top)]
    _run(cmd, dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_base_root", required=True, help="e.g. ../runs/csv_base_2.5_20")
    ap.add_argument("--out_root", default=None,
                    help="Shared best_archs_plots root. Default: <csv_base_root>/best_archs_plots")
    ap.add_argument("--base_configs", default=",".join(DEFAULT_BASE_CONFIGS))
    ap.add_argument("--folders", default=",".join(FOLDERS))
    ap.add_argument("--pair_csvs", default=",".join(DEFAULT_PAIR))
    ap.add_argument("--gm_ceiling", type=float, default=None,
                    help="default: write_findings.py's own per-folder default (1.4 if '_nogm' "
                         "in folder name, else 1.1)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel calls across all (folder, base_config) combos and folders "
                         "for the search/overall/comparison steps (default: 8) -- all cheap CSV "
                         "I/O, safe to raise high.")
    ap.add_argument("--format", choices=("pdf", "md", "both"), default="md",
                    help="plot_arch_hash.py --format for the newly-plotted tanh-only picks "
                         "(default: md).")
    ap.add_argument("--plot_workers", type=int, default=8,
                    help="parallel plot_arch_hash.py calls, shared across ALL folders at once "
                         "(default: 8) -- only for tanh-only picks not already plotted by "
                         "run_plot_consistent_archs.py's heterogeneous pass (same shared plots "
                         "tree, deduped). No training happens (loads already-trained weights, "
                         "runs inference/plotting), but still the heaviest step here.")
    ap.add_argument("--skip_plotting", action="store_true",
                    help="Run the search + overall + comparison steps only, skip plot_arch_hash.py "
                         "entirely (tanh-only arch cells whose plots don't already exist from the "
                         "heterogeneous pass stay unlinked).")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--html", action="store_true",
                    help="After everything else, also run render_html_view.py over --out_root "
                         "to mirror every .md file as an .html twin (real new-tab/new-window "
                         "links when clicked in a browser, unlike VS Code's in-place markdown "
                         "preview navigation). Doesn't touch the .md files themselves.")
    args = ap.parse_args()

    csv_base_root = Path(args.csv_base_root).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else csv_base_root / "best_archs_plots"
    base_configs = [b.strip() for b in args.base_configs.split(",") if b.strip()]
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    pair = tuple(c.strip() for c in args.pair_csvs.split(","))

    print(f"csv_base_root={csv_base_root}")
    print(f"out_root={out_root}")
    print(f"base_configs={base_configs}")
    print(f"folders={folders}")
    print(f"pair={pair}")

    all_json_paths = _search_pass(csv_base_root, out_root, base_configs, folders, pair,
                                   args.gm_ceiling, args.top, args.python_exe, args.dry_run,
                                   args.workers, "tanh-only search pass")

    all_picks = []  # (folder, arch_hash, goal, base_config)
    for folder in folders:
        json_paths = all_json_paths.get(folder) or {}
        if not json_paths:
            print(f"  SKIP {folder}: no base_config search succeeded")
            continue
        picks = _unique_hash_picks(json_paths, pair)
        to_plot = [(h, goal, bc) for h, goal, bc in picks
                   if find_arch_plot_dir(out_root, folder, h) is None]
        print(f"\n=== {folder}: {len(picks)} unique tanh-only arch_hash(es), "
              f"{len(to_plot)} not already plotted ===")
        for h, goal, bc in to_plot:
            print(f"  {h}  goal={goal}  from={bc}")
            all_picks.append((folder, h, goal, bc))

    print(f"\n=== plotting {len(all_picks)} tanh-only pick(s) not already covered by the "
          f"heterogeneous pass, across {len(folders)} folder(s) (plot_workers={args.plot_workers})"
          f"{' -- SKIPPED (--skip_plotting)' if args.skip_plotting else ''} ===")
    if args.skip_plotting:
        pass
    elif not args.dry_run and all_picks:
        with ThreadPoolExecutor(max_workers=args.plot_workers) as ex:
            futs = {ex.submit(_plot_one, csv_base_root, out_root, folder, h, goal, bc,
                               args.format, args.python_exe, False): (folder, h, goal)
                    for folder, h, goal, bc in all_picks}
            for fut in as_completed(futs):
                folder, h, goal = futs[fut]
                rc = fut.result()
                if rc != 0:
                    print(f"  FAILED plot_arch_hash.py for {folder}/{h} goal={goal} (rc={rc})")
    elif args.dry_run:
        for folder, h, goal, bc in all_picks:
            _plot_one(csv_base_root, out_root, folder, h, goal, bc, args.format, args.python_exe, True)

    if all_picks and not args.skip_plotting:
        _search_pass(csv_base_root, out_root, base_configs, folders, pair, args.gm_ceiling,
                     args.top, args.python_exe, args.dry_run, args.workers,
                     "tanh-only refresh pass (fills in category/links now that plots exist)")

    print(f"\n=== tanh-only overall summary + comparison: {len(folders)} folder(s) "
          f"(workers={args.workers}) ===")
    if args.dry_run:
        for folder in folders:
            _overall_and_comparison_one(out_root, folder, base_configs, args.top, args.python_exe, True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_overall_and_comparison_one, out_root, folder, base_configs,
                               args.top, args.python_exe, False) for folder in folders]
            for fut in as_completed(futs):
                fut.result()

    if args.html:
        print(f"\n=== html view ===")
        _run([args.python_exe, str(RENDER_HTML), "--out_root", str(out_root),
              "--workers", str(args.workers)], args.dry_run)

    print("\ndone.")


if __name__ == "__main__":
    main()
