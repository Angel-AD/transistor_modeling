"""
run_plot_consistent_archs.py -- step 2 of the consistency-analysis pipeline, across ALL
--folders at once:

  1. Run write_findings.py for every (folder, base_config) pair (writes consistency_summary_
     <base_config>.md/.json into the shared --out_root/<folder>/) -- this is the SEARCH pass,
     picking which arch_hashes matter (methods A/B/C/D, see find_consistent_archs.py). Runs in
     a worker pool (--search_workers) since these are cheap (CSV reads/writes, no GPU work).
  2. Collect every picked arch_hash across all folders/base_configs/methods, deduped by
     (folder, arch_hash, goal) -- the same arch_hash picked under the same goal by >1
     base_config (they draw from overlapping architecture pools) is plotted once, not once per
     base_config.
  3. Run plot_arch_hash.py --format <format> on each unique pick, writing into the SAME shared
     --out_root (so compliance.json / plots live at
     <out_root>/<folder>/<category>/<goal>/<arch_hash>/, independent of which base_config's
     search found it). ALL picks across ALL folders share ONE worker pool (--workers) -- this
     is the heaviest step (each plot_arch_hash.py call loads that architecture's already-
     trained weights via plot_saved_state.py and runs inference across every measurement csv to
     render plots + equation writeup -- no training/fitting happens here, the model was already
     trained when this architecture was originally swept), so a single shared pool caps total
     concurrency instead of multiplying if folders were also parallelized on top of it.
  4. Re-run write_findings.py once more per (folder, base_config) (now that compliance.json
     exists for the picks, the tables' "category" column -- previously "?" -- gets filled in).
     Same --search_workers pool as step 1.
  5. Run write_overall_findings.py per folder (cross-references picks across base_configs).

goal labels (matches the convention already used by hand in earlier folders):
    method A  (fit_quality_minmax)      -> best_fit_quality
    method B  (*_all_csvs)              -> best_consistent_with_all_csvs
    method C  (*_pair)                  -> best_consistent_with_<pair0_short>_and_<pair1_short>
    method D  (fit_quality_minmax_pair) -> best_fit_quality_pair

Usage:
    python run_plot_consistent_archs.py --csv_base_root ../runs/csv_base_2.5_20
    python run_plot_consistent_archs.py --csv_base_root ../runs/csv_base_2.5_20 --workers 16
    python run_plot_consistent_archs.py --csv_base_root ../runs/csv_base_2.5_20 \\
        --folders tanh_margin10 --dry_run
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_derived_configs_simplegate import FOLDERS  # noqa: E402
from find_consistent_archs import DEFAULT_PAIR  # noqa: E402

WRITE_FINDINGS = HERE / "write_findings.py"
WRITE_OVERALL = HERE / "write_overall_findings.py"
PLOT_ARCH_HASH = HERE / "plot_arch_hash_simplegate.py"
RENDER_HTML = HERE / "render_html_view.py"

DEFAULT_BASE_CONFIGS = [
    "best200ids_of_9069_byloss_rw0_2.5_20",
    "bothshapeok_of_9069_byshape_rw0_2.5_20",
    "best100_gmshapeok_of_9069_byloss_rw0_2.5_20",
]

# (json key, goal-label builder)
_METHOD_GOALS = [
    ("fit_quality_minmax", lambda pair: "best_fit_quality"),
    ("gmshape_ok_all_csvs", lambda pair: "best_consistent_with_all_csvs"),
    ("bothshape_ok_all_csvs", lambda pair: "best_consistent_with_all_csvs"),
    ("gmshape_ok_pair", lambda pair: f"best_consistent_with_{_short(pair[0])}_and_{_short(pair[1])}"),
    ("bothshape_ok_pair", lambda pair: f"best_consistent_with_{_short(pair[0])}_and_{_short(pair[1])}"),
    ("fit_quality_minmax_pair", lambda pair: "best_fit_quality_pair"),
]

_SHORT_RE = re.compile(r"new_(\d[\d.]*_\d+)_2_70W")


def _short(csv_name: str) -> str:
    """'cg2h40010_new_2.5_20_2_70W_center9' -> '2.5_20' (matches this project's own
    rw0_2.5_20-style suffix convention). Falls back to the full name if the pattern doesn't
    match (e.g. a differently-named measurement csv)."""
    m = _SHORT_RE.search(csv_name)
    return m.group(1) if m else csv_name


def _run(cmd: list[str], dry_run: bool) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def _write_findings_one(csv_base_root: Path, out_root: Path, base_config: str, folder: str,
                         pair: tuple[str, str], gm_ceiling: float | None, top: int,
                         python_exe: str, dry_run: bool) -> tuple[str, str, int]:
    """Returns (folder, base_config, returncode)."""
    root = csv_base_root / base_config
    if not root.is_dir():
        print(f"  SKIP {base_config}/{folder}: {root} does not exist")
        return folder, base_config, 1
    cmd = [python_exe, str(WRITE_FINDINGS), "--root", str(root), "--folder", folder,
           "--out_root", str(out_root), "--top", str(top), "--pair_csvs", ",".join(pair)]
    if gm_ceiling is not None:
        cmd += ["--gm_ceiling", str(gm_ceiling)]
    rc = _run(cmd, dry_run)
    if rc != 0:
        print(f"  FAILED write_findings.py for {base_config}/{folder}")
    return folder, base_config, rc


def _search_pass_all(csv_base_root: Path, out_root: Path, base_configs: list[str], folders: list[str],
                      pair: tuple[str, str], gm_ceiling: float | None, top: int,
                      python_exe: str, dry_run: bool, search_workers: int,
                      label: str) -> dict[str, dict[str, Path]]:
    """Runs write_findings.py for every (folder, base_config) pair in a worker pool (cheap, no
    GPU work). Returns {folder: {base_config: json_path}}."""
    print(f"\n=== {label}: {len(folders)} folder(s) x {len(base_configs)} base_config(s) "
          f"(search_workers={search_workers}) ===")
    tasks = [(bc, f) for f in folders for bc in base_configs]
    results: dict[str, dict[str, Path]] = {f: {} for f in folders}
    if dry_run:
        for bc, f in tasks:
            _write_findings_one(csv_base_root, out_root, bc, f, pair, gm_ceiling, top, python_exe, True)
        return results
    with ThreadPoolExecutor(max_workers=search_workers) as ex:
        futs = {ex.submit(_write_findings_one, csv_base_root, out_root, bc, f, pair, gm_ceiling,
                           top, python_exe, False): (bc, f) for bc, f in tasks}
        for fut in as_completed(futs):
            folder, base_config, rc = fut.result()
            if rc == 0:
                results[folder][base_config] = out_root / folder / f"consistency_summary_{base_config}.json"
    return results


def _collect_picks(json_paths: dict[str, Path], pair: tuple[str, str]):
    """[(arch_hash, goal, base_config)], deduped by (arch_hash, goal), first base_config wins
    (base_configs are iterated in json_paths' own dict order)."""
    seen = {}
    order = []
    for base_config, jp in json_paths.items():
        if not jp.is_file():
            continue
        doc = json.loads(jp.read_text(encoding="utf-8"))
        for json_key, goal_fn in _METHOD_GOALS:
            for r in doc.get(json_key) or []:
                h = r.get("arch_hash")
                if not h:
                    continue
                goal = goal_fn(pair)
                key = (h, goal)
                if key not in seen:
                    seen[key] = base_config
                    order.append(key)
    return [(h, goal, seen[(h, goal)]) for h, goal in order]


def _plot_one(csv_base_root: Path, out_root: Path, folder: str, arch_hash: str, goal: str,
              base_config: str, fmt: str, python_exe: str, dry_run: bool) -> int:
    root = csv_base_root / base_config
    cmd = [python_exe, str(PLOT_ARCH_HASH), "--root", str(root), "--folder", folder,
           "--arch_hash", arch_hash, "--goal", goal, "--out_dir", str(out_root),
           "--format", fmt]
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


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
    ap.add_argument("--top", type=int, default=5, help="write_findings.py's --top (per method)")
    ap.add_argument("--format", choices=("pdf", "md", "both"), default="md",
                    help="plot_arch_hash.py --format for the picks (default: md)")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel plot_arch_hash.py calls, shared across ALL folders at once "
                         "(default: 8) -- the heaviest step (loads already-trained weights + "
                         "runs inference/plotting, no training), so this is the one number that "
                         "actually caps total concurrent load.")
    ap.add_argument("--search_workers", type=int, default=8,
                    help="parallel write_findings.py calls for the search/refresh passes "
                         "(default: 8) -- cheap (CSV reads/writes only), safe to raise well "
                         "above --workers.")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--skip_plotting", action="store_true",
                    help="Run the search + refresh passes and overall summaries, but skip the "
                         "plot_arch_hash.py step entirely. Useful for cheaply regenerating "
                         "consistency_summary_*.md/.json (e.g. after a find_consistent_archs.py "
                         "formula/field change) without re-plotting images that haven't changed.")
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

    all_json_paths = _search_pass_all(csv_base_root, out_root, base_configs, folders, pair,
                                       args.gm_ceiling, args.top, args.python_exe, args.dry_run,
                                       args.search_workers, "search pass")

    all_picks = []  # (folder, arch_hash, goal, base_config)
    for folder in folders:
        json_paths = all_json_paths.get(folder) or {}
        if not json_paths:
            print(f"  SKIP {folder}: no base_config search succeeded")
            continue
        picks = _collect_picks(json_paths, pair)
        print(f"\n=== {folder}: {len(picks)} unique (arch_hash, goal) picks to plot ===")
        for h, goal, bc in picks:
            print(f"  {h}  goal={goal}  from={bc}")
            all_picks.append((folder, h, goal, bc))

    print(f"\n=== plotting {len(all_picks)} unique picks across {len(folders)} folder(s) "
          f"(workers={args.workers}){' -- SKIPPED (--skip_plotting)' if args.skip_plotting else ''} ===")
    if args.skip_plotting:
        pass
    elif not args.dry_run and all_picks:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
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

    _search_pass_all(csv_base_root, out_root, base_configs, folders, pair, args.gm_ceiling,
                      args.top, args.python_exe, args.dry_run, args.search_workers,
                      "refresh pass (fills in category column now that plots exist)")

    print(f"\n=== overall summaries ({len(folders)} folder(s)) ===")
    for folder in folders:
        if not all_json_paths.get(folder):
            continue
        cmd = [args.python_exe, str(WRITE_OVERALL), "--out_root", str(out_root), "--folder", folder,
               "--top", str(max(args.top * 4, 20))]
        _run(cmd, args.dry_run)

    if args.html:
        print(f"\n=== html view ===")
        _run([args.python_exe, str(RENDER_HTML), "--out_root", str(out_root),
              "--workers", str(args.workers)], args.dry_run)

    print("\ndone.")


if __name__ == "__main__":
    main()
