"""
write_overall_best_archs.py -- top-level (spanning ALL folders) leaderboard: for each of the 6
methods (A / B gmshape_ok / B bothshape_ok / C gmshape_ok / C bothshape_ok / D), which
<folder>/<arch_hash> is the single best pick in EACH folder (comparing that folder's 3
base_configs' own rank-#1 picks via find_consistent_archs.method_sort_key, keeping whichever is
actually best -- not just the first one found), then all folders ranked against each other by
that same criterion.

Uses the UNRESTRICTED (heterogeneous-activation) consistency_summary_<base_config>.json for
every folder -- this is the "what's the best architecture overall, across the whole project"
summary, not a tanh-only comparison (see <folder>/tanh_only_heterogeneus_comparisons/ for that,
per folder).

Written directly at the best_archs_plots ROOT (--out_root itself), not nested inside any
folder, since it spans all of them.

Run AFTER run_plot_consistent_archs.py has produced consistency_summary_<base_config>.json for
every folder x base_config you want included.

Usage:
    python write_overall_best_archs.py --out_root ../runs/csv_base_2.5_20/best_archs_plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_derived_configs import FOLDERS  # noqa: E402
from find_consistent_archs import METHOD_INFO, method_sort_key, format_ids_rmse, \
    format_combined_gm, format_worst_pct, format_best_pct, load_arch_by_hash, \
    find_arch_plot_dir, arch_link_cell, is_tanh_only  # noqa: E402

DEFAULT_BASE_CONFIGS = [
    "best200ids_of_9069_byloss_rw0_2.5_20",
    "bothshapeok_of_9069_byshape_rw0_2.5_20",
    "best100_gmshapeok_of_9069_byloss_rw0_2.5_20",
]


def _load(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _short_bc(bc: str) -> str:
    """'best200ids_of_9069_byloss_rw0_2.5_20' -> 'best200ids' -- the '_of_9069_by...' suffix is
    the same shared derivation-sweep boilerplate on every base_config, not distinguishing info."""
    return bc.split("_of_")[0]


_arch_cache: dict[tuple[str, str], dict[str, str]] = {}
_tanh_only_cache: dict[tuple[str, str], set[str]] = {}


def _arch_cell(root: str, out_root: Path, folder: str, arch_hash: str) -> str:
    key = (root, folder)
    if key not in _arch_cache:
        _arch_cache[key] = load_arch_by_hash(Path(root), folder)
    arch_dir = find_arch_plot_dir(out_root, folder, arch_hash)
    tanh = is_tanh_only(Path(root), folder, arch_hash, _tanh_only_cache)
    return arch_link_cell(arch_hash, _arch_cache[key], arch_dir, out_root, tanh)


def _stat_str(json_key: str, r: dict) -> str:
    if json_key in ("fit_quality_minmax", "fit_quality_minmax_pair"):
        pair_note = " (pair)" if json_key.endswith("_pair") else ""
        return (f"worst-case pct{pair_note} **{format_worst_pct(r)}**, best-case pct{pair_note} "
                f"**{format_best_pct(r)}**, ids_rmse{pair_note} {format_ids_rmse(r)}, "
                f"combined_gm{pair_note} {format_combined_gm(r)}")
    if json_key in ("gmshape_ok_all_csvs", "bothshape_ok_all_csvs"):
        return (f"hits **{r['count']}/{r['of']}**, worst-case pct {format_worst_pct(r)}, "
                f"best-case pct {format_best_pct(r)}, ids_rmse {format_ids_rmse(r)}, "
                f"combined_gm {format_combined_gm(r)}")
    return (f"worst-case pct (pair) {format_worst_pct(r)}, best-case pct (pair) {format_best_pct(r)}, "
            f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}")


def build(out_root: Path, folders: list[str], base_configs: list[str], top: int) -> Path:
    lines = ["# Overall best architectures across all folders (heterogeneous / unrestricted)", "",
             f"For each method, the single BEST architecture found in each of the {len(folders)} "
             f"folders (comparing that folder's {len(base_configs)} base_configs' own rank-#1 "
             f"picks via the method's own ranking criterion, keeping whichever is actually "
             f"better), then every folder ranked against every other folder by that same "
             f"criterion. Uses the unrestricted (heterogeneous-activation) search -- see each "
             f"folder's own `tanh_only_heterogeneus_comparisons/` for the tanh-only-vs-"
             f"heterogeneous comparison instead.", ""]

    for json_key, heading, _scope in METHOD_INFO:
        lines.append(f"## {heading}")
        lines.append("")
        folder_best = []  # (folder, base_config, root, row)
        for folder in folders:
            best_row = None
            best_bc = None
            best_root = None
            for bc in base_configs:
                doc = _load(out_root / folder / f"consistency_summary_{bc}.json")
                if not doc:
                    continue
                rows = doc.get(json_key) or []
                if not rows:
                    continue
                r = rows[0]
                if best_row is None or method_sort_key(json_key, r) < method_sort_key(json_key, best_row):
                    best_row, best_bc, best_root = r, bc, doc.get("root")
            if best_row is not None:
                folder_best.append((folder, best_bc, best_root, best_row))

        folder_best.sort(key=lambda t: method_sort_key(json_key, t[3]))
        if not folder_best:
            lines.append("(no picks in any folder for this method)")
        else:
            lines.append("| rank | folder | arch | base_config | stat |")
            lines.append("|---|---|---|---|---|")
            for i, (folder, bc, root, r) in enumerate(folder_best[:top], start=1):
                arch_cell = _arch_cell(root, out_root, folder, r["arch_hash"]) if root else f"`{r['arch_hash']}`"
                lines.append(f"| #{i} | [`{folder}`]({folder}/consistency_summary_{bc}.md) | "
                             f"{arch_cell} | {_short_bc(bc)} | {_stat_str(json_key, r)} |")
        lines.append("")

    out_path = out_root / "best_archs_summary_overall.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, help="e.g. ../runs/csv_base_2.5_20/best_archs_plots")
    ap.add_argument("--base_configs", default=",".join(DEFAULT_BASE_CONFIGS))
    ap.add_argument("--folders", default=",".join(FOLDERS))
    ap.add_argument("--top", type=int, default=8, help="how many folders to list per method (default: all 8)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out_root not found: {out_root}")
    base_configs = [b.strip() for b in args.base_configs.split(",") if b.strip()]
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]

    build(out_root, folders, base_configs, args.top)


if __name__ == "__main__":
    main()
