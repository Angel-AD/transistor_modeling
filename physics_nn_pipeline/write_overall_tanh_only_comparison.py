"""
write_overall_tanh_only_comparison.py -- top-level (spanning ALL folders) tanh-only vs
heterogeneous comparison, analogous to write_overall_best_archs.py's leaderboard: for each of
the 6 methods (A / B gmshape_ok / B bothshape_ok / C gmshape_ok / C bothshape_ok / D), reduces
each folder's 3 base_configs down to a single best pick (via find_consistent_archs.
method_sort_key) SEPARATELY for the heterogeneous search and the tanh-only search, then compares
the two per folder using the same same-hash-tie-aware logic write_tanh_only_comparison.py already
uses per base_config (a folder whose overall-best architecture happens to itself be tanh-only is
a TIE, not a spurious "winner", since it's the identical physical architecture on both sides).

Written directly at the best_archs_plots ROOT (--out_root itself), not nested inside any folder,
since it spans all of them -- same convention as best_archs_summary_overall.md.

Run AFTER run_plot_consistent_archs.py AND run_tanh_only_analysis.py have both produced their
consistency_summary_<base_config>.json (heterogeneous and tanh_only/) for every folder x
base_config you want included.

Usage:
    python write_overall_tanh_only_comparison.py --out_root ../runs/csv_base_2.5_20/best_archs_plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_derived_configs import FOLDERS  # noqa: E402
from find_consistent_archs import METHOD_INFO, method_sort_key, load_arch_by_hash, \
    find_arch_plot_dir, arch_link_cell, is_tanh_only  # noqa: E402
from write_tanh_only_comparison import _compare_rank1, _STAT_FN  # noqa: E402
from write_overall_best_archs import DEFAULT_BASE_CONFIGS, _load  # noqa: E402

_arch_cache: dict[tuple[str, str], dict[str, str]] = {}
_tanh_only_cache: dict[tuple[str, str], set[str]] = {}


def _arch_cell(root: str | None, out_root: Path, folder: str, arch_hash: str,
               mark_tanh_only: bool = False) -> str:
    if not root:
        return f"`{arch_hash}`"
    key = (root, folder)
    if key not in _arch_cache:
        _arch_cache[key] = load_arch_by_hash(Path(root), folder)
    arch_dir = find_arch_plot_dir(out_root, folder, arch_hash)
    tanh = mark_tanh_only and is_tanh_only(Path(root), folder, arch_hash, _tanh_only_cache)
    return arch_link_cell(arch_hash, _arch_cache[key], arch_dir, out_root, tanh)


def _best_across_bc(out_root: Path, folder: str, base_configs: list[str], json_key: str,
                     subdir: str | None = None) -> tuple[dict | None, str | None]:
    """Best rank-#1 pick for this folder+method across its base_configs' own consistency_summary
    json (optionally under subdir, e.g. 'tanh_only'), plus which root produced it -- same
    reduction write_overall_best_archs.py does for the unrestricted leaderboard, applied here to
    EITHER search variant."""
    best_row = best_root = None
    for bc in base_configs:
        rel = f"{subdir}/consistency_summary_{bc}.json" if subdir else f"consistency_summary_{bc}.json"
        doc = _load(out_root / folder / rel)
        if not doc:
            continue
        rows = doc.get(json_key) or []
        if not rows:
            continue
        r = rows[0]
        if best_row is None or method_sort_key(json_key, r) < method_sort_key(json_key, best_row):
            best_row, best_root = r, doc.get("root")
    return best_row, best_root


def build(out_root: Path, folders: list[str], base_configs: list[str]) -> Path:
    lines = ["# Tanh-only vs heterogeneous comparison -- overall (across all folders)", "",
             f"For each method, compares the single BEST heterogeneous pick against the single "
             f"BEST tanh-only pick in each of the {len(folders)} folders (each reduced across "
             f"that folder's {len(base_configs)} base_configs via the method's own ranking "
             f"criterion, same reduction as `best_archs_summary_overall.md`), then states the "
             f"verdict per folder (same same-hash-tie-aware comparison as each folder's own "
             f"`tanh_only_heterogeneus_comparisons/`).", ""]

    for json_key, heading, _scope in METHOD_INFO:
        stat_fn = _STAT_FN[json_key]
        het_better, tanh_better, tied, no_data = [], [], [], []
        rows_out = []
        for folder in folders:
            het_row, het_root = _best_across_bc(out_root, folder, base_configs, json_key)
            tanh_row, tanh_root = _best_across_bc(out_root, folder, base_configs, json_key,
                                                   subdir="tanh_only")
            verdict = _compare_rank1(json_key, [het_row] if het_row else [],
                                      [tanh_row] if tanh_row else [])
            {"heterogeneous": het_better, "tanh_only": tanh_better,
             "tie": tied, "no_data": no_data}[verdict].append(folder)
            rows_out.append((folder, het_row, het_root, tanh_row, tanh_root, verdict))

        lines.append(f"## {heading}")
        lines.append("")
        if het_better:
            lines.append("- Heterogeneous was better in: " + ", ".join(f"`{f}`" for f in het_better) + ".")
        if tanh_better:
            lines.append("- **Tanh-only was better in: " + ", ".join(f"`{f}`" for f in tanh_better) + ".**")
        if tied:
            lines.append("- Tied (identical rank-#1 architecture) in: " +
                          ", ".join(f"`{f}`" for f in tied) + ".")
        if no_data:
            lines.append("- No comparison possible (no pick in one or both searches) in: " +
                          ", ".join(f"`{f}`" for f in no_data) + ".")
        lines.append("")

        lines.append("| folder | tanh-only arch | tanh-only stat | heterogeneous arch | "
                      "heterogeneous stat | verdict |")
        lines.append("|---|---|---|---|---|---|")
        for folder, het_row, het_root, tanh_row, tanh_root, verdict in rows_out:
            t_cell = _arch_cell(tanh_root, out_root, folder, tanh_row["arch_hash"]) if tanh_row else "-"
            t_stat = stat_fn(tanh_row) if tanh_row else "(no tanh-only pick)"
            h_cell = _arch_cell(het_root, out_root, folder, het_row["arch_hash"], True) if het_row else "-"
            h_stat = stat_fn(het_row) if het_row else "(no heterogeneous pick)"
            lines.append(f"| [`{folder}`]({folder}/tanh_only_heterogeneus_comparisons/"
                         f"comparison_overall.md) | {t_cell} | {t_stat} | {h_cell} | {h_stat} | {verdict} |")
        lines.append("")

    out_path = out_root / "tanh_only_vs_heterogeneous_overall.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True, help="e.g. ../runs/csv_base_2.5_20/best_archs_plots")
    ap.add_argument("--base_configs", default=",".join(DEFAULT_BASE_CONFIGS))
    ap.add_argument("--folders", default=",".join(FOLDERS))
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out_root not found: {out_root}")
    base_configs = [b.strip() for b in args.base_configs.split(",") if b.strip()]
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]

    build(out_root, folders, base_configs)


if __name__ == "__main__":
    main()
