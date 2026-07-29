"""
write_tanh_only_comparison.py -- side-by-side comparison of the tanh-only consistency search
(write_findings.py --only_tanh, under <folder>/tanh_only/) against the unrestricted
heterogeneous-activation search (<folder>/consistency_summary_<base_config>.json), for each of
the 6 method sections (A / B gmshape_ok / B bothshape_ok / C gmshape_ok / C bothshape_ok / D).

At the top of every file, a verdict summary states which method(s) heterogeneous won (rank-#1
vs rank-#1, compared via find_consistent_archs.method_sort_key -- the SAME criterion each
search already sorts by, not a raw percentage eyeball) and calls out explicitly if tanh-only
won any -- that's the notable case worth a reader's attention, since heterogeneous winning is
the common case (see important_mds/bestids_across_allcsvs_analysis_steps.md).

Writes ONE file per base_config (comparison_<base_config>.md) plus ONE comparison_overall.md
that concatenates all base_configs' tables into a single document -- into
<out_root>/<folder>/tanh_only_heterogeneus_comparisons/.

Run AFTER both write_findings.py (unrestricted) AND write_findings.py --only_tanh have produced
their consistency_summary_<base_config>.json for every base_config you want compared.

Usage:
    python write_tanh_only_comparison.py --out_root ../runs/csv_base_2.5_20/best_archs_plots \\
        --folder tanh_margin10
    python write_tanh_only_comparison.py --out_root ... --folder tanh_margin10,softplus \\
        --base_configs best200ids_of_9069_byloss_rw0_2.5_20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from find_consistent_archs import METHOD_INFO, method_sort_key, format_ids_rmse, \
    format_combined_gm, format_worst_pct, format_best_pct, load_arch_by_hash, \
    find_arch_plot_dir, arch_link_cell, is_tanh_only  # noqa: E402

_arch_cache: dict[tuple[str, str], dict[str, str]] = {}
_tanh_only_cache: dict[tuple[str, str], set[str]] = {}


def _arch_cell(root: str | None, out_root: Path, folder: str, arch_hash: str, from_dir: Path,
               mark_tanh_only: bool = False) -> str:
    if not root:
        return f"`{arch_hash}`"
    key = (root, folder)
    if key not in _arch_cache:
        _arch_cache[key] = load_arch_by_hash(Path(root), folder)
    arch_dir = find_arch_plot_dir(out_root, folder, arch_hash)
    tanh = mark_tanh_only and is_tanh_only(Path(root), folder, arch_hash, _tanh_only_cache)
    return arch_link_cell(arch_hash, _arch_cache[key], arch_dir, from_dir, tanh)

# json_key -> stat formatter (heading/scope come from the shared METHOD_INFO)
_STAT_FN = {
    "fit_quality_minmax":
        lambda r: f"worst-case pct **{format_worst_pct(r)}**, best-case pct **{format_best_pct(r)}**, "
                  f"ids_rmse {format_ids_rmse(r)}, combined_gm {format_combined_gm(r)}",
    "gmshape_ok_all_csvs":
        lambda r: f"hits **{r['count']}/{r['of']}**, worst-case pct {format_worst_pct(r)}, "
                  f"best-case pct {format_best_pct(r)}, ids_rmse {format_ids_rmse(r)}, "
                  f"combined_gm {format_combined_gm(r)}",
    "bothshape_ok_all_csvs":
        lambda r: f"hits **{r['count']}/{r['of']}**, worst-case pct {format_worst_pct(r)}, "
                  f"best-case pct {format_best_pct(r)}, ids_rmse {format_ids_rmse(r)}, "
                  f"combined_gm {format_combined_gm(r)}",
    "gmshape_ok_pair":
        lambda r: f"worst-case pct (pair) {format_worst_pct(r)}, best-case pct (pair) {format_best_pct(r)}, "
                  f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}",
    "bothshape_ok_pair":
        lambda r: f"worst-case pct (pair) {format_worst_pct(r)}, best-case pct (pair) {format_best_pct(r)}, "
                  f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}",
    "fit_quality_minmax_pair":
        lambda r: f"worst-case pct (pair) **{format_worst_pct(r)}**, best-case pct (pair) **{format_best_pct(r)}**, "
                  f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}",
}


def _load(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _compare_rank1(json_key: str, het_rows: list, tanh_rows: list) -> str:
    """'heterogeneous' / 'tanh_only' / 'tie' / 'no_data', comparing each search's own rank-#1
    pick (already the best in that search, by that search's own sort order) via
    method_sort_key -- NOT by eyeballing e.g. worst-case pct directly, since a tanh-only
    percentile is computed against a much smaller candidate pool and isn't on the same scale
    (see format_worst_pct's pool-size note).

    If both searches land on the SAME arch_hash (common when the best architecture overall
    happens to itself be tanh-only, so it's simultaneously #1 in both the full population and
    the tanh-only-restricted one), this is a tie by definition -- it's the identical physical
    architecture, so comparing its own pool-size-dependent percentile against itself would
    otherwise misreport a "winner" that doesn't actually reflect two different architectures."""
    het_r = het_rows[0] if het_rows else None
    tanh_r = tanh_rows[0] if tanh_rows else None
    if het_r is None and tanh_r is None:
        return "no_data"
    if het_r is None:
        return "tanh_only"
    if tanh_r is None:
        return "heterogeneous"
    if het_r["arch_hash"] == tanh_r["arch_hash"]:
        return "tie"
    het_key = method_sort_key(json_key, het_r)
    tanh_key = method_sort_key(json_key, tanh_r)
    if het_key < tanh_key:
        return "heterogeneous"
    if tanh_key < het_key:
        return "tanh_only"
    return "tie"


def _verdict_summary_lines(het_doc: dict | None, tanh_doc: dict | None) -> list[str]:
    het_better, tanh_better, tied, no_data = [], [], [], []
    for json_key, heading, _scope in METHOD_INFO:
        het_rows = (het_doc.get(json_key) or []) if het_doc else []
        tanh_rows = (tanh_doc.get(json_key) or []) if tanh_doc else []
        verdict = _compare_rank1(json_key, het_rows, tanh_rows)
        {"heterogeneous": het_better, "tanh_only": tanh_better,
         "tie": tied, "no_data": no_data}[verdict].append(heading)

    lines = ["**Rank #1 verdict, method by method** (heterogeneous vs tanh-only, compared by "
             "each method's own ranking criterion):", ""]
    if het_better:
        lines.append(f"- Heterogeneous was better in: {'; '.join(het_better)}.")
    if tanh_better:
        lines.append(f"- **Tanh-only was better in: {'; '.join(tanh_better)}.**")
    if tied:
        lines.append(f"- Tied (identical rank-#1 architecture) in: {'; '.join(tied)}.")
    if no_data:
        lines.append(f"- No comparison possible (no pick in one or both searches) in: {'; '.join(no_data)}.")
    lines.append("")
    return lines


def _comparison_table(het_doc: dict | None, tanh_doc: dict | None, out_root: Path, folder: str,
                       from_dir: Path, top: int) -> list[str]:
    root = (het_doc or {}).get("root") or (tanh_doc or {}).get("root")
    lines = ["| method | rank | tanh-only arch | tanh-only stat | heterogeneous arch | heterogeneous stat |",
             "|---|---|---|---|---|---|"]
    any_row = False
    for json_key, heading, _scope in METHOD_INFO:
        stat_fn = _STAT_FN[json_key]
        het_rows = (het_doc.get(json_key) or []) if het_doc else []
        tanh_rows = (tanh_doc.get(json_key) or []) if tanh_doc else []
        n = min(top, max(len(het_rows), len(tanh_rows)))
        for i in range(n):
            any_row = True
            th = tanh_rows[i] if i < len(tanh_rows) else None
            hh = het_rows[i] if i < len(het_rows) else None
            t_cell = _arch_cell(root, out_root, folder, th["arch_hash"], from_dir) if th else "-"
            t_stat = stat_fn(th) if th else "(no tanh-only pick at this rank)"
            h_cell = _arch_cell(root, out_root, folder, hh["arch_hash"], from_dir, True) if hh else "-"
            h_stat = stat_fn(hh) if hh else "(no heterogeneous pick at this rank)"
            label = heading if i == 0 else ""
            lines.append(f"| {label} | #{i + 1} | {t_cell} | {t_stat} | {h_cell} | {h_stat} |")
    if not any_row:
        return ["(no picks in either search)"]
    return lines


def build_for_base_config(out_root: Path, folder: str, base_config: str, top: int,
                           comp_dir: Path) -> Path:
    het_doc = _load(out_root / folder / f"consistency_summary_{base_config}.json")
    tanh_doc = _load(out_root / folder / "tanh_only" / f"consistency_summary_{base_config}.json")

    lines = [f"# `{folder}` tanh-only vs heterogeneous comparison ({base_config})", ""]
    if het_doc is None:
        lines.append(f"_(no heterogeneous consistency_summary_{base_config}.json found -- run "
                      f"write_findings.py for this base_config first)_")
    if tanh_doc is None:
        lines.append(f"_(no tanh-only consistency_summary_{base_config}.json found -- run "
                      f"write_findings.py --only_tanh for this base_config first)_")
    if het_doc is None or tanh_doc is None:
        lines.append("")
    lines.extend(_verdict_summary_lines(het_doc, tanh_doc))
    lines.append(f"Top {top} of each method section, tanh-only search vs the unrestricted "
                 f"(heterogeneous-activation) search, same rank position side by side. See "
                 f"[`../tanh_only/consistency_summary_{base_config}.md`]"
                 f"(../tanh_only/consistency_summary_{base_config}.md) and "
                 f"[`../consistency_summary_{base_config}.md`](../consistency_summary_{base_config}.md) "
                 f"for the full un-paired tables.")
    lines.append("")
    lines.extend(_comparison_table(het_doc, tanh_doc, out_root, folder, comp_dir, top))
    lines.append("")

    out_path = comp_dir / f"comparison_{base_config}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def build_overall(out_root: Path, folder: str, base_configs: list[str], top: int,
                   comp_dir: Path) -> Path:
    lines = [f"# `{folder}` tanh-only vs heterogeneous comparison -- overall", "",
              f"Same comparison as each `comparison_<base_config>.md` below, concatenated into "
              f"one file for a single-scroll overview across all {len(base_configs)} "
              f"base_config(s).", ""]
    for base_config in base_configs:
        het_doc = _load(out_root / folder / f"consistency_summary_{base_config}.json")
        tanh_doc = _load(out_root / folder / "tanh_only" / f"consistency_summary_{base_config}.json")
        lines.append(f"## {base_config}")
        lines.append("")
        lines.append(f"([full comparison](comparison_{base_config}.md) -- "
                      f"[tanh-only summary](../tanh_only/consistency_summary_{base_config}.md) -- "
                      f"[heterogeneous summary](../consistency_summary_{base_config}.md))")
        lines.append("")
        lines.extend(_verdict_summary_lines(het_doc, tanh_doc))
        lines.extend(_comparison_table(het_doc, tanh_doc, out_root, folder, comp_dir, top))
        lines.append("")

    out_path = comp_dir / "comparison_overall.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True,
                    help="Shared best_archs_plots root (same one passed to write_findings.py "
                         "--out_root).")
    ap.add_argument("--folder", required=True, help="comma-separated for several at once")
    ap.add_argument("--base_configs", default=None,
                    help="comma-separated; default: every consistency_summary_*.json found "
                         "directly under <out_root>/<folder>/ (the heterogeneous ones)")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out_root not found: {out_root}")
    folders = [f.strip() for f in args.folder.split(",") if f.strip()]

    for folder in folders:
        if args.base_configs:
            base_configs = [b.strip() for b in args.base_configs.split(",") if b.strip()]
        else:
            base_configs = sorted(
                p.stem.removeprefix("consistency_summary_")
                for p in (out_root / folder).glob("consistency_summary_*.json")
            )
        if not base_configs:
            print(f"SKIP {folder}: no consistency_summary_*.json found under {out_root / folder}")
            continue
        comp_dir = out_root / folder / "tanh_only_heterogeneus_comparisons"
        for base_config in base_configs:
            build_for_base_config(out_root, folder, base_config, args.top, comp_dir)
        build_overall(out_root, folder, base_configs, args.top, comp_dir)


if __name__ == "__main__":
    main()
