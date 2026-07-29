"""
Auto-generate a consistency_summary_<root_name>.md for one (root, folder) from
find_consistent_archs.py's search results (fit-quality min-max; gmshape_ok/bothshape_ok x
goal=all_csvs/pair), plus each picked arch_hash's compact architecture summary (from the
'architecture' column) and its final category (both/gmshape/gdsshape/none) as written by
plot_arch_hash.py's compliance.json.

Written to <out_root>/<folder>/consistency_summary_<root.name>.md (override the filename with
--out; --out_root controls WHERE, default <root>/best_archs_plots -- pass --out_root explicitly
to point at a shared best_archs_plots tree when several derivation roots' picks are meant to
land in one place, e.g. --root .../csv_base_2.5_20/best200ids_of_9069_byloss_rw0_2.5_20
--out_root .../csv_base_2.5_20/best_archs_plots -- that's also where plot_arch_hash.py must have
been pointed via ITS OWN --out_dir for the compliance.json lookups below to find anything).
A companion consistency_summary_<root.name>.json is written alongside with the raw result data
(same numbers as the tables, machine-readable) -- consumed by write_overall_findings.py to
cross-reference picks across several roots/base_configs for the same folder without having to
re-parse markdown tables.

This is a template, not hand-written prose -- used to keep many folders x roots consistent
and tractable. Run AFTER plot_arch_hash.py has produced <out_root>/<folder>/<category>/
<goal>/<arch_hash>/compliance.json for every picked hash.

Usage:
    python write_findings.py --root ../runs/best200ids_of_9069_byloss --folder sigmoid_margin10
    python write_findings.py --root ../runs/csv_base_2.5_20/best100_gmshapeok_of_9069_byloss_rw0_2.5_20 \\
        --folder tanh_margin10 --out_root ../runs/csv_base_2.5_20/best_archs_plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_csv_row import _read_rows  # noqa: E402
from find_consistent_archs import load_population, search_fit_quality_minmax, \
    search_shape_all_csvs, search_shape_pair, DEFAULT_PAIR, \
    tanh_only_hashes, filter_population, format_ids_rmse, format_combined_gm, \
    format_worst_pct, format_best_pct, find_arch_plot_dir, arch_link_cell  # noqa: E402


def _compliance(arch_dir: Path | None) -> dict | None:
    if arch_dir is None:
        return None
    try:
        return json.loads((arch_dir / "compliance.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--pair_csvs", default=",".join(DEFAULT_PAIR))
    ap.add_argument("--gm_ceiling", type=float, default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out_root", default=None,
                    help="Where plot_arch_hash.py's <folder>/<category>/<goal>/<arch_hash>/ tree "
                         "lives (for compliance.json lookups) and where the summary md/json get "
                         "written. Default: <root>/best_archs_plots (matches plot_arch_hash.py's "
                         "own default --out_dir). Pass explicitly when plot_arch_hash.py was run "
                         "with a shared --out_dir across several roots.")
    ap.add_argument("--out", default=None,
                    help="Override the summary .md path. Default: "
                         "<out_root>/<folder>/consistency_summary_<root.name>.md (or .../"
                         "tanh_only/consistency_summary_<root.name>.md with --only_tanh)")
    ap.add_argument("--only_tanh", action="store_true",
                    help="Restrict the population to architectures using ONLY 'tanh' in every "
                         "layer (no swish/mish mixed in) before running any of the 5 searches. "
                         "Writes to <out_root>/<folder>/tanh_only/ instead of <out_root>/"
                         "<folder>/ -- compliance.json lookups still use the SHARED (non-tanh_"
                         "only) plots tree, since plots aren't duplicated per-filter.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    folder = args.folder
    gm_ceiling = args.gm_ceiling if args.gm_ceiling is not None else (1.4 if "_nogm" in folder else 1.1)
    pair = tuple(c.strip() for c in args.pair_csvs.split(","))
    out_root = Path(args.out_root).resolve() if args.out_root else root / "best_archs_plots"
    out_dir = (out_root / folder / "tanh_only") if args.only_tanh else (out_root / folder)

    pop = load_population(root, folder)
    if args.only_tanh:
        keep = tanh_only_hashes(root, folder)
        pop = filter_population(pop, keep)
        print(f"[only_tanh] restricting population to {len(keep)} tanh-only arch_hash(es)")
        tanh_set: set[str] = set()  # every row here is already tanh-only -- marker would be noise
    else:
        tanh_set = tanh_only_hashes(root, folder)
    n_shape = sum(1 for d in pop.values() if d["shape"] is not None)
    n_csvs = len(pop)

    arch_by_hash = {}
    for csv_name, d in pop.items():
        ranked_dir = root / csv_name / folder / "ranked_region_knee_vgs-3to0_vds0to15"
        base = sorted(ranked_dir.glob("*_ranked_by_region_knee_combined_gm.csv"))
        if not base:
            continue
        for r in _read_rows(base[0]):
            h = r.get("arch_hash")
            if h and h not in arch_by_hash:
                arch_by_hash[h] = r.get("architecture", "")

    fq = search_fit_quality_minmax(pop, gm_ceiling, args.top)
    used_ceiling = gm_ceiling
    while not fq and used_ceiling < 5.0:
        used_ceiling += 0.4
        fq = search_fit_quality_minmax(pop, used_ceiling, args.top)

    fq_pair = search_fit_quality_minmax(pop, gm_ceiling, args.top, csv_names=list(pair))
    used_ceiling_pair = gm_ceiling
    while not fq_pair and used_ceiling_pair < 5.0:
        used_ceiling_pair += 0.4
        fq_pair = search_fit_quality_minmax(pop, used_ceiling_pair, args.top, csv_names=list(pair))

    gm_all = search_shape_all_csvs(pop, "gmshape_ok", args.top)
    both_all = search_shape_all_csvs(pop, "bothshape_ok", args.top)
    gm_pair = search_shape_pair(pop, "gmshape_ok", pair, args.top)
    both_pair = search_shape_pair(pop, "bothshape_ok", pair, args.top)

    lines = []
    tanh_note = " -- tanh-only subset" if args.only_tanh else ""
    lines.append(f"# `{folder}` consistency findings ({root.name}){tanh_note}")
    lines.append("")
    if args.only_tanh:
        lines.append(f"**Restricted to architectures using ONLY `tanh` in every layer** "
                      f"({len(keep)} of the folder's full population). See the unrestricted "
                      f"`../consistency_summary_{root.name}.md` for the heterogeneous-activation "
                      f"population, or `../tanh_only_heterogeneus_comparisons/` for a side-by-"
                      f"side comparison.")
        lines.append("")
    lines.append(f"Population: {n_csvs} csvs, full-population shape analysis available for {n_shape}/{n_csvs}. "
                  f"Every architecture below is identified by `arch_hash` (content-stable across the "
                  f"independently-compiled csv folders), not the numeric `arch_id`. "
                  f"Pair goal = `{pair[0]}` AND `{pair[1]}`.")
    lines.append("")

    lines.append("## A) Fit-quality min-max search (no shape data needed)")
    lines.append("")
    lines.append(f"Worst-case `ids_rmse` percentile across all {n_csvs} csvs, within each csv's own "
                  f"`combined_gm<={used_ceiling:.1f}` pool" +
                  (f" (auto-escalated from {gm_ceiling:.1f} -- no arch qualified at the default ceiling)"
                   if used_ceiling != gm_ceiling else "") + ". Lower worst-case % = more reliably good everywhere. "
                  f"The `(ids_rmse rank R/N)` shown alongside each pct is a separate, purely informational "
                  f"number: this arch's plain `ids_rmse` rank against the FULL population of that csv's "
                  f"config (N = total architectures in the config, e.g. 200/100/52), with no `combined_gm` "
                  f"gate applied -- it doesn't decide qualification or ranking, only the pct does.")
    lines.append("")
    if fq:
        lines.append("| arch | worst-case pct | best-case pct | ids_rmse | combined_gm | category |")
        lines.append("|---|---|---|---|---|---|")
        for r in fq:
            arch_dir = find_arch_plot_dir(out_root, folder, r["arch_hash"])
            comp = _compliance(arch_dir)
            cat = comp.get("category", "?") if comp else "?"
            cell = arch_link_cell(r["arch_hash"], arch_by_hash, arch_dir, out_dir, r["arch_hash"] in tanh_set)
            lines.append(f"| {cell} | {format_worst_pct(r)} | "
                          f"{format_best_pct(r)} | {format_ids_rmse(r)} | {format_combined_gm(r)} | {cat} |")
    else:
        lines.append("No architecture is present in every csv's gm-capped pool even after escalating the ceiling.")
    lines.append("")

    for label, res in (("gmshape_ok", gm_all), ("bothshape_ok", both_all)):
        lines.append(f"## B) `{label}` consistency, goal=all_csvs")
        lines.append("")
        lines.append(f"Ranked by (count of csvs where `{label}`=True) desc, tie-break avg `ids_rmse` asc. "
                      f"The pct columns are informational only (not part of this ranking): this arch's "
                      f"plain `ids_rmse` rank against the full per-csv population (no `combined_gm` gate).")
        lines.append("")
        if res:
            lines.append("| arch | hits | worst-case ids_rmse pct | best-case ids_rmse pct | ids_rmse | combined_gm | category |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in res:
                arch_dir = find_arch_plot_dir(out_root, folder, r["arch_hash"])
                comp = _compliance(arch_dir)
                cat = comp.get("category", "?") if comp else "?"
                cell = arch_link_cell(r["arch_hash"], arch_by_hash, arch_dir, out_dir, r["arch_hash"] in tanh_set)
                lines.append(f"| {cell} | {r['count']}/{r['of']} | "
                              f"{format_worst_pct(r)} | {format_best_pct(r)} | {format_ids_rmse(r)} | "
                              f"{format_combined_gm(r)} | {cat} |")
        else:
            lines.append(f"No architecture passes `{label}` on any csv with shape data.")
        lines.append("")

    gm_all_hashes = {r["arch_hash"] for r in gm_all}
    both_all_hashes = {r["arch_hash"] for r in both_all}
    overlap_all = sorted(gm_all_hashes & both_all_hashes)
    if overlap_all:
        lines.append(f"**Overlap**: {len(overlap_all)} of {len(gm_all_hashes)} `gmshape_ok` picks above also "
                      f"appear in the `bothshape_ok` list: " + ", ".join(f"`{h}`" for h in overlap_all) + ".")
        lines.append("")

    for label, res in (("gmshape_ok", gm_pair), ("bothshape_ok", both_pair)):
        lines.append(f"## C) `{label}` consistency, goal=pair ({pair[0]} AND {pair[1]})")
        lines.append("")
        lines.append(f"Ranked by avg `ids_rmse` across the pair asc. The pct columns are informational only "
                      f"(not part of this ranking): this arch's plain `ids_rmse` rank against the full "
                      f"per-csv population (no `combined_gm` gate).")
        lines.append("")
        if res:
            lines.append("| arch | worst-case ids_rmse pct (pair) | best-case ids_rmse pct (pair) | ids_rmse (pair) | combined_gm (pair) | category |")
            lines.append("|---|---|---|---|---|---|")
            for r in res:
                arch_dir = find_arch_plot_dir(out_root, folder, r["arch_hash"])
                comp = _compliance(arch_dir)
                cat = comp.get("category", "?") if comp else "?"
                cell = arch_link_cell(r["arch_hash"], arch_by_hash, arch_dir, out_dir, r["arch_hash"] in tanh_set)
                lines.append(f"| {cell} | {format_worst_pct(r)} | "
                              f"{format_best_pct(r)} | {format_ids_rmse(r)} | {format_combined_gm(r)} | {cat} |")
        else:
            lines.append(f"No architecture passes `{label}` on BOTH csvs of this pair.")
        lines.append("")

    gm_pair_hashes = {r["arch_hash"] for r in gm_pair}
    both_pair_hashes = {r["arch_hash"] for r in both_pair}
    overlap_pair = sorted(gm_pair_hashes & both_pair_hashes)
    if overlap_pair:
        lines.append(f"**Overlap**: {len(overlap_pair)} of {len(gm_pair_hashes)} `gmshape_ok` pair picks above "
                      f"also appear in the `bothshape_ok` pair list: " +
                      ", ".join(f"`{h}`" for h in overlap_pair) + ".")
        lines.append("")

    lines.append("## D) Fit-quality min-max search, goal=pair (no shape data needed)")
    lines.append("")
    lines.append(f"Worst-case `ids_rmse` percentile across just `{pair[0]}` and `{pair[1]}`, within each csv's "
                  f"own `combined_gm<={used_ceiling_pair:.1f}` pool" +
                  (f" (auto-escalated from {gm_ceiling:.1f} -- no arch qualified at the default ceiling)"
                   if used_ceiling_pair != gm_ceiling else "") +
                  ". Lower worst-case % = more reliably good on BOTH of these two csvs. "
                  f"The `(ids_rmse rank R/N)` shown alongside each pct is informational only -- this arch's "
                  f"plain `ids_rmse` rank against the FULL config population, no `combined_gm` gate.")
    lines.append("")
    if fq_pair:
        lines.append("| arch | worst-case pct (pair) | best-case pct (pair) | ids_rmse (pair) | combined_gm (pair) | category |")
        lines.append("|---|---|---|---|---|---|")
        for r in fq_pair:
            arch_dir = find_arch_plot_dir(out_root, folder, r["arch_hash"])
            comp = _compliance(arch_dir)
            cat = comp.get("category", "?") if comp else "?"
            cell = arch_link_cell(r["arch_hash"], arch_by_hash, arch_dir, out_dir, r["arch_hash"] in tanh_set)
            lines.append(f"| {cell} | {format_worst_pct(r)} | "
                          f"{format_best_pct(r)} | {format_ids_rmse(r)} | {format_combined_gm(r)} | {cat} |")
    else:
        lines.append("No architecture is present in both pair csvs' gm-capped pool even after escalating the ceiling.")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    lines.append(f"6-panel IV/GM plots (+ eq-comparison, `.md` equation writeup, `.va` Verilog-A export) for "
                  f"every architecture above, one set per measurement csv, live under "
                  f"`{out_root.name}/{folder}/<category>/<goal>/<arch_hash>/`, generated via "
                  f"`physics_nn_pipeline/plot_arch_hash.py --goal ... --format md` (writes "
                  f"`all_csvs.md`, embedding just the `plot_saved_state_full.png` plots + the "
                  f"same shape-compliance/config summary as above -- pass `--format pdf` or "
                  f"`both` for the older `all_csvs.pdf` instead/as well). `<goal>` is "
                  f"`best_fit_quality` for group A picks or `best_consistent_with_all_csvs` / "
                  f"`best_consistent_with_2.5_20_and_2.9_28` for group B/C picks.")
    lines.append("")

    out_path = Path(args.out) if args.out else out_dir / f"consistency_summary_{root.name}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")

    # Companion machine-readable dump (same numbers as the tables above) -- lets
    # write_overall_findings.py cross-reference picks across several roots/base_configs for the
    # same folder without re-parsing markdown.
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "root": str(root), "base_config": root.name, "folder": folder,
        "pair_csvs": list(pair), "n_csvs": n_csvs, "n_shape": n_shape,
        "gm_ceiling": gm_ceiling, "gm_ceiling_used": used_ceiling, "gm_ceiling_used_pair": used_ceiling_pair,
        "fit_quality_minmax": fq, "fit_quality_minmax_pair": fq_pair,
        "gmshape_ok_all_csvs": gm_all, "bothshape_ok_all_csvs": both_all,
        "gmshape_ok_pair": gm_pair, "bothshape_ok_pair": both_pair,
    }, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
