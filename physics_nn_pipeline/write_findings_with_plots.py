"""
write_findings_with_plots.py -- companion to write_findings.py's consistency_summary_
<base_config>.md: ONE file per summary, same section structure (A / B gmshape_ok / B
bothshape_ok / C gmshape_ok / C bothshape_ok / D), but with each picked architecture's ACTUAL
plot_saved_state_full.png images (one per measurement csv, from plot_arch_hash.py's output
already on disk) embedded inline, plus the same identifying info shown in that architecture's
own all_csvs.md (architecture/config block, shape-compliance category, per-csv breakdown) --
so the "best" picks can be scrolled through and visually inspected in one place, without
opening 10-20+ separate all_csvs.md files one at a time.

Run AFTER write_findings.py (needs consistency_summary_<base_config>.json) AND
plot_arch_hash.py --format md (needs plot_saved_state_full.png + compliance.json + the
equation-writeup .md files already on disk for every picked arch_hash -- i.e. after
run_plot_consistent_archs.py's normal search-then-plot sequence).

The SAME arch_hash often recurs across sections (an architecture can be the fit-quality winner
AND a shape-consistency winner) -- its plots are only embedded once, the first time it's
encountered in the file; later occurrences get a short cross-reference note instead of
duplicating the images.

Usage:
    python write_findings_with_plots.py --out_root ../runs/csv_base_2.5_20/best_archs_plots \\
        --folder tanh_margin10
    python write_findings_with_plots.py --out_root ... --folder tanh_margin10,softplus \\
        --base_configs best200ids_of_9069_byloss_rw0_2.5_20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_arch_pdf import DEFAULT_PATTERN, _parse_config, _CATEGORY_LABELS, _load_compliance  # noqa: E402
from run_plot_consistent_archs import _METHOD_GOALS, _short  # noqa: E402
from find_consistent_archs import DEFAULT_PAIR, format_ids_rmse, format_combined_gm, \
    format_worst_pct, format_best_pct  # noqa: E402

# (json key, section heading, table-row formatter(rank, r) -> info line)
_SECTIONS = [
    ("fit_quality_minmax", "A) Fit-quality min-max search",
     lambda rank, r: f"worst-case pct **{format_worst_pct(r)}**, best-case pct **{format_best_pct(r)}**, "
                     f"ids_rmse {format_ids_rmse(r)}, combined_gm {format_combined_gm(r)}"),
    ("gmshape_ok_all_csvs", "B) `gmshape_ok` consistency, goal=all_csvs",
     lambda rank, r: f"hits **{r['count']}/{r['of']}**, worst-case pct {format_worst_pct(r)}, "
                     f"best-case pct {format_best_pct(r)}, ids_rmse {format_ids_rmse(r)}, "
                     f"combined_gm {format_combined_gm(r)}"),
    ("bothshape_ok_all_csvs", "B) `bothshape_ok` consistency, goal=all_csvs",
     lambda rank, r: f"hits **{r['count']}/{r['of']}**, worst-case pct {format_worst_pct(r)}, "
                     f"best-case pct {format_best_pct(r)}, ids_rmse {format_ids_rmse(r)}, "
                     f"combined_gm {format_combined_gm(r)}"),
    ("gmshape_ok_pair", "C) `gmshape_ok` consistency, goal=pair",
     lambda rank, r: f"worst-case pct (pair) {format_worst_pct(r)}, best-case pct (pair) {format_best_pct(r)}, "
                     f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}"),
    ("bothshape_ok_pair", "C) `bothshape_ok` consistency, goal=pair",
     lambda rank, r: f"worst-case pct (pair) {format_worst_pct(r)}, best-case pct (pair) {format_best_pct(r)}, "
                     f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}"),
    ("fit_quality_minmax_pair", "D) Fit-quality min-max search, goal=pair",
     lambda rank, r: f"worst-case pct (pair) **{format_worst_pct(r)}**, best-case pct (pair) **{format_best_pct(r)}**, "
                     f"ids_rmse (pair) {format_ids_rmse(r)}, combined_gm (pair) {format_combined_gm(r)}"),
]


def _find_arch_dir(out_root: Path, folder: str, goal: str, arch_hash: str) -> Path | None:
    matches = list((out_root / folder).glob(f"*/{goal}/{arch_hash}"))
    return matches[0] if matches else None


def _arch_block(arch_dir: Path, heading_level: str = "###") -> list[str]:
    """Full embedded block for one arch: category + per-csv compliance table + config, then
    one image per csv. Same content sources as make_arch_pdf.py's build_md/summary page."""
    lines = []
    compliance = _load_compliance(arch_dir)
    if compliance:
        category = compliance.get("category", "?")
        label, _ = _CATEGORY_LABELS.get(category, (category, None))
        lines.append(f"**Shape compliance:** {label}")
        goal = compliance.get("goal")
        if goal:
            lines.append(f"**Search goal:** {goal}")
        lines.append("")
        per_csv = compliance.get("per_csv") or {}
        if per_csv:
            lines.append("| csv | gmshape_ok | gdsshape_ok | bothshape_ok |")
            lines.append("|---|---|---|---|")
            mark = lambda b: "yes" if b else ("no" if b is not None else "-")
            for csv_name in sorted(per_csv):
                c = per_csv[csv_name]
                if c is None:
                    lines.append(f"| {csv_name} | not checked | not checked | not checked |")
                else:
                    lines.append(f"| {csv_name} | {mark(c.get('gmshape_ok'))} | "
                                  f"{mark(c.get('gdsshape_ok'))} | {mark(c.get('bothshape_ok'))} |")
            lines.append("")

    md_files = sorted(arch_dir.glob("*_plot_saved_state_full.md"))
    config = None
    for md in md_files:
        config = _parse_config(md)
        if config:
            break
    if config:
        lines.append("**Architecture / config:**")
        lines.append("")
        for key, value in config:
            lines.append(f"- **{key}:** `{value}`")
        lines.append("")

    pngs = sorted(arch_dir.glob(DEFAULT_PATTERN))
    for p in pngs:
        csv_label = p.stem.replace("_plot_saved_state_full", "")
        lines.append(f"{heading_level}# {csv_label}")
        lines.append("")
        # relative path from the OUTPUT file (out_root/<folder>/) down into
        # <category>/<goal>/<arch_hash>/<png>
        rel = Path(arch_dir.parts[-3]) / arch_dir.parts[-2] / arch_dir.parts[-1] / p.name
        lines.append(f"![{csv_label}]({rel.as_posix()})")
        lines.append("")
    return lines


def build_for_base_config(out_root: Path, folder: str, base_config: str, pair: tuple[str, str]) -> Path | None:
    json_path = out_root / folder / f"consistency_summary_{base_config}.json"
    if not json_path.is_file():
        print(f"SKIP {folder}/{base_config}: {json_path} not found (run write_findings.py first)")
        return None
    doc = json.loads(json_path.read_text(encoding="utf-8"))

    lines = [f"# `{folder}` consistency plots ({base_config})", "",
             f"Plots for every architecture in "
             f"[`consistency_summary_{base_config}.md`](consistency_summary_{base_config}.md) "
             f"(same sections, same rank order). An arch_hash's plots are embedded once, the "
             f"first time it's encountered below -- later occurrences just note where to find "
             f"them.", ""]

    seen: dict[str, str] = {}  # arch_hash -> section heading where first embedded
    n_embedded = n_refs = n_missing = 0

    for json_key, heading, info_fn in _SECTIONS:
        rows = doc.get(json_key) or []
        lines.append(f"## {heading}")
        lines.append("")
        if not rows:
            lines.append("(no picks in this section)")
            lines.append("")
            continue
        goal = dict(_METHOD_GOALS)[json_key](pair)
        for rank, r in enumerate(rows, start=1):
            arch_hash = r["arch_hash"]
            lines.append(f"### #{rank} — `{arch_hash}`")
            lines.append("")
            lines.append(info_fn(rank, r))
            lines.append("")
            if arch_hash in seen:
                lines.append(f"_(plots already shown above, under **{seen[arch_hash]}**)_")
                lines.append("")
                n_refs += 1
                continue
            arch_dir = _find_arch_dir(out_root, folder, goal, arch_hash)
            if arch_dir is None:
                lines.append(f"_(plot directory not found -- expected under `{folder}/*/{goal}/{arch_hash}/`)_")
                lines.append("")
                n_missing += 1
                continue
            lines.extend(_arch_block(arch_dir))
            seen[arch_hash] = heading
            n_embedded += 1

    out_path = out_root / folder / f"consistency_summary_{base_config}_plots.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}  (embedded={n_embedded} cross-ref={n_refs} missing={n_missing})")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True,
                    help="Shared best_archs_plots root (same one passed to write_findings.py "
                         "--out_root / plot_arch_hash.py --out_dir).")
    ap.add_argument("--folder", required=True, help="comma-separated for several at once")
    ap.add_argument("--base_configs", default=None,
                    help="comma-separated; default: every consistency_summary_*.json found "
                         "under <out_root>/<folder>/")
    ap.add_argument("--pair_csvs", default=",".join(DEFAULT_PAIR))
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out_root not found: {out_root}")
    pair = tuple(c.strip() for c in args.pair_csvs.split(","))
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
        for base_config in base_configs:
            build_for_base_config(out_root, folder, base_config, pair)


if __name__ == "__main__":
    main()
