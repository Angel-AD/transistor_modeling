"""
write_overall_findings.py -- combine several write_findings.py runs for the SAME folder (one per
--root/base_config, e.g. best200ids_of_9069_byloss_rw0_2.5_20, bothshapeok_of_9069_byshape_
rw0_2.5_20, best100_gmshapeok_of_9069_byloss_rw0_2.5_20) into one consistency_summary_overall.md,
written alongside them in <out_root>/<folder>/.

Reads each base_config's consistency_summary_<base_config>.json (written by write_findings.py --
NOT the .md, which is human prose; the json has the same numbers in machine-readable form) and
cross-references which arch_hashes turn up as a top pick under MORE THAN ONE base_config --
independent confirmation from separately-selected architecture pools is a stronger signal than
any single base_config's own ranking. Architectures are compared by arch_hash only (content-
stable, same physical model regardless of which base_config's search found it), so a hash that's
a top-5 pick under both best200 and best100_gmshapeok shows up once in the cross-reference table,
annotated with both base_configs and which method(s) (A/B/C/D) picked it in each.

Run AFTER write_findings.py has produced a consistency_summary_<base_config>.json for every
base_config you want included.

Usage:
    python write_overall_findings.py --out_root ../runs/csv_base_2.5_20/best_archs_plots \\
        --folder tanh_margin10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from find_consistent_archs import load_arch_by_hash, find_arch_plot_dir, \
    arch_link_cell, is_tanh_only  # noqa: E402

_arch_cache: dict[tuple[str, str], dict[str, str]] = {}
_tanh_only_cache: dict[tuple[str, str], set[str]] = {}


def _arch_cell(root: str | None, out_root: Path, folder: str, arch_hash: str,
               arch_dir: Path | None, from_dir: Path, mark_tanh_only: bool) -> str:
    if not root:
        return f"`{arch_hash}`"
    key = (root, folder)
    if key not in _arch_cache:
        _arch_cache[key] = load_arch_by_hash(Path(root), folder)
    tanh = mark_tanh_only and is_tanh_only(Path(root), folder, arch_hash, _tanh_only_cache)
    return arch_link_cell(arch_hash, _arch_cache[key], arch_dir, from_dir, tanh)

# (json key, markdown label, "pair" if goal=pair else "all")
_METHOD_SECTIONS = [
    ("fit_quality_minmax", "A) fit-quality min-max", "all"),
    ("gmshape_ok_all_csvs", "B) gmshape_ok, goal=all_csvs", "all"),
    ("bothshape_ok_all_csvs", "B) bothshape_ok, goal=all_csvs", "all"),
    ("gmshape_ok_pair", "C) gmshape_ok, goal=pair", "pair"),
    ("bothshape_ok_pair", "C) bothshape_ok, goal=pair", "pair"),
    ("fit_quality_minmax_pair", "D) fit-quality min-max, goal=pair", "pair"),
]


def _load_summaries(search_dir: Path) -> list[dict]:
    docs = []
    for jp in sorted(search_dir.glob("consistency_summary_*.json")):
        try:
            docs.append(json.loads(jp.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[warn] could not read {jp}: {e}")
    return docs


def _compliance(arch_dir: Path | None) -> dict | None:
    if arch_dir is None:
        return None
    try:
        return json.loads((arch_dir / "compliance.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def build_overall(out_root: Path, folder: str, top: int, subdir: str | None = None) -> Path | None:
    search_dir = (out_root / folder / subdir) if subdir else (out_root / folder)
    docs = _load_summaries(search_dir)
    if not docs:
        print(f"[skip] no consistency_summary_*.json found under {search_dir}")
        return None

    bc_to_root = {doc.get("base_config", "?"): doc.get("root") for doc in docs}

    # hash -> base_config -> [(section_label, rank_in_that_section)]
    hits: dict[str, dict[str, list[tuple[str, int]]]] = defaultdict(lambda: defaultdict(list))
    for doc in docs:
        base_config = doc.get("base_config", "?")
        for json_key, label, _scope in _METHOD_SECTIONS:
            for i, r in enumerate(doc.get(json_key) or [], start=1):
                h = r.get("arch_hash")
                if h:
                    hits[h][base_config].append((label, i))

    # Only architectures confirmed by MORE THAN ONE base_config are the actual point of this
    # file (a single base_config's own picks are already in its own consistency_summary_*.md).
    cross = {h: by_bc for h, by_bc in hits.items() if len(by_bc) > 1}
    # Rank by (number of distinct base_configs it appears under) desc, then (total number of
    # method-section hits across all of them) desc, then best (lowest) rank seen anywhere.
    def _sort_key(item):
        h, by_bc = item
        n_bc = len(by_bc)
        n_hits = sum(len(v) for v in by_bc.values())
        best_rank = min(r for entries in by_bc.values() for _, r in entries)
        return (-n_bc, -n_hits, best_rank)
    ranked = sorted(cross.items(), key=_sort_key)[:top]

    lines = []
    lines.append(f"# `{folder}` overall consistency summary")
    lines.append("")
    lines.append(f"Cross-references {len(docs)} base_config(s): " +
                  ", ".join(f"`{d.get('base_config', '?')}`" for d in docs) + ".")
    lines.append("")
    lines.append("Each base_config's own full picks are in its own "
                  "`consistency_summary_<base_config>.md` (same directory as this file). This "
                  "file only lists architectures picked by **more than one** base_config's "
                  "search -- independent confirmation from separately-selected architecture "
                  "pools, a stronger signal than any single base_config's own ranking.")
    lines.append("")

    lines.append("## Architectures confirmed by multiple base_configs")
    lines.append("")
    if ranked:
        lines.append("| arch | base_configs | hits | category | detail |")
        lines.append("|---|---|---|---|---|")
        for h, by_bc in ranked:
            arch_dir = find_arch_plot_dir(out_root, folder, h)
            comp = _compliance(arch_dir)
            cat = comp.get("category", "?") if comp else "?"
            n_bc = len(by_bc)
            n_hits = sum(len(v) for v in by_bc.values())
            root = next((bc_to_root[bc] for bc in by_bc if bc_to_root.get(bc)), None)
            arch_cell = _arch_cell(root, out_root, folder, h, arch_dir, search_dir, subdir != "tanh_only")
            detail_parts = []
            for bc in sorted(by_bc):
                section_bits = ", ".join(f"{label}#{rank}" for label, rank in by_bc[bc])
                detail_parts.append(f"**{bc}**: {section_bits}")
            lines.append(f"| {arch_cell} | {n_bc} | {n_hits} | {cat} | " + "; ".join(detail_parts) + " |")
    else:
        lines.append("No architecture was picked by more than one base_config's search "
                      "(every base_config's top picks were disjoint).")
    lines.append("")

    lines.append("## Per base_config summaries")
    lines.append("")
    for doc in docs:
        bc = doc.get("base_config", "?")
        lines.append(f"- [`{bc}`](consistency_summary_{bc}.md) -- "
                      f"{doc.get('n_csvs', '?')} csvs, "
                      f"{doc.get('n_shape', '?')} with full-population shape data")
    lines.append("")

    out_path = search_dir / "consistency_summary_overall.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", required=True,
                    help="Shared best_archs_plots root containing <folder>/consistency_summary_"
                         "<base_config>.json for each base_config (same --out_root passed to "
                         "write_findings.py for each of them).")
    ap.add_argument("--folder", required=True,
                    help="Equation-type folder name(s), comma-separated for several at once "
                         "(e.g. tanh_margin10,softplus).")
    ap.add_argument("--top", type=int, default=20,
                    help="Max rows in the cross-reference table (default: 20).")
    ap.add_argument("--subdir", default=None,
                    help="Look for consistency_summary_<base_config>.json under <out_root>/"
                         "<folder>/<subdir>/ instead of <out_root>/<folder>/, and write "
                         "consistency_summary_overall.md there too (e.g. --subdir tanh_only). "
                         "compliance.json lookups still use the SHARED (non-subdir) plots tree.")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if not out_root.is_dir():
        raise SystemExit(f"--out_root not found: {out_root}")

    folders = [f.strip() for f in args.folder.split(",") if f.strip()]
    for folder in folders:
        build_overall(out_root, folder, args.top, subdir=args.subdir)


if __name__ == "__main__":
    main()
