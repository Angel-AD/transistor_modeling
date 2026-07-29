"""
make_arch_pdf.py -- combine an arch_hash folder's per-csv plot_saved_state_full.png images
into a single multi-page PDF (--format pdf, the default) and/or a single markdown file
(--format md) with one section per measurement csv (each labeled/headed with its csv name so
pages/sections are identifiable at a glance), plus the architecture's config (parsed from one of
the companion "<csv>_plot_saved_state_full.md" equation writeups already sitting in the same
folder) -- so they can be flipped/scrolled through without opening 6+ separate files or
cross-referencing which plot belongs to which measurement csv.

Companion to plot_arch_hash.py, which produces the source PNGs/.md files (named
"<csv_name>_plot_saved_state_full.*") in best_archs_plots/<folder>/<arch_hash>/. This script
(and plot_arch_hash.py itself, via --format) just assembles what's already there -- it does no
plotting of its own.

Usage
-----
    # one arch_hash directory, PDF (default)
    python make_arch_pdf.py --dir ../runs/best200ids_of_9069_byloss/best_archs_plots/tanh_margin10/93fd5b66

    # same, but markdown instead
    python make_arch_pdf.py --dir ... --format md

    # both, in one pass
    python make_arch_pdf.py --dir ... --format both

    # every arch_hash directory under a folder
    python make_arch_pdf.py --root ../runs/best200ids_of_9069_byloss/best_archs_plots --folder tanh_margin10

    # every arch_hash directory under every folder (the whole best_archs_plots tree)
    python make_arch_pdf.py --root ../runs/best200ids_of_9069_byloss/best_archs_plots

    # only the eq-comparison variant instead of the main plot
    python make_arch_pdf.py --dir ... --pattern "*_plot_saved_state_full_eq_comparison.png"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_PATTERN = "*_plot_saved_state_full.png"
# csv_name is everything before "_plot_saved_state_full...": strip that suffix for the title
# bar drawn on each page and the label printed in progress output.
_SUFFIX_MARKER = "_plot_saved_state_full"

_TITLE_BAR_H = 70
_TITLE_FONT_SIZE = 30
_SUMMARY_FONT_SIZE = 22
_SUMMARY_MONO_SIZE = 20
_MARGIN = 60

# Windows ships these; fall back through the list, then to PIL's bitmap default (tiny but
# always available) so this never hard-fails on a machine without these exact fonts.
_FONT_CANDIDATES = ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
_MONO_FONT_CANDIDATES = ("consola.ttf", "Consolas.ttf", "DejaVuSansMono.ttf", "cour.ttf")


def _load_font(candidates: tuple[str, ...], size: int) -> ImageFont.ImageFont:
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _csv_label(png_path: Path) -> str:
    name = png_path.stem
    idx = name.find(_SUFFIX_MARKER)
    return name[:idx] if idx != -1 else name


def _with_title_bar(img: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    """New image with a white title bar of height _TITLE_BAR_H added above img, the csv name
    centered in it."""
    out = Image.new("RGB", (img.width, img.height + _TITLE_BAR_H), "white")
    draw = ImageDraw.Draw(out)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, (_TITLE_BAR_H - (bbox[3] - bbox[1])) // 2 - bbox[1]),
              title, fill="black", font=font)
    out.paste(img, (0, _TITLE_BAR_H))
    return out


# Matches "- key: value" lines under a markdown "## Config" section.
_CONFIG_LINE_RE = re.compile(r"^-\s*([^:]+):\s*(.+)$")


def _parse_config(md_path: Path) -> list[tuple[str, str]]:
    """[(key, value)] parsed from the "## Config" section of a plot_saved_state_full.md
    equation writeup (see plot_saved_state.py's generate_and_write_equation_md). Stops at the
    next "##" header. Returns [] if the file/section isn't found -- callers should handle that
    (e.g. by omitting the summary page) rather than treat it as fatal, since this is reused for
    older PDFs that may have a slightly different writeup format."""
    if not md_path.is_file():
        return []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "## Config") + 1
    except StopIteration:
        return []
    out = []
    for line in lines[start:]:
        if line.strip().startswith("##"):
            break
        m = _CONFIG_LINE_RE.match(line.strip())
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def _wrap(text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, max_w: int) -> list[str]:
    """Greedy word-wrap text to fit max_w pixels at the given font, splitting only on spaces
    (architecture strings/long values may still overflow a line -- acceptable for a summary
    page, not worth a char-level wrap for this)."""
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


_CATEGORY_LABELS = {
    "both": ("BOTH (gmshape + gdsshape compliant)", "#0a7a0a"),
    "gmshape": ("GMSHAPE ONLY (never bothshape compliant)", "#8a6d00"),
    "none": ("NOT COMPLIANT (shape was checked, but fails gmshape/both everywhere)", "#a30000"),
    "filter_only": ("FILTER ONLY -- no shape analysis available for this architecture", "#555555"),
}


def _load_compliance(arch_dir: Path) -> dict | None:
    """compliance.json written by plot_arch_hash.py (arch_hash/category/per_csv), or None if
    absent (PDFs built before that plumbing existed, or built standalone without it)."""
    p = arch_dir / "compliance.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _draw_compliance_section(draw: ImageDraw.ImageDraw, x: int, y: int, w: int,
                             compliance: dict, label_font, mono_font) -> int:
    """Draws the category line + a per-csv gmshape/gdsshape/bothshape table starting at (x, y).
    Returns the y coordinate just below what was drawn."""
    category = compliance.get("category", "?")
    label, color = _CATEGORY_LABELS.get(category, (category, "black"))
    draw.text((x, y), f"Shape compliance: {label}", fill=color, font=label_font)
    y += _SUMMARY_FONT_SIZE + 20

    goal = compliance.get("goal")
    if goal:
        draw.text((x, y), f"Search goal: {goal}", fill="#222222", font=label_font)
        y += _SUMMARY_FONT_SIZE + 20

    per_csv = compliance.get("per_csv") or {}
    line_h = _SUMMARY_MONO_SIZE + 8
    for csv_name in sorted(per_csv):
        c = per_csv[csv_name]
        if c is None:
            status = "not checked (no shape analysis)"
        else:
            parts = [k.replace("_ok", "") for k in ("gmshape_ok", "gdsshape_ok", "bothshape_ok") if c.get(k)]
            status = " + ".join(parts) if parts else "fails all checks"
        draw.text((x, y), f"{csv_name}: {status}", fill="#222222", font=mono_font)
        y += line_h
    return y + 20


def _build_summary_page(arch_dir: Path, page_size: tuple[int, int]) -> Image.Image | None:
    """Final PDF page: arch_hash, shape-compliance category + per-csv breakdown (from
    compliance.json, if present), and the config block parsed from the first available
    companion .md file (same for every csv -- it's one architecture, the .md just repeats the
    config per csv alongside that csv's own normalization). None only if NEITHER compliance.json
    NOR a parseable .md Config section is available (e.g. eq-comparison-only PDFs, or a
    directory that predates both conventions)."""
    compliance = _load_compliance(arch_dir)

    md_files = sorted(arch_dir.glob("*_plot_saved_state_full.md"))
    config = None
    for md in md_files:
        config = _parse_config(md)
        if config:
            break

    if not compliance and not config:
        return None

    w, h = page_size
    page = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(page)
    title_font = _load_font(_FONT_CANDIDATES, _TITLE_FONT_SIZE)
    label_font = _load_font(_FONT_CANDIDATES, _SUMMARY_FONT_SIZE)
    mono_font = _load_font(_MONO_FONT_CANDIDATES, _SUMMARY_MONO_SIZE)

    y = _MARGIN
    draw.text((_MARGIN, y), f"arch_hash: {arch_dir.name}", fill="black", font=title_font)
    y += _TITLE_FONT_SIZE + 30

    if compliance:
        y = _draw_compliance_section(draw, _MARGIN, y, w, compliance, label_font, mono_font)

    if not config:
        return page

    max_w = w - 2 * _MARGIN
    line_h = _SUMMARY_MONO_SIZE + 10
    for key, value in config:
        label = f"{key}:"
        draw.text((_MARGIN, y), label, fill="black", font=label_font)
        # architecture / long values wrap in mono under the label; short values sit inline
        inline_x = _MARGIN + draw.textbbox((0, 0), label, font=label_font)[2] + 15
        if draw.textbbox((0, 0), value, font=mono_font)[2] <= (w - inline_x - _MARGIN):
            draw.text((inline_x, y + 2), value, fill="#222222", font=mono_font)
            y += line_h + 6
        else:
            y += line_h
            for wrapped in _wrap(value, mono_font, draw, max_w - 20):
                draw.text((_MARGIN + 20, y), wrapped, fill="#222222", font=mono_font)
                y += line_h
            y += 6
        if y > h - _MARGIN - line_h:
            break  # ran out of page; rest of the config is still in the .md files themselves
    return page


def build_pdf(arch_dir: Path, pattern: str = DEFAULT_PATTERN, out_name: str = "all_csvs.pdf") -> Path | None:
    """One titled page per matching PNG in arch_dir (sorted by filename == by csv name, since
    plot_arch_hash.py prefixes every file with its csv name), plus a final config-summary page
    if a companion .md file is present. Returns the written path, or None if no matching PNGs
    were found."""
    pngs = sorted(arch_dir.glob(pattern))
    if not pngs:
        return None

    title_font = _load_font(_FONT_CANDIDATES, _TITLE_FONT_SIZE)
    images = []
    for p in pngs:
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(_with_title_bar(img, _csv_label(p), title_font))
        print(f"  page: {_csv_label(p)}  ({p.name})")

    summary = _build_summary_page(arch_dir, images[0].size)
    if summary is not None:
        images.append(summary)
        print("  page: [summary] arch config")

    out_path = arch_dir / out_name
    images[0].save(out_path, save_all=True, append_images=images[1:])
    print(f"wrote {out_path}  ({len(images)} pages)")
    return out_path


def build_md(arch_dir: Path, pattern: str = DEFAULT_PATTERN, out_name: str = "all_csvs.md") -> Path | None:
    """One markdown file with the same content as build_pdf's PDF -- shape-compliance category +
    per-csv table, the architecture/config block (both from the same compliance.json /
    plot_saved_state_full.md sources _build_summary_page reads), then one section per matching
    PNG in arch_dir (sorted by filename == by csv name) with the plot embedded via a relative
    image link (the PNGs live alongside the .md file, so just the filename). Only
    plot_saved_state_full.png images -- DEFAULT_PATTERN never matches the _eq_comparison/_val
    variants. Returns the written path, or None if no matching PNGs were found."""
    pngs = sorted(arch_dir.glob(pattern))
    if not pngs:
        return None

    lines = [f"# {arch_dir.name}", ""]

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
        lines.append("## Architecture / config")
        lines.append("")
        for key, value in config:
            lines.append(f"- **{key}:** `{value}`")
        lines.append("")

    lines.append("## Plots")
    lines.append("")
    for p in pngs:
        csv_label = _csv_label(p)
        lines.append(f"### {csv_label}")
        lines.append("")
        lines.append(f"![{csv_label}]({p.name})")
        lines.append("")
        print(f"  section: {csv_label}  ({p.name})")

    out_path = arch_dir / out_name
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}  ({len(pngs)} plots)")
    return out_path


def _find_arch_dirs(root: Path, folder: str | None) -> list[Path]:
    """Every arch_hash directory (a directory containing at least one file matching
    DEFAULT_PATTERN) under root, optionally restricted to one --folder subdirectory."""
    search_roots = [root / folder] if folder else [p for p in root.iterdir() if p.is_dir()]
    arch_dirs = []
    for folder_dir in search_roots:
        if not folder_dir.is_dir():
            continue
        for arch_dir in sorted(folder_dir.iterdir()):
            if arch_dir.is_dir() and list(arch_dir.glob(DEFAULT_PATTERN)):
                arch_dirs.append(arch_dir)
    return arch_dirs


def main():
    ap = argparse.ArgumentParser(
        description="Combine an arch_hash folder's per-csv plot_saved_state_full.png images "
                    "into a titled multi-page PDF and/or a markdown file (--format), each with "
                    "the arch's shape-compliance/config summary.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", help="A single arch_hash directory to build a PDF for.")
    src.add_argument("--root", help="best_archs_plots root -- builds a PDF for every arch_hash "
                                    "directory found (optionally restricted by --folder).")
    ap.add_argument("--folder", default=None,
                    help="With --root: only process this one equation-type subfolder "
                         "(e.g. tanh_margin10) instead of every folder under --root.")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help=f"Glob pattern for source PNGs within each arch_hash dir "
                         f"(default: {DEFAULT_PATTERN!r}).")
    ap.add_argument("--out_name", default=None,
                    help="Output filename, written inside each arch_hash dir. Default: "
                         "all_csvs.pdf / all_csvs.md (matching --format; with --format both, "
                         "an explicit --out_name would collide across formats, so it's ignored "
                         "and the per-format default is always used).")
    ap.add_argument("--format", choices=("pdf", "md", "both"), default="pdf",
                    help="Output format: 'pdf' (default, unchanged legacy behavior), 'md' "
                         "(markdown with embedded plot_saved_state_full.png images -- no PDF "
                         "assembly/PIL page-compositing needed), or 'both'.")
    args = ap.parse_args()

    if args.dir:
        arch_dirs = [Path(args.dir).resolve()]
    else:
        root = Path(args.root).resolve()
        if not root.is_dir():
            raise SystemExit(f"--root not found: {root}")
        arch_dirs = _find_arch_dirs(root, args.folder)
        if not arch_dirs:
            raise SystemExit(f"no arch_hash directories with files matching {args.pattern!r} "
                             f"found under {root}" + (f"/{args.folder}" if args.folder else ""))

    want_pdf = args.format in ("pdf", "both")
    want_md = args.format in ("md", "both")
    single_format = args.format != "both"
    pdf_name = args.out_name if (args.out_name and single_format and want_pdf) else "all_csvs.pdf"
    md_name = args.out_name if (args.out_name and single_format and want_md) else "all_csvs.md"

    ok = fail = 0
    failed_dirs = []
    for arch_dir in arch_dirs:
        print(f"=== {arch_dir} ===")
        try:
            pdf_result = build_pdf(arch_dir, pattern=args.pattern, out_name=pdf_name) if want_pdf else True
            md_result = build_md(arch_dir, pattern=args.pattern, out_name=md_name) if want_md else True
        except OSError as e:
            # Most commonly a locked/open-elsewhere output file (PDF viewer, cloud sync,
            # antivirus scan) -- don't let one locked file abort the whole batch.
            print(f"  FAILED: {e}")
            fail += 1
            failed_dirs.append(arch_dir)
            continue
        if pdf_result is None or md_result is None:
            print(f"  SKIP: no files matching {args.pattern!r}")
            fail += 1
            failed_dirs.append(arch_dir)
        else:
            ok += 1

    print(f"\ndone. ok={ok} fail={fail}")
    if failed_dirs:
        print("failed/skipped:")
        for d in failed_dirs:
            print(f"  {d}")


if __name__ == "__main__":
    main()
