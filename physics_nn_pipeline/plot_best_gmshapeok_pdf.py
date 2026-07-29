"""
plot_best_gmshapeok_pdf.py -- one-command version of "pick the best N architectures per folder
(same Approach 1 gate-then-rank selection as extract_derived_configs.py's best100_gmshapeok /
bothshapeok derivations), plot each one, and assemble every plot_saved_state_full.png into a
single labeled PDF" -- previously done by hand as a shell loop over plot_csv_row.py.

Two selection modes (--mode), both imported directly from extract_derived_configs.py so results
always match that script's derivations for the same --source_root/--gm_ceiling_start:

  gmshapeok (default): gmshape_ok rows, gate-then-ranked to a --pool_n-sized pool (default 100,
  matching the actual best100_gmshapeok_of_9069_byloss_<suffix> config), then top --top_n of
  that pool by ids_rmse.

  bothshapeok: bothshape_ok rows (the stricter check -- gmshape_ok AND gds_residual_bad_frac==0).
  --pool_n defaults to None here (uncapped), matching the actual bothshapeok_of_9069_byshape_
  <suffix> config, which by default keeps EVERY bothshape_ok row with no combined_gm gate --
  the full pool is just ranked by ids_rmse and the top --top_n taken. Pass --pool_n to instead
  gate-then-rank like gmshapeok (matches extract_derived_configs.py's --bothshapeok_top_n).

Each PDF page is one architecture's plot_saved_state_full.png with a caption bar above it
(folder, rank within selection, id/arch_id/arch_hash, config_name, ids_rmse/combined_gm --
both the raw run values and the region_knee_* values selection was actually based on -- the
measurement csv, and the selection gate used) so pages are identifiable without cross-
referencing anything else. No separate info-only pages -- just the plots, captioned.

Usage:
    # plots only, no PDF (default) -- best gmshapeok archs
    python plot_best_gmshapeok_pdf.py --source_root ../runs/pure_combined9069_rw0_2.5_20

    # best bothshapeok archs, plots + captioned PDF
    python plot_best_gmshapeok_pdf.py --source_root ../runs/pure_combined9069_rw0_2.5_20 \\
        --mode bothshapeok --pdf

    python plot_best_gmshapeok_pdf.py --source_root ../runs/pure_combined9069_rw0_2.5_20 --top_n 5 --pdf
    python plot_best_gmshapeok_pdf.py --source_root ../runs/pure_combined9069_rw0_2.5_20 \\
        --folders tanh_margin10,softplus --dry_run

Output (only with --pdf): <source_root>/out_pdfs/best<top_n>_<mode>.pdf
(override with --out_pdf_dir/--out_pdf_name).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plot_csv_row import _read_rows  # noqa: E402
from extract_derived_configs import (  # noqa: E402
    FOLDERS, gmshape_ok, bothshape_ok, _f, _base_ranked_csv, _ensure_shape_csv, _escalate_and_select,
)

_MODE_FILTERS = {"gmshapeok": gmshape_ok, "bothshapeok": bothshape_ok}

PLOT_CSV_ROW = HERE / "plot_csv_row.py"

_TITLE_FONT_CANDIDATES = ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf")
_MONO_FONT_CANDIDATES = ("consola.ttf", "Consolas.ttf", "DejaVuSansMono.ttf", "cour.ttf")
_TITLE_SIZE = 26
_LABEL_SIZE = 20
_MONO_SIZE = 18
_MARGIN = 20
_LINE_GAP = 6


def _load_font(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.ImageFont, draw: ImageDraw.ImageDraw, max_w: int) -> list[str]:
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


def _caption_page(img: Image.Image, title: str, info_lines: list[str], arch_text: str) -> Image.Image:
    """New image: a white caption bar (title + info lines + wrapped arch string) stacked above
    img."""
    title_font = _load_font(_TITLE_FONT_CANDIDATES, _TITLE_SIZE)
    label_font = _load_font(_TITLE_FONT_CANDIDATES, _LABEL_SIZE)
    mono_font = _load_font(_MONO_FONT_CANDIDATES, _MONO_SIZE)

    probe = ImageDraw.Draw(img)
    max_w = img.width - 2 * _MARGIN
    arch_lines = _wrap(f"architecture: {arch_text}", mono_font, probe, max_w) if arch_text else []

    line_h = _LABEL_SIZE + _LINE_GAP
    mono_line_h = _MONO_SIZE + _LINE_GAP
    bar_h = (_MARGIN + (_TITLE_SIZE + _LINE_GAP) + len(info_lines) * line_h
             + len(arch_lines) * mono_line_h + _MARGIN)

    out = Image.new("RGB", (img.width, img.height + bar_h), "white")
    draw = ImageDraw.Draw(out)
    y = _MARGIN
    draw.text((_MARGIN, y), title, fill="black", font=title_font)
    y += _TITLE_SIZE + _LINE_GAP
    for line in info_lines:
        draw.text((_MARGIN, y), line, fill="#222222", font=label_font)
        y += line_h
    for line in arch_lines:
        draw.text((_MARGIN, y), line, fill="#444444", font=mono_font)
        y += mono_line_h

    out.paste(img, (0, bar_h))
    return out


def _rank_by_ids_rmse(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: _f(r, "region_knee_ids_rmse")
                  if _f(r, "region_knee_ids_rmse") is not None else 1e9)


def _select_folder(folder: str, source_root: Path, mode: str, top_n: int, pool_n: int | None,
                    gm_ceiling_start: float, python_exe: str, dry_run: bool):
    """Returns (base_csv, base_by_id, chosen_rows, ok_count, gate_desc) or None if skipped.

    mode='gmshapeok': gate-then-ranks gmshape_ok rows to a pool_n-sized pool (matches
    extract_derived_configs.py's do_best100_gmshapeok when pool_n matches --gmshapeok_top_n),
    then returns the first top_n of it (already sorted ascending by ids_rmse).

    mode='bothshapeok': same, but over bothshape_ok rows. With pool_n=None (default), matches
    do_bothshapeok's own default (top_n=None -- uncapped, no combined_gm gate): the WHOLE
    bothshape_ok pool is ranked by ids_rmse and the top top_n taken. With an explicit pool_n,
    matches do_bothshapeok(top_n=pool_n) -- the same gate-then-rank as gmshapeok."""
    filter_fn = _MODE_FILTERS[mode]
    folder_root = source_root / folder
    if not folder_root.is_dir():
        print(f"SKIP {folder}: {folder_root} does not exist")
        return None
    base_csv = _base_ranked_csv(folder_root)
    if base_csv is None:
        print(f"SKIP {folder}: no base ranked csv found under {folder_root}")
        return None
    shape_csv = _ensure_shape_csv(folder, base_csv, python_exe, dry_run)
    if dry_run:
        pool_desc = f"best {pool_n}" if pool_n else "ALL (uncapped)"
        print(f"  (dry run) would select top {top_n} of {pool_desc} {mode} rows from "
              f"{shape_csv.name}, plot, and add to PDF")
        return None

    shape_rows = _read_rows(shape_csv)
    ok_rows = [r for r in shape_rows if r.get("id") and filter_fn(r)]
    if not ok_rows:
        print(f"SKIP {folder}: 0 {mode} rows")
        return None

    if pool_n is None:
        pool = _rank_by_ids_rmse(ok_rows)
        gate_desc = "no combined_gm gate (uncapped)"
    else:
        pool, gm_ceiling = _escalate_and_select(ok_rows, pool_n, gm_ceiling_start)
        gate_desc = f"combined_gm<={gm_ceiling}"
    chosen = pool[:top_n]
    print(f"  {folder}: {mode}={len(ok_rows)}/{len(shape_rows)}, {gate_desc} -> "
          f"best {len(pool)} (pool_n={pool_n}) by ids_rmse, plotting top {len(chosen)}")

    base_rows = _read_rows(base_csv)
    base_by_id = {r["id"]: r for r in base_rows}
    return base_csv, base_by_id, chosen, len(ok_rows), gate_desc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source_root", required=True,
                    help="pure_combined9069-style sweep root, e.g. ../runs/pure_combined9069_rw0_2.5_20")
    ap.add_argument("--folders", default=",".join(FOLDERS))
    ap.add_argument("--mode", choices=sorted(_MODE_FILTERS), default="gmshapeok",
                    help="which shape-compliance derivation to pick 'best' archs from: "
                         "gmshapeok (gm-smoothness only, matches best100_gmshapeok) or "
                         "bothshapeok (gmshape_ok + gds_residual_bad_frac==0, matches "
                         "bothshapeok_of_9069_byshape). Default: gmshapeok")
    ap.add_argument("--top_n", type=int, default=2, help="how many best archs per folder to plot")
    ap.add_argument("--pool_n", type=int, default=None,
                    help="gate-then-rank pool size. Default depends on --mode: 100 for gmshapeok "
                         "(matches extract_derived_configs.py's --gmshapeok_top_n / the actual "
                         "best100_gmshapeok config); None/uncapped for bothshapeok (matches "
                         "bothshapeok_of_9069_byshape's own default of no combined_gm gate -- "
                         "ranks the WHOLE bothshape_ok pool by ids_rmse). Pass explicitly to "
                         "override either default. --top_n only controls how many of this pool "
                         "get plotted.")
    ap.add_argument("--gm_ceiling_start", type=float, default=0.9)
    ap.add_argument("--out_pdf_dir", default=None, help="default: <source_root>/out_pdfs")
    ap.add_argument("--out_pdf_name", default=None, help="default: best<top_n>_<mode>.pdf")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--pdf", action="store_true",
                    help="Assemble the captioned PDF after plotting. Without this flag, the "
                         "script only selects and plots (via plot_csv_row.py) -- no PDF is built.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    source_root = Path(args.source_root).resolve()
    folders = [f.strip() for f in args.folders.split(",") if f.strip()]
    out_pdf_dir = Path(args.out_pdf_dir).resolve() if args.out_pdf_dir else source_root / "out_pdfs"
    out_pdf_name = args.out_pdf_name or f"best{args.top_n}_{args.mode}.pdf"
    pool_n = args.pool_n if args.pool_n is not None else (100 if args.mode == "gmshapeok" else None)

    pages = []  # (folder, rank, id, base_by_id_row, ok_count, gate_desc, base_csv)
    for folder in folders:
        print(f"=== {folder} ===")
        result = _select_folder(folder, source_root, args.mode, args.top_n, pool_n,
                                 args.gm_ceiling_start, args.python_exe, args.dry_run)
        if result is None:
            continue
        base_csv, base_by_id, chosen, ok_count, gate_desc = result
        ids = [r["id"] for r in chosen]

        cmd = [args.python_exe, str(PLOT_CSV_ROW), "--ranked_csv", str(base_csv), "--id", ",".join(ids)]
        print("  $ " + " ".join(cmd))
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"  WARNING: plot_csv_row.py exited {rc} for {folder} -- some plots may be missing")

        for rank, id_val in enumerate(ids, start=1):
            pages.append((folder, rank, id_val, base_by_id.get(id_val, {}), ok_count, gate_desc, base_csv))

    if args.dry_run:
        print("\n(dry run) no PDF written.")
        return

    if not args.pdf:
        print("\ndone (plots only -- pass --pdf to also assemble the captioned PDF).")
        return

    if not pages:
        raise SystemExit("No pages collected -- nothing to write.")

    images = []
    missing = []
    for folder, rank, id_val, row, ok_count, gate_desc, base_csv in pages:
        out_dir = base_csv.parent / "plotted_configs" / f"{base_csv.stem}_id{id_val}"
        png = out_dir / "plot_saved_state_full.png"
        if not png.is_file():
            missing.append((folder, id_val, png))
            continue

        img = Image.open(png)
        if img.mode != "RGB":
            img = img.convert("RGB")

        title = f"{folder}  --  rank #{rank}/{args.top_n} of best_{args.mode}"
        info_lines = [
            f"id={id_val}   arch_id={row.get('arch_id', '?')}   arch_hash={row.get('arch_hash', '?')}",
            f"config_name={row.get('config_name', '?')}",
            f"ids_rmse={row.get('ids_rmse', '?')}   combined_gm={row.get('combined_gm', '?')}   "
            f"region_knee_ids_rmse={row.get('region_knee_ids_rmse', '?')}   "
            f"region_knee_combined_gm={row.get('region_knee_combined_gm', '?')}",
            f"csv={Path(row.get('csv', '?')).name}   selection gate: {args.mode} & {gate_desc}   "
            f"({args.mode} pool={ok_count})",
        ]
        arch_text = row.get("architecture", "")
        images.append(_caption_page(img, title, info_lines, arch_text))
        print(f"  page: {title}  (id={id_val})")

    if missing:
        print("\nmissing plots (skipped):")
        for folder, id_val, png in missing:
            print(f"  {folder} id={id_val}: {png}")

    if not images:
        raise SystemExit("No plot images found -- nothing to write.")

    out_pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_pdf_dir / out_pdf_name
    images[0].save(out_path, save_all=True, append_images=images[1:])
    print(f"\nwrote {out_path}  ({len(images)} pages)")


if __name__ == "__main__":
    main()
