# `pure_combined9069` — running notes

Rationale and cross-folder context for the `runs/pure_combined9069/` sweep (8 folders:
`sigmoid_margin10`, `sigmoid_margin10_nogm`, `softplus`, `softplus_nogm`, `tanh_margin10`,
`tanh_margin10_nogm`, `vdsgate_aeff_quad_tanhm`, `vdsgate_aeff_quad_v3`) that doesn't belong in
any single folder's `filter_readme.md` (auto-maintained by `filter_results.py`, appended/
rewritten by the script itself — see "Why this isn't in `filter_readme.md`" below — so free-text
commentary doesn't survive there) or elsewhere. Added to over time as filters/comparisons are
run against this sweep.

## Filter `#5` — the ~200-row comparison population

Every folder got a `filter_results.py --filter_number 5` pass, producing a
`<hash>_ranked_by_region_knee_combined_gm_5.csv` (plus the usual 8-file shape-analysis set from
`analyze_shape.py` + `keep_columns.py`) in each.

### Goal
A single, comparably-sized population per folder for cross-folder shape analysis (which
`output_activation`/gate-mode/margin combo produces the most `bothshape_ok` survivors, etc. —
see the `vdsgate_aeff_quad_tanhm` vs `vdsgate_aeff_quad_v3` comparison this filter fed into).
Comparing folders is only meaningful if each contributes roughly the same number of candidates,
so `region_knee_ids_rmse` was tuned per folder (under a `region_knee_combined_gm` quality cap)
to land at ~200 rows, using the same `_5` suffix everywhere.

`--filter_number 5` (not `--auto_number`) was used deliberately: it lets each folder use its own
tuned threshold value (the criteria text differs per folder) while keeping the same suffix
number across all of them, which `--auto_number` can't do (it assigns numbers by matching
criteria text, so different thresholds would get different auto-numbers per folder).

### Per-folder criteria and row counts

| Folder | Criteria | Rows | Notes |
|---|---|---|---|
| `sigmoid_margin10` | `combined_gm<=0.9 AND ids_rmse<=0.01143` | 198 | — |
| `sigmoid_margin10_nogm` | `combined_gm<=1.36 AND ids_rmse<=0.01886` | 200 | `combined_gm` capped at `1.36` not `0.9` — only 1 row here reaches `0.9`. Reused this folder's own filter `#4` cap. |
| `softplus` | `combined_gm<=0.9 AND ids_rmse<=0.01115` | 199 | — |
| `softplus_nogm` | `combined_gm<=1.39 AND ids_rmse<=0.02251` | 200 | `combined_gm` capped at `1.39` not `0.9` — 0 rows here reach `0.9`. Reused this folder's own filter `#4` cap. |
| `tanh_margin10` | `combined_gm<=0.9 AND ids_rmse<=0.00871` | 201 | Jumped from filter `#3` straight to `#5` (skipped `#4`) to keep the suffix aligned with the other folders. |
| `tanh_margin10_nogm` | `combined_gm<=0.9 AND ids_rmse<=0.01146` | 200 | Same `#3`→`#5` skip as `tanh_margin10`. |
| `vdsgate_aeff_quad_tanhm` | `combined_gm<=0.9 AND ids_rmse<=0.01501` | 197 | Only had filters `#1`/`#2` before this; jumped straight to `#5` (skipped `#3`/`#4`). |
| `vdsgate_aeff_quad_v3` | `combined_gm<=0.9 AND ids_rmse<=0.01356` | 200 | Same `#1`/`#2`→`#5` skip as `vdsgate_aeff_quad_tanhm`. |

All criteria are on `region_knee_combined_gm` / `region_knee_ids_rmse` (the
`ranked_region_knee_vgs-3to0_vds0to15` ranking), applied via `filter_results.py`'s
`ranked_*/*_ranked_by_*.csv` auto-discovery under each folder's root.

## Filter `#6` — pushing for >=100 `bothshape_ok` per folder

Goal: widen the fit-quality filter (from the `#5` ~200-row population) enough that every folder
contributes at least 100 `bothshape_ok` survivors, so cross-folder shape comparisons aren't
starved by small-sample noise. Same `--filter_number 6` pattern as `#5` (own tuned criteria per
folder, same suffix number).

### What happened
`ids_rmse` relaxation alone (keeping each folder's established `combined_gm` cap: `0.9`, or the
relaxed `1.36`/`1.39` for the two low-combined_gm-ceiling `_nogm` folders) was tried first —
using effectively the ENTIRE existing-quality-tier pool per folder. Result: only 2 of 8 folders
reached 100 this way (`softplus`: 11→126, `tanh_margin10_nogm`: 55→62 at that step). The other 6
barely moved (e.g. `vdsgate_aeff_quad_tanhm`: 5→5 across a 24x larger pool) — proving
`region_knee_ids_rmse` was NOT the limiting factor for those folders; `region_knee_combined_gm`
was.

`combined_gm` cap was then roughly doubled (`0.9`→`1.8`, `1.36`→`2.72`, `1.39`→`2.78`) for the 6
still-short folders. This revealed a sharp cliff: `bothshape_ok` hit rate *within* the original
`<=0.9`-ish tier was roughly flat (~1-9% across low/mid/high `combined_gm`, no discernible trend
with fit quality in the folders that reached the target), but crossing above that tier the hit
rate crashed ~10x on most folders (e.g. `sigmoid_margin10`: ~2.15%→~0.18%,
`vdsgate_aeff_quad_v3`: ~1.47%→~0.036%). So `combined_gm<=0.9` isn't an arbitrary quality
threshold here — it approximates a real compliance boundary for this shape-checking formula.

All 8 folders were then pushed to their ENTIRE architecture population (9068/9069/9056 rows,
i.e. no `combined_gm`/`ids_rmse` cap at all) to get exact answers. The 4 folders run first this
way (`tanh_margin10`, `vdsgate_aeff_quad_v3`) confirmed 100 is unreachable via this filter; the
remaining 4 (`sigmoid_margin10`, `sigmoid_margin10_nogm`, `softplus_nogm`,
`vdsgate_aeff_quad_tanhm`) were then also run to full population (in parallel, ~9-10 min wall
time instead of sequential ~66 min, since `analyze_shape.py` has no incremental/resume mode --
`--ranked_csv` always re-evaluates every row from scratch, so reaching full population means a
full re-run, not just evaluating the previously-missing rows) and produced **identical**
`bothshape_ok` counts to their partial-population runs -- the additional rows (the worst-fitting
~50-91% of each pool, `combined_gm` above the doubled cap) contributed ZERO new compliant rows.
So all 6 non-target-reaching folders are confirmed hard ceilings, not just extrapolated ones. See
`important_mds/shape_analysis_rules.md` for the compliance formula itself -- this filter only
controls which rows get EVALUATED against it, not the formula.

`softplus` and `tanh_margin10_nogm` were subsequently ALSO pushed to their full population (they'd
only been run at their reached-100 population before) to get exhaustively-confirmed numbers for
every folder, not just the 6 that fell short. Result: `softplus` grew 126→155 (found more
compliant rows in the wider pool -- it was NOT at a ceiling, just hadn't been pushed further since
100 was already cleared); `tanh_margin10_nogm` stayed EXACTLY at 258 (confirms it independently
hit its own hard ceiling too, same as the other 6 -- it just happens to sit well above 100).

### Final per-folder criteria and `bothshape_ok` results (ALL 8 at full population, exhaustive)

| Folder | Pop. rows | `bothshape_ok` | Status |
|---|---|---|---|
| `sigmoid_margin10` | 9069 / 9069 | 33 | **hard ceiling** |
| `sigmoid_margin10_nogm` | 9069 / 9069 | 18 | **hard ceiling** |
| `softplus` | 9069 / 9069 | **155** | **hard ceiling** (>=100, not growth-limited) |
| `softplus_nogm` | 9069 / 9069 | 16 | **hard ceiling** |
| `tanh_margin10` | 9068 / 9068 | 94 | **hard ceiling** |
| `tanh_margin10_nogm` | 9069 / 9069 | **258** | **hard ceiling** (>=100, not growth-limited) |
| `vdsgate_aeff_quad_tanhm` | 9069 / 9069 | 5 | **hard ceiling** |
| `vdsgate_aeff_quad_v3` | 9056 / 9056 | 50 | **hard ceiling** |

Final outcome: every folder's `bothshape_ok` count is now its true, exhaustively-verified ceiling
(100% of its trained architectures evaluated, `filter #6` = no `combined_gm`/`ids_rmse` cap at
all). **2 of 8 folders clear 100** (`softplus`: 155, `tanh_margin10_nogm`: 258); the other 6 fall
permanently short, ascending: `vdsgate_aeff_quad_tanhm` (5) << `softplus_nogm` (16) <
`sigmoid_margin10_nogm` (18) < `sigmoid_margin10` (33) < `vdsgate_aeff_quad_v3` (50) <
`tanh_margin10` (94, closest miss). 100 is not achievable for these 6 via
`combined_gm`/`ids_rmse` filtering under any threshold, since the filter is already at "no
threshold at all."

## Why this file exists (not `filter_readme.md`)
`filter_readme.md` (one per folder) is auto-maintained by `filter_results.py` (`--auto_number`
appends a line in-place; `--filter_number` fully rewrites the file from parsed
`N: desc [in: dirs]` entries — see `physics_nn_pipeline/filter_results.py`'s
`get_or_assign_number`/`register_filter_number`/`record_applied_dirs`). A hand-added trailing
comment on one of its lines survives routine `--auto_number` calls (append-only, untouched
lines) but would silently be **deleted** by a future `--filter_number` re-run with different
criteria under the same number on the same folder (the whole line gets overwritten), and would
get mangled (duplicate `[in: ...]` annotations) by a `record_applied_dirs` update, since that
rewrite path re-parses each line expecting it to end in `[in: ...]`. Keeping rationale/commentary
here instead means it survives any future filter reruns.
