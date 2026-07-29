# Finding architectures consistent across all measurement csvs — steps and scripts

How to find, plot, and document architectures that hold up well across every measurement csv
(not just one), for a `csv_base_root` containing several derivation roots trained against all 6
measurement csvs (e.g. `runs/csv_base_2.5_20/`, built per
`important_mds/base_9069_and_bestids_creation.md`). All commands run from `physics_nn_pipeline/`
unless noted.

## Prerequisite
Every `<base_config>/<csv_name>/<folder>/` leaf under `csv_base_root` must already be trained
AND compiled (`compiled_results*.csv` / `ranked_region_knee_vgs-3to0_vds0to15/*_ranked_by_
region_knee_combined_gm.csv` present) — see `base_9069_and_bestids_creation.md` §4. Everything
below reads `region_knee_combined_gm`/`region_knee_ids_rmse` (the knee-region-restricted
metrics, `vgs∈[-3,0]`/`vds∈[0,15]`) from that base ranked csv — not the plain global
`combined_gm`/`ids_rmse` columns also present in the same file.

## Step 1 — full-population shape analysis
`find_consistent_archs.py`'s shape-based search methods (B/C below) need per-architecture
gm-smoothness/gds-residual metrics computed FIRST, for every architecture, for every csv, for
every folder — this is what "shape analysis available for N/6 csvs" in the summaries refers to.

```bash
python run_shape_analysis_csv_base.py --csv_base_root ../runs/csv_base_2.5_20
```
Loops every `(base_config, csv, folder)` leaf (3 × 6 × 8 = 144 by default), reusing
`extract_derived_configs.py`'s `_ensure_shape_csv` (`analyze_shape.py` full-population pass +
`keep_columns.py` slimming, same convention used everywhere else in this project). Resumable —
skips any leaf whose `<folder>_ranked_by_region_knee_combined_gm_shape.csv` already exists. To
force a clean recompute (e.g. after a change to `analyze_shape.py`'s compliance formula), delete
the existing `*_ranked_by_region_knee_combined_gm_shape*` files first (NOT the numbered/filtered
`_1_shape`/`_5_shape` variants, which come from something else and use a different glob).

See `important_mds/shape_analysis_rules.md` for the `gmshape_ok`/`bothshape_ok` compliance
formula itself — this step only computes the metrics the formula reads.

Leaves are independent (one `analyze_shape.py` subprocess each), so this runs in a worker pool:
`--workers` (default 8), safe to raise well above that since it's CPU/IO-bound, not GPU-bound.

## Step 2 — search, plot, and summarize
```bash
python run_plot_consistent_archs.py --csv_base_root ../runs/csv_base_2.5_20
# defaults: all 8 folders, all 3 base_configs, --format md, --workers 8, --search_workers 8, --top 5
```
Runs across ALL folders at once (not one folder at a time), in two separately-tunable worker
pools: `--search_workers` for the cheap `write_findings.py` calls (steps 1/4 below, CSV I/O
only — safe to raise high, e.g. 24+), and `--workers` for the expensive `plot_arch_hash.py`
calls (step 3, real model training/eval) shared across every folder's picks at once (so raising
`--workers` is the one number that actually controls total concurrent load, rather than
multiplying if folders were parallelized on top of it independently). Pass `--skip_plotting` to
run only steps 1/4/5 (e.g. to cheaply regenerate `consistency_summary_*` after a
`find_consistent_archs.py` formula/field change, without re-plotting images that haven't
changed).

Steps, in order:

1. **Search** (`write_findings.py`, once per folder × base_config) — runs
   `find_consistent_archs.py`'s 5 methods against that base_config's population:
   - **A) fit-quality min-max** (no shape data needed): for every `arch_hash` present in EVERY
     csv's own `combined_gm<=gm_ceiling` pool, take its `ids_rmse` PERCENTILE within that csv's
     pool, then the WORST (max) percentile across all csvs — sort ascending. Finds the single
     most reliably-good-everywhere architecture; threshold-free by design (a narrow top-20-per-
     csv overlap search can miss a consistently-decent-but-never-top-20 architecture). Also
     reports the BEST (min) percentile, and `ids_rmse`/`combined_gm` as avg + per-csv min-max
     range (`find_consistent_archs.format_ids_rmse`/`format_combined_gm`).
   - **B) `gmshape_ok`/`bothshape_ok`, goal=all_csvs**: ranked by (count of csvs where the check
     passes) desc, tie-break avg `ids_rmse` asc. Same `ids_rmse`/`combined_gm` avg+range detail.
   - **C) `gmshape_ok`/`bothshape_ok`, goal=pair**: same checks, restricted to a specific PAIR
     of csvs (default: `cg2h40010_new_2.5_20_2_70W_center9` AND
     `cg2h40010_new_2.9_28_2_70W_center9`, override with `--pair_csvs`).
   - **D) fit-quality min-max, goal=pair**: method A restricted to just that pair.

   Writes `consistency_summary_<base_config>.md` (human-readable tables) and a companion
   `consistency_summary_<base_config>.json` (same numbers, machine-readable) to
   `<out_root>/<folder>/`, where `out_root` defaults to `<csv_base_root>/best_archs_plots`
   (SHARED across all base_configs, not nested inside each one).

2. **Collect + dedupe picks** — every `(folder, arch_hash, goal)` triple across all folders × 6
   method-sections × all base_configs, deduped (the SAME arch_hash picked under the SAME goal
   by more than one base_config is plotted once, not once per base_config — they're the same
   physical architecture regardless of which search found it). Goal labels:
   `best_fit_quality` (A), `best_consistent_with_all_csvs` (B),
   `best_consistent_with_<pair0>_and_<pair1>` (C), `best_fit_quality_pair` (D).

3. **Plot** (`plot_arch_hash.py --format md`, one call per unique pick) — plots the architecture
   across every csv subfolder, writes `compliance.json` (per-csv `gmshape_ok`/`gdsshape_ok`/
   `bothshape_ok` + overall category), and assembles `all_csvs.md` (or `.pdf`/both, via
   `--format`) into:
   ```
   <out_root>/<folder>/<category>/<goal>/<arch_hash>/
   ```
   `<category>` (from `compliance.json`, priority `both > gmshape > none > filter_only`):
   `both` (bothshape_ok on ≥1 csv), `gmshape` (gmshape_ok but never bothshape_ok),
   `none` (shape checked, never passed), `filter_only` (no shape data at all for this arch —
   picked purely by fit quality). `<goal>` is which search method (above) picked it.

4. **Refresh** — re-runs step 1's `write_findings.py` calls so the "category" column (blank/`?`
   until `compliance.json` exists) gets filled in now that the plots exist.

5. **Overall summary** (`write_overall_findings.py`, once per folder) — cross-references all the
   base_configs' `consistency_summary_<base_config>.json` files, writing
   `consistency_summary_overall.md`: which `arch_hash`es were picked by MORE THAN ONE
   base_config's search (independent confirmation from separately-selected architecture pools —
   a stronger signal than any single base_config's own ranking), plus links to each individual
   summary.

Note: `run_plot_consistent_archs.py` does NOT run step 6 below automatically -- run it as a
separate pass after (needs every pick's plots/compliance.json already on disk, i.e. after step
3 above has finished for all base_configs in that folder).

6. **Plots-inlined companion file** (`write_findings_with_plots.py`, once per base_config) --
   `consistency_summary_<base_config>.md` only has text tables (arch_hash, stats, no images);
   this produces a companion `consistency_summary_<base_config>_plots.md` with the SAME section
   structure but each picked architecture's actual `plot_saved_state_full.png` (one per csv)
   embedded inline, plus its config/architecture block and per-csv compliance table (same
   sources `all_csvs.md` itself uses). An arch_hash's plots are embedded only the FIRST time
   it's encountered in the file (recurring across sections is common -- a fit-quality winner is
   often also a shape-consistency winner); later occurrences get a short cross-reference note
   instead of duplicating images.
   ```bash
   python write_findings_with_plots.py --out_root <csv_base_root>/best_archs_plots \
       --folder tanh_margin10
   # --base_configs defaults to every consistency_summary_*.json found under that folder
   ```

## Output layout
```
<csv_base_root>/best_archs_plots/<folder>/
    consistency_summary_<base_config>.md / .json      (one pair per base_config)
    consistency_summary_<base_config>_plots.md         (same sections, images inlined -- step 6)
    consistency_summary_overall.md                     (cross-references all base_configs)
    <category>/<goal>/<arch_hash>/
        <csv_name>_plot_saved_state_full.png           (one per csv the arch was plotted on)
        compliance.json
        all_csvs.md   (and/or all_csvs.pdf, per --format)
```

## Running pieces individually (not the full batch)
- **Quick look, no writing anything**: `python find_consistent_archs.py --root
  <csv_base_root>/<base_config> --folder tanh_margin10` — prints all 5 result tables to stdout.
- **One specific architecture**: `python plot_arch_hash.py --root <csv_base_root>/<base_config>
  --folder tanh_margin10 --arch_hash <hash> --goal <label> --out_dir <csv_base_root>/
  best_archs_plots --format md`
- **One base_config's summary only**: `python write_findings.py --root <csv_base_root>/
  <base_config> --folder tanh_margin10 --out_root <csv_base_root>/best_archs_plots`
- **Just the overall cross-reference** (after summaries exist): `python
  write_overall_findings.py --out_root <csv_base_root>/best_archs_plots --folder tanh_margin10`
- **Just the plots-inlined companion file** (after that base_config's plots exist): `python
  write_findings_with_plots.py --out_root <csv_base_root>/best_archs_plots
  --folder tanh_margin10 --base_configs best200ids_of_9069_byloss_rw0_2.5_20`

## Tanh-only sub-analysis: does restricting to pure-tanh architectures cost anything?

A separate, optional pass: restricts the SAME 5 search methods to architectures using ONLY
`tanh` in every layer (no `swish`/`mish` mixed in), and compares the result against the
unrestricted (heterogeneous-activation) search above. Reuses everything above -- same
`find_consistent_archs.py` search functions, just filtered first.

```bash
python run_tanh_only_analysis.py --csv_base_root ../runs/csv_base_2.5_20
# defaults: all 8 folders, all 3 base_configs, --workers 8 -- everything here is cheap CSV I/O
# (no plotting), so --workers is safe to raise well above the plotting-step's default.
```
Runs across every (folder, base_config) combination and every folder's overall/comparison step
in one shared worker pool (`--workers`), same reasoning as step 2's `--search_workers` above.

For each folder, per base_config:
1. `write_findings.py --only_tanh` -- filters the population (via `find_consistent_archs.py`'s
   `tanh_only_hashes()` + `filter_population()`) to tanh-only architectures BEFORE running the
   5 searches, writes `consistency_summary_<base_config>.md/.json` into `<folder>/tanh_only/`
   (SAME filenames as the unrestricted summaries, just nested one level deeper -- `compliance.
   json` lookups still use the SHARED, non-nested plots tree, since plots aren't duplicated
   per-filter).
2. `write_overall_findings.py --subdir tanh_only` -- cross-references the tanh-only picks
   across base_configs, same as the unrestricted overall summary, written to
   `<folder>/tanh_only/consistency_summary_overall.md`.
3. `write_tanh_only_comparison.py` -- reads BOTH the tanh-only and unrestricted
   `consistency_summary_<base_config>.json` for each base_config, and writes a side-by-side
   comparison table (rank-for-rank, same method sections) to `<folder>/
   tanh_only_heterogeneus_comparisons/comparison_<base_config>.md`, plus one
   `comparison_overall.md` concatenating all base_configs into a single-scroll overview.

   Every comparison file OPENS with a **verdict summary**: which method(s) heterogeneous won,
   and -- called out explicitly since it's the notable/rarer case -- which (if any) tanh-only
   won. "Won" is decided per-method by `find_consistent_archs.method_sort_key` (the SAME
   criterion each search already sorts its own results by -- worst-case percentile for A/D,
   hit-count then avg `ids_rmse` for B, avg `ids_rmse` for C), comparing each search's own
   rank-#1 pick. **If both searches land on the identical `arch_hash`** (common -- the single
   best architecture overall sometimes just happens to itself be tanh-only, so it's #1 in both
   the unrestricted AND tanh-only-restricted search), the verdict is a TIE by definition, not a
   "win" for either side -- comparing that one architecture's own percentile against itself
   would otherwise misreport a winner purely from the pool-size difference (see the correctness
   note below), even though there's no real difference in which architecture was found.

Output layout addition:
```
<csv_base_root>/best_archs_plots/<folder>/
    tanh_only/
        consistency_summary_<base_config>.md / .json   (tanh-only-restricted searches)
        consistency_summary_overall.md
    tanh_only_heterogeneus_comparisons/
        comparison_<base_config>.md                     (tanh-only vs heterogeneous, per derivation)
        comparison_overall.md                           (all base_configs concatenated)
```

Empirical result so far (`tanh_margin10`, `best200ids_of_9069_byloss_rw0_2.5_20`): tanh-only
architectures are meaningfully worse on every method -- e.g. method A's best tanh-only pick has
a 50.0% worst-case percentile vs 19.5% for the best heterogeneous pick. Mixed activations are
not just marginally better here. Not universal though -- some folders' verdict summaries show
ties (same architecture winning both searches) or even a genuine tanh-only win on a specific
method; read the verdict line rather than assuming the general pattern holds everywhere.

## Step 4 (optional) -- cross-folder leaderboard
```bash
python write_overall_best_archs.py --out_root ../runs/csv_base_2.5_20/best_archs_plots
```
Spans ALL folders (unlike everything above, which is per-folder): for each of the 6 methods,
finds each folder's own best pick (comparing that folder's 3 base_configs' rank-#1 picks via
`method_sort_key`, keeping whichever is actually better -- not just the first one found), then
ranks all folders against each other by that same criterion. Uses the unrestricted
(heterogeneous) search only. Written directly at `<out_root>/best_archs_summary_overall.md`
(the `best_archs_plots` root itself, not nested in any folder). Links back to each folder's own
`consistency_summary_<base_config>.md` for the winning entries.

## A correctness note worth knowing before trusting shape-analysis numbers
`analyze_shape.py`'s residual-wobble check (`gds_residual_bad_frac`, half of `bothshape_ok`) and
`plot_saved_state.py`'s Ids-vs-Vds/Ids-vs-Vgs panels both evaluate the model at a matched
measurement's ACTUAL mean Vgs/Vds (not the nominal `--plot_vgs_list`/`--plot_vds_list` target
grid value) — real sweeps rarely land exactly on the nominal grid (~0.03-0.05V off for Vgs;
Vds trace points are irregularly spaced with larger gaps). This was fixed in both places
together (`optim_utils/per_neuron_plotting.py::generate_physics_plot_data`,
`physics_nn_pipeline/analyze_shape.py::measured_vds_traces`) so the two stay consistent with
each other — a mismatch here previously changed the computed residual by ~2x. Any `_shape.csv`
computed BEFORE this fix used the old (target-Vgs) convention; step 1 above must be rerun
(after deleting the stale `_shape.csv`/derivatives) for `bothshape_ok`/`gds_residual_bad_frac`
numbers to reflect the corrected formula.
