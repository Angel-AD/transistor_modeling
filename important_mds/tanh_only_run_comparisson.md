# Tanh-only vs heterogeneous: derive, train, analyze, compare

Dedicated tanh-only search (best200/best100_gmshapeok/bothshapeok picked FROM a tanh-only-only
population, not a post-hoc filter of the heterogeneous picks) for all 10 folders:
- the 8 original folders (`sigmoid_margin10`, `sigmoid_margin10_nogm`, `softplus`,
  `softplus_nogm`, `tanh_margin10`, `tanh_margin10_nogm`, `vdsgate_aeff_quad_tanhm`,
  `vdsgate_aeff_quad_v3`) — original (non-simplegate) mechanism
- `vdsgate_v3`/`vdsgate_tanhm` — SIMPLEGATE mechanism

then compared against the existing heterogeneous results. See
`important_mds/old_vs_simplegate_scripts.md` for why these two groups need different scripts.

All commands run from `physics_nn_pipeline\runner_helpers\` unless noted.

## 1. Derive + train + analyze the tanh-only population

```powershell
.\run_tanhonly_derivatives_and_best_archs_plots.ps1
.\run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1
```

Run **both** — one covers the 8 original folders, the other covers `vdsgate_v3`/`vdsgate_tanhm`.

The original-mechanism script runs, in order: `extract_derived_configs.py --tanh_only` (derives
best200/best100_gmshapeok/bothshapeok from a tanh-only-restricted population) → trains those
derived configs across all 6 measurement CSVs + compiles
(`run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1`) → `run_shape_analysis_
csv_base.py` → `run_plot_consistent_archs.py` → `render_html_view.py`.

The simplegate script does the same with the `_simplegate` variant of every script, plus an
extra first step to (re-)compile the base 9069 sweep itself (`compile_overall_simplegate.ps1`),
since that's a prerequisite `extract_derived_configs_simplegate.py --tanh_only` needs.

Both write everything under `<folder>_tanhonly` (e.g. `sigmoid_margin10_tanhonly`,
`vdsgate_v3_tanhonly`) — never touch the existing heterogeneous folders/data. Resumable; re-run
safely if interrupted. Useful flags on both: `-Workers 16`, `-SkipDerive`,
`-SkipTrainAll6Csvs`, `-SkipShapeAnalysis`, `-SkipPlotConsistentArchs`, `-SkipHtml` (the
simplegate one also has `-SkipCompileBaseSweep`).

## 2. Create the tanh_only junctions

```powershell
.\create_tanhonly_junctions.ps1
```

One script, covers all 10 folders. Creates `best_archs_plots\<folder>\tanh_only` as a directory
junction pointing at `best_archs_plots\<folder>_tanhonly`. This is what lets the existing
`write_tanh_only_comparison.py` (which hardcodes `<folder>/tanh_only/consistency_summary_
<base_config>.json` as its lookup path) find the dedicated tanh-only data without any code
changes. Safe to re-run — skips any junction that already exists, and skips any folder whose
`_tanhonly` target doesn't exist yet (so it's fine to run this before both pipelines above have
finished, or before you've decided to run one of them at all).

## 3. Run the comparison

```bash
python write_tanh_only_comparison.py --out_root "C:\Users\acost\repos\new_opts_2\runs\csv_base_2.5_20_rw4\best_archs_plots" --folder sigmoid_margin10,sigmoid_margin10_nogm,softplus,softplus_nogm,tanh_margin10,tanh_margin10_nogm,vdsgate_aeff_quad_tanhm,vdsgate_aeff_quad_v3,vdsgate_v3,vdsgate_tanhm --base_configs best200ids_of_9069_byloss_rw4_2.5_20,bothshapeok_of_9069_byshape_rw4_2.5_20,best100_gmshapeok_of_9069_byloss_rw4_2.5_20
```

Run from `physics_nn_pipeline\`. Same script, one call, all 10 folders — the comparison itself
does no model reconstruction (just reads already-computed summary JSON/CSV), so it's identical
either way regardless of which folders are original-mechanism vs SIMPLEGATE. Writes the
side-by-side comparison to `best_archs_plots\<folder>\tanh_only_heterogeneus_comparisons\
comparison_<base_config>.md` (plus a `comparison_overall.md` per folder), for each folder.

## Why bothshapeok is included even though it can't add new architectures

`bothshapeok` has no cap and no cross-architecture competition — it already includes every
tanh-only `bothshape_ok` architecture that exists (verified empirically: 0 missing across all 8
original folders). Running it here anyway is a deliberate consistency check, not wasted-on-a-
guess: it confirms the tanh-only selection is picking up everything expected, rather than
assuming it.

## Relevant scripts

- `extract_derived_configs.py` / `extract_derived_configs_simplegate.py` — `--tanh_only` flag (new)
- `runner_helpers\run_tanhonly_derivatives_and_best_archs_plots.ps1` (new)
- `runner_helpers\run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1` (new)
- `runner_helpers\create_tanhonly_junctions.ps1` (new, covers all 10 folders)
- `write_tanh_only_comparison.py` (existing, unmodified)
