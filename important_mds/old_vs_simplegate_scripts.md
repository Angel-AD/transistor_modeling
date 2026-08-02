# Which scripts to use: old (original) folders vs SIMPLEGATE folders

`csv_base_2.5_20_rw4` contains folders trained by two different mechanisms. Using the wrong
script for a folder silently reconstructs the model wrong (double-squashes/mismatches the gate)
— region metrics, shape compliance, and plots would all be quietly incorrect. This doc is the
reference for which script goes with which folder.

## Folder -> mechanism

| Mechanism | Folders |
|---|---|
| **Old** (`output_activation: linear` + separate `vdsgate_output_activation` key, or plain `output_activation`) | `sigmoid_margin10`, `sigmoid_margin10_nogm`, `softplus`, `softplus_nogm`, `tanh_margin10`, `tanh_margin10_nogm`, `vdsgate_aeff_quad_tanhm`, `vdsgate_aeff_quad_v3`, and their `_tanhonly` derivatives (`sigmoid_margin10_tanhonly`, etc.) |
| **SIMPLEGATE** (`output_activation` alone IS the gate, no separate key) | `vdsgate_v3`, `vdsgate_tanhm`, and their `_tanhonly` derivatives (`vdsgate_v3_tanhonly`, `vdsgate_tanhm_tanhonly`) |

## Scripts for the OLD folders (+ their `_tanhonly` derivatives)

| Step | Script |
|---|---|
| Derive best200/best100_gmshapeok/bothshapeok | `extract_derived_configs.py` (`--tanh_only` for the dedicated tanh-only versions) |
| Train across 6 csvs + compile | `runner_helpers\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1` |
| Compile (region metrics + rank + shape) | `compile_helpers\compile_overall.ps1` -> `compute_region_metrics.py`, `analyze_shape.py` |
| Shape analysis (cross-csv) | `run_shape_analysis_csv_base.py` |
| Search/plot/summarize | `run_plot_consistent_archs.py` -> `plot_arch_hash.py` -> `plot_saved_state.py` |
| End-to-end tanh-only pipeline | `runner_helpers\run_tanhonly_derivatives_and_best_archs_plots.ps1` |

## Scripts for the SIMPLEGATE folders (`vdsgate_v3`, `vdsgate_tanhm`, + their `_tanhonly` derivatives)

| Step | Script |
|---|---|
| Derive best200/best100_gmshapeok/bothshapeok | `extract_derived_configs_simplegate.py` (`--tanh_only` for the dedicated tanh-only versions) |
| Train across 6 csvs + compile | `runner_helpers\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs_simplegate.ps1` |
| Compile (region metrics + rank + shape) | `compile_helpers\compile_overall_simplegate.ps1` -> `compute_region_metrics_simplegate.py`, `analyze_shape_simplegate.py` |
| Shape analysis (cross-csv) | `run_shape_analysis_csv_base_simplegate.py` |
| Search/plot/summarize | `run_plot_consistent_archs_simplegate.py` -> `plot_arch_hash_simplegate.py` -> `plot_saved_state_simplegate.py` |
| End-to-end pipeline | `runner_helpers\run_vdsgate_v3_tanhm_derivatives_and_best_archs_plots.ps1` |
| End-to-end tanh-only pipeline | `runner_helpers\run_tanhonly_derivatives_and_best_archs_plots_simplegate.ps1` |

## Not mechanism-specific (safe either way, same script for both)

- `write_findings.py`, `write_overall_findings.py`, `write_tanh_only_comparison.py`,
  `write_overall_best_archs.py`, `run_tanh_only_analysis.py`, `render_html_view.py`,
  `compile_results_csv.py`, `compile_master_best.py`, `compile_ranked.py`, `filter_results.py`,
  `keep_columns.py`, `gen_config_from_rows.py` — none of these load/reconstruct a trained model,
  they only read already-computed CSVs/JSON.
- `runner_helpers\create_tanhonly_junctions.ps1` — one script, covers all 10 folders' `_tanhonly`
  junctions (old + SIMPLEGATE alike), since it's pure filesystem plumbing, no model involved.

## The one rule that keeps this safe

Each script's own `FOLDERS` default is already scoped correctly (old scripts default to the 8
original folders only; SIMPLEGATE scripts default to `vdsgate_v3`/`vdsgate_tanhm` only; neither
default includes the `_tanhonly` folders). **Always pass `--folders` explicitly** rather than
typing a folder name into the wrong script by hand — the defaults are safe, but nothing stops a
manual typo from pointing the wrong script at the wrong folder.
