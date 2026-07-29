# Creating `pure_combined9069_rw0_2.5_20` and its 3 "best ids" derivations

How `runs/pure_combined9069_rw0_2.5_20/` and the three derived config sets trained under
`runs/csv_base_2.5_20/` (`best200ids_of_9069_byloss_rw0_2.5_20`,
`bothshapeok_of_9069_byshape_rw0_2.5_20`, `best100_gmshapeok_of_9069_byloss_rw0_2.5_20`) were
built, step by step, with the exact scripts/commands used. All paths below are relative to
`physics_nn_pipeline/` unless noted; `runs/` is an NTFS junction to `D:\new_opts_2_runs\`.

## 1. `pure_combined9069_rw0_2.5_20` — the source sweep

Starting point: the original `runs/pure_combined9069/` sweep (8 folders — `sigmoid_margin10`,
`sigmoid_margin10_nogm`, `softplus`, `softplus_nogm`, `tanh_margin10`, `tanh_margin10_nogm`,
`vdsgate_aeff_quad_tanhm`, `vdsgate_aeff_quad_v3` — each independently trained on the SAME
~9069-architecture search space, `region_weight=4`, against
`cg2h40010_new_2.4_5_2_70W_center9.csv`). See `important_mds/pure_combined9069.md` for that
sweep's own history.

**Goal**: replicate the exact same 8-folder/9069-architecture sweep, but with `region_weight=0`
(the weighted-MSE region-box loss OFF — confirmed via
`per_neuron_simple_angelov_nn_test.py`'s `if args.region_weight and args.region_weight > 0.0:`
gate, `0` genuinely disables the branch, not a redundant `1+0` multiplier) and against a
DIFFERENT measurement csv, `cg2h40010_new_2.5_20_2_70W_center9.csv`.

### 1a. Region-weight-0 config copies
For each of the 8 folders' 3 archived batch configs (`runs/pure_combined9069/<folder>/
base_files/*.json`), a deep copy was made with only `"region_weights": [4.0]` overridden to
`[0.0]` — everything else (`nn_architectures`, gm weights, epochs, `base_configs`, etc.)
byte-identical, verified via deep structural diff. Written to
`runs/pure_combined9069_rw0_2.5_20/_configs/` (24 files: 8 folders × 3 batches).

### 1b. Training command
```powershell
cd physics_nn_configs
.\run_pure_combined9069_2_5_20_all_batches.ps1
# defaults: -Workers 46 -Csv ..\csvs\cg2h40010_new_2.5_20_2_70W_center9.csv
#           -OutSuffix rw0_2.5_20  (-> runs\pure_combined9069_rw0_2.5_20\)
```
Same batching pattern as the original sweep: 8 folders × 3 batches, all 3 batches per folder
landing in one shared root (distinct `_batchN` experiment-name suffix avoids collision),
compiled once per folder after all 3 batches finish (`compile_overall.ps1`, no `-ShortName`
since this is a flat `<folder>/` layout, not `<csv>/<folder>/`).

**Result**: `runs/pure_combined9069_rw0_2.5_20/<folder>/` × 8, each with ~9069 trained
architectures (region_weight=0, `cg2h40010_new_2.5_20_2_70W_center9.csv`), compiled with a
`ranked_region_knee_vgs-3to0_vds0to15/<folder>_ranked_by_region_knee_combined_gm.csv` per
folder.

## 2. Deriving the 3 "best ids" config sets

`extract_derived_configs.py` reproduces the same derivation methodology originally used to turn
`runs/pure_combined9069/` into `runs/best200ids_of_9069_byloss/` and
`runs/bothshapeok_of_9069_byshape/`, generalized to run against ANY `pure_combined9069`-style
source root:

```bash
cd physics_nn_pipeline
python extract_derived_configs.py --source_root ../runs/pure_combined9069_rw0_2.5_20
```
(all defaults: all 8 folders, `--top_n 200`, `--gmshapeok_top_n 100`,
`--bothshapeok_top_n` unset/uncapped, `--gm_ceiling_start 0.9`, `--filter_number 5`)

Three independent derivations, run per folder:

| Derivation | Method | Output root |
|---|---|---|
| `best200` | Gate-then-rank ("Approach 1"): auto-escalate a `region_knee_combined_gm` ceiling (×1.5 from 0.9) until ≥200 rows qualify, then top 200 of those by `region_knee_ids_rmse` (via `filter_results.py --top_n 200 --sort_by region_knee_ids_rmse`) | `runs/best200ids_of_9069_byloss_rw0_2.5_20/_configs/` |
| `bothshapeok` | Fully mechanical: full-population `analyze_shape.py` → every `bothshape_ok` id (uncapped — no gate) | `runs/bothshapeok_of_9069_byshape_rw0_2.5_20/_configs/` |
| `best100_gmshapeok` | Same gate-then-rank as `best200`, but restricted to the `gmshape_ok` subset, capped at 100 | `runs/best100_gmshapeok_of_9069_byloss_rw0_2.5_20/_configs/` |

`bothshapeok`'s gate-then-rank helper (`_escalate_and_select`) is SHARED with `best200`/
`best100_gmshapeok` so all three derivations use the identical selection criterion when a cap
applies — confirmed different from (and preferred over) a naive
`df.sort_values(["region_knee_combined_gm", "region_knee_ids_rmse"]).head(N)`: the gate-then-
rank approach uses `combined_gm` as a hard quality GATE (physically justified — shape-compliance
hit rate crashes ~10x above certain `combined_gm` thresholds, see `pure_combined9069.md`), THEN
ranks by `ids_rmse` within that gated pool, rather than a lexicographic sort where the second
column only breaks EXACT ties on the first (and `region_knee_combined_gm` is stored rounded to
4 decimals, so ties are common — ~65% of rows share a value with another row).

**Result**: 24 config files (8 folders × 3 derivations), each with `nn_architectures` = the
selected subset, `region_weights: [0.0]` inherited unchanged from the source.

## 3. Config review — `adamw_avoid_localmin` fix

Double-checking the 24 derived configs against the actual CLI args logged in a real source run
(`run_log.txt.gz`) found one real gap: `"adamw_avoid_localmin": true` (cosine-annealing-with-
warm-restarts AdamW LR schedule — confirmed via the run's own `CMD`/`Args` log that this WAS
used for every run in the source sweep) is never logged into `run_loss_*.json`
(`per_neuron_simple_angelov_nn_test.py`'s `run_info` dict never includes it), so
`gen_config_from_rows.py` has no way to recover it — a pre-existing gap in the tool, not
specific to this derivation. Patched into all 24 configs' `DEFAULTS` block directly (one key
added, nothing else touched).

Other apparent differences against the source configs were checked and confirmed BENIGN (no
functional effect, since the code paths that would consume them are already gated off for this
population — `region_weight=0`, `vds_loss=None` everywhere):
- `region_vds_hi/lo`, `region_vgs_hi/lo` — absent in derived configs (never logged when
  `region_weight=0`); inert either way.
- `gm1_weights`/`lbfgs_gm_aware` in the `_nogm` folders — derived configs correctly show the
  POST-gating effective values (`0.0`/`False`, forced whenever `use_gm=False`), vs. the raw
  (never-actually-used) CLI values in the source config.
- `base_configs` dict key naming (`vdsgate_aeff_quad_tanhm`/`_v3` folders only) — cosmetic;
  only affects output folder naming, not model selection (`equation_type` inside the value dict
  is identical either way).

## 4. Training all 24 configs against all 6 measurement csvs

```powershell
cd physics_nn_configs
.\run_derived_configs_from_pure_combined9069_rw0_2.5_20_all6csvs.ps1
# defaults: -Workers 46, all 8 folders, all 3 derivations, -OutputParent csv_base_2.5_20
```
Loops derivation × csv × folder (3 × 6 × 8 = 144 `multi_experiment_runner.py` calls), reading
each config from its ORIGINAL `_configs/` location (unaffected by `-OutputParent`), writing
results nested under one shared parent:
```
runs/csv_base_2.5_20/best200ids_of_9069_byloss_rw0_2.5_20/<csv-name>/<folder>/
runs/csv_base_2.5_20/bothshapeok_of_9069_byshape_rw0_2.5_20/<csv-name>/<folder>/
runs/csv_base_2.5_20/best100_gmshapeok_of_9069_byloss_rw0_2.5_20/<csv-name>/<folder>/
```
Compiles every one of the 144 leaf folders (`compile_overall.ps1 -ShortName`) once ALL training
finishes (not interleaved per-folder). Resumable — `multi_experiment_runner.py` itself skips any
`exp_*` dir that already has both `run_loss_*` and `weights_loss_*`.

**Scale**: architecture counts per folder are 200 (`best200`), 100 (`best100_gmshapeok`), and
uncapped for `bothshapeok` (4 to 1147 depending on folder — `softplus`/`vdsgate_aeff_quad_v3`
dominate). Total: 4,530 runs per csv × 6 csvs = **27,180 training runs**.

## Final state
3 derivation roots under `runs/csv_base_2.5_20/`, each with 8 folders × 6 measurement csvs, all
fully trained and compiled (`compiled_results*.csv` / `ranked_region_knee_.../` present in every
leaf). This is the population the consistency analysis in
`important_mds/bestids_across_allcsvs_analysis_steps.md` runs against.
