# transistor_modeling

Tooling for fitting GaN HEMT I–V characterization data (`Ids` vs `Vgs`, `Vds`,
and the transconductance derivatives `gm1/gm2/gm3`) with two complementary
approaches:

- **Physics-only (SLSQP):** fit an Angelov-family analytic transistor model
  directly (`scipy` SLSQP), producing an optimized-parameter seed.
- **NN / physics+NN (gradient training):** train a small neural net — alone,
  or combined with the Angelov physics model and constrained to a tight prior
  around an SLSQP seed — to fit `Ids` and, optionally, its derivatives.

Both paths are config-driven (JSON sweep files expand into a batch of parallel
training runs) and share a common post-processing pipeline: compile results to
CSV, rank by whichever metric matters (global or region-localized RMSE), and
re-plot the winners.

For the full user guide — CLI flags, config schema, every runner feature
(gradient surgery modes, loss normalization, tight-prior seeding, etc.),
output layout, the recommended coarse-to-fine search workflow, and smoke
tests — see **[docs/README.md](docs/README.md)**. Past experiment rounds,
baselines, and the standing conclusions they drove are in
**[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)**.

## Repo layout

| path | contents |
|---|---|
| [`physics_nn_pipeline/`](physics_nn_pipeline) | Main pipeline: `multi_experiment_runner.py` (NN / physics+NN training), plus compile/rank/filter/plot tooling. |
| [`slsqp_scripts/`](slsqp_scripts) | Physics-only SLSQP runner (`slsqp_experiment_runner.py`) that produces the seed files `use_opt_params` configs consume. |
| [`mod_scripts/`](mod_scripts) | Standalone/earlier copies of several pipeline and compile/plot scripts. |
| [`optim_utils/`](optim_utils) | Shared library: model definitions, physics params/equations, normalization, plotting helpers, solver-space utilities. |
| [`physics_nn_configs/`](physics_nn_configs) | The sweep configs (`opt_configs_*.json`) and PowerShell orchestration scripts (`run_*.ps1`) actually used to launch runs — this is the active config set. |
| [`important_mds/`](important_mds) | Working notes on specific analyses: cross-CSV architecture consistency, shape-compliance rules, and how specific derived config sets (`pure_combined9069`, `bestpicks10`, ...) were built. |
| [`docs/`](docs) | Full user guide and the experiment log. |

`physics_nn_configs/` is a sibling of `physics_nn_pipeline/`, not nested
inside it — several `run_*.ps1` scripts `cd` into `physics_nn_pipeline/`
before invoking the runner by bare filename, and resolve config/csv paths
relative to that directory.

> **Note:** this repo is a curated export of scripts, utilities, and configs
> from a larger working directory. Run outputs (`runs/`), measurement CSVs
> (`csvs/`), and an older/superseded config directory (`physnn_cfgs/`) are
> intentionally not included. Some paths referenced in `docs/README.md` (e.g.
> `slsqp_scripts/slsqp_configs/...`, `../csvs/...`) assume that fuller layout
> and won't resolve as-is here — treat the guide as the reference for how the
> pipeline works, not a runnable checkout.

## Quick start

```bash
cd physics_nn_pipeline

# (a) Physics-only seed fit (only needed if your NN config uses use_opt_params=true)
python ../slsqp_scripts/slsqp_experiment_runner.py \
    --config <slsqp_sweep_config>.json \
    --master_root_path runs/slsqp

# (b) NN / physics+NN sweep
python multi_experiment_runner.py \
    --config ../physics_nn_configs/opt_configs_smoke.json \
    --csv <measurements>.csv \
    --master_root_path runs/nn \
    --min_vgs=-4.0

# (c) Compile + rank + plot the winners
python compile_results_csv.py --root runs/nn
python compile_master_best.py --root runs/nn --metric gm1_rmse --top_n 10
python plot_best_configs.py --dir runs/nn/<experiment> --csv <measurements>.csv \
    --top 5 --sort_by gm1_rmse --regen
```

See [docs/README.md](docs/README.md) for the config schema, the full feature
reference, and the recommended search workflow.
