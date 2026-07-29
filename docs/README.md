# Optimization Runners — User Guide

How to run the device-model optimization pipeline: NN / physics+NN training
sweeps and physics-only SLSQP seed fits, plus the compile/rank/plot tooling.

This guide is meant to be self-contained — a new person (or a fresh AI session)
should be able to read it and drive the whole pipeline.

---

## Table of contents
1. [Pipeline overview](#1-pipeline-overview)
2. [Quick start](#2-quick-start)
3. [The NN runner (`multi_experiment_runner.py`)](#3-the-nn-runner-multi_experiment_runnerpy)
4. [The SLSQP runner (`slsqp_experiment_runner.py`)](#4-the-slsqp-runner-slsqp_experiment_runnerpy)
5. [Feature reference](#5-feature-reference)
6. [Output layout & `run_loss_*.json` fields](#6-output-layout--run_loss_json-fields)
7. [Post-processing: compile, rank, plot](#7-post-processing-compile-rank-plot)
8. [Recommended workflow (how to actually search)](#8-recommended-workflow-how-to-actually-search)
9. [Reproducibility](#9-reproducibility)
10. [Tests](#10-tests)
11. [Gotchas / FAQ](#11-gotchas--faq)

---

## 1. Pipeline overview

The data is a GaN HEMT I–V characterization CSV (`Ids` vs `Vgs`, `Vds`). Two
modelling paths:

- **Physics-only (SLSQP):** fit an Angelov-family analytic model to the data →
  produces a `slsqp_seed.json` of optimized physics parameters.
- **NN / physics+NN (gradient training):** train a small neural net, optionally
  combined with an Angelov physics model, to fit `Ids` and (optionally) its
  transconductance derivatives `gm1 = dIds/dVgs`, `gm2`, `gm3`.

```
                 ┌─────────────────────────┐
   measurements  │ slsqp_experiment_runner │  (physics-only, scipy SLSQP)
   CSV  ───────► │   per_neuron_slsqp_single│ ─► slsqp_seed.json (+ auto-plot)
                 └─────────────────────────┘            │  (optimized physics params)
                                                        ▼  used as opt_params seed
                 ┌─────────────────────────┐
                 │ multi_experiment_runner │  (NN / physics+NN, AdamW + L-BFGS)
   CSV  ───────► │ per_neuron_simple_…test │ ─► run_loss_*.json + weights_*.pt + run_log.txt.gz
                 └─────────────────────────┘            │
                                                        ▼
   compile_results_csv ─► compile_master_best ─► compile_results_to_xlsx
                                                        │
                                            plot_best_configs (re-plot top-N,
                                                  collect into best_n_configs/)
```

Everything below lives in `physics_nn_pipeline/`. Paths in this doc are relative
to that folder unless noted. Both runners are **config-driven**: you write a JSON
config describing a sweep, and the runner expands it into individual training
jobs run in parallel.

---

## 2. Quick start

```bash
cd physics_nn_pipeline

# (a) Generate physics seeds (only needed if your NN config uses use_opt_params=true)
python slsqp_scripts/slsqp_experiment_runner.py \
    --config slsqp_scripts/slsqp_configs/opt_configs_smoke.json \
    --master_root_path runs/slsqp

# (b) Run an NN / physics+NN sweep
python multi_experiment_runner.py \
    --config physics_nn_configs/opt_configs_smoke.json \
    --csv ../csvs/<your_measurements>.csv \
    --master_root_path runs/nn \
    --min_vgs=-4.0

# (c) Compile + rank + plot the winners
python compile_results_csv.py     --root runs/nn
python compile_master_best.py     --root runs/nn --metric gm1_rmse --top_n 10
python plot_best_configs.py        --dir runs/nn/<experiment> --csv ../csvs/<...>.csv \
                                   --top 5 --sort_by gm1_rmse --regen
```

> **Important:** values that begin with `-` (e.g. `--min_vgs -4.0`,
> `--plot_vgs_list -3.5,...`) must use the `--opt=value` form
> (`--min_vgs=-4.0`), or argparse treats them as flags.

---

## 3. The NN runner (`multi_experiment_runner.py`)

### CLI
| flag | required | default | meaning |
|---|---|---|---|
| `--config` | yes | – | path to the sweep JSON. **Takes one or more paths** (`nargs="+"`) — passing several loads and runs all of their experiments together under one shared worker pool. |
| `--csv` | yes | – | measurement CSV forwarded to every worker |
| `--master_root_path` | no | env `MASTER_ROOT_PATH` or built-in | output root |
| `--opt_params_path` | no | env `OPT_PARAMS_PATH` | fallback seed path (overridden per-variant) |
| `--max_workers` | no | (config's own `max_workers`, or their max if multiple configs) | global override for the shared `ProcessPoolExecutor` pool size |
| `--min_vgs` | no | `-4.0` | extrapolation floor passed downstream |
| `--plot_vds_list` | no | `0,5,10,15,20,28` | (only used if a worker plots) |
| `--plot_vgs_list` | no | `-3.5,...,0` | " |

### Config schema

```jsonc
{
  "DEFAULTS": {                         // applied to every experiment unless overridden
    "base_configs": { "<name>": { ...base_config... } },
    "knee_alpha_scales": [1.0],         // physics knee scale (physics modes only)
    "knee_combiners":    ["sum"],       // sum|product|residual|sum_gated_vgs|max|min (physics modes only)
    "learning_rates":    [0.01],
    "nn_architectures":  ["[[\"tanh\",\"sin\"],[\"tanh\",\"sin\"]]"],  // or "@GENERATE@"
    "output_activations":["linear"],    // linear|softplus|...
    "gm1_weights": [0.0], "gm2_weights": [0.0], "gm3_weights": [0.0],
    "gm_surgery_modes":  ["none"],      // swept: none|no-bounded|mag-bounded|soft|element-wise|
                                        //   element-wise-percent|element-wise-bounded|element-wise-soft|drop-conflict
    "gm_max_ratios":     [1.0],         // swept: surgery multiplier/cap (see §5). >1 amplifies in
                                        //   element-wise-percent; raises the cap in the bounded modes.
    "epochs": 5000,                     // AdamW epochs (scalar)
    "lbfgs_epochs": 5,                  // L-BFGS polish outer steps (scalar; was hardcoded 5)
    "lbfgs_max_iter": 200,              // L-BFGS inner max_iter per step (scalar; was hardcoded 200)
    "lbfgs_gm_aware": null,             // null = AUTO (on whenever use_gm); true/false to force.
                                        //   gm-aware: L-BFGS polishes plain combined Ids+gm (surgery is AdamW-only)
    "loss_norm": "none",               // none | nmse. nmse: scale-balance every loss term (Ids & each gm)
                                        //   by mean(target^2) so equal weights balance the gms automatically
    "seeds": [27],                      // swept dimension -> seed-averaging
    "deterministic": true,              // stricter reproducibility
    "mixed_init": "per_activation",     // xavier | per_activation (mixed layers)
    "max_workers": 16
  },
  "EXPERIMENTS": {
    "<experiment_name>": {
      "base_configs": { "<name>": { ...base_config... } },
      // ...any DEFAULTS key can be overridden here...
      "opt_params_variants": [ ... ]    // only for use_opt_params=true (see below)
    }
  }
}
```

> **Not exhaustive.** `build_tasks()` in `multi_experiment_runner.py` reads roughly a dozen more
> keys not shown above — swept: `gm_max_ratios`, `knee_vgs_thrs`, `gm_vds_mins`, `gm_vgs_mins`,
> `ids_region_weights`, `region_weights`, `vds_losses`, `ids_out_margins`,
> `vdsgate_output_activations`; scalar/global: `gm_warmup_epochs`, `gm_warmup_lr`,
> `ids_constraint`, `ids_target`, `ids_lambda`, `knee_vgs_tau`, `knee_max_correction`,
> `add_zero_vds`, `ids_region_center/width/lo/hi`, `knee_lr_scale`, `region_vgs_lo/hi`,
> `region_vds_lo/hi`, `adamw_avoid_localmin`. Several show up in [`gen_config_from_rows.py`](#post-processing-compile-rank-plot)'s
> `--set` table below; check `build_tasks()` directly (or the verification one-liner just below)
> if you need the full, current list.

**`base_config` fields** (one per entry in `base_configs`):

| field | meaning |
|---|---|
| `equation_type` | `"pure"` (NN only) or `"noNN_knee:<eq>"` (physics+NN). `<eq>` ∈ `mod1_angelov`, `classic_angelov`, `angelov_6_term`, `angelov_9_term`. |
| `use_gm` | `true` enables gm gradient-matching loss (uses `gm*_weights`, `gm_surgery_modes`). |
| `use_opt_params` | `true` initializes physics params from an SLSQP seed **and constrains each to a tight ±10% prior box around its seed value** (`width_percent=0.10`, sigmoid-bounded so a param can never leave the box; floor `δ=1e-6` for near-zero params). Requires `opt_params_variants` or `--opt_params_path`. `false` uses wide default bounds. See [§5 Tight priors](#tight-priors-from-slsqp-seeds-use_opt_params--freeze_physics). |
| `freeze_physics` | `true` **locks** physics params exactly at the seed values → NN-only training. `false` lets them train **within the ±10% box**. Empirically trained > frozen for gm (see [EXPERIMENT_LOG](EXPERIMENT_LOG.md) R7). Implementation note: for the `noNN_knee:<eq>` models used throughout this doc, a frozen param is stored as a plain Python float (never wrapped as an `nn.Parameter`, so there's no `requires_grad` to set) — `requires_grad=False` is only literally what happens in the separate hybrid-PINN code path. Either way the param does not train. |

**`opt_params_variants`** (only when `use_opt_params=true`) — fans the experiment
out over several seeds; each is a separate run:
```jsonc
"opt_params_variants": [
  { "label": "mod1",    "path": "../tests/slsqp_mod1_physics_seed_cfg9.json",
    "equation_type": "noNN_knee:mod1_angelov" },
  { "label": "classic", "path": "../tests/slsqp_classic_physics_seed_cfg9.json",
    "equation_type": "noNN_knee:classic_angelov" }
]
```
Relative `path`s resolve against `physics_nn_pipeline/`.

### How many configs does a sweep produce?

Per experiment it's the **Cartesian product**:
```
knee_alpha_scales × knee_combiners × learning_rates × nn_architectures ×
output_activations × gm1_weights × gm2_weights × gm3_weights ×
gm_surgery_modes × gm_max_ratios × knee_vgs_thrs × gm_vds_mins × gm_vgs_mins ×
ids_region_weights × region_weights × vds_losses × ids_out_margins ×
vdsgate_output_activations × seeds × opt_params_variants × base_configs
```
(the last 9 dimensions before `seeds` all default to a single-element list, so they're
invisible unless you actually sweep them — that's the common case shown in the schema above.)
Check before launching:
```bash
python -c "import json,multi_experiment_runner as m;from types import SimpleNamespace as S; \
cfg=json.load(open('physics_nn_configs/opt_configs.json')); a=S(min_vgs=-4.0,plot_vds_list=None,plot_vgs_list=None); \
print(sum(len(m.build_tasks(n,c,cfg['DEFAULTS'],'/tmp/x','x','x.csv',a)[0]) for n,c in cfg['EXPERIMENTS'].items()))"
```
> `nn_architectures: "@GENERATE@"` expands to **316 architectures** — combined
> with the other dimensions this easily reaches **hundreds of thousands** of
> configs (months of compute). See [§8](#8-recommended-workflow-how-to-actually-search).

---

## 4. The SLSQP runner (`slsqp_experiment_runner.py`)

Fits the **physics-only** Angelov model (no NN) and writes a `slsqp_seed.json`
per run, then auto-plots it via `plot_saved_state.py --seed`.

### CLI
| flag | default | meaning |
|---|---|---|
| `--config` | `config.json` | sweep JSON |
| `--master_root_path` | built-in | output root |
| `--max_workers` | (config) | global worker override |
| `--only` | all | run only the named experiments |
| `--min_vgs` | `-4.0` | extrapolation floor |
| `--plot_vds_list`/`--plot_vgs_list` | preset | plot targets |

> The CSV is **not** a CLI flag here — it comes from the config’s
> `GLOBAL_DEFAULTS.csv` (or the runner’s auto-detected default).

### Config schema
Top-level keys are experiments (everything except `GLOBAL_DEFAULTS`):
```jsonc
{
  "GLOBAL_DEFAULTS": { "csv": "...", "test_percent": 0.0, "timeout": 14400, "max_workers": 10 },
  "<experiment>": {
    "eq_names":        ["mod1_angelov"],           // mod1_angelov|classic_angelov|angelov_6_term|angelov_9_term
    "config_keys":     [9],
    "n_restarts_list": [5],                          // random SLSQP restarts (best kept)
    "gm_modes":        [["0", 0.0, 0.0, 0.0]]        // [use_gm_spec, gm1_w, gm2_w, gm3_w]
  }
}
```
`gm_modes` spec: `"0"` = off, `"1"` = gm1, `"1,2"`, `"1,2,3"`. **Do not** put a
`_comment` key in this config — every top-level key is treated as an experiment.

> `test_percent: 0.0` (no validation split) is supported: the fit falls back to
> the training data for best-restart selection.

---

## 5. Feature reference

### `seeds` — seed-averaging
`seeds` is a swept dimension. `[27]` = single fixed seed (reproducible).
`[1,2,3,4,5]` runs every config 5× (one per seed) so you can average out
init-luck — **essential** when comparing things that differ by <~15%. Each run
dir is suffixed `_s{seed}`.

### `deterministic` — strict reproducibility
`true` forwards `--deterministic` to each worker:
`torch.use_deterministic_algorithms(True, warn_only=True)` + single-threaded
torch. The runner already pins BLAS threads (`OMP/MKL/OPENBLAS_NUM_THREADS=1`),
so on CPU this mostly adds a safety net (it **warns** on a non-deterministic op
rather than aborting). Same seed + same config ⇒ identical result on the same
machine.

### `mixed_init` — weight initialization for mixed-activation layers
The NN uses **matched initialization** keyed to each layer’s activation
(gains = `1/RMS(act(z)), z~N(0,1)`):
- **Homogeneous layer** (one activation): Kaiming for relu-like, Xavier
  otherwise, scaled by that activation’s gain. Always on.
- **Mixed layer** (multiple activations in one layer): controlled by `mixed_init`:
  - `"xavier"` — one Xavier(gain=1) for the whole matrix.
  - `"per_activation"` — each neuron’s row scaled by **its own** activation gain.
    (Empirically the better default for gm; production default.)

Init only sets the *starting* weights — the forward pass and the extracted
equation are unchanged.

### Activations
Supported in architecture strings: `tanh, sin, cos, sinh, cosh, sigmoid,
softplus, swish, mish, relu, linear`. Avoid `sinh`/`cosh` as hidden activations
(exponential tails). `sin` gives clean derivatives (good for gm) and a clean
symbolic equation.

### Architecture string format
`[[<layer-1 activations>], [<layer-2 activations>], ...]`, one activation **per
neuron**. Examples:
- `[["tanh","tanh"],["tanh","tanh"]]` — 2 layers × 2 tanh neurons (homogeneous).
- `[["tanh","sin","swish"]]` — 1 hidden layer, 3 neurons, mixed.
- `"@GENERATE@"` — auto-generate ~316 small architectures (use sparingly).

### Tight priors from SLSQP seeds (`use_opt_params` / `freeze_physics`)
When `use_opt_params: true`, each physics parameter is initialized from the
SLSQP seed and given a **tight prior box** built around its optimized value
`v`: `delta = |v| · width_percent` (default `width_percent = 0.10` → **±10%**;
floored at `1e-6` so params that optimized to ~0 still have a sliver of room).
The box `[v−delta, v+delta]` is enforced through a sigmoid reparametrization, so
during training a param can only *refine within ±10%* — it can never drift away
from the physically-meaningful SLSQP fit. Two regimes:
- **`freeze_physics: false`** (trained): params move freely inside the ±10% box.
- **`freeze_physics: true`** (frozen): params are locked at `v` (`requires_grad=False`);
  only the NN trains.

`width_percent` is currently **hardcoded** at `0.10`
(`per_neuron_simple_angelov_nn_test.py`, in the `get_physics_config(...)` call) —
it is not yet a CLI/config knob. Empirically (EXPERIMENT_LOG R7) **trained beats
frozen** for gm on the good equation families.

### gm losses & gradient surgery (single- and multi-gm)
`use_gm: true` adds `gmN_weight · MSE(gmN_pred, gmN_true)` to the Ids loss for
each gm whose weight is > 0. gm targets are Savitzky–Golay-smoothed empirical
derivatives; the model's gm is its exact autograd derivative (`gm2`/`gm3` are
2nd/3rd-order autograd — progressively more expensive).

**Multi-gm:** you can penalize **several gms at once** — set any combination of
`gm1_weights`, `gm2_weights`, `gm3_weights` non-zero in the same run. The trainer
builds a *list* of gm losses, and the surgery step projects **each** gm gradient
against the base Ids gradient (and combines them), so PCGrad operates on the full
multi-objective set, not just one gm. This is the setting where gradient surgery
matters most (multiple conflicting derivative objectives).

**`gm_surgery_modes`** — how each gm gradient is combined with the base Ids
gradient (`apply_gradient_surgery`):
| mode | behavior |
|---|---|
| `none` | plain weighted sum (`base + Σ gmN`); no projection. The baseline. |
| `no-bounded` | **canonical PCGrad** — if a gm gradient conflicts (dot < 0) with base, remove the conflicting component. |
| `mag-bounded` | PCGrad projection + cap gm-grad magnitude at `base_norm · gm_max_ratio`. |
| `soft` | scale each gm grad by `clamp(cosine_sim, 0)` (down-weight conflict), then cap. |
| `element-wise` | zero only the individual gm-grad elements that conflict with base. |
| `element-wise-percent` | element-wise + scale surviving grad by `gm_max_ratio`. |
| `element-wise-bounded` | element-wise zero-conflict + clamp each element to `±|base|·gm_max_ratio`. |
| `element-wise-soft` | per-element sigmoid alignment weight + clamp. |
| `drop-conflict` | drop the *entire* gm grad if it conflicts with base. |

**`gm_max_ratio`** (config `gm_max_ratios`, swept; CLI `--gm_max_ratio`, default 1.0)
is used **three ways** depending on mode:
- **multiplier** (amplifies gm): `element-wise-percent` does `g_gm × ratio`, so
  `ratio > 1` pushes gm *past* its natural magnitude. At `ratio = 1.0` it is a no-op,
  so `element-wise-percent` == `element-wise` — which is why it looked inert until now.
- **cap/ceiling**: `mag-bounded`, `soft` (norm cap `‖g_gm‖ ≤ base_norm·ratio`),
  `element-wise-bounded`, `element-wise-soft` (per-element cap `|g| ≤ |base|·ratio`).
  `ratio > 1` only *un-caps* — it lets gm reach, but never exceed, its natural value.
- **ignored**: `no-bounded`, `element-wise`, `drop-conflict` (purely directional);
  and `none` (except it scales the gm *loss* when ratio < 1.0).

Empirically: PCGrad is **marginal for pure NN** (helps gm3 via `soft`) but
**helps in physics+NN** (`no-bounded`); on the multi-gm triple it is decisive
(−17% / prevents divergence) — see [EXPERIMENT_LOG](EXPERIMENT_LOG.md).

### L-BFGS polish (`lbfgs_epochs`, `lbfgs_max_iter`, `lbfgs_gm_aware`)
After AdamW, an L-BFGS pass polishes the model. All three controls are config keys
(previously the step count and `max_iter` were hardcoded at 5 / 200):
- `lbfgs_epochs` — number of outer L-BFGS steps.
- `lbfgs_max_iter` — `max_iter` per step (strong-Wolfe line search).
- `lbfgs_gm_aware` — when on **and** `use_gm`, the L-BFGS closure optimizes the
  **plain combined Ids + gm loss** (`Ids + Σ gmN_weight·gm_loss`), so extra L-BFGS
  epochs *sharpen gm* rather than trading it away for lower Ids. **Default is
  `null` = AUTO**: gm-aware turns ON automatically whenever `use_gm`. Set `false`
  (CLI `--no-lbfgs_gm_aware`) for the legacy Ids-only polish, or `true` to force it on.
  **Gradient surgery (`gm_surgery_mode` / `gm_max_ratio`) is applied ONLY in the AdamW
  phase**, never inside L-BFGS: a surgically-modified gradient (projection/clamping)
  is inconsistent with the loss the closure returns and makes strong-Wolfe thrash/hang
  on the bounded modes. L-BFGS polishing the *plain* combined loss uses the true
  gradient, so the line search stays consistent for every surgery mode.

Regardless of `lbfgs_gm_aware`, **best-weights are selected on the combined objective**
(`Ids + Σ gmN_weight·gm_loss`) whenever `use_gm` — so the polish can never silently
discard gm for a lower Ids. (For non-gm runs everything reduces to the old Ids-only behavior.)

### Loss normalization (`loss_norm`) — auto-balancing the gms
gm signals differ wildly in scale (gm1 ~0.05, gm2 ~0.2, gm3 ~1.0), so the **absolute**
MSE of gm3 dwarfs gm1's and dominates training unless you hand-tune the per-gm weights.
`loss_norm: "nmse"` removes that manual treadmill: each term's MSE (Ids **and** every gm)
is divided by its target's **mean-square**, `mean(target²)`, making all terms
dimensionless and **scale-balanced**. Then equal nominal weights contribute equally and
the `gm*_weights` become pure *priorities* rather than scale-correction factors.

- Normalization uses a **global per-signal scale** (a constant), never a per-point
  `(pred−target)/target` — gm2/gm3 cross zero, so per-point relative error would blow up.
- The Ids term is normalized too, so the tiny absolute Ids MSE isn't swamped by the gms.
- **Reported** `ids_rmse` / `gm*_rmse` stay in **absolute** units (computed from residuals,
  not the loss), so all cross-round comparisons remain valid — only the *training* loss
  and the combined selection objective are normalized.
- `loss_norm: "none"` (default) keeps the legacy absolute MSE.

---

## 6. Output layout & `run_loss_*.json` fields

```
<master_root>/
  base_files/                            # snapshot of the CSV + config used (reproducibility)
  _run_meta.json                        # records the measurement CSV / config paths etc.
  <experiment_name>/
    multi_experiment_runner.py            # copy of the runner used
    _provenance.json                      # lineage: which base experiment/row this was derived from, if any
    exp_001_<surgery>_W<g1>-<g2>-<g3>_s<seed>_rw<region_weight>/
      run_loss_<loss>_<tag>.json          # metrics + full config
      weights_loss_<loss>_<tag>.pt        # torch state_dict
      run_log.txt.gz                       # gzipped stdout/stderr of the run
```
(the `_rw<region_weight>` suffix exists so runs differing only in `region_weight` don't collide
in the same directory.)

Key `run_loss_*.json` fields:

| field | meaning |
|---|---|
| `ids_rmse` | **true Ids RMSE** = `sqrt(mean((pred-y)^2))`, computed directly (loss-agnostic). |
| `mse_loss` | true Ids MSE (`ids_rmse^2`). |
| `objective_loss` | the value the optimizer minimized. Plain Ids MSE for non-gm runs; **when `use_gm` is on, it's the combined Ids + Σ gmN_weight·gm_loss objective**, not Ids MSE alone. |
| `gm1_rmse`,`gm2_rmse`,`gm3_rmse` | RMSE of model gm vs smoothed empirical gm. |
| `seed`, `deterministic`, `mixed_init` | reproducibility settings used. |
| `architecture`, `output_activation`, `learning_rate`, `epochs` | NN config. |
| `equation_type`, `knee_combiner`, `knee_alpha_scale`, `freeze_physics` | physics config. |
| `use_opt_params`, `opt_params_path` | seed used (lets `plot_saved_state` rebuild the exact tight-prior config). |
| `use_gm`, `gm1_weight`, `gm_surgery_mode` | gm-loss settings. |
| `weights_path` | absolute path to the `.pt`. |

---

## 7. Post-processing: compile, rank, plot

### `compile_results_csv.py` — runs → CSV
```bash
python compile_results_csv.py --dir  <experiment_dir>          # one CSV for one experiment
python compile_results_csv.py --root <master_root>             # one CSV per immediate subdir
python compile_results_csv.py --root <root> --sort_by ids_rmse,gm1_rmse
```
Recursively scans `**/*.json` (skips `best_n_configs/`, and for `--root` also
`plotted_configs/`, `base_files/`, and any `ranked_*` folder). **`--dir` writes plain
`compiled_results.csv`; `--root` writes `compiled_results_<experiment>.csv` inside each
subdir** — the two modes name their output differently, so check which one you ran.
Also handles SLSQP `{"results": [...]}` seed files, not just `run_loss_*.json`.

Columns are far more numerous than a curated highlight list can usefully show — beyond
`best_loss, ids_rmse, gm1_rmse, gm2_rmse, gm3_rmse, config_name, architecture, lr,
output_activation, gm_surgery_mode, file_path`, the real output also includes provenance/lineage
(`id, run_id, run_hash, arch_id, arch_hash, base_exp, base_id, base_csv`), every physics/knee/gm
config field the runner accepts (`use_gm, gm*_weight, gm_max_ratio, freeze_physics,
use_opt_params, knee_*, ids_constraint/target/lambda, gm_warmup_*, gm_vds_min, gm_vgs_min,
ids_region_*, deterministic, mixed_init, vds_loss, lbfgs_*, loss_norm, csv, epochs, seed`), and a
union of any `region_*` columns present. Check the script's own `--help`/source for the exact,
current column list rather than treating this doc as authoritative.

### `compile_master_best.py` — best-N across experiments
```bash
python compile_master_best.py --root <master_root> --metric gm1_rmse --top_n 10
python compile_master_best.py --root <root> --sort_by ids_rmse,gm1_rmse --top_n 5
```
Reads `compiled_results*.csv` from each subdir, keeps the top-N per experiment,
writes `master_best_<sort>_top<N>_<root>.csv` with an `experiment` column.

### `compile_results_to_xlsx.py` — styled workbook
```bash
python compile_results_to_xlsx.py --root <master_root>     # all CSVs under root
python compile_results_to_xlsx.py --csv  <one>.csv
```
(Needs `openpyxl`.)

### `compute_region_metrics.py` — region-localized Ids/gm error → into the JSONs

```bash
# one run
python compute_region_metrics.py --dir <run_dir> --csv ../csvs/<m>.csv \
    --region "knee:vgs=-3..0,vds=0..10"

# every run under a master root, two regions
python compute_region_metrics.py --root <master_root> --csv ../csvs/<m>.csv \
    --region "knee:vgs=-3..0,vds=0..10" --region "sat:vgs=-1..0,vds=15..28"
```

Loads each run's optimized weights (same mechanism as `plot_saved_state.py`), evaluates
on the measurement set, and computes Ids + gm RMSE/MAE **restricted to one or more
(Vgs, Vds) windows**.  Run it **after training, before compile** — it writes the metrics
back into each `run_loss_*.json`, and `compile_results_csv.py` / `compile_ranked.py` then
expose them as columns automatically (union across runs; blank for runs not processed).
Works for every output head, including the `vdsgate*` structured wrappers (it reuses the
model loader + autograd gm).

Why: global Ids RMSE can rank a wiggly-but-low-error curve above a smooth one.  A region
metric isolates error where physical behavior matters (e.g. the turn-on knee).  The
**gm2/gm3 region RMSE is the sharpest "little bump" detector** — a wiggle is a curvature
spike — so ranking by `region_<name>_gm2_rmse` surfaces smooth-knee models even when their
global Ids RMSE is similar.

- **`--region`** (repeatable): `name:vgs=lo..hi,vds=lo..hi`.  Name optional (auto `r1`,
  `r2`).  Range separator may be `..`, `to`, or `:` (so `vgs=-3 to 0` works).
- Writes nested `region_metrics[name] = {vgs, vds, n, ids_rmse, ids_mae, gm1_rmse,
  gm2_rmse, gm3_rmse}` plus flat `region_<name>_<metric>` keys (the columns compile reads).
- Flags: `--add_zero_vds` (match training), `--min_vgs`, `--no_gm` (Ids only, faster),
  `--dry_run`.  Measurement data + gm-truth are computed once and reused across all runs.
- Downstream: `compile_ranked.py` auto-detects the region RMSE columns and writes a
  ranked CSV per metric, **grouped into folders** under the root: whole-curve metrics in
  `ranked_global/`, each region's in `ranked_region_<name>/` (e.g.
  `ranked_region_knee/<root>_ranked_by_region_knee_gm2_rmse.csv` = smoothest-knee first).
  Pick exactly which metrics with `compile_ranked.py --metrics region_knee_gm2_rmse,ids_rmse`.
  You can also sort in Excel, `compile_results_csv.py --sort_by region_<name>_gm2_rmse`, or
  feed the columns to `gen_config_from_rows.py --filter`/`--top` to seed the next round.

### `filter_results.py` — filter the compiled CSVs by metric thresholds

```bash
# filter every ranked CSV under a root (ranked_global/, ranked_region_*/), + xlsx
python filter_results.py --root base_arch \
    --filter combined_gm<0.8 --filter region_knee_combined_gm<0.8 --xlsx

# one CSV
python filter_results.py --csv base_arch/ranked_region_knee/base_arch_ranked_by_region_knee_combined_gm.csv \
    --filter region_knee_ids_rmse<0.01
```

Post-compile row filter. It reads the **CSVs** produced by `compile_ranked.py` and writes a
`*_filtered.csv` next to each (and `*_filtered.xlsx` with `--xlsx`). It **never reads or
modifies any `run_loss_*.json`** — filtering is pure row selection on the compiled output,
so every column is already present (global `combined_gm`/`ids_rmse`/`gm*` *and* any
`region_*`). Order: `compute_region_metrics` → `compile_ranked` → **`filter_results`**.

- `--filter COL<op>VAL` (repeatable, AND; op in `=,<,>,<=,>=`). Inequalities are numeric
  (blank/non-numeric cell → dropped). `=` also works on **text** columns, e.g.
  `--filter output_activation=softplus --filter gm_surgery_mode=none` (case-insensitive).
- `--root` prefers the grouped `ranked_*/` subfolders; a CSV that lacks a filter column
  (e.g. a global ranked CSV filtered on `region_knee_*`, or a stale pre-region CSV) is
  **skipped with a warning**, not an error.
- `--xlsx` converts each filtered CSV via `compile_results_to_xlsx.py`; `--suffix` changes
  the `_filtered` output suffix.

### `gen_config_from_rows.py` — generate a runner config from ranked CSV rows

```bash
python gen_config_from_rows.py \
    --csv  refine6/refine6_ranked_by_ids_rmse.csv \
    --ids  1-40 \
    --set  add_zero_vds=true \
    --out  physics_nn_configs/opt_configs_refine8.json \
    --prefix refine8 \
    --max_workers 42
```

Reads a compiled/ranked CSV, selects rows by `--ids`, loads each run's JSON for
the full hyperparameter set, groups rows by their unique non-architecture params,
and writes a ready-to-run config JSON.

**`--ids`** — which rows to include (by the `id` column):

| form | meaning |
|---|---|
| `1` | single row |
| `1-40` | inclusive range |
| `1,3,7` | explicit list |
| `all` | every row |

**`--set KEY=VALUE`** (repeatable, JSON-parsed value; `VALUE` can also be `@path/to/file.json`
to load a large value, e.g. a big `nn_architectures` list, from a file) — **every `--set` is a
flat override, and it *replaces*, it does not merge or add to, the source rows' values.**
(An earlier design intended additive "expansion" for the list-typed dimensions below — the
docstring language survives from that — but that is not what the current code does.)

For the keys in the table below, overriding the value also changes how source rows are
*grouped*: rows that differed only in that field now collapse into a single group (since the
field is dropped from the group's identity key), and the group's generated experiment gets
`VALUE` as its swept list wholesale — **the original per-row values for that field are
discarded, not kept alongside the new ones.** So `--set gm_vds_mins=[2.6,3.0,3.4,3.8,4.0]` does
not add these five values to whatever `gm_vds_min` values the selected rows had — it sets the
experiment's `gm_vds_min` sweep to exactly those five, full stop.

| `--set` key | groups by dropping |
|---|---|
| `gm_vds_mins=[2.6,3.0,3.4,3.8,4.0]` | `gm_vds_min` per config |
| `seeds=[27,42]` | `seed` |
| `gm1_weights=[0.1,0.3]` | `gm1_weight` |
| `gm2_weights=[0.1]` | `gm2_weight` |
| `gm3_weights=[0.0]` | `gm3_weight` |
| `gm_surgery_modes=[...]` | `gm_surgery_mode` |
| `learning_rates=[...]` | `lr` |
| `gm_max_ratios=[...]` | `gm_max_ratio` |
| `knee_alpha_scales=[...]` | `knee_alpha_scale` |
| `knee_combiners=[...]` | `knee_combiner` |
| `output_activations=[...]` | `output_activation` |

All other keys (`add_zero_vds`, `epochs`, `lbfgs_epochs`, `loss_norm`, `ids_constraint`,
`max_workers`, …) are simple global overrides applied as-is to every generated experiment, with
no grouping effect. Two special cases: `--set equation_type=X` (or
`equation_types=[X,Y]`) fans out into one experiment per equation type rather than overriding in
place; `use_gm`/`equation_type`/`use_opt_params` route into each experiment's `base_configs`
rather than its top-level dict.

**`--prefix`** sets the experiment-name prefix, which also becomes the subdirectory
name under `--master_root_path` when the runner executes (e.g. `refine8__w0p1-0p1-0__ewbounded`).

**`--has_gm`** drops rows where all three gm weights are 0.

If the source CSV has a `gm_vds_min` column (present in ranked CSVs produced by
refine6+), the value is read per-row and included in the group key — so each
unique `(gm_weights, surgery_mode, gm_vds_min)` becomes its own experiment rather
than a single experiment sweeping all values.

### `plot_csv_row.py` — re-plot specific rows from a ranked CSV

```bash
python plot_csv_row.py --ranked_csv refine6/refine6_ranked_by_ids_rmse.csv \
    --csv ../csvs/<measurements>.csv --id 1
python plot_csv_row.py --ranked_csv ... --csv ... --id 1,2,3
python plot_csv_row.py --ranked_csv ... --csv ... --row 5
python plot_csv_row.py --ranked_csv ... --csv ... --config_name PhysNN_W1.0-1.0-1.0_none
```

Selects one or more rows by `--id` (the stored `id` column — survives Excel
re-sorting), `--row` (current physical row number, 1-based), or `--config_name`,
then calls `plot_saved_state.py` for each.  Copies the resulting
`plot_saved_state_full.png`, `plot_saved_state_full_eq_comparison.png`,
`plot_saved_state_full_val.png`, and the run JSON into
`<ranked_csv_dir>/plotted_configs/<csv_stem>_id<N>/`.

Optional flags forwarded to `plot_saved_state.py`:
- `--val interpolation` / `--val 0.3` / `--val <path>` — validation plot
- `--add_zero_vds` — pin each TN curve to the origin by overwriting its first (lowest-Vds) row to Vds=0, Ids=0 (in place, no row added); mirrors the training-time flag
- `--plot_vds_list` / `--plot_vgs_list` — voltage axes

### `plot_best_configs.py` — re-plot winners + collect them
```bash
python plot_best_configs.py --dir <experiment_dir> --csv ../csvs/<...>.csv \
    --top 5 --sort_by gm1_rmse --regen --min_vgs=-4.0
```
Ranks the experiment’s runs, re-runs `plot_saved_state.py --dir` for the top-N
(produces `plot_saved_state_full.png` in each run dir), **and copies each
winner’s files (plot + json + weights + log) into
`<experiment_dir>/best_n_configs/<rank>_<run_dir>/`**. Flags: `--regen`
(rebuild the CSV first), `--no_best_copy`, `--dry_run`.

### `plot_saved_state.py` — 6-panel plot from one run or seed
```bash
python plot_saved_state.py --dir  <run_dir>      --csv ../csvs/<...>.csv --min_vgs=-4.0
python plot_saved_state.py --seed <slsqp_seed>.json --eq_name mod1_angelov --config_key 9 --csv ...
```
Rebuilds the exact model (reads knee config + `opt_params_path` from the JSON),
prints a **calculated-vs-saved RMSE check** (`OK`/`MISMATCH` per metric), and
fills the plot’s info box with both the saved and recomputed RMSEs.

---

## 8. Recommended workflow (how to actually search)

The Cartesian sweep explodes fast. **Do not** point a runner at a full
`@GENERATE@` × all-dimensions grid (≈10⁵–10⁶ configs = months). Instead, search
**coarse-to-fine**, one question at a time, in small batches:

1. **Isolate one variable per round.** Hold everything else fixed; only sweep the
   thing you’re studying (init mode, then architecture, then lr, then gm weights…).
2. **Seed-average** (`seeds: [1,2,3,4,5]`). A single seed cannot resolve effects
   below ~15% — they drown in init noise.
3. **Keep epochs modest for ranking** (e.g. 400–600). You need *relative* order,
   not converged models. Refine the top few later with more epochs.
4. **Rank by the right metric** in `compile_master_best` (`gm1_rmse` for gm goals,
   not `best_loss`).
5. **Architecture dominates.** Empirically, architecture changes gm by ~10×;
   init/lr tweaks are ~10%. Spend the budget on architecture first.
6. **Pure NN first, then NN+physics.** Settle architecture/init on cheap pure-NN
   runs, then bring the winners to the physics+NN path.

A round is typically **tens to a few hundred** configs, finishing in minutes.

---

## 9. Reproducibility

- Each worker seeds `torch` + `numpy` from `--seed` (config `seeds`) **before**
  building the model. Same seed + same config ⇒ same result on the same machine.
- The runner pins BLAS to one thread per worker (fixed reduction order).
- `deterministic: true` adds `use_deterministic_algorithms(True, warn_only=True)`
  + single-threaded torch.
- **Cross-machine** bit-identity is *not* guaranteed (different BLAS / CPU /
  PyTorch build can reorder float ops); same-machine is solid.
- The seed (and `deterministic`, `mixed_init`) are logged in `run_loss_*.json`.

---

## 10. Tests

> **This section previously described a `test_overall_smoke.py`-orchestrated suite
> (`test_multi_runner_smoke.py`, `test_compile_results_csv_smoke.py`,
> `test_compile_results_to_xlsx_smoke.py`, `test_plot_best_configs_smoke.py`, and a
> `smoke_paths.py` path module) that does not exist in this repo — only 2 of those 7 files are
> actually present. What follows describes what's really here.**

```bash
python slsqp_scripts/test_slsqp_runner_smoke.py     # SLSQP runner -> seeds + fixtures. Works standalone.
python test_region_weight.py                        # unit tests for the region-weight loss math,
                                                      # with a source guard that fails loudly if
                                                      # per_neuron_simple_angelov_nn_test.py's
                                                      # formula changes underneath it
python test_runner_e2e.py                            # runs the REAL multi_experiment_runner.py end-to-end,
                                                      # one tiny experiment per feature (region weight,
                                                      # vds_loss, gm mask, surgery mode, ids-region, loss_norm,
                                                      # ...), checking both that each option is recorded in
                                                      # run_loss_*.json AND that the debug log shows it fired.
                                                      # Needs the real measurement CSV; skips if absent.
python test_compile_master_best_smoke.py             # currently BROKEN in this repo — imports a
                                                      # `smoke_paths` module and shells out to a
                                                      # `test_compile_results_csv_smoke.py`, neither of
                                                      # which exists here. Fix or remove before relying on it.
```
The SLSQP smoke is config-driven (`slsqp_configs/opt_configs_smoke.json`), exercising the same
code path as production.

---

## 11. Gotchas / FAQ

- **`--opt=value` for negative/comma args.** `--min_vgs=-4.0`,
  `--plot_vgs_list=-3.5,...` — the space form is parsed as a flag.
- **`@GENERATE@` is huge** (316 archs). Almost never what you want for a full grid.
- **`use_opt_params: true` needs seeds.** The SLSQP runner generates them; if the
  paths in `opt_params_variants` don’t exist, `build_tasks` raises `FileNotFoundError`.
- **SLSQP config: no `_comment` key** (every top-level key = an experiment).
- **gm targets are smoothed** (Savitzky–Golay); gm RMSE depends on the smoothing
  windows and assumes each `Step_Index` group is a monotonic Vgs sweep (guarded —
  it raises if not).
- **`ids_rmse` is loss-agnostic** (computed from residuals, not `sqrt(loss)`), so
  it stays correct if you change the loss function.
- **Reproducing an old artifact after code changes:** init/RNG changes mean
  newly-trained weights won’t match pre-change ones; the saved JSON/weights still
  reload and reconstruct fine.
- **Resuming:** a run dir that already contains `run_loss_*.json` + `weights_*.pt`
  is skipped, so re-launching a sweep only fills in missing configs.
```
