"""
Multi-Experiment NN Hyperparameter Sweep Runner
================================================

This script acts as a parallelized orchestrator to execute large hyperparameter sweeps
for the underlying neural network training script ('per_neuron_simple_angelov_nn_test.py').
It reads a single JSON configuration file, constructs a Cartesian product (sweep matrix) 
of all parameter variants, and distributes individual training executions across multiple 
CPU cores using a ProcessPoolExecutor.

Accepted CLI Arguments
----------------------
* --config (Required)
    Path to the JSON configuration file containing DEFAULTS and EXPERIMENTS definitions.
* --csv (Required)
    Path to the measurement CSV data file forwarded to every training worker.
* --master_root_path (Optional)
    Root folder where each experiment subdirectory is created. 
    Overrides defaults or environment variables ($MASTER_ROOT_PATH).
* --opt_params_path (Optional)
    Fallback path to the physics-parameters JSON seed file.
    Defaults to the path set in $OPT_PARAMS_PATH or internal relative fallback.
* --min_vgs (Optional, Default: -4.0)
    Minimum Vgs value cutoff passed downstream to the NN training script.
* --plot_vds_list (Optional, Default: "0,5,10,15,20,28")
    Comma-separated string of Vds values to target for plotting output.
* --plot_vgs_list (Optional, Default: "-3.5,-3,-2.5,-2,-1.5,-1,-.5,0")
    Comma-separated string of Vgs values to target for plotting output.

JSON Configuration Structure
----------------------------
The configuration JSON file must contain two top-level keys:
1. "DEFAULTS": Contains fallback parameter lists applied to all sweeps unless overridden.
2. "EXPERIMENTS": A dictionary of distinct named sweeps. Each named sweep can override 
   global defaults or introduce specialized configuration lists.

Example JSON (`sweep_config.json`):
```json
{
  "DEFAULTS": {
    "knee_alpha_scales": [1.0, 1.5],
    "knee_combiners": ["add"],
    "learning_rates": [0.001, 0.01],
    "nn_architectures": "@GENERATE@", 
    "output_activations": ["linear"],
    "gm1_weights": [1.0],
    "gm2_weights": [0.0],
    "gm3_weights": [0.0],
    "gm_surgery_modes": ["1"],
    "epochs": 150,
    "max_workers": 4,
    "base_configs": {
      "physics_enabled": {
        "freeze_physics": false,
        "use_opt_params": true,
        "use_gm": true
      }
    }
  },
  "EXPERIMENTS": {
    "architecture_and_lr_sweep": {
      "learning_rates": [0.001, 0.005, 0.01],
      "max_workers": 8
    }
  }
}
"""

import os
import sys
import gzip
import json
import argparse
import subprocess
import itertools
import shutil
from concurrent.futures import ProcessPoolExecutor

# ------------------------------------------------------------------
# Paths Handling (Supports absolute paths & relative to this script)
# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

def resolve_path(path_str):
    """Resolves path string. If relative, resolves it relative to this script."""
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(HERE, path_str))

# Portable absolute fallbacks
SCRIPTS_ROOT_DIR = os.environ.get(
    "SCRIPTS_ROOT_DIR",
    os.path.abspath(os.path.join(HERE, "..", "..")),
)
DEFAULT_MASTER_ROOT = os.environ.get(
    "MASTER_ROOT_PATH",
    os.path.join(SCRIPTS_ROOT_DIR, "master_experiments_dirs"),
)
SCRIPT_PATH = os.path.join(HERE, "per_neuron_simple_angelov_nn_test.py")
PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)
OPT_PARAMS_PATH = os.environ.get(
    "OPT_PARAMS_PATH",
    os.path.join(SCRIPTS_ROOT_DIR, "parallel_optimization", "tests", "slsqp_fast_physics_seed.json"),
)

# Architecture-generation configurations
_NON_TANH_ACTIVATIONS = ["sigmoid", "sin", "cos", "sinh", "cosh", "linear", "swish", "mish", "softplus"]
_ARCH_MIN_LAYERS = 1
_ARCH_MAX_LAYERS = 2
_ARCH_MIN_NEURONS_PER_LAYER = 2
_ARCH_MAX_NEURONS_PER_LAYER = 8
_ARCH_MAX_TOTAL_NEURONS = 8
_ARCH_MIN_TANH_PER_LAYER = 1

def _build_architectures():
    archs = []
    seen = set()

    def _add(arch):
        s = json.dumps(arch)
        if s not in seen:
            seen.add(s)
            archs.append(s)

    layer_counts = range(_ARCH_MIN_LAYERS, _ARCH_MAX_LAYERS + 1)

    # 1) Homogeneous tanh
    for n_layers in layer_counts:
        max_size_by_total = _ARCH_MAX_TOTAL_NEURONS // n_layers
        max_size = min(_ARCH_MAX_NEURONS_PER_LAYER, max_size_by_total)
        for size in range(_ARCH_MIN_NEURONS_PER_LAYER, max_size + 1):
            _add([["tanh"] * size] * n_layers)

    # 2) Heterogeneous
    for x in _NON_TANH_ACTIVATIONS:
        for n_layers in layer_counts:
            max_size_by_total = _ARCH_MAX_TOTAL_NEURONS // n_layers
            max_size = min(_ARCH_MAX_NEURONS_PER_LAYER, max_size_by_total)
            for size in range(_ARCH_MIN_NEURONS_PER_LAYER, max_size + 1):
                for n_tanh in range(_ARCH_MIN_TANH_PER_LAYER, size):
                    n_x = size - n_tanh
                    if n_x < 1:
                        continue
                    layer = ["tanh"] * n_tanh + [x] * n_x
                    _add([layer] * n_layers)
    return archs

# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------
def run_experiment(task):
    try:
        return _run_experiment_impl(task)
    except Exception:
        import traceback
        idx = task[-1] if isinstance(task, tuple) and task else "?"
        print(f"\n[{idx}] WORKER EXCEPTION:\n{traceback.format_exc()}\n", flush=True)
        return

def _run_experiment_impl(task):
    (master_dir, base_name, base_args, alpha_scale, combiner, lr,
     architecture_str, out_act, gm1_w, gm2_w, gm3_w, gm_mode, epochs,
     opt_params_path, csv_path, min_vgs, plot_vds_list, plot_vgs_list,
     seed, deterministic, mixed_init, gm_max_ratio,
     lbfgs_epochs, lbfgs_max_iter, lbfgs_gm_aware, loss_norm,
     gm_warmup_epochs, gm_warmup_lr, ids_constraint, ids_target, ids_lambda,
     gm_vds_min, gm_vgs_min, vgs_thr, knee_vgs_tau, knee_max_correction, add_zero_vds, ids_region, knee_lr_scale, region, vds_loss, index) = task

    print(f"\n[{index}] {os.path.basename(master_dir)} :: Starting {base_name} "
          f"| GMw=({gm1_w},{gm2_w},{gm3_w}) | Mode_{gm_mode} | LR_{lr} | seed_{seed} | Arch_{architecture_str}")

    exp_dir = os.path.join(master_dir, f"exp_{index:03d}_{gm_mode}_W{gm1_w}-{gm2_w}-{gm3_w}_s{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    existing = os.listdir(exp_dir)
    if any(f.startswith("run_loss_") for f in existing) and any(f.startswith("weights_loss_") for f in existing):
        print(f"[{index}] Skipping (already finished)")
        return

    cmd = [PYTHON_EXE, SCRIPT_PATH, "--csv", csv_path]
    if base_args.get("freeze_physics"):
        cmd.append("--freeze_physics")
    if base_args.get("use_opt_params"):
        cmd.extend(["--opt_params_path", opt_params_path])
    else:
        cmd.append("--no_opt_params")
    if "equation_type" in base_args:
        cmd.extend(["--equation_type", base_args["equation_type"]])
    if base_args.get("use_gm"):
        cmd.append("--use_gm")
        cmd.extend(["--gm_surgery_mode", gm_mode])
        cmd.extend(["--gm1_weight", str(gm1_w), "--gm2_weight", str(gm2_w), "--gm3_weight", str(gm3_w)])
        cmd.extend(["--gm_max_ratio", str(gm_max_ratio)])

    run_tag = f"{base_name}_W{gm1_w}-{gm2_w}-{gm3_w}_{gm_mode}"
    cmd.extend(["--config_name", run_tag, "--knee_alpha_scale", str(alpha_scale), "--knee_combiner", combiner])
    cmd.extend(["--lr", str(lr), "--architecture", architecture_str, "--output_activation", out_act, "--output_dir", exp_dir])
    cmd.extend(["--epochs", str(epochs), "--hide_plot", "--no_plot", "--no_script_copy"])
    cmd.extend(["--seed", str(seed)])
    if deterministic:
        cmd.append("--deterministic")
    cmd.extend(["--mixed_init", mixed_init])
    cmd.extend(["--lbfgs_epochs", str(lbfgs_epochs), "--lbfgs_max_iter", str(lbfgs_max_iter)])
    if lbfgs_gm_aware is True:
        cmd.append("--lbfgs_gm_aware")
    elif lbfgs_gm_aware is False:
        cmd.append("--no-lbfgs_gm_aware")
    # lbfgs_gm_aware None -> pass nothing; trainer auto-enables when use_gm
    cmd.extend(["--loss_norm", loss_norm])
    # Ids-preserving gm training (only emit flags when set, so defaults stay legacy)
    if gm_warmup_epochs:
        cmd.extend(["--gm_warmup_epochs", str(gm_warmup_epochs)])
    if gm_warmup_lr is not None:
        cmd.extend(["--gm_warmup_lr", str(gm_warmup_lr)])
    if ids_constraint:
        cmd.append("--ids_constraint")
    if ids_target is not None:
        cmd.extend(["--ids_target", str(ids_target), "--ids_lambda", str(ids_lambda)])
    if gm_vds_min > 0.0:
        cmd.extend(["--gm_vds_min", str(gm_vds_min)])
    if gm_vgs_min is not None:
        cmd.extend(["--gm_vgs_min", str(gm_vgs_min)])
    _ir_c, _ir_w, _ir_wt, _ir_lo, _ir_hi = ids_region
    _ir_band = _ir_lo is not None and _ir_hi is not None
    if _ir_wt and _ir_wt > 0.0 and (_ir_band or _ir_c is not None):
        cmd.extend(["--ids_region_weight", str(_ir_wt)])
        if _ir_band:
            cmd.extend(["--ids_region_lo", str(_ir_lo), "--ids_region_hi", str(_ir_hi)])
        else:
            cmd.extend(["--ids_region_center", str(_ir_c), "--ids_region_width", str(_ir_w)])
    if knee_lr_scale != 1.0:
        cmd.extend(["--knee_lr_scale", str(knee_lr_scale)])
    # 2-D box region weighting (Ids + gm). Emit only when weight>0 and a bound is set.
    _rg_vgs_lo, _rg_vgs_hi, _rg_vds_lo, _rg_vds_hi, _rg_wt = region
    if _rg_wt and _rg_wt > 0.0 and any(b is not None for b in (_rg_vgs_lo, _rg_vgs_hi, _rg_vds_lo, _rg_vds_hi)):
        cmd.extend(["--region_weight", str(_rg_wt)])
        if _rg_vgs_lo is not None: cmd.extend(["--region_vgs_lo", str(_rg_vgs_lo)])
        if _rg_vgs_hi is not None: cmd.extend(["--region_vgs_hi", str(_rg_vgs_hi)])
        if _rg_vds_lo is not None: cmd.extend(["--region_vds_lo", str(_rg_vds_lo)])
        if _rg_vds_hi is not None: cmd.extend(["--region_vds_hi", str(_rg_vds_hi)])
    # gds>=0 knee-bump monotonicity penalty (uses the region box above). Emit only when on.
    if vds_loss and vds_loss > 0.0:
        cmd.extend(["--vds_loss", str(vds_loss)])
    # Vgs window (residual combiner). thr None -> gate disabled (flag omitted).
    if vgs_thr is not None:
        cmd.extend(["--knee_vgs_thr", str(vgs_thr)])
    cmd.extend(["--knee_vgs_tau", str(knee_vgs_tau), "--knee_max_correction", str(knee_max_correction)])
    if add_zero_vds:
        cmd.append("--add_zero_vds")

    if min_vgs is not None:
        cmd.append(f"--min_vgs={min_vgs}")
    if plot_vds_list:
        cmd.append(f"--plot_vds_list={plot_vds_list}")
    if plot_vgs_list:
        cmd.append(f"--plot_vgs_list={plot_vgs_list}")

    run_env = os.environ.copy()
    run_env.update({"PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})

    # Per-config log is gzip-compressed on write (logs are highly compressible
    # and there is one per sweep cell; .gz keeps experiment dirs small).
    log_path = os.path.join(exp_dir, "run_log.txt.gz")
    try:
        result = subprocess.run(cmd, env=run_env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        try:
            with gzip.open(log_path, "wt", encoding="utf-8") as f:
                f.write(f"CMD: {cmd}\n\n[runner] subprocess.run raised:\n{err}")
        except Exception: pass
        print(f"[{index}] LAUNCH FAILED: {run_tag}\n{err}", flush=True)
        return

    try:
        with gzip.open(log_path, "wt", encoding="utf-8") as f:
            f.write(f"CMD: {cmd}\n\n--- returncode = {result.returncode} ---\n--- STDOUT ---\n"
                    f"{result.stdout or ''}\n--- STDERR ---\n{result.stderr or ''}")
    except Exception: pass

    if result.returncode != 0:
        print(f"\n[{index}] FAILED: {run_tag} (rc={result.returncode})\ncmd: {cmd}\nlog: {log_path}\n", flush=True)
    else:
        print(f"[{index}] Finished: {run_tag}", flush=True)

def _resolve_variants(exp_name, merged, fallback_opt_params_path):
    needs_opt = any(ba.get("use_opt_params") for ba in merged["base_configs"].values())
    if "opt_params_variants" in merged:
        variants = list(merged["opt_params_variants"])
    else:
        variants = [{"label": "", "path": merged.get("opt_params_path", fallback_opt_params_path)}]

    if needs_opt:
        for v in variants:
            p = v.get("path")
            if not p:
                raise ValueError(f"Experiment '{exp_name}': variant missing 'path'.")
            # Resolve paths dynamically relative to script location if relative
            v["path"] = resolve_path(p)
            if not os.path.exists(v["path"]):
                raise FileNotFoundError(f"Opt params path not found: {v['path']}")
    return variants

def build_tasks(exp_name, cfg, defaults, master_root, opt_params_path, csv_path, args):
    merged = {**defaults, **cfg}
    master_dir = os.path.join(master_root, exp_name)
    os.makedirs(master_dir, exist_ok=True)

    variants = _resolve_variants(exp_name, merged, opt_params_path)
    
    # Process dynamic architectures placeholder
    if merged.get("nn_architectures") == "@GENERATE@":
        merged["nn_architectures"] = _build_architectures()

    # `seeds` is a swept dimension (default [27]) so a config can be seed-averaged;
    # `deterministic` is a global toggle forwarded to every worker.
    seeds = merged.get("seeds", [27])
    deterministic = bool(merged.get("deterministic", False))
    mixed_init = merged.get("mixed_init", "xavier")  # mixed-layer init scheme (global per run)
    # `gm_max_ratios` is a swept dimension (default [1.0]) -> forwarded as
    # --gm_max_ratio. It caps the gm gradient magnitude relative to the base Ids
    # gradient in the bounded surgery modes (soft, mag-bounded, element-wise-bounded,
    # element-wise-soft); >1.0 lets gm push harder than base. Ignored by the
    # projection-only modes (no-bounded, element-wise, drop-conflict).
    gm_max_ratios = merged.get("gm_max_ratios", [1.0])
    # L-BFGS polish controls (scalars, per-experiment): outer steps, inner iters, and
    # whether the polish optimizes the combined Ids+gm objective (gm-aware) vs Ids-only.
    lbfgs_epochs = merged.get("lbfgs_epochs", 5)
    lbfgs_max_iter = merged.get("lbfgs_max_iter", 200)
    # None (unset) -> auto: the trainer turns gm-aware L-BFGS ON whenever use_gm.
    # Set true/false in the config to force on/off.
    lbfgs_gm_aware = merged.get("lbfgs_gm_aware", None)
    loss_norm = merged.get("loss_norm", "none")   # none | nmse (scale-balance the losses)
    # Ids-preserving gm training (scalars, per-experiment):
    #   gm_warmup_epochs  -> Ids-only AdamW epochs before gm enters (two-phase warm-start)
    #   gm_warmup_lr      -> LR for the post-warmup gm phase (None = keep lr)
    #   ids_constraint    -> epsilon-constraint loss (minimize gm s.t. Ids <= ids_target)
    #   ids_target        -> Ids RMSE ceiling (epsilon + best-weights selection cap)
    #   ids_lambda        -> constraint penalty strength
    gm_warmup_epochs = merged.get("gm_warmup_epochs", 0)
    gm_warmup_lr = merged.get("gm_warmup_lr", None)
    ids_constraint = bool(merged.get("ids_constraint", False))
    ids_target = merged.get("ids_target", None)
    ids_lambda = merged.get("ids_lambda", 100.0)
    # gm_vds_mins is a SWEPT dimension: each value excludes gm loss for Vds < that value.
    # Falls back to scalar gm_vds_min (legacy), then to [0.0] (all points, no masking).
    gm_vds_mins = merged.get("gm_vds_mins", [merged.get("gm_vds_min", 0.0)])
    # gm_vgs_mins is a SWEPT dimension: each value excludes gm loss for Vgs < that value.
    # Falls back to scalar gm_vgs_min, then to [None] (no Vgs masking). NOTE: 0.0 is a real
    # Vgs threshold, so the "off" sentinel is None (unlike gm_vds_min where 0 means off).
    gm_vgs_mins = merged.get("gm_vgs_mins", [merged.get("gm_vgs_min", None)])
    # Vgs window (residual combiner): knee_vgs_thrs is SWEPT (None = gate off); tau/max are scalars.
    knee_vgs_thrs = merged.get("knee_vgs_thrs", [None])
    knee_vgs_tau = merged.get("knee_vgs_tau", 0.3)
    knee_max_correction = merged.get("knee_max_correction", 1.0)
    add_zero_vds = bool(merged.get("add_zero_vds", False))
    # Per-region Ids-loss weighting. Region shape (center/width or lo/hi) is per-experiment;
    # ids_region_weights is a SWEPT dimension (e.g. [4,3,2]) -> one run per weight value.
    # Falls back to scalar ids_region_weight, then [0.0] (off).
    ids_region_center = merged.get("ids_region_center", None)
    ids_region_width  = merged.get("ids_region_width", 0.3)
    ids_region_lo     = merged.get("ids_region_lo", None)
    ids_region_hi     = merged.get("ids_region_hi", None)
    ids_region_weights = merged.get("ids_region_weights", [merged.get("ids_region_weight", 0.0)])
    knee_lr_scale = merged.get("knee_lr_scale", 1.0)  # LR multiplier for vdsgate knee params (<1 = slower)
    # 2-D (Vgs x Vds) box region weighting (Ids + gm). Box bounds are per-experiment scalars;
    # region_weights is a SWEPT dimension (e.g. [2,3]) -> one run per weight. Falls back to
    # scalar region_weight, then [0.0] (off). Any bound None = unbounded on that side.
    region_vgs_lo = merged.get("region_vgs_lo", None)
    region_vgs_hi = merged.get("region_vgs_hi", None)
    region_vds_lo = merged.get("region_vds_lo", None)
    region_vds_hi = merged.get("region_vds_hi", None)
    region_weights = merged.get("region_weights", [merged.get("region_weight", 0.0)])
    # gds>=0 knee-bump penalty weight. SWEPT dimension (e.g. [0,5]) -> one run per value.
    # Falls back to scalar vds_loss, then [0.0] (off).
    vds_losses = merged.get("vds_losses", [merged.get("vds_loss", 0.0)])
    sweep = list(itertools.product(
        merged["knee_alpha_scales"], merged["knee_combiners"], merged["learning_rates"],
        merged["nn_architectures"], merged["output_activations"], merged["gm1_weights"],
        merged["gm2_weights"], merged["gm3_weights"], merged["gm_surgery_modes"],
        seeds, gm_max_ratios, knee_vgs_thrs, gm_vds_mins, gm_vgs_mins, ids_region_weights,
        region_weights, vds_losses,
    ))

    tasks = []
    idx = 1
    for variant in variants:
        v_label, v_path, v_eqtype = variant.get("label", ""), variant.get("path"), variant.get("equation_type")
        for base_name, base_args in merged["base_configs"].items():
            eff_base_args = {**base_args, "equation_type": v_eqtype} if v_eqtype is not None else base_args
            eff_base_name = f"{base_name}_{v_label}" if v_label else base_name
            for (alpha, combiner, lr, arch, out_act, w1, w2, w3, mode, seed, gmr, vgs_thr, vds_min, vgs_min, ids_rw, region_wt, vds_loss) in sweep:
                ids_region = (ids_region_center, ids_region_width, ids_rw, ids_region_lo, ids_region_hi)
                region = (region_vgs_lo, region_vgs_hi, region_vds_lo, region_vds_hi, region_wt)
                tasks.append((master_dir, eff_base_name, eff_base_args, alpha, combiner, lr, arch, out_act,
                              w1, w2, w3, mode, merged["epochs"], v_path, csv_path, args.min_vgs,
                              args.plot_vds_list, args.plot_vgs_list, seed, deterministic, mixed_init, gmr,
                              lbfgs_epochs, lbfgs_max_iter, lbfgs_gm_aware, loss_norm,
                              gm_warmup_epochs, gm_warmup_lr, ids_constraint, ids_target, ids_lambda,
                              vds_min, vgs_min, vgs_thr, knee_vgs_tau, knee_max_correction, add_zero_vds, ids_region, knee_lr_scale, region, vds_loss, idx))
                idx += 1
    return tasks, master_dir, merged["max_workers"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-experiment NN sweep.")
    parser.add_argument("--config", required=True, nargs="+",
                        help="Path(s) to JSON config file(s). Multiple configs are loaded "
                             "and all their experiments run in parallel under one shared pool.")
    parser.add_argument("--master_root_path", help="Root folder where each experiment subdir is created (Overrides config/env default).")
    parser.add_argument("--csv", required=True, help="Path to the measurement CSV forwarded to every worker.")
    parser.add_argument("--opt_params_path", default=OPT_PARAMS_PATH, help="Fallback path to physics-params JSON.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Override the worker-pool size for ALL configs combined. "
                             "Default: use the largest max_workers found across all experiments.")
    parser.add_argument("--min_vgs", type=float, default=-4.0)
    parser.add_argument("--plot_vds_list", type=str, default="0,5,10,15,20,28")
    parser.add_argument("--plot_vgs_list", type=str, default="-3.5,-3,-2.5,-2,-1.5,-1,-.5,0")
    args = parser.parse_args()

    # Finalize critical path resolution
    master_root = resolve_path(args.master_root_path if args.master_root_path else DEFAULT_MASTER_ROOT)
    opt_params_path = resolve_path(args.opt_params_path)
    csv_path = resolve_path(args.csv)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    os.makedirs(master_root, exist_ok=True)

    print(f"Master root: {master_root}")
    print(f"CSV path:    {csv_path}")

    # Collect tasks from all configs/experiments into one list so they all
    # run in parallel under a single shared pool (no waiting between configs).
    all_tasks = []
    pool_size = 4  # floor; raised by each experiment's max_workers below

    for config_path_str in args.config:
        config_file_path = resolve_path(config_path_str)
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        DEFAULTS = config_data.get("DEFAULTS", {})
        EXPERIMENTS = config_data.get("EXPERIMENTS", {})

        print(f"\nConfig: {config_file_path}  ({len(EXPERIMENTS)} experiments)")

        for exp_name, cfg in EXPERIMENTS.items():
            tasks, master_dir, max_workers = build_tasks(
                exp_name, cfg, DEFAULTS, master_root, opt_params_path, csv_path, args
            )
            shutil.copy2(__file__, os.path.join(master_dir, os.path.basename(__file__)))
            print(f"  {exp_name}: {len(tasks)} tasks  (config max_workers={max_workers})")
            all_tasks.extend(tasks)
            pool_size = max(pool_size, max_workers)

    # CLI --max_workers overrides the per-config values
    if args.max_workers is not None:
        pool_size = args.max_workers

    workers = min(os.cpu_count() or 4, pool_size)
    print(f"\nTotal tasks: {len(all_tasks)}  |  Pool size: {workers}")
    print("=" * 60)

    from concurrent.futures import as_completed
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_experiment, t) for t in all_tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                import traceback
                print(f"[runner] task raised:\n{traceback.format_exc()}", flush=True)

    print(f"\nAll done. {len(all_tasks)} tasks completed.")