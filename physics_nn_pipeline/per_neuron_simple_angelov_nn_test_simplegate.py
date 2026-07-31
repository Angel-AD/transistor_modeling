"""
Single-config trainer for the physics+NN pipeline.

Workhorse child script invoked once per task by `multi_experiment_runner.py`
(can also be run standalone). Loads a measurement CSV, builds either a pure
NN, a physics-only model, or a physics+NN hybrid (`PhysicsInformedNN`),
trains it with Adam, optionally adds gradient-matching (GM) losses on Gm1/
Gm2/Gm3 with PCGrad surgery, and writes the best-loss artifacts to
--output_dir.

Per-run artifacts written to --output_dir:
    run_loss_*.json     metrics + config + paths
    weights_loss_*.pt   best model weights
    script_loss_*.py    snapshot of this script (skipped if --no_script_copy)
    plot_loss_*.png     6-panel IV/GM plot   (skipped if --no_plot)

Key CLI args (run --help for the full list, including all GM tunables):
    --csv                  measurement CSV (REQUIRED)
    --output_dir           where artifacts are written (default 'best_runs')
    --config_name          tag appended to output filenames
    --epochs               training epochs (default 10000)
    --lr                   Adam learning rate (default 1e-2)

    # Model selection
    --equation_type        "pure" | "noNN_knee:mod1_angelov" |
                           "noNN_knee:classic_angelov" | ...
    --freeze_physics       freeze physics params (NN only is trained)
    --no_opt_params        skip loading any physics seed
    --opt_params_path      load physics seed JSON produced by SLSQP runner

    # NN architecture
    --architecture         JSON activations, e.g. '[["tanh","sin"],["tanh"]]'
                           overrides --hidden_layers/--neurons_per_layer/--activation
    --output_activation    "linear" | "softplus" | ...
    --knee_combiner        "sum" | "product" | "max" | "min" | "residual" | "sum_gated_vgs"
    --knee_alpha_scale     float (default 1.0)

    # Gradient matching (loss-shaping over Gm1/Gm2/Gm3)
    --use_gm               enable GM losses
    --gm1_weight / --gm2_weight / --gm3_weight
    --gm_surgery_mode      PCGrad mode (e.g. "soft", "element-wise-bounded")

    # Plotting / output toggles
    --hide_plot            do not open the PNG viewer at the end
    --no_plot              skip generating/saving plot_loss_*.png entirely
    --no_script_copy       skip the per-run script snapshot
    --plot_vds_list        comma-separated Vds targets for the 6-panel plot
    --plot_vgs_list        comma-separated Vgs targets for the 6-panel plot
    --min_vgs              extrapolation floor passed to meas_load

Examples
--------
# Pure NN, no physics, no GM:
python per_neuron_simple_angelov_nn_test.py --csv path/to/meas.csv \
    --no_opt_params --equation_type pure \
    --architecture '[["tanh","tanh","sin"]]' --output_activation softplus \
    --epochs 5000 --output_dir runs/pureNN --hide_plot

# Physics+NN with frozen physics seed and Gm1 matching (PCGrad):
python per_neuron_simple_angelov_nn_test.py --csv path/to/meas.csv \
    --freeze_physics --opt_params_path tests/slsqp_mod1_physics_seed_cfg9.json \
    --equation_type noNN_knee:mod1_angelov \
    --use_gm --gm1_weight 1.0 --gm_surgery_mode soft \
    --architecture '[["tanh","tanh"]]' --epochs 5000 \
    --output_dir runs/physicsNN_gm1 --hide_plot
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import json
import shutil
import argparse
import matplotlib
matplotlib.use('Agg') # Strictly disable GUI to prevent thread hang on Windows
import matplotlib.pyplot as plt

# Adjust imports to sibling optim_utils/ package (portable: resolved relative to this file)
_HERE = os.path.dirname(os.path.abspath(__file__))
_OPTIM_UTILS = os.environ.get(
    'OPTIM_UTILS_DIR',
    os.path.abspath(os.path.join(_HERE, '..', 'optim_utils')),
)
if _OPTIM_UTILS not in sys.path:
    sys.path.insert(0, _OPTIM_UTILS)
# Back-compat: data-file defaults still resolve via parallel_optimization/
optim_dir = os.path.abspath(os.path.join(_HERE, '..'))

from measurements_load import meas_load
from per_neuron_models_simplegate import DynamicNN, PhysicsInformedNN, set_mixed_init_mode
from per_neuron_normalization import USED_NORMALIZATIONS
from per_neuron_physics_params import PHYSICS_PARAM_CONFIG
from per_neuron_noNN import NONN_MODELS_CONFIG
from per_neuron_plotting import smooth_derivative

# ---------------------------------------------------------------------------
# Optional debug tracing. Silent and zero-cost unless PNN_DEBUG (or the legacy
# PNN_REGION_DEBUG) is set -- the smoke tests set it to trace the whole path.
# Emits "[trace] ..." lines to stdout, captured in each run's run_log.txt.gz,
# so a test (or a human) can confirm each stage ran with the expected values.
# `once` de-dups messages emitted inside training loops.
# ---------------------------------------------------------------------------
_DBG = bool(os.environ.get("PNN_DEBUG") or os.environ.get("PNN_REGION_DEBUG"))
_DBG_SEEN = set()
def _dbg(msg, once=None):
    if not _DBG:
        return
    if once is not None:
        if once in _DBG_SEEN:
            return
        _DBG_SEEN.add(once)
    print(f"[trace] {msg}", flush=True)

def create_gms_for_train(T_train):
    T_train = T_train.copy()
    T_train['Step_Index'] = T_train.groupby('TN').cumcount()
    T_train = T_train.sort_values(by=['Step_Index', 'TN']).reset_index(drop=True)
    
    X_train = np.column_stack((T_train['Vgs_meas'], T_train['Vds']))
    y_train = np.array(T_train['Ids']).reshape(-1, 1)

    gm1_true_list, gm2_true_list, gm3_true_list = [], [], []
    
    for step_val, group in T_train.groupby('Step_Index'):
        v_pts = group['Vgs_meas'].values
        i_pts = group['Ids'].values
        n_pts = len(v_pts)
        
        w_pre = min(11, n_pts if n_pts % 2 != 0 else n_pts - 1)
        w_post = min(13, n_pts if n_pts % 2 != 0 else n_pts - 1)
        
        if n_pts >= 4 and w_pre >= 3 and w_post >= 3:
            # gm-truth is dⁿIds/dVgsⁿ via smooth_derivative -> np.gradient(Ids, Vgs),
            # which is only valid if Vgs is a strictly monotonic sweep within the
            # group (each Step_Index group should be a transfer curve: one point
            # per TN, TNs ordered by increasing Vgs). Guard against a different
            # measurement layout (e.g. near-constant Vgs) that would silently
            # produce garbage gm-truth (zero/oscillating spacing -> inf/noise).
            dv = np.diff(v_pts)
            if not (np.all(dv > 0) or np.all(dv < 0)):
                raise ValueError(
                    f"create_gms_for_train: Step_Index group {step_val} is not "
                    f"strictly monotonic in Vgs (n={n_pts}, "
                    f"Vgs span={v_pts.max() - v_pts.min():.4f}, "
                    f"min|dVgs|={np.abs(dv).min():.3e}). gm-truth requires a sorted, "
                    "monotonic Vgs sweep per group; the data layout likely differs "
                    "from the expected (each TN = a Vds sweep at fixed Vgs, TNs "
                    "ordered by Vgs). Refusing to compute gm-truth on an invalid sweep."
                )
            gm1 = smooth_derivative(v_pts, i_pts, order=1, win_pre=w_pre, win_post=w_post)
            gm2 = smooth_derivative(v_pts, i_pts, order=2, win_pre=w_pre, win_post=w_post)
            gm3 = smooth_derivative(v_pts, i_pts, order=3, win_pre=w_pre, win_post=w_post)
        else:
            gm1, gm2, gm3 = np.zeros(n_pts), np.zeros(n_pts), np.zeros(n_pts)
            
        gm1_true_list.extend(gm1)
        gm2_true_list.extend(gm2)
        gm3_true_list.extend(gm3)
        
    return np.array(gm1_true_list), np.array(gm2_true_list), np.array(gm3_true_list), X_train, y_train

def apply_gradient_surgery(optimizer, loss_base, gm_losses, parameters, mode="no-bounded", max_gm_ratio=1.0):
    params = list(parameters)
    optimizer.zero_grad()

    if mode == "none" or len(gm_losses) == 0:
        if mode == "none" and max_gm_ratio < 1.0:
            # Scale GM losses by max_gm_ratio even in "none" mode
            scaled_gm_losses = [loss * max_gm_ratio for loss in gm_losses]
            total_loss = loss_base + sum(scaled_gm_losses)
        else:
            total_loss = loss_base + sum(gm_losses)
        total_loss.backward()
        return

    loss_base.backward(retain_graph=True)
    grad_base = [p.grad.view(-1).clone() if p.grad is not None else torch.zeros_like(p).view(-1) for p in params]
    g_base_vec = torch.cat(grad_base)
    g_base_norm_sq = torch.dot(g_base_vec, g_base_vec) + 1e-8
    base_norm = torch.sqrt(g_base_norm_sq)
    g_final = g_base_vec.clone()

    for i, loss_gm in enumerate(gm_losses):
        optimizer.zero_grad()
        is_last = (i == len(gm_losses) - 1)
        loss_gm.backward(retain_graph=not is_last)
        
        grad_gm = [p.grad.view(-1).clone() if p.grad is not None else torch.zeros_like(p).view(-1) for p in params]
        g_gm_vec = torch.cat(grad_gm)
        
        if mode in ["no-bounded", "mag-bounded"]:
            dot_product = torch.dot(g_gm_vec, g_base_vec)
            if dot_product < 0:
                g_gm_vec = g_gm_vec - (dot_product / g_base_norm_sq) * g_base_vec

            if mode == "mag-bounded":
                gm_norm = torch.norm(g_gm_vec) + 1e-8
                limit = base_norm * max_gm_ratio
                if gm_norm > limit:
                    g_gm_vec = g_gm_vec * (limit / gm_norm)
                    
        elif mode == "soft":
            dot_product = torch.dot(g_gm_vec, g_base_vec)
            gm_norm = torch.norm(g_gm_vec) + 1e-8
            cos_sim = dot_product / (base_norm * gm_norm)
            alignment_weight = torch.clamp(cos_sim, min=0.0)
            g_gm_vec = g_gm_vec * alignment_weight
            gm_norm_after_soft = torch.norm(g_gm_vec) + 1e-8
            limit = base_norm * max_gm_ratio
            if gm_norm_after_soft > limit:
                g_gm_vec = g_gm_vec * (limit / gm_norm_after_soft)
                
        elif mode in ["element-wise", "element-wise-percent"]:
            conflicts = (g_gm_vec * g_base_vec) < 0
            g_gm_vec[conflicts] = 0.0
            if mode == "element-wise-percent":
                g_gm_vec = g_gm_vec * max_gm_ratio
                
        elif mode == "drop-conflict":
            dot_product = torch.dot(g_gm_vec, g_base_vec)
            if dot_product < 0:
                g_gm_vec = torch.zeros_like(g_gm_vec)
                
        elif mode == "element-wise-bounded":
            conflicts = (g_gm_vec * g_base_vec) < 0
            g_gm_vec[conflicts] = 0.0
            limit = torch.abs(g_base_vec) * max_gm_ratio
            g_gm_vec = torch.clamp(g_gm_vec, min=-limit, max=limit)
            
        elif mode == "element-wise-soft":
            element_products = g_gm_vec * g_base_vec
            temperature = 1.0 / (base_norm + 1e-8)
            alignment_weight = torch.sigmoid(element_products * temperature)
            g_gm_vec = g_gm_vec * alignment_weight
            limit = torch.abs(g_base_vec) * max_gm_ratio
            g_gm_vec = torch.clamp(g_gm_vec, min=-limit, max=limit)
            
        g_final += g_gm_vec

    optimizer.zero_grad()
    idx = 0
    for p in params:
        length = p.numel()
        if p.requires_grad:
            p.grad = g_final[idx : idx + length].view_as(p).clone()
        idx += length

def get_physics_config(use_previously_optimized_params, previously_optimized_params_path=None, freeze_physics=False, base_key=9, width_percent=0.10, equation_type=None):
    """Build tight priors around an SLSQP-optimized seed.

    For noNN / noNN_knee modes the param set comes from
    ``NONN_MODELS_CONFIG[base_key][eq_name]['bounds']`` (matching exactly what the
    noNN model's function signature requires, and what the SLSQP fit produced).
    For PINN/hybrid modes the param set comes from ``PHYSICS_PARAM_CONFIG[base_key]``.
    """
    if not use_previously_optimized_params:
        # The Old Way: standard wide bounds, no tight priors, PyTorch starts from random center (~0.5)
        return base_key

    if equation_type is None:
        raise ValueError(
            "get_physics_config requires 'equation_type' when use_previously_optimized_params=True "
            "so the tight-prior param set can be sourced from the right config "
            "(NONN_MODELS_CONFIG for noNN/noNN_knee modes, PHYSICS_PARAM_CONFIG otherwise)."
        )

    if previously_optimized_params_path is None or not os.path.exists(previously_optimized_params_path):
        raise FileNotFoundError(f"Optimized params path not found: {previously_optimized_params_path}")

    with open(previously_optimized_params_path, 'r') as f:
        data = json.load(f)

    results = data.get('results', data) if isinstance(data, dict) else data
    if not (isinstance(results, list) and len(results) > 0):
        raise ValueError(
            f"Optimized params file has no results entries: {previously_optimized_params_path}"
        )
    optimums = results[0].get('optimized_params', {})
    if not isinstance(optimums, dict) or not optimums:
        raise ValueError(
            f"Optimized params file missing 'optimized_params' dict: {previously_optimized_params_path}"
        )

    # Parse mode + eq_name from equation_type ("noNN_knee:mod1_angelov", "pure", "classic_angelov", ...)
    if ':' in equation_type:
        mode_str, eq_name = equation_type.split(':', 1)
    else:
        mode_str, eq_name = equation_type, None
    is_nonn = mode_str in ('noNN', 'noNN_knee')

    if is_nonn:
        if eq_name is None:
            raise ValueError(
                f"noNN/noNN_knee equation_type must be of the form 'noNN[_knee]:<eq_name>', got {equation_type!r}"
            )
        if base_key not in NONN_MODELS_CONFIG:
            raise KeyError(
                f"base_key={base_key} not in NONN_MODELS_CONFIG. Known: {sorted(NONN_MODELS_CONFIG.keys())}"
            )
        if eq_name not in NONN_MODELS_CONFIG[base_key]:
            raise KeyError(
                f"noNN equation {eq_name!r} not under NONN_MODELS_CONFIG[{base_key}]. "
                f"Available: {sorted(NONN_MODELS_CONFIG[base_key].keys())}"
            )
        source_bounds = NONN_MODELS_CONFIG[base_key][eq_name]['bounds']
        source_label = f"NONN_MODELS_CONFIG[{base_key}][{eq_name!r}]['bounds']"
    else:
        if base_key not in PHYSICS_PARAM_CONFIG:
            raise KeyError(
                f"base_key={base_key} not in PHYSICS_PARAM_CONFIG. Known: {sorted(PHYSICS_PARAM_CONFIG.keys())}"
            )
        source_bounds = PHYSICS_PARAM_CONFIG[base_key]
        source_label = f"PHYSICS_PARAM_CONFIG[{base_key}]"

    expected_keys = set(source_bounds.keys())
    missing = [k for k in expected_keys if k not in optimums
               or not isinstance(optimums[k], (float, int))]
    if missing:
        raise KeyError(
            f"Optimized params file {previously_optimized_params_path} is missing "
            f"required params for {source_label}: {sorted(missing)}. "
            f"Refusing to silently fall back to wide default bounds."
        )

    # Dynamic tight priors around the loaded optimum values
    tight_config = {}
    for k, v in source_bounds.items():
        opt_val = float(optimums[k])
        if freeze_physics:
            # Fully locked physical parameter
            tight_config[k] = opt_val
        else:
            # Narrow boundary box
            if not isinstance(v, dict) or 'lr' not in v:
                raise KeyError(
                    f"Source bounds for param {k!r} in {source_label} is not a dict with 'lr'; "
                    f"got {v!r}. Cannot derive tight prior."
                )
            delta = abs(opt_val) * width_percent
            if delta < 1e-6: delta = 1e-6
            tight_config[k] = {"min": opt_val - delta, "max": opt_val + delta, "lr": v["lr"]}

    # Hash the file path to create a unique dictionary key
    import hashlib
    dynamic_key = f"tight_{hashlib.md5(previously_optimized_params_path.encode()).hexdigest()[:8]}"

    if is_nonn:
        # Clone the eq config but swap in the tight bounds
        base_eq_cfg = NONN_MODELS_CONFIG[base_key][eq_name]
        NONN_MODELS_CONFIG.setdefault(dynamic_key, {})
        NONN_MODELS_CONFIG[dynamic_key][eq_name] = {**base_eq_cfg, 'bounds': tight_config}
        # Also expose any sibling equations under the same base_key (with their original bounds)
        # so user code that references them under dynamic_key still works.
        for other_eq, other_cfg in NONN_MODELS_CONFIG[base_key].items():
            NONN_MODELS_CONFIG[dynamic_key].setdefault(other_eq, other_cfg)
    else:
        PHYSICS_PARAM_CONFIG[dynamic_key] = tight_config
        # Mirror noNN equations under the dynamic key as well (legacy behavior)
        if base_key in NONN_MODELS_CONFIG:
            NONN_MODELS_CONFIG[dynamic_key] = NONN_MODELS_CONFIG[base_key]

    return dynamic_key

def get_gm_rmse_metrics(model, X_train_t, gm1_true, gm2_true, gm3_true):
    # Enable grad to check the model derivatives
    X_req = X_train_t.clone().detach().requires_grad_(True)
    out = model(X_req)
    
    # Calculate GMs prediction from graph
    gm1_pred = torch.autograd.grad(outputs=out, inputs=X_req, grad_outputs=torch.ones_like(out), create_graph=True)[0][:, 0]
    gm2_pred = torch.autograd.grad(outputs=gm1_pred, inputs=X_req, grad_outputs=torch.ones_like(gm1_pred), create_graph=True)[0][:, 0]
    gm3_pred = torch.autograd.grad(outputs=gm2_pred, inputs=X_req, grad_outputs=torch.ones_like(gm2_pred))[0][:, 0]
    
    # Detach for RMSE computation against truth vectors
    rmse_1 = torch.sqrt(torch.mean((gm1_pred.detach() - torch.tensor(gm1_true, device=X_train_t.device))**2)).item() if gm1_true is not None and len(gm1_true) > 0 else 0.0
    rmse_2 = torch.sqrt(torch.mean((gm2_pred.detach() - torch.tensor(gm2_true, device=X_train_t.device))**2)).item() if gm2_true is not None and len(gm2_true) > 0 else 0.0
    rmse_3 = torch.sqrt(torch.mean((gm3_pred.detach() - torch.tensor(gm3_true, device=X_train_t.device))**2)).item() if gm3_true is not None and len(gm3_true) > 0 else 0.0
    
    return rmse_1, rmse_2, rmse_3, gm1_pred.detach().cpu().numpy(), gm2_pred.detach().cpu().numpy(), gm3_pred.detach().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="NN+Physics Test Framework")
    parser.add_argument("--csv", required=True, type=str,
                        help="Path to the measurement CSV (required).")
    parser.add_argument("--freeze_physics", action="store_true", help="Freeze physics parameters")
    parser.add_argument("--no_opt_params", action="store_true", help="Don't use previously optimized params")
    parser.add_argument("--opt_params_path", type=str, default=None,
                        help="Path to the previously-optimized physics-params JSON. "
                             "Defaults to <SCRIPTS_ROOT_DIR>/parallel_optimization/tests/slsqp_fast_physics_seed.json.")
    parser.add_argument("--knee_alpha_scale", type=float, default=1.0,
                        help="Knee gate width in Vds: gate = 1 - tanh(|alpha|/k * Vds). Larger k widens the "
                             "NN's active window into saturation. (The 'Vds window'.)")
    parser.add_argument("--knee_combiner", type=str, default="sum", choices=["sum", "product", "max", "min", "residual", "sum_gated_vgs"], help="Knee combiner logic")
    # Vgs window (only used by the 'residual' combiner): an extra sigmoid gate that
    # suppresses the NN below a Vgs threshold -> fixes the Gm bump at low Vgs.
    #   g_Vgs = sigmoid((Vgs - knee_vgs_thr) / knee_vgs_tau)
    #   final = phys * (1 + knee_max_correction * gate_Vds * g_Vgs * tanh(NN))
    parser.add_argument("--knee_vgs_thr", type=float, default=None,
                        help="Vgs threshold [V] for the residual-combiner Vgs gate. None = gate disabled "
                             "(NN active for all Vgs). Below it the NN is suppressed (physics only).")
    parser.add_argument("--knee_vgs_tau", type=float, default=0.3,
                        help="Softness [V, >0] of the Vgs gate edge (sigmoid scale). Smaller = sharper window.")
    parser.add_argument("--knee_max_correction", type=float, default=1.0,
                        help="Caps the NN amplitude in the residual combiner, in (0,1]. 1.0 = no cap.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=27,
                        help="RNG seed for torch + numpy (weight init, sampling). Same seed "
                             "+ same config => reproducible run. Default 27.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Stricter reproducibility: torch.use_deterministic_algorithms("
                             "True, warn_only=True) + single-threaded torch. Slower; warns if an "
                             "op has no deterministic implementation.")
    parser.add_argument("--mixed_init", type=str, default="xavier",
                        choices=["xavier", "per_activation"],
                        help="Init scheme for MIXED-activation layers (homogeneous layers are "
                             "always activation-matched). 'xavier': one Xavier(gain=1) for the "
                             "layer. 'per_activation': each neuron's row scaled by its own "
                             "activation gain. Default: xavier.")
    parser.add_argument("--hide_plot", action="store_true", help="Don't show the plot window at the end")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip generating/saving the 6-panel IV+GM plot entirely. "
                             "Implies --hide_plot. Useful for large batch sweeps.")
    parser.add_argument("--no_script_copy", action="store_true",
                        help="Skip copying this script into the per-run output dir. "
                             "Useful when the runner already snapshots itself once "
                             "per experiment.")
    parser.add_argument("--config_name", type=str, default="", help="Optional base config name for saved file tags")
    parser.add_argument("--equation_type", type=str, default="noNN_knee:mod1_angelov", help="Equation mode (e.g. pure, noNN_knee:mod1_angelov, etc.)")
    
    # Gradient Matching (GM)
    parser.add_argument("--use_gm", action="store_true", help="Enable Gm1 matching during training")
    parser.add_argument("--gm1_weight", type=float, default=0.1, help="Weight of Gm1 loss")
    parser.add_argument("--gm2_weight", type=float, default=0.0, help="Weight of Gm2 loss")
    parser.add_argument("--gm3_weight", type=float, default=0.0, help="Weight of Gm3 loss")
    parser.add_argument("--gm_surgery_mode", type=str, default="no-bounded", choices=["none", "no-bounded", "mag-bounded", "soft", "element-wise", "element-wise-percent", "drop-conflict", "element-wise-bounded", "element-wise-soft"], help="Gradient surgery mode for PCGrad")
    parser.add_argument("--gm_max_ratio", type=float, default=1.0, help="Max ratio for gm gradient norm relative to base gradient norm")
    parser.add_argument("--lbfgs_epochs", type=int, default=5, help="Number of L-BFGS polishing steps (outer loop). Each runs up to --lbfgs_max_iter iters.")
    parser.add_argument("--lbfgs_max_iter", type=int, default=200, help="max_iter per L-BFGS step")
    parser.add_argument("--lbfgs_gm_aware", action=argparse.BooleanOptionalAction, default=None,
                        help="Polish the COMBINED Ids+gm objective (with surgery) in L-BFGS instead of Ids-only. "
                             "Default (unset): auto-ON whenever --use_gm, so AdamW and L-BFGS BOTH apply gm surgery. "
                             "Pass --no-lbfgs_gm_aware to force the legacy Ids-only polish.")
    parser.add_argument("--loss_norm", type=str, default="none", choices=["none", "nmse"],
                        help="Loss normalization. 'nmse': divide each term's MSE (Ids and every gm) by its target's "
                             "mean-square so all terms are dimensionless / scale-balanced -> equal weights balance the "
                             "gms automatically (no manual per-gm weight tuning). 'none': raw absolute MSE (legacy).")

    # --- Ids-preserving gm training: warm-start (two-phase) + epsilon-constraint ---
    parser.add_argument("--gm_warmup_epochs", type=int, default=0,
                        help="Warm-start: train Ids-ONLY for this many AdamW epochs before gm enters the loss, so "
                             "the net reaches the Ids floor first, then fine-tunes toward gm. 0 = off (legacy).")
    parser.add_argument("--gm_warmup_lr", type=float, default=None,
                        help="LR for the post-warmup (gm) phase. Default: keep --lr. Use a smaller value to take "
                             "bounded steps away from the Ids floor.")
    parser.add_argument("--ids_constraint", action=argparse.BooleanOptionalAction, default=False,
                        help="Epsilon-constraint loss: minimize the (weighted) gm losses subject to Ids, i.e. "
                             "total = sum(gm) + ids_lambda*max(0, ids_mse/ids_target^2 - 1)^2. The penalty is ZERO "
                             "while Ids is within the band, so gm improves with no pull on Ids. Replaces weighted sum.")
    parser.add_argument("--ids_target", type=float, default=None,
                        help="Ids RMSE ceiling. Used as the epsilon in --ids_constraint AND as the cap for "
                             "best-weights selection: keep the lowest-gm epoch whose ids_rmse <= ids_target.")
    parser.add_argument("--ids_lambda", type=float, default=100.0,
                        help="Penalty strength for the Ids constraint (relative-overage units; only used with "
                             "--ids_constraint).")
    parser.add_argument("--gm_vds_min", type=float, default=0.0,
                        help="Exclude gm loss contributions for points where Vds < this value [V]. "
                             "0 = include all points (default). Ids loss is always computed over all points. "
                             "Useful when low-Vds gm is noisy/unreliable and hurts training.")
    parser.add_argument("--gm_vgs_min", type=float, default=None,
                        help="Exclude gm loss contributions for points where Vgs < this value [V]. "
                             "None = include all points (default). Unlike --gm_vds_min, 0.0 is a real "
                             "threshold (Vgs spans negative values), so the 'off' value is None, not 0. "
                             "Ids loss is always computed over all points. Combined with --gm_vds_min "
                             "via logical AND. Useful when deep-subthreshold gm is noisy and hurts training.")
    parser.add_argument("--ids_region_center", type=float, default=None,
                        help="Vgs [V] center of a Gaussian up-weighting bump on the Ids loss "
                             "(e.g. -2.2 to emphasize the turn-on knee). None = uniform weighting (off).")
    parser.add_argument("--ids_region_width", type=float, default=0.3,
                        help="Std-dev [V] of the Ids-loss weighting bump around --ids_region_center.")
    parser.add_argument("--ids_region_weight", type=float, default=0.0,
                        help="Extra weight in the up-weighted region (0 = off). gm losses are unaffected. "
                             "Gaussian: 1 + weight*exp(-((Vgs-center)^2)/(2*width^2)). "
                             "Band (if --ids_region_lo/hi set): 1 + weight for Vgs in [lo, hi], else 1.")
    parser.add_argument("--ids_region_lo", type=float, default=None,
                        help="Lower Vgs [V] bound of a FLAT up-weighting band on the Ids loss. Set with "
                             "--ids_region_hi for a uniform boost on [lo, hi] (overrides the Gaussian center/width). "
                             "Use for a bad plateau, e.g. --ids_region_lo -3.0 --ids_region_hi -1.8.")
    parser.add_argument("--ids_region_hi", type=float, default=None,
                        help="Upper Vgs [V] bound of the flat Ids-loss band (use with --ids_region_lo).")
    # 2-D (Vgs x Vds) box region weighting, applied to BOTH the Ids loss AND the gm losses
    # (mirrors compute_region_metrics' region box). Points inside the box are weighted
    # (1 + region_weight); points outside keep weight 1. Any bound left None is unbounded
    # on that side. Unlike gm_vds_min/gm_vgs_min (which EXCLUDE points), this UP-WEIGHTS them,
    # so it can emphasize the knee box without discarding the rest of the curve.
    parser.add_argument("--region_vgs_lo", type=float, default=None,
                        help="Lower Vgs [V] bound of the 2-D weighting box (None = unbounded).")
    parser.add_argument("--region_vgs_hi", type=float, default=None,
                        help="Upper Vgs [V] bound of the 2-D weighting box (None = unbounded).")
    parser.add_argument("--region_vds_lo", type=float, default=None,
                        help="Lower Vds [V] bound of the 2-D weighting box (None = unbounded).")
    parser.add_argument("--region_vds_hi", type=float, default=None,
                        help="Upper Vds [V] bound of the 2-D weighting box (None = unbounded).")
    parser.add_argument("--region_weight", type=float, default=0.0,
                        help="Extra weight for points inside the (Vgs,Vds) box, on BOTH Ids and gm "
                             "losses (0 = off). Weighted MSE: 1 + region_weight inside, 1 outside. "
                             "E.g. --region_vgs_lo -3 --region_vgs_hi 0 --region_vds_lo 0 --region_vds_hi 15 "
                             "--region_weight 3 emphasizes the knee box.")
    parser.add_argument("--knee_lr_scale", type=float, default=1.0,
                        help="LR multiplier for the vdsgate structural/knee params (a0..a3, l0..l2, "
                             "alpha/lamb) in the AdamW phase. <1 slows them for stability (e.g. 0.3). "
                             "1.0 = off. Only affects vdsgate* wrappers.")
    parser.add_argument("--vds_loss", type=float, default=0.0,
                        help="Weight of the output-conductance monotonicity penalty: penalizes "
                             "gds=dIds/dVds < 0 (a knee bump) inside the --region_* box (0 = off). "
                             "Reuses the region_vgs/vds bounds; relu(-gds)^2 so it is zero wherever "
                             "the curve is already monotonic. Applied in BOTH AdamW and L-BFGS phases.")
    parser.add_argument("--adamw_avoid_localmin", action="store_true",
                        help="Opt-in AdamW-phase upgrade to reduce getting stuck in / settling into a "
                             "bad basin: (1) cosine-annealing-with-warm-restarts LR schedule (periodic "
                             "LR resets to re-explore, annealing down within each cycle) instead of a "
                             "flat --lr, and (2) gradient-norm clipping (max_norm=1.0). Off by default -- "
                             "legacy runs (flat LR, no clipping) are completely unaffected unless this "
                             "flag is passed. Scoped to the AdamW phase only; L-BFGS is unchanged.")

    # NN Hyperparameters
    parser.add_argument("--architecture", type=str, default="", help="JSON string of activation lists, e.g. '[[\"tanh\", \"sin\"], [\"relu\"]]'")
    parser.add_argument("--hidden_layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument("--neurons_per_layer", type=int, default=4, help="Neurons in each hidden layer")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function to use")
    parser.add_argument("--output_activation", type=str, default="linear",
                        help="Activation function for the output layer. SIMPLEGATE: for "
                             "vdsgate*/vdsgate_aeff*/vdsgate_vdsk* wrappers, this IS the gate -- "
                             "there is no separate gate-mode setting. 'softplus' -> legacy "
                             "sign(Ids)=sign(Vds)-guaranteed gate; 'tanh' -> old 'tanhm' behavior "
                             "(no sign guarantee, better gradient at pinch-off; pair with "
                             "--ids_out_margin); 'linear' -> old vdsgatelin behavior.")
    parser.add_argument("--ids_out_margin", type=float, default=0.0,
                        help="Output-scale margin for BOUNDED output activations (sigmoid/tanh, "
                             "range capped near +-1): the model predicts scale*activation(NN(...)) "
                             "with scale=(1+margin)*max(Ids) in the training data, computed once. "
                             "0 (default) = off, scale=1.0, reproduces exact legacy behavior -- "
                             "required for bounded activations to reach the data's real range at "
                             "all (e.g. Ids up to 2.7A can't be represented by sigmoid's (0,1) "
                             "output without this). Irrelevant for unbounded activations "
                             "(linear/softplus), which don't need it. A larger margin leaves more "
                             "headroom before the activation's asymptote (avoids vanishing-gradient "
                             "saturation at the top of the training range) at the cost of using less "
                             "of the activation's dynamic range for the actual data.")
    parser.add_argument("--output_dir", type=str, default="best_runs", help="Directory to save the best runs")
    parser.add_argument("--min_vgs", type=float, default=None,
                        help="If set (e.g. -4.0), meas_load auto-extrapolates extra start-groups so the most-negative Vgs reaches this floor.")
    parser.add_argument("--add_zero_vds", action="store_true",
                        help="Overwrite the first (lowest-Vds) row of every TN group, forcing Vds=0, Ids=0 "
                             "to pin each curve to the origin. In place and unconditional — no row is added "
                             "and the group's original first sample is discarded.")
    parser.add_argument("--plot_vds_list", type=str, default=None,
                        help="Comma-separated Vds targets for the plot (e.g. '0,10,20,30'). "
                             "Overrides the auto-selected first/middle/last picks.")
    parser.add_argument("--plot_vgs_list", type=str, default=None,
                        help="Comma-separated Vgs targets for the plot (e.g. '-4,-3,-2,-1,0'). "
                             "Overrides the auto-selected 5-point picks.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Mixed-activation layers: choose Xavier-only vs per-activation init.
    set_mixed_init_mode(args.mixed_init)
    print(f"[init] mixed-activation init mode = {args.mixed_init}")
    if args.deterministic:
        # Stricter reproducibility: deterministic op implementations + single
        # thread (fixed reduction order). warn_only=True so a missing
        # deterministic kernel warns instead of aborting the run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        print(f"[determinism] use_deterministic_algorithms(True, warn_only=True) + 1 thread")
    print(f"[seed] torch+numpy seed = {args.seed}")

    print("Setting up simple training loop for Angelov + NN correction ...")
    print(f"Args: {vars(args)}")

    # 1. Data loading
    CSV_PATH = os.path.abspath(args.csv)
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    meas_load_kwargs = {
        "file_type": "auriga",
        "keep_every_N_group": 0,
        "remove_negative_vds": 0,
        "remove_negative_ids": 0,
        "test_percent": 0.0,
        "num_extrapolate_groups_start": 0,
        "min_vgs": args.min_vgs,
        "add_zero_vds": args.add_zero_vds,
    }

    _dbg(f"START run: eq={getattr(args, 'equation_type', None)} use_gm={args.use_gm} "
         f"gm_w=({args.gm1_weight},{args.gm2_weight},{args.gm3_weight}) "
         f"region_weight={args.region_weight} box=Vgs[{args.region_vgs_lo},{args.region_vgs_hi}]"
         f"/Vds[{args.region_vds_lo},{args.region_vds_hi}] vds_loss={args.vds_loss} "
         f"epochs={args.epochs} lbfgs={args.lbfgs_epochs}x{args.lbfgs_max_iter} "
         f"loss_norm={args.loss_norm} seed={args.seed}")
    print(f"Loading data from: {CSV_PATH}")
    T_train, _ = meas_load(CSV_PATH, **meas_load_kwargs)

    # Always extract true GM arrays to evaluate true model capacity natively afterwards
    gm1_true_arr, gm2_true_arr, gm3_true_arr, X_train_gm, y_train_gm = create_gms_for_train(T_train)
    
    if args.use_gm:
        X_train = X_train_gm
        y_train = y_train_gm
    else:
        X_train = np.column_stack((T_train['Vgs_meas'], T_train['Vds']))
        y_train = np.array(T_train['Ids']).reshape(-1, 1)

    device = torch.device("cpu")
    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float64, device=device)
    
    # We must flag requires_grad consistently so autograd works during eval.
    X_train_t.requires_grad_(True)
    
    if args.use_gm:
        gm1_target_t = torch.tensor(gm1_true_arr, dtype=torch.float64, device=device)
        gm2_target_t = torch.tensor(gm2_true_arr, dtype=torch.float64, device=device)
        gm3_target_t = torch.tensor(gm3_true_arr, dtype=torch.float64, device=device)

    if _DBG:
        _vg = X_train_t[:, 0].detach(); _vd = X_train_t[:, 1].detach()
        _dbg(f"data loaded: X={tuple(X_train_t.shape)} y={tuple(y_train_t.shape)} "
             f"Vgs[{float(_vg.min()):.2f},{float(_vg.max()):.2f}] "
             f"Vds[{float(_vd.min()):.2f},{float(_vd.max()):.2f}] use_gm={args.use_gm}")

    # gm-loss point masks: exclude low-Vds and/or low-Vgs points from gm losses.
    # Ids loss is always full (no masking). X_train_t columns: [:, 0] = Vgs, [:, 1] = Vds.
    # A Vds cut excludes entire low-Vds transfer curves (each Step_Index group shares one
    # Vds); a Vgs cut excludes the most-negative (deep-subthreshold) Vgs columns where gm
    # is noisy. When both are set they are combined with logical AND. gm_vds_min=0 means
    # off (Vds >= 0 always); gm_vgs_min uses None for off (0.0 is a real Vgs threshold).
    gm_loss_mask = None
    if args.use_gm:
        _masks, _crit = [], []
        if args.gm_vds_min > 0.0:
            _masks.append(X_train_t[:, 1].detach() >= args.gm_vds_min)
            _crit.append(f"Vds>={args.gm_vds_min}V")
        if args.gm_vgs_min is not None:
            _masks.append(X_train_t[:, 0].detach() >= args.gm_vgs_min)
            _crit.append(f"Vgs>={args.gm_vgs_min}V")
        if _masks:
            gm_loss_mask = _masks[0]
            for _mk in _masks[1:]:
                gm_loss_mask = gm_loss_mask & _mk
            n_excluded = int((~gm_loss_mask).sum().item())
            print(f"[gm mask: {', '.join(_crit)}] excluding {n_excluded}/{len(gm_loss_mask)} points from gm loss")
    _dbg(f"gm_loss_mask: {'set -> '+str(int(gm_loss_mask.sum()))+'/'+str(gm_loss_mask.numel())+' points kept' if gm_loss_mask is not None else 'None (all points enter gm loss)'}")

    # 2. Instantiate Angelov wrapper + NN
    # Build dynamic NN arguments based on parsed inputs
    if args.architecture:
        import ast
        try:
            parsed_arch = ast.literal_eval(args.architecture)
        except Exception as _arch_err:
            raise ValueError(
                f"Could not parse --architecture {args.architecture!r}: {_arch_err}. "
                "Pass a valid Python literal, e.g. \"[['tanh','tanh'],['tanh','tanh']]\"."
            ) from _arch_err
        n_layers = len(parsed_arch)
        neurons_per_layer = [len(layer) for layer in parsed_arch]
        activations_per_layer = parsed_arch
        
        # Build compact signature: e.g. "L_2ta1si_4ta"
        arch_sig = []
        for l in parsed_arch:
            counts = {}
            for act in l:
                counts[act] = counts.get(act, 0) + 1
            layer_str = "".join([f"{v}{k[:2]}" for k, v in counts.items()])
            arch_sig.append(layer_str)
        arch_tag = "L_" + "_".join(arch_sig)
    else:
        n_layers = args.hidden_layers
        neurons_per_layer = [args.neurons_per_layer] * n_layers
        activations_per_layer = [[args.activation] * args.neurons_per_layer for _ in range(n_layers)]
        arch_tag = f"L{n_layers}x{args.neurons_per_layer}{args.activation}"
    
    base_model = DynamicNN(
        input_dim=2,
        neurons_per_layer=neurons_per_layer,
        activations_per_layer=activations_per_layer,
        output_dim=1,
        output_activation=args.output_activation
    ).double()

    model_arch = args.equation_type
    freeze_physics = args.freeze_physics
    use_previously_optimized_params = not args.no_opt_params
    if use_previously_optimized_params:
        if not args.opt_params_path:
            raise ValueError(
                "--opt_params_path is required unless --no_opt_params is set. "
                "Refusing to silently default to a hardcoded test seed."
            )
        previously_optimized_params_path = args.opt_params_path
    else:
        previously_optimized_params_path = None

    # Hard fail if the optimized-params seed is required but missing.
    if use_previously_optimized_params and not os.path.exists(previously_optimized_params_path):
        raise FileNotFoundError(
            f"Optimized physics params file not found: {previously_optimized_params_path}. "
            "Provide the file or pass --no_opt_params to use wide default bounds."
        )

    physics_c = get_physics_config(
        use_previously_optimized_params=use_previously_optimized_params,
        previously_optimized_params_path=previously_optimized_params_path,
        freeze_physics=freeze_physics, 
        base_key=9, 
        width_percent=0.10,
        equation_type=model_arch,
    )
    
    _norm_key = 'cgh40010f_vgs4_vds45'
    if _norm_key not in USED_NORMALIZATIONS:
        raise KeyError(
            f"Normalization key {_norm_key!r} not in USED_NORMALIZATIONS. "
            f"Available: {sorted(USED_NORMALIZATIONS.keys())}"
        )
    norm_c = dict(USED_NORMALIZATIONS[_norm_key])
    norm_c['hybrid'] = True  # FIX: NN gets [-1,1], Physics gets raw data

    model = PhysicsInformedNN(
        base_nn=base_model,
        equation_type=model_arch,
        normalization_config=norm_c,
        config_key=physics_c,
    ).double()
    
    model.knee_alpha_scale = args.knee_alpha_scale
    model.knee_use_alpha_eff = True
    model.knee_combiner = args.knee_combiner
    # Vgs window (residual combiner): see per_neuron_models noNN_knee forward.
    model.knee_vgs_thr = args.knee_vgs_thr
    model.knee_vgs_tau = args.knee_vgs_tau
    model.knee_max_correction = args.knee_max_correction
    model.freeze_physics = freeze_physics

    with torch.no_grad():
        last_layer_idx = len(model.base_nn.net) - 1
        model.base_nn.net[last_layer_idx].linear.weight.uniform_(-1e-6, 1e-6)
        model.base_nn.net[last_layer_idx].linear.bias.uniform_(-1e-6, 1e-6)

    model.to(device)
    if _DBG:
        _n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _dbg(f"model built: {type(model).__name__}(base={type(base_model).__name__}) "
             f"eq={model_arch} trainable_params={_n_par} "
             f"knee(combiner={args.knee_combiner},alpha_scale={args.knee_alpha_scale}) freeze_physics={freeze_physics}")

    # --ids_out_margin: output-scale margin for BOUNDED output activations (sigmoid/tanh).
    # Patch model.forward ONCE here so every downstream use (AdamW/L-BFGS preds, gm-loss/gm-metric
    # autograd, final RMSE, plotting) transparently sees scale*activation(NN(...)) instead of the
    # raw (0,1)/(-1,1)-bounded output -- state_dict/attributes/parameters are all untouched (only
    # the instance's forward is shadowed), and autograd scales derivatives correctly via the chain
    # rule (verified: d(scale*f(x))/dx == scale*df/dx), so gm-loss needs no separate handling.
    ids_scale = 1.0
    if args.ids_out_margin and args.ids_out_margin > 0:
        ids_scale = float((1.0 + args.ids_out_margin) * y_train_t.detach().max().item())
        _orig_forward = model.forward
        model.forward = lambda x, _f=_orig_forward, _s=ids_scale: _s * _f(x)
        _dbg(f"ids_out_margin={args.ids_out_margin}: ids_scale={ids_scale:.6f} "
             f"(= (1+{args.ids_out_margin})*max(Ids)={float(y_train_t.max()):.6f})")

    # 3. Simple Training Loop
    criterion = nn.MSELoss()
    trainable_params = lambda: filter(lambda p: p.requires_grad, model.parameters())
    # Optional lower LR for the vdsgate structural/knee params (a0..a3, l0..l2, alpha/lamb): they
    # have outsized leverage (each scales the whole Vds curve via tanh) and can destabilize training.
    _knee_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vdsgate_")]
    if _knee_p and args.knee_lr_scale != 1.0:
        _knee_ids = {id(p) for p in _knee_p}
        _other_p = [p for p in trainable_params() if id(p) not in _knee_ids]
        optimizer = optim.AdamW([{"params": _other_p, "lr": args.lr},
                                 {"params": _knee_p,  "lr": args.lr * args.knee_lr_scale}])
        print(f"[knee_lr] {len(_knee_p)} structural params at lr={args.lr * args.knee_lr_scale:g} "
              f"({args.knee_lr_scale}x base)")
    else:
        optimizer = optim.AdamW(trainable_params(), lr=args.lr)

    epochs = args.epochs
    best_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())  # safe default; replaced by selection

    # --adamw_avoid_localmin: cosine-annealing-with-warm-restarts LR schedule for the AdamW phase.
    # Periodic restarts (back up to the base LR every ~epochs/_N_LR_RESTARTS steps) actively kick
    # the weights out of a settled basin to explore, annealing down again within each cycle for
    # fine convergence -- unlike a flat LR or a plateau-only drop. None (off) reproduces the exact
    # legacy flat-LR behavior. Each optimizer param group anneals relative to ITS OWN base LR, so
    # this is correct even with --knee_lr_scale's two-group optimizer.
    _N_LR_RESTARTS = 4
    scheduler = None
    if args.adamw_avoid_localmin:
        _t0 = max(epochs // _N_LR_RESTARTS, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=_t0, T_mult=1, eta_min=args.lr * 0.01)
        _dbg(f"adamw_avoid_localmin: ON -- cosine warm restarts (T_0={_t0}, eta_min={args.lr*0.01:.2e}) "
             f"+ grad-norm clipping (max_norm=1.0)")

    # --- Loss normalization (NMSE) ---------------------------------------------
    # loss_norm='nmse' divides each term's MSE by mean(target^2) so Ids and every gm
    # become dimensionless / scale-balanced -> equal weights balance the gms
    # automatically (no manual per-gm weight tuning). The divisors are constants
    # (no gradient effect). 'none' -> all divisors 1.0 (legacy absolute MSE).
    # NOTE: normalize by a GLOBAL per-signal scale, never per-point (gm2/gm3 cross
    # zero, so element-wise relative error would blow up).
    use_nmse = (args.loss_norm == "nmse")
    _eps = 1e-12
    ids_norm = (y_train_t.detach().pow(2).mean().item() + _eps) if use_nmse else 1.0
    gm1_norm = gm2_norm = gm3_norm = 1.0
    if use_nmse and args.use_gm:
        # Normalize over the same subset that enters the loss (masked if gm_vds_min/gm_vgs_min is set).
        _m = gm_loss_mask  # None = all points, bool tensor = masked subset
        gm1_norm = (gm1_target_t[_m] if _m is not None else gm1_target_t).detach().pow(2).mean().item() + _eps
        gm2_norm = (gm2_target_t[_m] if _m is not None else gm2_target_t).detach().pow(2).mean().item() + _eps
        gm3_norm = (gm3_target_t[_m] if _m is not None else gm3_target_t).detach().pow(2).mean().item() + _eps
    _dbg(f"norms ({'nmse' if use_nmse else 'none'}): ids_norm={ids_norm:.4e} "
         f"gm1_norm={gm1_norm:.4e} gm2_norm={gm2_norm:.4e} gm3_norm={gm3_norm:.4e}")

    # Per-region Ids-loss weighting: up-weight a Vgs region where the amplitude/NMSE loss is
    # otherwise weak (e.g. the subthreshold plateau from ~-1.8 down to Vth). Two shapes
    # (gm losses untouched):
    #   band     (ids_region_lo/hi set): weight = 1 + w for Vgs in [lo, hi], else 1   (flat).
    #   gaussian (else, center set):     weight = 1 + w*exp(-((Vgs-c)^2)/(2*width^2)).
    # Off (uniform) when w <= 0 or neither band nor center given. Weights are constants
    # (no gradient); X_train_t[:, 0] = Vgs works in both gm and no-gm orderings.
    ids_weight = None
    _wt = args.ids_region_weight
    if _wt and _wt > 0.0:
        _vgs_col = X_train_t[:, 0].detach()
        if args.ids_region_lo is not None and args.ids_region_hi is not None:
            _in = ((_vgs_col >= args.ids_region_lo) & (_vgs_col <= args.ids_region_hi)).to(X_train_t.dtype)
            ids_weight = (1.0 + _wt * _in).reshape(-1, 1)
            print(f"[ids_region] band Vgs in [{args.ids_region_lo}, {args.ids_region_hi}] weight=+{_wt}: "
                  f"{int(_in.sum())}/{len(_vgs_col)} pts boosted to {1.0 + _wt:.1f}x")
        elif args.ids_region_center is not None:
            _bump = torch.exp(-((_vgs_col - args.ids_region_center) ** 2) / (2.0 * args.ids_region_width ** 2))
            ids_weight = (1.0 + _wt * _bump).reshape(-1, 1)
            print(f"[ids_region] gaussian center={args.ids_region_center}V width={args.ids_region_width} "
                  f"weight=+{_wt}: peak weight={float(ids_weight.max()):.2f}")

    # 2-D (Vgs x Vds) box region weight, applied to BOTH Ids and gm losses (see --region_*).
    # Per-point weight aligned with X_train_t rows: 1 + region_weight inside the box, else 1.
    region_w = None
    if args.region_weight and args.region_weight > 0.0:
        _vgs = X_train_t[:, 0].detach()
        _vds = X_train_t[:, 1].detach()
        _inbox = torch.ones_like(_vgs, dtype=torch.bool)
        if args.region_vgs_lo is not None: _inbox &= (_vgs >= args.region_vgs_lo)
        if args.region_vgs_hi is not None: _inbox &= (_vgs <= args.region_vgs_hi)
        if args.region_vds_lo is not None: _inbox &= (_vds >= args.region_vds_lo)
        if args.region_vds_hi is not None: _inbox &= (_vds <= args.region_vds_hi)
        region_w = 1.0 + args.region_weight * _inbox.to(X_train_t.dtype)
        print(f"[region] box Vgs[{args.region_vgs_lo},{args.region_vgs_hi}] "
              f"Vds[{args.region_vds_lo},{args.region_vds_hi}] weight=+{args.region_weight}: "
              f"{int(_inbox.sum())}/{len(_vgs)} pts at {1.0 + args.region_weight:.1f}x (Ids+gm)")
        # Exact numbers a test can verify: w.sum() must equal n + weight*in_box (i.e. 1 outside,
        # 1+weight inside), and in_box must equal the box-mask count.
        _dbg(f"region_w built: n={region_w.numel()} in_box={int(_inbox.sum())} "
             f"out_box={int((~_inbox).sum())} weight={args.region_weight:g} "
             f"w.sum={float(region_w.sum()):.4f}")

    def ids_loss(preds):
        # Combine the (Vgs-only) ids_region weight and the (2-D) region weight when present.
        w = ids_weight
        if region_w is not None:
            w = region_w.reshape(-1, 1) if w is None else w * region_w.reshape(-1, 1)
        if w is not None:
            loss = (w * (preds - y_train_t) ** 2).sum() / w.sum() / ids_norm
            if _DBG:
                _plain = ((preds - y_train_t) ** 2).mean() / ids_norm
                _dbg(f"ids_loss WEIGHTED: w.sum={float(w.sum()):.4f} n={w.numel()} "
                     f"weighted={float(loss):.6e} plain_mean={float(_plain):.6e} ids_norm={ids_norm:.4e}", once="ids")
            return loss
        if _DBG:
            _dbg(f"ids_loss PLAIN (region off): loss={float(criterion(preds, y_train_t) / ids_norm):.6e} ids_norm={ids_norm:.4e}", once="ids")
        return criterion(preds, y_train_t) / ids_norm

    def compute_gm_losses(preds):
        """Weighted, optionally NMSE-normalized gm1/gm2/gm3 losses via input autograd
        (create_graph for higher orders). Returns the weighted loss terms (weight > 0).
        When gm_loss_mask is set, ALL gm losses (gm1, gm2, gm3) are masked: only points
        passing the gm mask (Vds >= gm_vds_min and/or Vgs >= gm_vgs_min) enter criterion().
        Excluded points are dropped from every gm term. (PyTorch requires computing the full
        gm tensor before indexing, but only the masked elements are passed to criterion and
        hence to backward.)"""
        gm_losses = []
        # Region weight restricted to the same points the gm loss sees (after gm_loss_mask).
        # None -> plain MSE (criterion); else a weighted MSE that up-weights the box.
        _rw = None
        if region_w is not None:
            _rw = region_w[gm_loss_mask] if gm_loss_mask is not None else region_w
        def _mse(p, t):
            return (_rw * (p - t) ** 2).sum() / _rw.sum() if _rw is not None else criterion(p, t)
        gm1_pred = torch.autograd.grad(
            outputs=preds, inputs=X_train_t, grad_outputs=torch.ones_like(preds),
            create_graph=True, retain_graph=True)[0][:, 0]
        if args.gm1_weight > 0:
            _p1 = gm1_pred[gm_loss_mask] if gm_loss_mask is not None else gm1_pred
            _t1 = gm1_target_t[gm_loss_mask] if gm_loss_mask is not None else gm1_target_t
            gm_losses.append(args.gm1_weight * _mse(_p1, _t1) / gm1_norm)
        if args.gm2_weight > 0 or args.gm3_weight > 0:
            gm2_pred = torch.autograd.grad(
                outputs=gm1_pred, inputs=X_train_t, grad_outputs=torch.ones_like(gm1_pred),
                create_graph=True, retain_graph=True)[0][:, 0]
            if args.gm2_weight > 0:
                _p2 = gm2_pred[gm_loss_mask] if gm_loss_mask is not None else gm2_pred
                _t2 = gm2_target_t[gm_loss_mask] if gm_loss_mask is not None else gm2_target_t
                gm_losses.append(args.gm2_weight * _mse(_p2, _t2) / gm2_norm)
            if args.gm3_weight > 0:
                gm3_pred = torch.autograd.grad(
                    outputs=gm2_pred, inputs=X_train_t, grad_outputs=torch.ones_like(gm2_pred),
                    create_graph=True, retain_graph=True)[0][:, 0]
                _p3 = gm3_pred[gm_loss_mask] if gm_loss_mask is not None else gm3_pred
                _t3 = gm3_target_t[gm_loss_mask] if gm_loss_mask is not None else gm3_target_t
                gm_losses.append(args.gm3_weight * _mse(_p3, _t3) / gm3_norm)
        if _DBG:
            _info = f"_rw.sum={float(_rw.sum()):.4f} n={_rw.numel()}" if _rw is not None else "none"
            _dbg(f"gm_losses: region_w applied={_rw is not None} {_info} "
                 f"terms={[round(float(x), 8) for x in gm_losses]} "
                 f"norms=({gm1_norm:.3e},{gm2_norm:.3e},{gm3_norm:.3e})", once="gm")
        return gm_losses

    def vds_mono_loss(preds):
        """Output-conductance monotonicity penalty (--vds_loss): penalize negative
        gds = dIds/dVds (a knee bump) inside the --region_* box. Returns a scalar tensor
        (0 when --vds_loss is off). gds is column 1 of the input-gradient (column 0 is
        dIds/dVgs = gm1); X_train_t already carries requires_grad. relu(-gds)^2 is exactly
        zero wherever the curve is already monotonic, so it only acts on a real violation."""
        if not args.vds_loss or args.vds_loss <= 0.0:
            if _DBG: _dbg("vds_mono_loss: OFF (vds_loss<=0)", once="vds")
            return preds.new_zeros(())
        gds = torch.autograd.grad(
            outputs=preds, inputs=X_train_t, grad_outputs=torch.ones_like(preds),
            create_graph=True, retain_graph=True)[0][:, 1]
        _vgs = X_train_t[:, 0].detach(); _vds = X_train_t[:, 1].detach()
        m = torch.ones_like(_vgs, dtype=torch.bool)
        if args.region_vgs_lo is not None: m &= (_vgs >= args.region_vgs_lo)
        if args.region_vgs_hi is not None: m &= (_vgs <= args.region_vgs_hi)
        if args.region_vds_lo is not None: m &= (_vds >= args.region_vds_lo)
        if args.region_vds_hi is not None: m &= (_vds <= args.region_vds_hi)
        viol = torch.relu(-gds[m])
        if viol.numel() == 0:
            return preds.new_zeros(())
        pen = args.vds_loss * viol.pow(2).mean() / ids_norm   # /ids_norm scale-balances vs Ids loss
        if _DBG:
            _dbg(f"vds_mono_loss: ON weight={args.vds_loss:g} box_pts={int(m.sum())} "
                 f"viol_pts={int((gds[m] < 0).sum())} penalty={float(pen):.6e}", once="vds")
        return pen

    # --- Ids-preserving extensions: epsilon-constraint penalty + Ids-aware selection -
    ids_cap = args.ids_target                                  # RMSE ceiling (None disables)
    ids_target_mse = (ids_cap ** 2) if ids_cap is not None else None
    def ids_penalty(base_loss):
        """lambda * max(0, ids_mse/ids_target_mse - 1)^2: zero while Ids is inside the
        band, dimensionless overage outside it (so lambda is O(1-100) at any data scale)."""
        raw_mse = base_loss * ids_norm                         # undo nmse -> actual MSE
        over = torch.clamp(raw_mse / (ids_target_mse + 1e-30) - 1.0, min=0.0)
        return args.ids_lambda * over * over
    use_constraint = bool(args.ids_constraint) and args.use_gm and ids_target_mse is not None

    # Best-weights selection. With an Ids cap (--ids_target): keep the lowest-gm epoch
    # whose ids_rmse <= cap (Ids stays at the floor, gm is squeezed under it); without a
    # cap: keep the lowest combined Ids+gm objective (legacy combined_objective). Ids-only
    # warm-up epochs (gm_on False) are skipped so their gm_sum=0 can't masquerade as best.
    _sel = {"tier": 1, "score": float("inf")}
    def consider(base_loss, gm_losses, gm_on):
        nonlocal best_loss, best_weights
        if args.use_gm and not gm_on:
            return                                             # skip Ids-only warm-up epochs
        ids_r = (base_loss.item() * ids_norm) ** 0.5
        gm_sum = sum(l.item() for l in gm_losses)
        if ids_cap is not None:
            tier, score = (0, gm_sum) if ids_r <= ids_cap else (1, ids_r)
            if (tier, score) < (_sel["tier"], _sel["score"]):
                _sel["tier"], _sel["score"] = tier, score
                best_loss = base_loss.item() + gm_sum
                best_weights = copy.deepcopy(model.state_dict())
        else:
            obj = base_loss.item() + gm_sum
            if obj < best_loss:
                best_loss = obj
                best_weights = copy.deepcopy(model.state_dict())

    warmup_lr_applied = False
    _dbg(f"AdamW phase: epochs={epochs} lr={args.lr} gm_warmup_epochs={args.gm_warmup_epochs} "
         f"surgery_mode={args.gm_surgery_mode} use_constraint={use_constraint}")
    for epoch in range(epochs + 1):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = ids_loss(preds)
        vds_term = vds_mono_loss(preds)        # gds>=0 knee-bump penalty (0 when --vds_loss off)
        gm_losses = []
        gm_on = args.use_gm and epoch >= args.gm_warmup_epochs
        if gm_on:
            if (not warmup_lr_applied) and args.gm_warmup_lr is not None and args.gm_warmup_epochs > 0:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.gm_warmup_lr               # drop LR for the gm fine-tune phase
                warmup_lr_applied = True
                if scheduler is not None:
                    # the scheduler would otherwise overwrite pg["lr"] back to its own stored
                    # base_lrs on the next .step() -- rebase it onto the new warmup LR instead.
                    _remaining = max(epochs - epoch, 1)
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=max(_remaining // _N_LR_RESTARTS, 1), T_mult=1,
                        eta_min=args.gm_warmup_lr * 0.01)
            gm_losses = compute_gm_losses(preds)
            if use_constraint:
                # minimize gm subject to Ids <= target (plain true gradient, no surgery)
                total = sum(gm_losses) + ids_penalty(loss) + vds_term
                total.backward()
            else:
                # vds_term is a constraint, not a competing task -> fold into the base loss
                # so gradient surgery gives it a true (unprojected) gradient.
                apply_gradient_surgery(
                    optimizer, loss + vds_term, gm_losses, trainable_params(),
                    mode=args.gm_surgery_mode, max_gm_ratio=args.gm_max_ratio)
        else:
            (loss + vds_term).backward()                       # Ids-only (warm-up or use_gm off)

        if args.adamw_avoid_localmin:
            torch.nn.utils.clip_grad_norm_(trainable_params(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        consider(loss, gm_losses, gm_on)

        if epoch % 5000 == 0:
            with torch.no_grad():
                max_err = torch.max(torch.abs(preds - y_train_t)).item()
                idx = torch.argmax(torch.abs(preds - y_train_t))
                vds_err = X_train_t[idx, 1].item()
            print(f"Epoch {epoch:5d} | MSE Loss: {loss.item():.6e} | Max Abs Error: {max_err:.6f} at Vds={vds_err:.2f}")

    model.load_state_dict(best_weights)
    _dbg(f"AdamW phase done: best_loss={best_loss:.6e}")

    # L-BFGS POLISHING
    # --lbfgs_gm_aware: polish the combined Ids+gm objective so extra L-BFGS epochs
    #   sharpen gm instead of undoing it. Default (None) -> AUTO-ON whenever use_gm;
    #   pass --no-lbfgs_gm_aware for the legacy Ids-only polish. Either way, best-weights
    #   are selected on the combined objective when use_gm, so the polish can't trade gm away.
    #
    # IMPORTANT: gradient surgery is applied ONLY in the AdamW phase. The L-BFGS closure
    #   polishes the PLAIN combined loss (Ids + Σ gm) with its TRUE gradient -- it does NOT
    #   call apply_gradient_surgery. A surgically-modified gradient (projection/clamping in
    #   the bounded modes) is inconsistent with the returned loss, which makes L-BFGS's
    #   strong-Wolfe line search thrash/hang. Polishing the plain combined objective keeps
    #   gradient and loss consistent, so the line search behaves for ALL surgery modes,
    #   while surgery still does its conflict-resolution job during the bulk (AdamW) training.
    # --lbfgs_epochs / --lbfgs_max_iter: outer steps / inner iters (were hardcoded 5/200).
    if args.lbfgs_gm_aware is None:
        lbfgs_gm_aware = args.use_gm                       # auto: gm-aware whenever gm is on
    else:
        lbfgs_gm_aware = bool(args.lbfgs_gm_aware) and args.use_gm
    lbfgs_opt = optim.LBFGS(trainable_params(), max_iter=args.lbfgs_max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        lbfgs_opt.zero_grad()
        preds = model(X_train_t)
        loss = ids_loss(preds)
        vds_term = vds_mono_loss(preds)        # keep the gds>=0 penalty in the polish too
        if lbfgs_gm_aware:
            # plain combined objective -> true gradient (no surgery here); consistent
            # loss/grad so strong-Wolfe never thrashes regardless of gm_surgery_mode.
            # In epsilon-constraint mode the polish targets the same constrained objective.
            if use_constraint:
                total = sum(compute_gm_losses(preds)) + ids_penalty(loss) + vds_term
            else:
                total = loss + sum(compute_gm_losses(preds)) + vds_term
            total.backward()
            return total
        loss = loss + vds_term
        loss.backward()
        return loss

    _dbg(f"L-BFGS phase: steps={args.lbfgs_epochs} max_iter={args.lbfgs_max_iter} gm_aware={lbfgs_gm_aware}")
    for step in range(args.lbfgs_epochs):
        lbfgs_opt.step(closure)
        preds = model(X_train_t)
        base = ids_loss(preds)
        gm_losses = compute_gm_losses(preds) if args.use_gm else []
        consider(base, gm_losses, args.use_gm)   # L-BFGS is post-warmup -> gm_on = use_gm

    _dbg(f"L-BFGS phase done: best_loss={best_loss:.6e}")
    print(f"Final polished objective: {best_loss:.6e}")

    model.load_state_dict(best_weights)

    # Ids RMSE computed DIRECTLY from predictions (sqrt(mean((pred - y)^2))),
    # NOT as best_loss ** 0.5. best_loss is the optimization objective; that only
    # equals MSE while `criterion` is nn.MSELoss(). Deriving the RMSE from the
    # residuals keeps it correct if the loss is ever changed (MAE/Huber/weighted)
    # and makes it match plot_saved_state.py, which recomputes it the same way.
    with torch.no_grad():
        _ids_preds = model(X_train_t)
        ids_mse = float(torch.mean((_ids_preds - y_train_t) ** 2).item())
        ids_rmse = float(ids_mse ** 0.5)

    eval_gm1, eval_gm2, eval_gm3 = (0.0, 0.0, 0.0)
    # GM RMSEs must be evaluated on the SAME row ordering as the gm-truth arrays
    # (create_gms_for_train sorts rows by Step_Index, TN). When use_gm is False,
    # X_train_t is in the original/unsorted order, so the gm_pred would be
    # misaligned with gm_true. Always evaluate on a freshly gm-ordered tensor so
    # the saved gm RMSEs match what plot_saved_state.py recomputes. (When use_gm
    # is True, X_train_gm == X_train, so this is identical to the old path.)
    if len(gm1_true_arr) > 0:
        X_train_gm_t = torch.tensor(X_train_gm, dtype=torch.float64, device=device)
        eval_gm1, eval_gm2, eval_gm3, pred_gm1, pred_gm2, pred_gm3 = get_gm_rmse_metrics(
            model, X_train_gm_t, gm1_true_arr, gm2_true_arr, gm3_true_arr
        )

    # --- SAVE THE BEST CONFIG AND WEIGHTS ---
    # Put exactly where asked (relative to root)
    save_dir = os.path.join(args.output_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    run_info = {
        'seed': args.seed,
        'deterministic': bool(args.deterministic),
        'mixed_init': args.mixed_init,
        'objective_loss': best_loss,   # the value `criterion` minimized (MSE today)
        'mse_loss': ids_mse,           # true Ids MSE, loss-agnostic
        'ids_rmse': ids_rmse,          # true Ids RMSE, loss-agnostic
        'gm1_rmse': eval_gm1,
        'gm2_rmse': eval_gm2,
        'gm3_rmse': eval_gm3,
        'learning_rate': args.lr,
        'epochs': args.epochs,
        'architecture': args.architecture if args.architecture else arch_tag,
        'output_activation': args.output_activation,   # SIMPLEGATE: also IS the vdsgate* gate,
                                   # no separate field needed; plot_saved_state.py reads this
                                   # back to reconstruct the model
        'ids_out_margin': args.ids_out_margin,
        'ids_scale': ids_scale,   # the actual multiplier used (1.0 if ids_out_margin<=0); needed
                                   # to reproduce predictions later (plot_saved_state.py reads this)
        'hybrid_normalization': True,
        'knee_combiner': args.knee_combiner,
        'knee_alpha_scale': args.knee_alpha_scale,
        'knee_vgs_thr': args.knee_vgs_thr,
        'knee_vgs_tau': args.knee_vgs_tau,
        'knee_max_correction': args.knee_max_correction,
        'optimizer': 'AdamW + L-BFGS',
        'freeze_physics': freeze_physics,
        'use_opt_params': use_previously_optimized_params,
        # Seed path so plot_saved_state.py can rebuild the SAME dynamic tight-prior
        # physics config (bounds around this seed) when use_opt_params is true.
        'opt_params_path': os.path.abspath(previously_optimized_params_path)
                            if previously_optimized_params_path else None,
        'use_gm': args.use_gm,
        'gm1_weight': args.gm1_weight if args.use_gm else 0.0,
        'gm2_weight': args.gm2_weight if args.use_gm else 0.0,
        'gm3_weight': args.gm3_weight if args.use_gm else 0.0,
        'gm_surgery_mode': args.gm_surgery_mode if args.use_gm else 'none',
        'gm_warmup_epochs': args.gm_warmup_epochs,
        'gm_warmup_lr': args.gm_warmup_lr,
        'ids_constraint': bool(args.ids_constraint),
        'ids_target': args.ids_target,
        'ids_lambda': args.ids_lambda if args.ids_constraint else None,
        'gm_vds_min': args.gm_vds_min,
        'gm_vgs_min': args.gm_vgs_min,
        'ids_region_center': args.ids_region_center,
        'ids_region_width': args.ids_region_width,
        'ids_region_weight': args.ids_region_weight,
        'ids_region_lo': args.ids_region_lo,
        'ids_region_hi': args.ids_region_hi,
        'region_vgs_lo': args.region_vgs_lo,
        'region_vgs_hi': args.region_vgs_hi,
        'region_vds_lo': args.region_vds_lo,
        'region_vds_hi': args.region_vds_hi,
        'region_weight': args.region_weight,
        'vds_loss': args.vds_loss,
        'equation_type': args.equation_type,
        # Config/DEFAULTS-level knobs that used to live only in run_log's CMD line.
        # Recorded here so the JSON is a complete, structured record of the run
        # (older runs lack these -> compile falls back to run_log.txt.gz).
        'gm_max_ratio': args.gm_max_ratio,
        'lbfgs_epochs': args.lbfgs_epochs,
        'lbfgs_max_iter': args.lbfgs_max_iter,
        'lbfgs_gm_aware': lbfgs_gm_aware,
        'loss_norm': args.loss_norm,
        'adamw_avoid_localmin': bool(args.adamw_avoid_localmin),
        'csv': CSV_PATH,
    }
    if args.config_name:
        run_info['config_name'] = args.config_name
    
    # Create a unique signature using the hyperparameters and the loss
    name_str = f"_{args.config_name}" if args.config_name else f"_Frz{int(freeze_physics)}_Opt{int(use_previously_optimized_params)}"
    signature = f"{name_str}_{arch_tag}_a{args.knee_alpha_scale}_{args.knee_combiner}_lr{args.lr}"
    loss_str = f"{best_loss:.4e}".replace('.', '_') + signature
    # Sanitize: equation_type/config_name may contain ':' (e.g. 'pure:vdsgate', 'noNN_knee:...')
    # or other Windows-illegal chars. NTFS treats 'name:x' as an alternate data stream, which
    # silently writes the content to an invisible stream and leaves a 0-byte, extensionless file.
    # Map any illegal filename char to '-'.
    for _ch in ':*?"<>|/\\':
        loss_str = loss_str.replace(_ch, '-')

    json_path = os.path.join(save_dir, f'run_loss_{loss_str}.json')
    weights_path = os.path.join(save_dir, f'weights_loss_{loss_str}.pt')
    script_save_path = os.path.join(save_dir, f'script_loss_{loss_str}.py')
    plot_save_path = os.path.join(save_dir, f'plot_loss_{loss_str}.png')
    
    run_info['weights_path'] = os.path.abspath(weights_path)
    run_info['plot_path'] = os.path.abspath(plot_save_path)
    run_info['script_backup_path'] = os.path.abspath(script_save_path)
    run_info['experiment_runner_script'] = os.path.abspath(os.path.join(save_dir, '..', 'run_experiments_used.py'))
    
    import json # Make absolutely sure it's globally imported here in main scope
    
    with open(json_path, 'w') as f:
        json.dump(run_info, f, indent=4)

    torch.save(best_weights, weights_path)
    _dbg(f"SAVED: run_loss -> {os.path.basename(json_path)} | weights -> {os.path.basename(weights_path)} | "
         f"region_weight={run_info.get('region_weight')} vds_loss={run_info.get('vds_loss')}")
    if not args.no_script_copy:
        shutil.copy2(os.path.abspath(__file__), script_save_path)

    print(f"Saved run config to {json_path}")
    print(f"Saved model weights to {weights_path}")
    if not args.no_script_copy:
        print(f"Saved script backup to {script_save_path}")
    else:
        print("[no_script_copy] Skipped per-run script backup.")

    # --- PLOTTING ---
    if args.no_plot:
        print("[no_plot] Skipped plot generation.")
        return
    print("Generating comprehensive 6-panel IV and GM curves plot...")
    from per_neuron_plotting import generate_physics_plot_data, plot_grid
    
    with torch.no_grad():
        final_preds = model(X_train_t).cpu().numpy().flatten()
        max_err = np.max(np.abs(y_train.flatten() - final_preds))
        
    vds_vals = sorted(list(np.unique(T_train['Vds'])))
    unique_vds_nominal = [vds_vals[0], vds_vals[len(vds_vals)//2], vds_vals[-1]] if len(vds_vals) > 3 else vds_vals
    
    vgs_vals = sorted(list(np.unique(T_train['Vgs'])))
    unique_vgs_nominal = [vgs_vals[0], vgs_vals[len(vgs_vals)//4], vgs_vals[len(vgs_vals)//2], vgs_vals[(len(vgs_vals)*3)//4], vgs_vals[-1]] if len(vgs_vals) > 5 else vgs_vals

    # CLI overrides for plot sweep targets
    if args.plot_vds_list:
        unique_vds_nominal = [float(x) for x in args.plot_vds_list.split(',') if x.strip()]
    if args.plot_vgs_list:
        unique_vgs_nominal = [float(x) for x in args.plot_vgs_list.split(',') if x.strip()]
    
    # Use the directly-computed Ids RMSE (loss-agnostic), consistent with
    # run_loss_*.json['ids_rmse'] and plot_saved_state.py.
    calc_metrics = {
        'rmse_ids': ids_rmse,
        'rmse_gm1': eval_gm1,
        'rmse_gm2': eval_gm2,
        'rmse_gm3': eval_gm3
    }
    eq_metrics = {'mae': max_err, 'rmse': ids_rmse}
    
    saved_metrics = {k: None for k in ['new_ids', 'new_gm1', 'new_gm2', 'new_gm3', 'base_ids', 'base_gm1', 'base_gm2', 'base_gm3']}
    
    plot_data = generate_physics_plot_data(
          model, T_train, 'Vgs_meas', 'Vds', 'Ids', unique_vds_nominal, unique_vgs_nominal, device
    )
    
    title_str = f"Performance: {arch_tag} | alpha: {args.knee_alpha_scale} | lr: {args.lr}"
    plot_grid(plot_data, calc_metrics, saved_metrics, eq_metrics, plot_save_path, 
              title_str, unique_vds_nominal, unique_vgs_nominal, use_real_voltages=False)

    print(f"Saved plot to: {plot_save_path}")
    
    if not args.hide_plot:
        try:
            # We open the image via PIL so we don't break GUI threads on headless contexts.
            from PIL import Image
            img = Image.open(plot_save_path)
            img.show()
        except Exception as _img_err:
            print(f"[warn] Failed to open plot via PIL: {_img_err}")

    with torch.no_grad():
        real_p = model.get_real_params()
        if not real_p:
            # Pure-NN modes have no physics params, so the gate-window report
            # is not applicable. Skip silently — this is informational only.
            print("\n--- GATE WINDOW INFO ---")
            print("Skipped: model has no physics params (pure-NN mode).")
        else:
            if 'alphaR' in real_p:
                _alpha_val = real_p['alphaR']
            elif 'alpha' in real_p:
                _alpha_val = real_p['alpha']
            else:
                raise KeyError(
                    "Expected 'alphaR' or 'alpha' in get_real_params() for gate-window report, "
                    f"but got keys: {sorted(real_p.keys())}"
                )
            # Static params come back as plain floats; trainable params come back as tensors.
            alpha = _alpha_val.item() if hasattr(_alpha_val, 'item') else float(_alpha_val)
            print(f"\n--- GATE WINDOW INFO ---")
            print(f"Learned Alpha parameter: {alpha:.4f}")

if __name__ == "__main__":
    main()
