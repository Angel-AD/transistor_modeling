"""
Generate 6-panel IV/GM plots either from:

  (a) a trained-model directory containing weights_loss_*.pt + run_*.json
      (the usual NN / physics+NN output of `per_neuron_simple_angelov_nn_test.py`)
      → use --dir <path>

  (b) a physics-only SLSQP seed JSON (produced by per_neuron_runner_8_slsqp_v2.py
      or copied to parallel_optimization/tests/slsqp_fast_physics_seed.json)
      → use --seed <path>   (no base NN; pure physics model)

Examples
--------
# Plot a trained run:
python plot_saved_state.py --dir master_experiments_dirs/physics_nn_no_gm/exp_001_none_W0.0-0.0-0.0

# Plot a fresh SLSQP seed:
python plot_saved_state.py --seed parallel_optimization/tests/slsqp_fast_physics_seed.json
"""
import argparse
import ast
import glob
import json
import math
import os
import sys

import numpy as np
import torch

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
from per_neuron_models import DynamicNN, PhysicsInformedNN
from per_neuron_normalization import USED_NORMALIZATIONS
from per_neuron_plotting import generate_physics_plot_data, plot_grid
# Reuse the exact gm-truth / gm-rmse logic from the NN training pipeline so that
# the SLSQP plots report Calculated RMSE values comparable to physics+NN plots.
sys.path.insert(0, _HERE)
from per_neuron_simple_angelov_nn_test import create_gms_for_train, get_gm_rmse_metrics, get_physics_config


# --------------------------------------------------------------------- #
# Builders: two ways to construct the model + decide the output path     #
# --------------------------------------------------------------------- #
def _build_model_from_dir(save_dir, device):
    """Load architecture from run_*.json and weights from weights_loss_*.pt."""
    run_json_files = glob.glob(os.path.join(save_dir, 'run_*.json'))
    if not run_json_files:
        raise FileNotFoundError(
            f"No run_*.json file found in {save_dir}. Cannot reconstruct architecture/equation_type "
            "without it; refusing to silently assume a default 4x4 tanh net."
        )
    run_json_path = max(run_json_files, key=os.path.getmtime)
    with open(run_json_path, 'r') as f:
        run_data = json.load(f)
    if 'equation_type' not in run_data:
        run_data['equation_type'] = 'pure'  # back-compat for older runs that didn't log this
    for _req in ('architecture', 'output_activation', 'equation_type'):
        if _req not in run_data:
            raise KeyError(
                f"{run_json_path} is missing required key {_req!r}. "
                "Refusing to silently fall back to a default."
            )
    arch_list = ast.literal_eval(run_data['architecture'])
    neurons_per_layer = [len(layer) for layer in arch_list]
    activations_per_layer = arch_list
    out_activation = run_data['output_activation']
    print(f"Detected architecture from config: {neurons_per_layer} with {out_activation}")
    config_name = run_data.get('config_name', '')
    model_arch = run_data['equation_type']

    weights_files = glob.glob(os.path.join(save_dir, 'weights_loss_*.pt'))
    if not weights_files:
        raise FileNotFoundError(f"No weights found in {save_dir} folder!")
    weights_path = max(weights_files, key=os.path.getmtime)
    print(f"Loading latest weights from: {weights_path}")

    base_model = DynamicNN(
        input_dim=2,
        neurons_per_layer=neurons_per_layer,
        activations_per_layer=activations_per_layer,
        output_dim=1,
        output_activation=out_activation,
    ).double()

    _norm_key = 'cgh40010f_vgs4_vds45'
    if _norm_key not in USED_NORMALIZATIONS:
        raise KeyError(
            f"Normalization key {_norm_key!r} not in USED_NORMALIZATIONS. "
            f"Available: {sorted(USED_NORMALIZATIONS.keys())}"
        )
    norm_c = dict(USED_NORMALIZATIONS[_norm_key])
    norm_c['hybrid'] = True

    # Reconstruct the SAME physics config_key the run was TRAINED with. For
    # use_opt_params=true runs the trainer builds a dynamic tight-prior config
    # (bounds around the SLSQP seed) via get_physics_config and uses that key, NOT
    # 9. Rebuilding it here (from the seed path logged in run_*.json) is required
    # or the raw->real parameter bounds differ and the predictions/RMSE won't
    # match. use_opt_params=false runs resolve to base_key=9 (unchanged).
    use_opt = bool(run_data.get('use_opt_params', False))
    freeze_phys = bool(run_data.get('freeze_physics', False))
    opt_params_path = run_data.get('opt_params_path')
    if use_opt and opt_params_path and os.path.exists(opt_params_path):
        physics_c = get_physics_config(
            use_previously_optimized_params=True,
            previously_optimized_params_path=opt_params_path,
            freeze_physics=freeze_phys,
            base_key=9,
            width_percent=0.10,
            equation_type=model_arch,
        )
    else:
        if use_opt:
            print(f"[plot_saved_state] WARNING: run used use_opt_params=true but its "
                  f"seed path is missing/unavailable ({opt_params_path!r}); falling back "
                  "to base config 9. Reconstructed RMSE may not match.")
        physics_c = 9

    # vdsgate gate mode (softplus=legacy default, tanhm=raw-NN gate) -- orthogonal to
    # equation_type; older runs predate this field and default to the legacy 'softplus'.
    vdsgate_gate_mode = run_data.get('vdsgate_output_activation', 'softplus')
    model = PhysicsInformedNN(
        base_nn=base_model,
        equation_type=model_arch,
        normalization_config=norm_c,
        config_key=physics_c,
        vdsgate_gate_mode=vdsgate_gate_mode,
    ).double()
    # Reconstruct the knee configuration the run was TRAINED with, rather than
    # hardcoding defaults. A run trained with e.g. knee_combiner="product" must
    # be evaluated the same way or its predictions (and RMSE) will not match
    # run_loss_*.json. Fall back to the historical defaults only if absent.
    model.knee_alpha_scale = float(run_data.get('knee_alpha_scale', 1.0))
    model.knee_use_alpha_eff = True
    model.knee_combiner = run_data.get('knee_combiner', 'sum')
    model.freeze_physics = bool(run_data.get('freeze_physics', False))

    sd = torch.load(weights_path, map_location=device)
    # strict=False so legacy checkpoints with extra/renamed buffers still load,
    # but surface any mismatch: silently dropped tensors would change the model
    # (and its RMSE) without warning.
    incompatible = model.load_state_dict(sd, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(f"[plot_saved_state] WARNING: state_dict mismatch loading {weights_path}\n"
              f"    missing_keys    = {list(incompatible.missing_keys)}\n"
              f"    unexpected_keys = {list(incompatible.unexpected_keys)}")
    model.eval().to(device)

    # --ids_out_margin (bounded output activations, e.g. sigmoid/tanh): the run predicted
    # ids_scale*activation(NN(...)), not the raw activation output. Reapply the SAME scaling
    # here so every caller (region metrics, shape analysis, plotting, RMSE) sees real-unit
    # predictions -- state_dict/attributes are untouched, only the instance's forward is
    # shadowed (mirrors the training-time patch in per_neuron_simple_angelov_nn_test.py).
    ids_scale = float(run_data.get('ids_scale', 1.0))
    if ids_scale != 1.0:
        _orig_forward = model.forward
        model.forward = lambda x, _f=_orig_forward, _s=ids_scale: _s * _f(x)
        print(f"[plot_saved_state] ids_scale={ids_scale:.6f} reapplied (ids_out_margin="
              f"{run_data.get('ids_out_margin')})")

    return model


def _to_raw(real_val, name, mode, config_key, eq_name):
    """Invert real = min + sigmoid(raw)*(max-min) → raw = logit((real-min)/(max-min))."""
    cfg = PhysicsInformedNN.calc_bound_config(name, mode, config_key, eq_name)
    if not isinstance(cfg, dict):
        return None  # static / fixed param
    lo, hi = cfg["min"], cfg["max"]
    eps = 1e-6
    p = (float(real_val) - lo) / (hi - lo)
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _build_model_from_seed(seed_path, eq_name, config_key, device):
    """Build a noNN PhysicsInformedNN and inject SLSQP-optimized params."""
    with open(seed_path, 'r') as f:
        data = json.load(f)
    opt_params = data['results'][0]['optimized_params']

    _norm_key = 'cgh40010f_vgs4_vds45'
    if _norm_key not in USED_NORMALIZATIONS:
        raise KeyError(
            f"Normalization key {_norm_key!r} not in USED_NORMALIZATIONS. "
            f"Available: {sorted(USED_NORMALIZATIONS.keys())}"
        )
    norm_c = dict(USED_NORMALIZATIONS[_norm_key])
    norm_c['hybrid'] = True
    model = PhysicsInformedNN(
        base_nn=None,
        equation_type=f"noNN:{eq_name}",
        normalization_config=norm_c,
        config_key=config_key,
    ).double().to(device)

    injected, skipped = [], []
    for k, v in opt_params.items():
        raw_attr = f"{k}_raw"
        if not hasattr(model, raw_attr):
            skipped.append((k, "no _raw attr")); continue
        raw = _to_raw(v, k, "noNN", config_key, eq_name)
        if raw is None:
            skipped.append((k, "static")); continue
        getattr(model, raw_attr).data = torch.tensor(raw, dtype=torch.float64, device=device)
        injected.append(k)
    model.eval()
    print(f"Injected {len(injected)}/{len(opt_params)} params; skipped={skipped}")
    return model


def _load_T_train(csv, min_vgs, device, add_zero_vds=False):
    """Load measurement table with the canonical plot/eval kwargs."""
    meas_load_kwargs = {
        "file_type": "auriga", "keep_every_N_group": 0,
        "remove_negative_vds": 0, "remove_negative_ids": 0,
        "test_percent": 0.0, "num_extrapolate_groups_start": 0,
        "min_vgs": min_vgs,
        "add_zero_vds": add_zero_vds,
    }
    T_train, _ = meas_load(csv, **meas_load_kwargs)
    return T_train


def evaluate_dir_ids_rmse(save_dir, csv, min_vgs=None, device=None):
    """Rebuild the trained model from `save_dir` and return its Ids RMSE on the
    full measurement set.

    This is the SAME quantity stored in run_loss_*.json as 'ids_rmse'
    (= sqrt(best MSE) during training), so callers can assert the reconstructed
    model reproduces the logged metric. RMSE is order-invariant and
    create_gms_for_train only reorders rows, so this matches both the plain and
    gm-ordered conventions.
    """
    device = device or torch.device("cpu")
    T_train = _load_T_train(csv, min_vgs, device)
    X_train = np.column_stack((T_train['Vgs_meas'], T_train['Vds']))
    y_train = np.array(T_train['Ids']).reshape(-1, 1)
    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    model = _build_model_from_dir(os.path.abspath(save_dir), device)
    with torch.no_grad():
        preds = model(X_train_t).cpu().numpy().flatten()
    return float(np.sqrt(np.mean((y_train.flatten() - preds) ** 2)))


def evaluate_seed_ids_rmse(seed_path, eq_name, config_key, csv, min_vgs=None, device=None):
    """Inject an SLSQP seed's optimized_params into a noNN PhysicsInformedNN and
    return its Ids RMSE on the full measurement set.

    This is the same quantity per_neuron_slsqp_single.py stores in the seed JSON
    as results[0]['ids_rmse'], so callers can assert the --seed reconstruction
    reproduces the logged metric (modulo the raw<->real sigmoid round-trip).
    """
    device = device or torch.device("cpu")
    T_train = _load_T_train(csv, min_vgs, device)
    X_train = np.column_stack((T_train['Vgs_meas'], T_train['Vds']))
    y_train = np.array(T_train['Ids']).reshape(-1, 1)
    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)
    model = _build_model_from_seed(os.path.abspath(seed_path), eq_name, config_key, device)
    with torch.no_grad():
        preds = model(X_train_t).cpu().numpy().flatten()
    return float(np.sqrt(np.mean((y_train.flatten() - preds) ** 2)))


def _saved_metrics_from_dir(save_dir):
    """Read the RMSEs the trainer logged into the latest run_*.json (the values
    shown as 'Saved JSON base RMSE' in the plot info box)."""
    run_json_files = glob.glob(os.path.join(save_dir, 'run_*.json'))
    if not run_json_files:
        return {}
    with open(max(run_json_files, key=os.path.getmtime), 'r') as f:
        d = json.load(f)
    return {'ids': d.get('ids_rmse'), 'gm1': d.get('gm1_rmse'),
            'gm2': d.get('gm2_rmse'), 'gm3': d.get('gm3_rmse')}


def _saved_metrics_from_seed(seed_path):
    """Read the RMSEs per_neuron_slsqp_single.py logged into the seed JSON."""
    with open(seed_path, 'r') as f:
        d = json.load(f)
    e = d['results'][0]
    return {'ids': e.get('ids_rmse'), 'gm1': e.get('gm1_rmse'),
            'gm2': e.get('gm2_rmse'), 'gm3': e.get('gm3_rmse')}


# --------------------------------------------------------------------- #
# NN Equation helpers                                                    #
# --------------------------------------------------------------------- #

def _act_numpy(name, x):
    """Numpy implementation of every activation DynamicNN supports."""
    n = name.lower()
    if n in ('linear', 'identity'):  return x
    if n == 'tanh':                  return np.tanh(x)
    if n == 'sigmoid':               return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    if n == 'relu':                  return np.maximum(0.0, x)
    if n == 'mish':                  return x * np.tanh(np.log1p(np.exp(np.clip(x, -500, 500))))
    if n == 'softplus':              return np.log1p(np.exp(np.clip(x, -500, 500)))
    if n == 'swish':                 return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))
    if n == 'sin':                   return np.sin(x)
    if n == 'cos':                   return np.cos(x)
    if n == 'sinh':                  return np.sinh(x)
    if n == 'cosh':                  return np.cosh(x)
    raise ValueError(f"[equation] Unknown activation: {name!r}")


def _extract_nn_layers(base_nn):
    """Return list of (W, b, activations_list) for every PerNeuronLinear in base_nn.net."""
    from per_neuron_models import PerNeuronLinear
    layers = []
    for layer in base_nn.net:
        if isinstance(layer, PerNeuronLinear):
            W    = layer.linear.weight.detach().cpu().numpy()   # [n_out, n_in]
            b    = layer.linear.bias.detach().cpu().numpy()     # [n_out]
            acts = list(layer.activations_list)
            layers.append((W, b, acts))
    return layers


def _eval_eq_numpy(layers_info, X_norm):
    """Pure-numpy forward pass — same arithmetic as the NN. X_norm: [N, n_in]."""
    h = X_norm.astype(np.float64)
    for W, b, acts in layers_info:
        pre = h @ W.T + b                       # [N, n_out]
        h   = np.empty_like(pre)
        for j, act in enumerate(acts):
            h[:, j] = _act_numpy(act, pre[:, j])
    return h.flatten()


def _fw(v):
    """Format weight/bias in scientific notation with 8 decimal places."""
    return f"{float(v):+.8e}"


def generate_and_write_equation_md(model, X_train_np, calc_metrics,
                                    plot_save_path, run_data=None, device=None):
    """Fold normalization into first-layer weights so the equation accepts raw
    Vgs / Vds.  Verify by evaluating the 8-decimal string coefficients against
    the full torch model.  Writes .md alongside the plot PNG.

    Returns eq_metrics {'mae', 'rmse'} or {} for non-DynamicNN models.
    """
    device = device or torch.device('cpu')
    from per_neuron_models import DynamicNN, aeff_spec, vdsk_spec

    base_nn = getattr(model, 'base_nn', None)
    if base_nn is None or not isinstance(base_nn, DynamicNN):
        return {}

    # --- Normalization params (from the PhysicsInformedNN wrapper) ---
    vgs_min   = float(getattr(model, 'vgs_min',   0.0))
    vgs_scale = float(getattr(model, 'vgs_scale', 1.0))   # 2 / (vgs_max - vgs_min)
    vds_min   = float(getattr(model, 'vds_min',   0.0))
    vds_scale = float(getattr(model, 'vds_scale', 1.0))
    vgs_max   = vgs_min + 2.0 / vgs_scale if vgs_scale else vgs_min + 4.2
    vds_max   = vds_min + 2.0 / vds_scale if vds_scale else vds_min + 45.2

    # --ids_out_margin: the trained model predicts ids_scale*activation(NN(...)). The torch
    # `model` passed in already has this folded into its (patched) forward, so torch_preds
    # below is automatically correct -- but eq_preds is a manual numpy reimplementation that
    # bypasses model.forward entirely, so it needs the multiplier applied explicitly.
    ids_scale = float((run_data or {}).get('ids_scale', 1.0))

    # --- Extract per-layer (W, b, activations) ---
    layers_info = _extract_nn_layers(base_nn)

    # --- vdsgate* structured output families ---
    #   vdsgate / vdsgatelin  : Ids = gate(NN) * tanh(alpha*Vds) * (1 + lamb*Vds)          [scalar alpha, lamb]
    #   vdsgate_aeff*         : Ids = softplus(NN) * tanh(a_eff(Vgs)*Vds) * (1 + l_eff(Vgs)*Vds)  [Vgs-poly alpha, lamb]
    #   vdsgate_vdsk*         : Ids = softplus(NN) * tanh(Vds/Vds_knee(Vgs)) * (1 + l_eff(Vgs)*Vds)
    _wrapper = getattr(model, 'nn_wrapper', 'identity')
    _is_vdsgate_simple = _wrapper in ('vdsgate', 'vdsgatelin')
    _is_vdsgate_aeff   = _wrapper.startswith('vdsgate_aeff')
    _is_vdsgate_vdsk   = _wrapper.startswith('vdsgate_vdsk')
    _is_vdsgate = _is_vdsgate_simple or _is_vdsgate_aeff or _is_vdsgate_vdsk
    _vg_alpha = _vg_lamb = None
    _a_coefs = _l_coefs = None
    _use_sig = _free_lam = False
    # Gate mode (softplus=legacy default, tanhm=raw-NN gate) is a model-level attribute
    # (see vdsgate_gate_mode / --vdsgate_output_activation), orthogonal to the wrapper string --
    # applies to ALL vdsgate*/vdsgate_aeff*/vdsgate_vdsk* families (not vdsgatelin, already raw).
    _tanhm_gate = getattr(model, 'vdsgate_gate_mode', 'softplus') == 'tanhm'
    if _is_vdsgate_simple:
        _vg_alpha = float(torch.nn.functional.softplus(model.vdsgate_alpha_raw).item())
        _vg_lamb  = float(model.vdsgate_lamb_raw.item())
    elif _is_vdsgate_aeff:
        _a_ord, _use_sig, _l_ord, _free_lam = aeff_spec(_wrapper)
        _a_coefs = [float(getattr(model, f'vdsgate_a{i}').item()) for i in range(_a_ord + 1)]
        _l_coefs = [float(getattr(model, f'vdsgate_l{j}').item()) for j in range(_l_ord + 1)]
    elif _is_vdsgate_vdsk:
        _a_ord, _l_ord, _free_lam = vdsk_spec(_wrapper)
        _a_coefs = [float(getattr(model, f'vdsgate_a{i}').item()) for i in range(_a_ord + 1)]
        _l_coefs = [float(getattr(model, f'vdsgate_l{j}').item()) for j in range(_l_ord + 1)]

    # --- Fold normalization into layer 0 so equation uses raw Vgs/Vds ---
    # pre_j = W0[j,0]*vgs_n + W0[j,1]*vds_n + b0[j]
    #   where vgs_n = (Vgs - vgs_min)*vgs_scale - 1
    # => pre_j = (W0[j,0]*vgs_scale)*Vgs + (W0[j,1]*vds_scale)*Vds
    #          + b0[j] - W0[j,0]*(vgs_scale*vgs_min + 1) - W0[j,1]*(vds_scale*vds_min + 1)
    W0, b0, acts0 = layers_info[0]
    W0_eff = np.empty_like(W0)
    b0_eff = np.empty_like(b0)
    for j in range(W0.shape[0]):
        W0_eff[j, 0] = W0[j, 0] * vgs_scale
        W0_eff[j, 1] = W0[j, 1] * vds_scale
        b0_eff[j] = (b0[j]
                     - W0[j, 0] * (vgs_scale * vgs_min + 1.0)
                     - W0[j, 1] * (vds_scale * vds_min + 1.0))
    layers_raw = [(W0_eff, b0_eff, acts0)] + list(layers_info[1:])

    # --- Truncate to 8 decimal places — exactly what the written string encodes ---
    def _trunc(arr):
        return np.array([float(f"{v:.8f}") for v in arr.flat],
                        dtype=np.float64).reshape(arr.shape)
    layers_trunc = [(_trunc(W), _trunc(b), acts) for W, b, acts in layers_raw]

    # --- Evaluate the 8-decimal equation on raw (unnormalized) inputs ---
    X_raw = X_train_np.astype(np.float64)
    eq_preds = _eval_eq_numpy(layers_trunc, X_raw)
    if _is_vdsgate_simple:
        _vds_col = X_raw[:, 1]
        if _wrapper == 'vdsgatelin':
            _gate_np = eq_preds                        # always raw, gate_mode ignored
        elif _tanhm_gate:
            _gate_np = np.tanh(eq_preds)
        else:
            _gate_np = np.log1p(np.exp(eq_preds))       # legacy softplus
        eq_preds = _gate_np * np.tanh(_vg_alpha * _vds_col) * (1.0 + _vg_lamb * _vds_col)
    elif _is_vdsgate_aeff or _is_vdsgate_vdsk:
        _vds_col = X_raw[:, 1]
        _vgs_norm_np = (X_raw[:, 0] - vgs_min) * vgs_scale - 1.0
        # aeff/vdsk gate via softplus(NN) (legacy) or tanh(NN) ('tanhm' gate mode) -- see
        # vdsgate_gate_mode / --vdsgate_output_activation. Self-contained: NOT dependent on
        # output_activation (use 'linear' for both modes, same as legacy).
        _gate_np = np.tanh(eq_preds) if _tanhm_gate else np.log1p(np.exp(eq_preds))
        _poly_a_np = sum(_a_coefs[i] * _vgs_norm_np ** i for i in range(len(_a_coefs)))
        _poly_l_np = sum(_l_coefs[j] * _vgs_norm_np ** j for j in range(len(_l_coefs)))
        _l_eff_np = _poly_l_np if _free_lam else np.log1p(np.exp(_poly_l_np))
        if _is_vdsgate_aeff:
            _a_eff_np = (2.0 / (1.0 + np.exp(-_poly_a_np))) if _use_sig else np.log1p(np.exp(_poly_a_np))
            eq_preds = _gate_np * np.tanh(_a_eff_np * _vds_col) * (1.0 + _l_eff_np * _vds_col)
        else:
            _vds_knee_np = np.log1p(np.exp(_poly_a_np)) + 1e-4
            eq_preds = _gate_np * np.tanh(_vds_col / _vds_knee_np) * (1.0 + _l_eff_np * _vds_col)
    if ids_scale != 1.0:
        eq_preds = eq_preds * ids_scale

    # --- Compare against full torch model (normalises internally) ---
    with torch.no_grad():
        torch_preds = model(
            torch.tensor(X_raw, dtype=torch.float64, device=device)
        ).cpu().numpy().flatten()

    eq_mae  = float(np.mean(np.abs(eq_preds - torch_preds)))
    eq_rmse = float(np.sqrt(np.mean((eq_preds - torch_preds) ** 2)))
    eq_metrics = {'mae': eq_mae, 'rmse': eq_rmse}

    # --- ADS-compatible activation string ---
    def _act_str(act, expr):
        """Expand activation to exponential form supported by ADS."""
        a = act.lower()
        if a in ('linear', 'identity'): return f"({expr})"
        if a == 'tanh':                 return f"tanh({expr})"
        if a == 'sin':                  return f"sin({expr})"
        if a == 'cos':                  return f"cos({expr})"
        if a == 'sinh':                 return f"sinh({expr})"
        if a == 'cosh':                 return f"cosh({expr})"
        if a == 'relu':                 return f"max(0,({expr}))"
        if a == 'sigmoid':              return f"(1/(1+exp(-({expr}))))"
        if a == 'softplus':             return f"ln(1+exp({expr}))"
        if a == 'swish':                return f"(({expr})/(1+exp(-({expr}))))"
        if a == 'mish':                 return f"(({expr})*tanh(ln(1+exp({expr}))))"
        return f"{act}({expr})"

    def _poly_expr(coefs, x_expr):
        """c0 + c1*x + c2*x*x + ... as a string, x_expr substituted for x."""
        terms = []
        for i, c in enumerate(coefs):
            if i == 0:
                terms.append(f"({_fw(c)})")
            else:
                terms.append(f"({_fw(c)})*({'*'.join([x_expr] * i)})")
        return "(" + " + ".join(terms) + ")"

    _vgs_norm_ads = f"((_v1 - ({vgs_min:.8f})) * ({vgs_scale:.8f}) - 1.0)"

    def _wrap_out_ads(nn_raw_expr):
        """Wrap the bare NN expression with the vdsgate structured output (ADS form, _v2 = Vds),
        and the ids_scale multiplier (--ids_out_margin) if this run used one."""
        if _is_vdsgate_simple:
            if _wrapper == 'vdsgatelin':
                gate = f"({nn_raw_expr})"
            elif _tanhm_gate:
                gate = f"tanh({nn_raw_expr})"
            else:
                gate = f"ln(1+exp({nn_raw_expr}))"
            out = f"({gate}) * tanh(({_vg_alpha:.8f})*_v2) * (1 + ({_vg_lamb:.8f})*_v2)"
        elif _is_vdsgate_aeff or _is_vdsgate_vdsk:
            gate = f"tanh({nn_raw_expr})" if _tanhm_gate else f"ln(1+exp({nn_raw_expr}))"
            poly_a = _poly_expr(_a_coefs, _vgs_norm_ads)
            poly_l = _poly_expr(_l_coefs, _vgs_norm_ads)
            l_eff = poly_l if _free_lam else f"ln(1+exp({poly_l}))"
            if _is_vdsgate_aeff:
                a_eff = f"(2/(1+exp(-{poly_a})))" if _use_sig else f"ln(1+exp({poly_a}))"
                out = f"({gate}) * tanh(({a_eff})*_v2) * (1 + ({l_eff})*_v2)"
            else:
                vds_knee = f"(ln(1+exp({poly_a})) + 0.0001)"
                out = f"({gate}) * tanh(_v2/({vds_knee})) * (1 + ({l_eff})*_v2)"
        else:
            out = nn_raw_expr
        return f"({ids_scale:.8f}) * ({out})" if ids_scale != 1.0 else out

    # --- Build full expanded equation (all layers substituted, ADS form) ---
    def _build_full_eq(layers):
        exprs = ['_v1', '_v2']
        for W, b, acts in layers[:-1]:
            new_exprs = []
            for j in range(W.shape[0]):
                inner = " + ".join(f"({_fw(W[j, k])})*{exprs[k]}"
                                   for k in range(len(exprs)))
                inner += f" + ({_fw(b[j])})"
                new_exprs.append(_act_str(acts[j], inner))
            exprs = new_exprs
        W_o, b_o, acts_o = layers[-1]
        inner = " + ".join(f"({_fw(W_o[0, k])})*{exprs[k]}"
                           for k in range(len(exprs)))
        inner += f" + ({_fw(b_o[0])})"
        return _wrap_out_ads(_act_str(acts_o[0], inner))

    full_eq_str = _build_full_eq(layers_raw)

    # --- Build MD ---
    md = []
    md.append("# NN Model Equation  (raw inputs — normalization baked in)\n")
    md.append("    _v1 = Vgs  [V]")
    md.append("    _v2 = Vds  [V]\n")

    if run_data:
        md.append("## Config\n")
        for k in ('config_name', 'equation_type', 'architecture',
                  'gm_surgery_mode', 'gm1_weight', 'gm2_weight', 'gm3_weight',
                  'output_activation', 'lr', 'seed', 'epochs',
                  'gm_vds_min', 'gm_vgs_min', 'gm_warmup_epochs', 'ids_constraint',
                  'ids_target', 'ids_lambda'):
            v = run_data.get(k)
            if v is not None:
                md.append(f"- {k}: {v}")
        md.append("")

    md.append("## Normalization reference  (already baked into Layer 0 weights)\n")
    md.append(f"    vgs_norm = (_v1 - ({vgs_min:.4f})) * {vgs_scale:.8e} - 1.0")
    md.append(f"             [training domain: _v1 in [{vgs_min:.2f}, {vgs_max:.2f}] V -> [-1,+1]]")
    md.append(f"    vds_norm = (_v2 - ({vds_min:.4f})) * {vds_scale:.8e} - 1.0")
    md.append(f"             [training domain: _v2 in [{vds_min:.2f}, {vds_max:.2f}] V -> [-1,+1]]")
    md.append("\nEquations below accept RAW _v1, _v2 (Volts). Output Ids is in Amperes.\n")

    n_hidden = [W.shape[0] for W, _, _ in layers_raw[:-1]]
    out_act  = layers_raw[-1][2][0]
    md.append(f"## Architecture: hidden layers {n_hidden}, output = {out_act}\n")

    # Layer-by-layer form (ADS-expanded activations)
    prev = ['_v1', '_v2']
    for li, (W, b, acts) in enumerate(layers_raw[:-1]):
        n_out = W.shape[0]
        md.append(f"## Layer {li}  ({n_out} neurons)\n")
        for j in range(n_out):
            inner = " + ".join(f"({_fw(W[j, k])})*{prev[k]}"
                               for k in range(len(prev)))
            inner += f" + ({_fw(b[j])})"
            md.append(f"    h{li}_{j} = {_act_str(acts[j], inner)}")
        md.append("")
        prev = [f"h{li}_{j}" for j in range(n_out)]

    W_o, b_o, acts_o = layers_raw[-1]
    inner = " + ".join(f"({_fw(W_o[0, k])})*{prev[k]}" for k in range(len(prev)))
    inner += f" + ({_fw(b_o[0])})"
    if _is_vdsgate_simple:
        _gate_label = "NN" if _wrapper == 'vdsgatelin' else ("tanh(NN)" if _tanhm_gate else "softplus(NN)")
        _out_label = f"{acts_o[0]} -> {_gate_label}*tanh(a*Vds)*(1+l*Vds)  [{_wrapper}]"
    elif _is_vdsgate_aeff:
        _gate_label = "tanh(NN)" if _tanhm_gate else "softplus(NN)"
        _out_label = f"{acts_o[0]} -> {_gate_label}*tanh(a_eff(Vgs)*Vds)*(1+l_eff(Vgs)*Vds)  [{_wrapper}]"
    elif _is_vdsgate_vdsk:
        _gate_label = "tanh(NN)" if _tanhm_gate else "softplus(NN)"
        _out_label = f"{acts_o[0]} -> {_gate_label}*tanh(Vds/Vds_knee(Vgs))*(1+l_eff(Vgs)*Vds)  [{_wrapper}]"
    else:
        _out_label = acts_o[0]
    md.append(f"## Output  (activation = {_out_label})\n")
    md.append(f"    Ids [A] = {_wrap_out_ads(_act_str(acts_o[0], inner))}\n")
    if _is_vdsgate_simple:
        md.append(f"    [vdsgate params: alpha={_vg_alpha:.6f} (>0, softplus), lamb={_vg_lamb:.6f}]\n")
    elif _is_vdsgate_aeff or _is_vdsgate_vdsk:
        _a_str = ", ".join(f"a{i}={c:.6f}" for i, c in enumerate(_a_coefs))
        _l_str = ", ".join(f"l{j}={c:.6f}" for j, c in enumerate(_l_coefs))
        _a_kind = "alpha(Vgs_norm) poly coefs" if _is_vdsgate_aeff else "Vds_knee(Vgs_norm) poly coefs (pre-softplus)"
        _l_kind = "raw (unconstrained)" if _free_lam else "pre-softplus (l_eff=softplus(poly)>=0)"
        md.append(f"    [{_wrapper} params: {_a_kind}: {_a_str}]")
        md.append(f"    [lamb(Vgs_norm) poly coefs, {_l_kind}: {_l_str}]\n")

    # Full expanded equation (all layers substituted)
    md.append("## Full Equation  (all layers substituted, only _v1 and _v2 as inputs)\n")
    md.append(f"    Ids [A] = {full_eq_str}\n")

    md.append("## Verification: equation string (8 decimals) vs torch model\n")
    md.append("Equation coefficients truncated to 8 decimal places, evaluated on the")
    md.append("full training set with raw _v1/_v2, compared against the torch model:\n")
    md.append(f"    RMSE (eq_string - torch): {eq_rmse:.4e} A")
    md.append(f"    MAE  (eq_string - torch): {eq_mae:.4e}  A")
    md.append("\n(Residual is truncation error from 8 decimal places — should be ~1e-6 or smaller.)\n")

    def _fm(v):
        return f"{float(v):.4e}" if v not in (None, '', 'N/A') else 'N/A'

    md.append("## Model RMSEs vs Measured Data\n")
    md.append(f"    Ids RMSE : {_fm(calc_metrics.get('rmse_ids'))}  A")
    md.append(f"    Gm1 RMSE : {_fm(calc_metrics.get('rmse_gm1'))}  A/V")
    md.append(f"    Gm2 RMSE : {_fm(calc_metrics.get('rmse_gm2'))}  A/V^2")
    md.append(f"    Gm3 RMSE : {_fm(calc_metrics.get('rmse_gm3'))}  A/V^3\n")

    md_path = os.path.splitext(plot_save_path)[0] + '.md'
    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(md) + '\n')
    print(f"[plot_saved_state] Equation MD written: {md_path}")

    # --- Verilog-A ---
    def _act_str_va(act, expr):
        """Activation in standard Verilog-A syntax (ln = natural log)."""
        a = act.lower()
        if a in ('linear', 'identity'): return f"({expr})"
        if a == 'tanh':                 return f"tanh({expr})"
        if a == 'sin':                  return f"sin({expr})"
        if a == 'cos':                  return f"cos({expr})"
        if a == 'sinh':                 return f"sinh({expr})"
        if a == 'cosh':                 return f"cosh({expr})"
        if a == 'relu':                 return f"(({expr}) > 0.0 ? ({expr}) : 0.0)"
        if a == 'sigmoid':              return f"(1.0/(1.0+exp(-({expr}))))"
        if a == 'softplus':             return f"ln(1.0+exp({expr}))"
        if a == 'swish':                return f"(({expr})/(1.0+exp(-({expr}))))"
        if a == 'mish':                 return f"(({expr})*tanh(ln(1.0+exp({expr}))))"
        return f"{act}({expr})"

    _vgs_norm_va = f"((Vgs - ({vgs_min:.8f})) * ({vgs_scale:.8f}) - 1.0)"

    def _wrap_out_va(nn_raw_expr):
        """Wrap the bare NN expression with the vdsgate structured output (Verilog-A form),
        and the ids_scale multiplier (--ids_out_margin) if this run used one."""
        if _is_vdsgate_simple:
            if _wrapper == 'vdsgatelin':
                gate = f"({nn_raw_expr})"
            elif _tanhm_gate:
                gate = f"tanh({nn_raw_expr})"
            else:
                gate = f"ln(1.0+exp({nn_raw_expr}))"
            out = f"({gate}) * tanh(({_vg_alpha:.8f})*Vds) * (1.0 + ({_vg_lamb:.8f})*Vds)"
        elif _is_vdsgate_aeff or _is_vdsgate_vdsk:
            gate = f"tanh({nn_raw_expr})" if _tanhm_gate else f"ln(1.0+exp({nn_raw_expr}))"
            poly_a = _poly_expr(_a_coefs, _vgs_norm_va)
            poly_l = _poly_expr(_l_coefs, _vgs_norm_va)
            l_eff = poly_l if _free_lam else f"ln(1.0+exp({poly_l}))"
            if _is_vdsgate_aeff:
                a_eff = f"(2.0/(1.0+exp(-{poly_a})))" if _use_sig else f"ln(1.0+exp({poly_a}))"
                out = f"({gate}) * tanh(({a_eff})*Vds) * (1.0 + ({l_eff})*Vds)"
            else:
                vds_knee = f"(ln(1.0+exp({poly_a})) + 0.0001)"
                out = f"({gate}) * tanh(Vds/({vds_knee})) * (1.0 + ({l_eff})*Vds)"
        else:
            out = nn_raw_expr
        return f"({ids_scale:.8f}) * ({out})" if ids_scale != 1.0 else out

    import re as _re
    config_name = (run_data or {}).get('config_name', 'nn_model')
    mod_name = _re.sub(r'[^a-zA-Z0-9_]', '_', config_name)
    if mod_name and mod_name[0].isdigit():
        mod_name = '_' + mod_name

    va = []
    va.append(f"// Verilog-A GaN HEMT NN model")
    va.append(f"// Config  : {config_name}")
    va.append(f"// Vgs domain: [{vgs_min:.2f}, {vgs_max:.2f}] V")
    va.append(f"// Vds domain: [{vds_min:.2f}, {vds_max:.2f}] V")
    va.append(f"// Normalization baked into Layer 0 weights — inputs are raw Vgs/Vds [V], output Ids [A]")
    va.append("")
    va.append('`include "disciplines.vams"')
    va.append('`include "constants.vams"')
    va.append("")
    va.append(f"module {mod_name}(d, g, s);")
    va.append("")
    va.append("  inout d, g, s;")
    va.append("  electrical d, g, s;")
    va.append("")
    va.append("  real Vgs, Vds, Ids;")
    hidden_layers = layers_raw[:-1]
    for li, (W, b, acts) in enumerate(hidden_layers):
        vars_str = ", ".join(f"h{li}_{j}" for j in range(W.shape[0]))
        va.append(f"  real {vars_str};")
    va.append("")
    va.append("  analog begin")
    va.append("")
    va.append("    Vgs = V(g,s);")
    va.append("    Vds = V(d,s);")
    va.append("")
    prev_va = ['Vgs', 'Vds']
    for li, (W, b, acts) in enumerate(hidden_layers):
        va.append(f"    // Layer {li}  ({W.shape[0]} neurons)")
        for j in range(W.shape[0]):
            inner = " + ".join(f"({_fw(W[j,k])})*{prev_va[k]}" for k in range(len(prev_va)))
            inner += f" + ({_fw(b[j])})"
            va.append(f"    h{li}_{j} = {_act_str_va(acts[j], inner)};")
        va.append("")
        prev_va = [f"h{li}_{j}" for j in range(W.shape[0])]
    W_o, b_o, acts_o = layers_raw[-1]
    inner = " + ".join(f"({_fw(W_o[0,k])})*{prev_va[k]}" for k in range(len(prev_va)))
    inner += f" + ({_fw(b_o[0])})"
    if _is_vdsgate_simple:
        _gate_label_va = "NN" if _wrapper == 'vdsgatelin' else ("tanh(NN)" if _tanhm_gate else "softplus(NN)")
        va.append(f"    // Output  ({_wrapper} gate, alpha={_vg_alpha:.6f}, lamb={_vg_lamb:.6f}):  Ids = {_gate_label_va}*tanh(alpha*Vds)*(1+lamb*Vds)")
    elif _is_vdsgate_aeff:
        _gate_label_va = "tanh(NN)" if _tanhm_gate else "softplus(NN)"
        va.append(f"    // Output  ({_wrapper} gate):  Ids = {_gate_label_va}*tanh(a_eff(Vgs)*Vds)*(1+l_eff(Vgs)*Vds)")
    elif _is_vdsgate_vdsk:
        _gate_label_va = "tanh(NN)" if _tanhm_gate else "softplus(NN)"
        va.append(f"    // Output  ({_wrapper} gate):  Ids = {_gate_label_va}*tanh(Vds/Vds_knee(Vgs))*(1+l_eff(Vgs)*Vds)")
    else:
        va.append(f"    // Output layer  (activation = {acts_o[0]})")
    va.append(f"    Ids = {_wrap_out_va(_act_str_va(acts_o[0], inner))};")
    va.append("")
    va.append("    I(d,s) <+ Ids;")
    va.append("    I(g,s) <+ 0.0;")
    va.append("")
    va.append("  end")
    va.append("")
    va.append("endmodule")

    va_path = os.path.splitext(plot_save_path)[0] + '.va'
    with open(va_path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(va) + '\n')
    print(f"[plot_saved_state] Verilog-A written: {va_path}")

    eq_metrics['_eq_preds']     = eq_preds
    eq_metrics['_layers_trunc'] = layers_trunc
    return eq_metrics


def _act_torch(name, x):
    """Torch implementation of every activation used by DynamicNN."""
    n = name.lower()
    if n in ('linear', 'identity'): return x
    if n == 'tanh':     return torch.tanh(x)
    if n == 'sigmoid':  return torch.sigmoid(x)
    if n == 'relu':     return torch.relu(x)
    if n == 'mish':     return x * torch.tanh(torch.nn.functional.softplus(x))
    if n == 'softplus': return torch.nn.functional.softplus(x)
    if n == 'swish':    return x * torch.sigmoid(x)
    if n == 'sin':      return torch.sin(x)
    if n == 'cos':      return torch.cos(x)
    if n == 'sinh':     return torch.sinh(x)
    if n == 'cosh':     return torch.cosh(x)
    raise ValueError(f"[_act_torch] Unknown activation: {name!r}")


class _EqTorchModel:
    """Wraps truncated-weight (raw-input) layers as a torch-autograd-compatible model.

    layer 0 already has normalization folded in, so this expects raw Vgs/Vds.
    Autograd works through every layer, so Gm panels are computed correctly.
    """
    def __init__(self, layers_trunc, device):
        self.layers = [
            (torch.tensor(W, dtype=torch.float64, device=device),
             torch.tensor(b, dtype=torch.float64, device=device),
             acts)
            for W, b, acts in layers_trunc
        ]

    def __call__(self, x):
        h = x
        for W, b, acts in self.layers:
            pre = h @ W.T + b
            h = torch.cat([_act_torch(acts[j], pre[:, j:j+1])
                           for j in range(pre.shape[1])], dim=1)
        return h


# --------------------------------------------------------------------- #
# Validation set builder                                                 #
# --------------------------------------------------------------------- #
def _build_val_set(val_opt, original_csv, min_vgs):
    """Return a T_val DataFrame (same structure as T_train) for the chosen option:

    'interpolation' — 2D linear interpolation of T_train on a 3× denser Vgs×Vds grid.
    '0.3' / '0.2'  — use meas_load(original_csv, test_percent=pct); returns the test split.
    '/path/to.csv' — use meas_load(path, test_percent=0); returns the full dataset.
    """
    from scipy.interpolate import griddata as _griddata
    import pandas as pd

    _kwargs = dict(file_type="auriga", keep_every_N_group=0,
                   remove_negative_vds=0, remove_negative_ids=0,
                   min_vgs=min_vgs)

    # --- File path ---
    if os.path.isfile(str(val_opt)):
        T_val, _ = meas_load(val_opt, test_percent=0, **_kwargs)
        print(f"[val] loaded from file: {val_opt}  "
              f"({len(T_val)} rows, {T_val['TN'].nunique()} groups)")
        return T_val

    # --- Float: use meas_load test split ---
    try:
        pct = float(val_opt)
        if not 0.0 < pct < 1.0:
            raise ValueError("must be between 0 and 1")
        _, T_val = meas_load(original_csv, test_percent=pct, **_kwargs)
        print(f"[val] {pct:.0%} test split from original CSV: "
              f"{len(T_val)} rows, {T_val['TN'].nunique()} groups")
        return T_val
    except ValueError:
        pass

    # --- Interpolation ---
    if val_opt == 'interpolation':
        T_train_full, _ = meas_load(original_csv, test_percent=0, **_kwargs)
        pts = T_train_full[['Vgs_meas', 'Vds']].values.astype(np.float64)
        ids = T_train_full['Ids'].values.astype(np.float64)

        vgs_u = np.sort(T_train_full['Vgs_meas'].unique())
        vds_u = np.sort(T_train_full['Vds'].unique())
        # Use nominal sweep counts (TN groups × median Vds-per-group) so the
        # grid stays O(thousands), not O(millions) of measured unique values.
        n_vgs_nom = T_train_full['TN'].nunique()
        n_vds_nom = int(T_train_full.groupby('TN')['Vds'].nunique().median())
        n_vgs = n_vgs_nom * 3
        n_vds = n_vds_nom * 3
        print(f"[val] interpolation grid: {n_vgs}x{n_vds} "
              f"(from {n_vgs_nom} Vgs groups x {n_vds_nom} Vds/group, 3x)")
        vgs_d = np.linspace(vgs_u.min(), vgs_u.max(), n_vgs)
        # Always include Vds=0 in the grid so the Vds=0 panel is plotted.
        vds_start = min(0.0, float(vds_u.min()))
        vds_d = np.linspace(vds_start, vds_u.max(), n_vds)

        VGS, VDS = np.meshgrid(vgs_d, vds_d)
        ids_g = _griddata(pts, ids,
                          np.column_stack([VGS.ravel(), VDS.ravel()]),
                          method='linear').reshape(len(vds_d), len(vgs_d))

        rows = []
        for j, vgs_v in enumerate(vgs_d):
            for i, vds_v in enumerate(vds_d):
                ids_v = ids_g[i, j]
                if np.isnan(ids_v):
                    # Force Ids=0 at Vds=0 (physics) even if outside interpolation hull
                    if abs(vds_v) < 1e-9:
                        ids_v = 0.0
                    else:
                        continue
                elif abs(vds_v) < 1e-9:
                    ids_v = 0.0  # enforce exactly at Vds=0
                rows.append({'TN': j, 'Vgs': vgs_v, 'Vgs_meas': vgs_v,
                             'Vds': float(vds_v), 'Ids': float(ids_v), 'Igs': 0.0})
        T_val = pd.DataFrame(rows)
        T_val['index'] = np.arange(1, len(T_val) + 1)
        print(f"[val] interpolation: {n_vgs}x{n_vds} grid -> {len(T_val)} valid points")
        return T_val

    raise ValueError(
        f"--val: cannot parse {val_opt!r}. "
        "Use 'interpolation', a float 0<f<1, or a path to a CSV file."
    )


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Generate 6-panel IV/GM plots from trained weights OR an SLSQP seed.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir",  type=str, help="Trained run directory (weights_loss_*.pt + run_*.json).")
    src.add_argument("--seed", type=str, help="Physics-only SLSQP seed JSON.")
    parser.add_argument("--csv", required=True, type=str,
                        help="Measurement CSV (required).")
    parser.add_argument("--eq_name", default="mod1_angelov",
                        help="(--seed mode only) Physics equation name. Default: mod1_angelov.")
    parser.add_argument("--config_key", type=int, default=9,
                        help="(--seed mode only) PHYSICS_PARAM_CONFIG key. Default: 9.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path. Defaults to <source_dir>/plot_saved_state_full.png "
                             "(--dir) or <seed_dir>/plot_slsqp_seed.png (--seed).")
    parser.add_argument("--min_vgs", type=float, default=None,
                        help="If set (e.g. -4.0), meas_load auto-extrapolates extra start-groups so the most-negative Vgs reaches this floor.")
    parser.add_argument("--plot_vds_list", type=str, default=None,
                        help="Comma-separated Vds targets for the plot (e.g. '0,10,20,30'). "
                             "Overrides the auto-selected picks.")
    parser.add_argument("--plot_vgs_list", type=str, default=None,
                        help="Comma-separated Vgs targets for the plot (e.g. '-4,-3,-2,-1,0'). "
                             "Overrides the auto-selected picks.")
    parser.add_argument("--legend_decimals", type=int, default=1,
                        help="Rounding for the Actual/Target Vds/Vgs legend labels (e.g. 1 -> "
                             "'1.1', '2.3'). Default: 1.")
    parser.add_argument("--extrapolate", action="store_true",
                        help="Generate model curves over the full --plot_vds_list / --plot_vgs_list "
                             "range even where no measured data exists (for extrapolation testing).")
    parser.add_argument("--val", type=str, default=None,
                        help="Validation set option: 'interpolation' (3× denser interpolated grid), "
                             "a float 0<f<1 (e.g. '0.3') to use that fraction as held-out test split, "
                             "or a path to a separate validation CSV.")
    parser.add_argument("--add_zero_vds", action="store_true",
                        help="Overwrite the first (lowest-Vds) row of every TN group in the plotted "
                             "training data, forcing Vds=0, Ids=0 (in place, unconditional; no row added). "
                             "Mirrors the add_zero_vds flag used during training.")
    args = parser.parse_args()

    device = torch.device("cpu")

    # ----- Data (shared) -----
    T_train = _load_T_train(args.csv, args.min_vgs, device, add_zero_vds=args.add_zero_vds)
    print(f"[plot_saved_state] args.min_vgs={args.min_vgs}, "
          f"T_train Vgs_meas range = [{T_train['Vgs_meas'].min():.3f}, {T_train['Vgs_meas'].max():.3f}], "
          f"unique TNs = {T_train['TN'].nunique()}")
    X_train = np.column_stack((T_train['Vgs_meas'], T_train['Vds']))
    y_train = np.array(T_train['Ids']).reshape(-1, 1)
    X_train_t = torch.tensor(X_train, dtype=torch.float64, device=device)

    # ----- Build model -----
    if args.dir:
        save_dir = os.path.abspath(args.dir)
        print(f"Setting up model from saved state in: {save_dir}")
        model = _build_model_from_dir(save_dir, device)
        saved_json_metrics = _saved_metrics_from_dir(save_dir)
        title_str = "Performance Evaluated from Saved Weights"
        default_out = os.path.join(save_dir, 'plot_saved_state_full.png')
        # Re-read run JSON so we can include config in the MD file
        _rj = glob.glob(os.path.join(save_dir, 'run_*.json'))
        _run_data = json.load(open(max(_rj, key=os.path.getmtime))) if _rj else {}
    else:
        seed_path = os.path.abspath(args.seed)
        print(f"Setting up physics-only model from SLSQP seed: {seed_path}")
        model = _build_model_from_seed(seed_path, args.eq_name, args.config_key, device)
        saved_json_metrics = _saved_metrics_from_seed(seed_path)
        title_str = f"SLSQP Physics-Only Seed ({args.eq_name})\n{os.path.basename(seed_path)}"
        default_out = os.path.join(os.path.dirname(seed_path), 'plot_slsqp_seed.png')
        _run_data = {}

    plot_save_path = args.out or default_out

    # ----- Metrics -----
    print("Generating comprehensive 6-panel IV and GM curves plot...")
    with torch.no_grad():
        final_preds = model(X_train_t).cpu().numpy().flatten()
    residuals = y_train.flatten() - final_preds
    r2 = float(1 - np.sum(residuals**2) / np.sum((y_train.flatten() - np.mean(y_train.flatten()))**2))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # --- Calculated Ids + Gm RMSEs (same convention as physics+NN path) ---
    # `create_gms_for_train` re-sorts the rows by (Step_Index, TN); we must
    # feed `get_gm_rmse_metrics` an X_train_t built from the SAME ordering so
    # gm_pred lines up elementwise with gm_true.
    gm1_true_arr, gm2_true_arr, gm3_true_arr, X_train_gm, y_train_gm = create_gms_for_train(T_train)
    X_train_gm_t = torch.tensor(X_train_gm, dtype=torch.float64, device=device)
    with torch.no_grad():
        ids_pred_gm = model(X_train_gm_t).cpu().numpy().flatten()
    rmse_ids = float(np.sqrt(np.mean((y_train_gm.flatten() - ids_pred_gm) ** 2)))
    rmse_gm1, rmse_gm2, rmse_gm3, _, _, _ = get_gm_rmse_metrics(
        model, X_train_gm_t, gm1_true_arr, gm2_true_arr, gm3_true_arr
    )

    # Info box: "Saved JSON base RMSE" <- the RMSEs logged in the run/seed JSON;
    # "Calculated RMSE" <- the RMSEs recomputed here from the saved weights.
    # (No 'new_*' keys, so only the base block is rendered.)
    saved_metrics = {
        'base_ids': saved_json_metrics.get('ids'),
        'base_gm1': saved_json_metrics.get('gm1'),
        'base_gm2': saved_json_metrics.get('gm2'),
        'base_gm3': saved_json_metrics.get('gm3'),
    }
    calc_metrics = {"rmse": rmse, "r2": r2, "base_rmse": np.nan, "base_r2": np.nan,
                    "rmse_ids": rmse_ids, "rmse_gm1": rmse_gm1,
                    "rmse_gm2": rmse_gm2, "rmse_gm3": rmse_gm3}

    # Generate symbolic equation, write .md, compare numpy vs torch
    _eq_result = generate_and_write_equation_md(
        model, X_train, calc_metrics,
        plot_save_path, run_data=_run_data, device=device,
    ) or {}
    eq_metrics     = {k: v for k, v in _eq_result.items()
                      if k not in ('_eq_preds', '_layers_trunc')}
    _layers_trunc  = _eq_result.get('_layers_trunc')

    # --- Compare calculated RMSEs against the values saved in the JSON ---
    # The recomputed values should match what the trainer/SLSQP logged; a
    # mismatch means the model was not faithfully reconstructed (e.g. knee config
    # or weights) or the metric conventions diverged.
    _RMSE_RTOL = 1e-2
    print("[plot_saved_state] RMSE check (calculated vs saved JSON):")
    _any_mismatch = False
    for _name, _calc, _saved in (
        ("Ids", rmse_ids, saved_metrics['base_ids']),
        ("Gm1", rmse_gm1, saved_metrics['base_gm1']),
        ("Gm2", rmse_gm2, saved_metrics['base_gm2']),
        ("Gm3", rmse_gm3, saved_metrics['base_gm3']),
    ):
        if _saved is None:
            print(f"    {_name}: calc={_calc:.6e} saved=N/A (not in JSON)")
            continue
        _rel = abs(_calc - float(_saved)) / max(abs(float(_saved)), 1e-12)
        _status = "OK" if _rel <= _RMSE_RTOL else "MISMATCH"
        if _status == "MISMATCH":
            _any_mismatch = True
        print(f"    {_name}: calc={_calc:.6e} saved={float(_saved):.6e} "
              f"rel_err={_rel:.2e} {_status}")
    if _any_mismatch:
        print("[plot_saved_state] WARNING: calculated RMSE(s) do not match saved JSON values.")

    # ----- Sweep targets -----
    if 'Vds_meas_set' in T_train.columns:
        unique_vds_nominal = np.sort(T_train['Vds_meas_set'].unique())
    else:
        vds_vals = sorted(list(np.unique(T_train['Vds'])))
        if len(vds_vals) > 5:
            unique_vds_nominal = [vds_vals[0], vds_vals[len(vds_vals)//4],
                                  vds_vals[len(vds_vals)//2],
                                  vds_vals[(len(vds_vals)*3)//4], vds_vals[-1]]
        else:
            unique_vds_nominal = vds_vals

    if 'Vgs_meas_set' in T_train.columns:
        unique_vgs_nominal = np.sort(T_train['Vgs_meas_set'].unique())
    else:
        vgs_vals = sorted(list(np.unique(T_train['Vgs'])))
        if len(vgs_vals) > 6:
            indices = np.linspace(0, len(vgs_vals) - 1, 6).astype(int)
            unique_vgs_nominal = [vgs_vals[i] for i in indices]
        else:
            unique_vgs_nominal = vgs_vals

    # CLI overrides for plot sweep targets
    if args.plot_vds_list:
        unique_vds_nominal = [float(x) for x in args.plot_vds_list.split(',') if x.strip()]
    if args.plot_vgs_list:
        unique_vgs_nominal = [float(x) for x in args.plot_vgs_list.split(',') if x.strip()]

    plot_data = generate_physics_plot_data(
        model, T_train, 'Vgs_meas', 'Vds', 'Ids',
        unique_vds_nominal, unique_vgs_nominal, device,
        tol_vds=0.5, tol_vgs=0.5,
        force_target_range=args.extrapolate,
    )
    for sw in plot_data.get('vds_sweeps', []):
        tv = sw.get('true_vgs')
        if tv is not None and len(tv):
            print(f"[plot_saved_state] vds_sweep target_vds={sw.get('target_vds')}: "
                  f"true_vgs n={len(tv)} range=[{tv.min():.3f}, {tv.max():.3f}]")

    plot_grid(plot_data, calc_metrics, saved_metrics, eq_metrics,
              plot_save_path, title_str,
              list(unique_vds_nominal), list(unique_vgs_nominal),
              use_real_voltages=True, legend_decimals=args.legend_decimals)
    print(f"6-panel Plot successfully saved to: {plot_save_path}")

    # --- Equation string comparison plot (same layout, dashed eq curves overlaid) ---
    if _layers_trunc is not None:
        _eq_model = _EqTorchModel(_layers_trunc, device)
        eq_plot_data = generate_physics_plot_data(
            _eq_model, T_train, 'Vgs_meas', 'Vds', 'Ids',
            unique_vds_nominal, unique_vgs_nominal, device,
            tol_vds=0.5, tol_vgs=0.5,
            force_target_range=args.extrapolate,
        )
        _cmp_path = os.path.splitext(plot_save_path)[0] + '_eq_comparison.png'
        plot_grid(plot_data, calc_metrics, saved_metrics, eq_metrics,
                  _cmp_path, title_str + "  —  NN (solid) vs Eq String (dashed)",
                  list(unique_vds_nominal), list(unique_vgs_nominal),
                  use_real_voltages=True, legend_decimals=args.legend_decimals, eq_plot_data=eq_plot_data)
        print(f"Eq comparison plot saved to: {_cmp_path}")

    # --- Validation plot ---
    if args.val:
        print(f"[val] building validation set from: {args.val!r}")
        T_val = _build_val_set(args.val, args.csv, args.min_vgs)

        gm1_val, gm2_val, gm3_val, X_val_gm, y_val_gm = create_gms_for_train(T_val)
        X_val_gm_t = torch.tensor(X_val_gm, dtype=torch.float64, device=device)
        with torch.no_grad():
            ids_val_pred = model(X_val_gm_t).cpu().numpy().flatten()
        val_rmse_ids = float(np.sqrt(np.mean((y_val_gm.flatten() - ids_val_pred) ** 2)))
        val_rmse_gm1, val_rmse_gm2, val_rmse_gm3, _, _, _ = get_gm_rmse_metrics(
            model, X_val_gm_t, gm1_val, gm2_val, gm3_val
        )
        print(f"[val] Ids RMSE={val_rmse_ids:.4e}  Gm1={val_rmse_gm1:.4e}  "
              f"Gm2={val_rmse_gm2:.4e}  Gm3={val_rmse_gm3:.4e}")

        calc_metrics_val = dict(calc_metrics)
        calc_metrics_val['val_rmse_ids']  = val_rmse_ids
        calc_metrics_val['val_rmse_gm1'] = val_rmse_gm1
        calc_metrics_val['val_rmse_gm2'] = val_rmse_gm2
        calc_metrics_val['val_rmse_gm3'] = val_rmse_gm3
        calc_metrics_val['val_source']   = str(args.val)

        val_plot_data = generate_physics_plot_data(
            model, T_val, 'Vgs_meas', 'Vds', 'Ids',
            unique_vds_nominal, unique_vgs_nominal, device,
            tol_vds=0.5, tol_vgs=0.5,
            force_target_range=args.extrapolate,
        )
        _val_path = os.path.splitext(plot_save_path)[0] + '_val.png'
        plot_grid(val_plot_data, calc_metrics_val, saved_metrics, eq_metrics,
                  _val_path, title_str + "  —  Validation",
                  list(unique_vds_nominal), list(unique_vgs_nominal),
                  use_real_voltages=True, legend_decimals=args.legend_decimals)
        print(f"Validation plot saved to: {_val_path}")


if __name__ == "__main__":
    main()
