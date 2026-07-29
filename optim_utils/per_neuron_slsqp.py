"""per_neuron_slsqp.py
====================
SLSQP-based constrained optimisation for noNN PhysicsInformedNN models.

Extracted from per_neuron_trainer_8.py to keep that file manageable.

Trainer-specific symbols (DynamicNN, PhysicsInformedNN, evaluate_physics, etc.)
are imported lazily inside optimize_config_nonn_slsqp() to avoid a circular
dependency at module load time.
"""

import copy
import math
import os
import json
import pickle
import time
import types
from datetime import datetime, timedelta

import numpy as np
import torch
from scipy.optimize import minimize as scipy_minimize

# ---------------------------------------------------------------------------
# Module-level helpers that do NOT create circular imports
# NOTE: new_models must already be on sys.path (per_neuron_trainer_8.py sets
#       this before importing us).
# ---------------------------------------------------------------------------
from per_neuron_normalization import USED_NORMALIZATIONS


# =============================================================================
#  SCIPY SLSQP — constrained optimisation for noNN models (no Optuna)
# =============================================================================

def _build_slsqp_constraints(model, pack_fn, unpack_fn):
    """Build a list of scipy SLSQP constraint dicts for the noNN physics constraints.

    SLSQP convention: 'ineq' means fun(x) >= 0 (satisfied when positive).
    Works for mod1_angelov only; returns [] for other equation names.

    Args:
        model:      A PhysicsInformedNN instance in noNN mode.
        pack_fn:    Callable() → np.ndarray.  Reads current raw params from model.
        unpack_fn:  Callable(x: np.ndarray) → None.  Writes x into model raw params.
    """
    if model.eq_name != "mod1_angelov":
        return []

    def _get_real(x):
        """Load x into the model and return real-space params as plain floats."""
        unpack_fn(x)
        with torch.no_grad():
            rp = model.get_real_params()
        return {k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                for k, v in rp.items()}

    def c_P31(x):                   # P31 >= 0
        return _get_real(x)['P31']

    def c_P32(x):                   # P32 >= 0
        return _get_real(x)['P32']

    def c_disc1(x):
        # Discriminant of Vgspk cubic root: need -(P21 + sqrt(P21^2 - 3*P31*P1)) >= 0
        # i.e. P21 + sqrt(relu(A1)) <= 0  →  -(P21 + sqrt(relu(A1))) >= 0
        rp = _get_real(x)
        P21, P31, P1 = rp['P21'], rp['P31'], rp['P1']
        A1 = P21 ** 2 - 3.0 * P31 * P1
        safe_A1 = max(A1, 0.0)
        return -(P21 + math.sqrt(safe_A1) + 1e-6)

    def c_disc2(x):
        # Discriminant of Vdspk cubic root: need -(-P22 + sqrt(P22^2-3*P32*P1')) >= 0
        # i.e. -P22 + sqrt(relu(A2)) <= 0  →  P22 - sqrt(relu(A2)) >= 0
        # P1_prime = P111 = P1 * Ipk / (Ipk1 + eps)  — derived inside mod1_angelov
        rp = _get_real(x)
        P22, P32 = rp['P22'], rp['P32']
        P1p = rp['P1'] * rp['Ipk'] / (rp['Ipk1'] + 1e-12)
        A2 = P22 ** 2 - 3.0 * P32 * P1p
        safe_A2 = max(A2, 0.0)
        return P22 - math.sqrt(safe_A2) - 1e-6

    return [
        {'type': 'ineq', 'fun': c_P31},
        {'type': 'ineq', 'fun': c_P32},
        {'type': 'ineq', 'fun': c_disc1},
        {'type': 'ineq', 'fun': c_disc2},
    ]


def train_model_slsqp(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    criterion,
    n_restarts=5,
    timeout=None,
    verbose=True,
    gm1_true_arr=None,
    gm2_true_arr=None,
    gm3_true_arr=None,
):
    """Train a noNN PhysicsInformedNN model using scipy SLSQP.

    Replaces the PyTorch training loop for noNN models.  No Optuna required.
    GM weighted losses (Gm1, Gm2, Gm3) are added to the scalar objective when
    USE_GM=1 and GMx_WEIGHT > 0, using autograd-based derivatives d^n(Ids)/d(Vgs)^n.
    No PCGrad surgery — SLSQP requires a single scalar, so it's a weighted sum.

    Returns:
        (model, best_loss, history)  — same tuple shape as train_model().
        history is a list of dicts with keys 'epoch' and 'val_loss'.
    """
    assert model.mode == "noNN", "train_model_slsqp is only for noNN models."

    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    X_train_t = torch.tensor(X_train, dtype=dtype, device=device) if not isinstance(X_train, torch.Tensor) else X_train.to(device=device, dtype=dtype)
    y_train_t = torch.tensor(y_train, dtype=dtype, device=device) if not isinstance(y_train, torch.Tensor) else y_train.to(device=device, dtype=dtype)
    X_val_t   = torch.tensor(X_val,   dtype=dtype, device=device) if not isinstance(X_val,   torch.Tensor) else X_val.to(device=device, dtype=dtype)
    y_val_t   = torch.tensor(y_val,   dtype=dtype, device=device) if not isinstance(y_val,   torch.Tensor) else y_val.to(device=device, dtype=dtype)

    # When no validation split is provided (e.g. test_percent=0), criterion() over
    # an empty tensor returns nan. That makes every restart's val_loss nan, so no
    # restart ever beats best_val_loss=inf and the model is left at its initial
    # (unoptimized) params. Fall back to the training data for restart selection.
    if y_val_t.numel() == 0 or X_val_t.numel() == 0:
        if verbose:
            print("[SLSQP] No validation set (test_percent=0?) — using training "
                  "data for restart selection.")
        X_val_t = X_train_t
        y_val_t = y_train_t

    # ------------------------------------------------------------------ #
    #  GM weights — read from env vars once at construction time           #
    #  USE_GM format matches the Optuna path: "0" | "1" | "1,2" | "1,2,3"
    #  All four env vars MUST be set explicitly by the caller; we refuse
    #  silent defaults because a missing var would silently disable GM.
    # ------------------------------------------------------------------ #
    _required_env = ('USE_GM', 'GM1_WEIGHT', 'GM2_WEIGHT', 'GM3_WEIGHT')
    _missing_env = [k for k in _required_env if k not in os.environ]
    if _missing_env:
        raise EnvironmentError(
            f"train_model_slsqp requires env vars {_required_env} to be set explicitly "
            f"(missing: {_missing_env}). Set USE_GM='0' and GM*_WEIGHT='0.0' to disable GM."
        )
    _use_gm_env = os.environ['USE_GM']
    _active_gms = (
        [int(x) for x in _use_gm_env.split(',') if x.strip().isdigit()]
        if _use_gm_env != '0' else []
    )
    gm1_w = float(os.environ['GM1_WEIGHT']) if 1 in _active_gms else 0.0
    gm2_w = float(os.environ['GM2_WEIGHT']) if 2 in _active_gms else 0.0
    gm3_w = float(os.environ['GM3_WEIGHT']) if 3 in _active_gms else 0.0

    def _to_t(arr):
        return torch.tensor(arr.reshape(-1, 1), dtype=dtype, device=device)

    gm1_true_t = _to_t(gm1_true_arr) if (gm1_w > 0 and gm1_true_arr is not None) else None
    gm2_true_t = _to_t(gm2_true_arr) if (gm2_w > 0 and gm2_true_arr is not None) else None
    gm3_true_t = _to_t(gm3_true_arr) if (gm3_w > 0 and gm3_true_arr is not None) else None
    _any_gm    = (gm1_true_t is not None or gm2_true_t is not None or gm3_true_t is not None)

    if verbose and _any_gm:
        print(f"[SLSQP] GM losses active — gm1_w={gm1_w}, gm2_w={gm2_w}, gm3_w={gm3_w}")

    # ------------------------------------------------------------------ #
    #  Parameter pack / unpack helpers (numpy <-> model)
    # ------------------------------------------------------------------ #
    # Collect ONLY learnable parameters (requires_grad=True)
    _param_list = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    _shapes     = [p.shape for _, p in _param_list]
    _sizes      = [p.numel() for _, p in _param_list]
    _total      = sum(_sizes)

    def pack() -> np.ndarray:
        """Read current model params → flat float64 numpy vector."""
        parts = [p.detach().cpu().numpy().ravel() for _, p in _param_list]
        return np.concatenate(parts).astype(np.float64)

    def unpack(x: np.ndarray):
        """Write flat float64 numpy vector → model params (in-place)."""
        offset = 0
        for (_, p), size, shape in zip(_param_list, _sizes, _shapes):
            chunk = x[offset: offset + size].reshape(shape)
            with torch.no_grad():
                p.copy_(torch.tensor(chunk, dtype=p.dtype, device=p.device))
            offset += size

    # ------------------------------------------------------------------ #
    #  Objective: returns (loss_float, grad_np_array) in one backward pass
    # ------------------------------------------------------------------ #
    def loss_and_grad(x: np.ndarray):
        unpack(x)
        model.zero_grad()

        if _any_gm:
            # Vgs must carry requires_grad so autograd can diff Ids w.r.t. Vgs
            vgs_t = X_train_t[:, 0:1].detach().requires_grad_(True)
            vds_t = X_train_t[:, 1:2].detach()
            X_in  = torch.cat([vgs_t, vds_t], dim=1)
            preds = model(X_in)
            loss  = criterion(preds, y_train_t)

            if gm1_true_t is not None:
                # Gm1 = d(Ids)/d(Vgs)
                gm1_pred = torch.autograd.grad(
                    preds.sum(), vgs_t, create_graph=True, retain_graph=True
                )[0]
                loss = loss + gm1_w * criterion(gm1_pred, gm1_true_t)

                if gm2_true_t is not None:
                    # Gm2 = d^2(Ids)/d(Vgs)^2
                    gm2_pred = torch.autograd.grad(
                        gm1_pred.sum(), vgs_t, create_graph=True, retain_graph=True
                    )[0]
                    loss = loss + gm2_w * criterion(gm2_pred, gm2_true_t)

                    if gm3_true_t is not None:
                        # Gm3 = d^3(Ids)/d(Vgs)^3
                        gm3_pred = torch.autograd.grad(
                            gm2_pred.sum(), vgs_t, create_graph=True
                        )[0]
                        loss = loss + gm3_w * criterion(gm3_pred, gm3_true_t)
        else:
            preds = model(X_train_t)
            loss  = criterion(preds, y_train_t)

        loss.backward()
        grad_parts = [p.grad.detach().cpu().numpy().ravel()
                      if p.grad is not None else np.zeros(_sizes[i])
                      for i, (_, p) in enumerate(_param_list)]
        return float(loss.item()), np.concatenate(grad_parts).astype(np.float64)

    # ------------------------------------------------------------------ #
    #  Validation loss helper
    # ------------------------------------------------------------------ #
    def val_loss(x: np.ndarray) -> float:
        unpack(x)
        with torch.no_grad():
            preds = model(X_val_t)
            loss  = criterion(preds, y_val_t)
        return float(loss.item())

    # ------------------------------------------------------------------ #
    #  Constraints
    # ------------------------------------------------------------------ #
    constraints = _build_slsqp_constraints(model, pack, unpack)

    # ------------------------------------------------------------------ #
    #  Timeout callback
    # ------------------------------------------------------------------ #
    _start_time = [time.time()]

    def _timeout_cb(_x):
        if timeout is not None and (time.time() - _start_time[0]) > timeout:
            raise StopIteration("SLSQP timeout")

    # ------------------------------------------------------------------ #
    #  Multi-restart loop
    # ------------------------------------------------------------------ #
    best_x        = None
    best_val_loss = float('inf')
    history       = []

    for restart in range(n_restarts):
        _start_time[0] = time.time()

        if restart == 0:
            x0 = pack()                                          # use current (pre-sampled) init
        else:
            x0 = np.random.uniform(-2.0, 2.0, _total).astype(np.float64)
            unpack(x0)

        if verbose:
            print(f"\n[SLSQP] Restart {restart + 1}/{n_restarts} — x0 norm = {np.linalg.norm(x0):.4f}")

        try:
            result = scipy_minimize(
                loss_and_grad,
                x0,
                jac=True,
                method='SLSQP',
                constraints=constraints,
                options={
                    'maxiter': 2000,
                    'ftol':    1e-14,
                    'eps':     1.5e-8,
                    'disp':    False,
                },
                callback=_timeout_cb,
            )
            final_x = result.x
        except StopIteration:
            if verbose:
                print(f"[SLSQP] Restart {restart + 1} timed out — using best-so-far params.")
            final_x = pack()
        except Exception as e:
            if verbose:
                print(f"[SLSQP] Restart {restart + 1} failed: {e} — skipping.")
            continue

        vl = val_loss(final_x)
        if verbose:
            print(f"[SLSQP] Restart {restart + 1} val_loss = {vl:.8g}")

        history.append({'epoch': restart, 'val_loss': vl})

        if vl < best_val_loss:
            best_val_loss = vl
            best_x        = final_x.copy()

        # Timeout across the full call (not per-restart)
        if timeout is not None and (time.time() - _start_time[0]) > timeout:
            if verbose:
                print("[SLSQP] Global timeout reached — stopping restarts.")
            break

    if best_x is not None:
        unpack(best_x)
    else:
        if verbose:
            print("[SLSQP] WARNING: All restarts failed — model left at initial params.")
        best_val_loss = val_loss(pack())

    if verbose:
        print(f"\n[SLSQP] Best val_loss across {n_restarts} restarts: {best_val_loss:.8g}")

    return model, best_val_loss, history


def optimize_config_nonn_slsqp(
    config_id,
    base_config,
    optimizer_space_id,
    optimizer_space,
    path_to_csv,
    environment,
    meas_load_kwargs,
    user_metadata=None,
    n_trials=10,          # repurposed as n_restarts (overridable via SLSQP_N_RESTARTS)
    n_best_results=3,
    val_interval=10,      # unused — kept for signature compatibility
    **kwargs,
):
    """Drop-in replacement for optimize_config() for noNN models using scipy SLSQP.

    Skips Optuna entirely. Runs train_model_slsqp() with multiple restarts, saves
    the best models to a PKL, and writes a results JSON compatible with the standard
    downstream tooling.

    Returns:
        (None, out_json)  — same shape as optimize_config().  The None placeholder
        means the caller must not try to access .best_value on it.
    """
    # --- Lazy imports to avoid circular dependency with per_neuron_trainer_8 ---
    import new_models.per_neuron_optim.per_neuron_trainer_8 as _t
    DynamicNN                       = _t.DynamicNN
    PhysicsInformedNN               = _t.PhysicsInformedNN
    get_criterion                   = _t.get_criterion
    set_deterministic_seed          = _t.set_deterministic_seed
    create_gms_for_train            = _t.create_gms_for_train
    extract_single_trial_data       = _t.extract_single_trial_data
    evaluate_physics                = _t.evaluate_physics
    plot_create_eq_best_predictions = _t.plot_create_eq_best_predictions
    target_vds_list                 = _t.target_vds_list
    meas_load                       = _t.meas_load
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------- #
    #  Setup — mirrors optimize_config() boilerplate                         #
    # --------------------------------------------------------------------- #
    csv_basename = os.path.basename(path_to_csv)
    csv_dir      = os.path.dirname(path_to_csv)
    results_dir  = os.path.join(csv_dir, f"{environment}_results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(
        results_dir,
        f"results_base{config_id}_opt{optimizer_space_id}.json",
    )

    current_env_info = {
        'environment': environment,
        'meas_load_kwargs': meas_load_kwargs,
        'csv_path': None,
        "from_env": False,
    }

    # --------------------------------------------------------------------- #
    #  PLOT_EQ_ONLY: skip optimization, re-plot/re-equation from saved data  #
    # --------------------------------------------------------------------- #
    if os.environ.get("PLOT_EQ_ONLY", "") == "True":
        print("[SLSQP] PLOT_EQ_ONLY=True — skipping optimization, plotting from saved results.")
        if not os.path.isfile(out_json):
            print(f"[SLSQP] PLOT_EQ_ONLY: No existing JSON at {out_json}, cannot plot.")
            return None, out_json
        try:
            plot_create_eq_best_predictions(
                base_json_path=os.path.dirname(os.path.dirname(out_json)),
                base_env=environment,
                base_config_no=out_json,
                space_config_no=optimizer_space_id,
                env_to_train_info=current_env_info,
                early_stopping_metric="mean",
                n_best=n_best_results,
                save_plot=True,
                save_equations=True,
                val_interval=val_interval,
            )
            print("[SLSQP] PLOT_EQ_ONLY: Post-processing complete.")
        except Exception as e:
            print(f"[SLSQP] PLOT_EQ_ONLY: Error in post-processing: {e}")
            import traceback
            traceback.print_exc()
        return None, out_json

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Number of restarts (env var wins over the n_trials argument)
    n_restarts = int(os.environ.get('SLSQP_N_RESTARTS', str(n_trials)))
    global_timeout = int(os.environ.get('OPTIMIZE_CONFIG_TIMEOUT', '3600'))

    # --------------------------------------------------------------------- #
    #  Data loading                                                           #
    # --------------------------------------------------------------------- #
    input_dim      = base_config['input_dim']
    output_dim     = base_config['output_dim']
    current2train  = user_metadata['current2train']
    train_data_mode = user_metadata['train_data_mode']
    early_stopping_metric = user_metadata.get('early_stopping_metric', 'mean')

    T_train, T_test = meas_load(path_to_csv, **meas_load_kwargs)
    gm1_train, gm2_train, gm3_train, X_train, y_train, X_val, Y_val = \
        create_gms_for_train(T_train, T_test)

    TEMP_MODEL_DIR = os.path.join(
        results_dir, 'temp_models_storage',
        f'proc_base{config_id}_opt{optimizer_space_id}',
    )
    os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

    physics_c  = optimizer_space.get('general_opts', {}).get('physics_config', None)
    model_arch = optimizer_space.get('model_arch', 'dynamic')

    norm_c_space = optimizer_space.get('general_opts', {}).get('normalization', None)
    gms_info     = optimizer_space.get('general_opts', {}).get('GMs', None)

    os.environ['USE_PCGRAD'] = gms_info.get('use_pcgrad', '0') if gms_info else '0'
    os.environ['USE_GM']     = gms_info.get('use_gm',     '0') if gms_info else '0'

    # --------------------------------------------------------------------- #
    #  Parse GM weight search ranges from the optimizer space                #
    #  A tuple value (lo, hi) means Optuna should search that range;         #
    #  a scalar / string means a fixed weight.                               #
    # --------------------------------------------------------------------- #
    _gm_ranges = {}
    if gms_info:
        for _gm_key in ('gm1_weight', 'gm2_weight', 'gm3_weight'):
            _v = gms_info.get(_gm_key)
            if _v is None:
                continue
            if isinstance(_v, (tuple, list)) and len(_v) == 2:
                _gm_ranges[_gm_key] = (float(_v[0]), float(_v[1]))
            else:
                _gm_ranges[_gm_key] = float(_v)   # fixed scalar
    _has_gm_search = any(isinstance(v, tuple) for v in _gm_ranges.values())

    # resolve norm_c  (identical logic to optimize_config)
    norm_c = None
    if norm_c_space is not None and norm_c_space != 'False':
        try:
            norm_k = norm_c_space[1]
        except (IndexError, TypeError):
            norm_k = None
        if norm_k is not None:
            norm_c = USED_NORMALIZATIONS[norm_k]
        if isinstance(norm_c_space, (list, tuple)) and norm_c_space[0] == 'Hybrid' and norm_c:
            norm_c['hybrid'] = True

    # --------------------------------------------------------------------- #
    #  Build noNN model (empty DynamicNN wrapper — NN layers are not used)  #
    # --------------------------------------------------------------------- #
    set_deterministic_seed(42)

    neurons_per_layer    = []          # noNN: no hidden layers
    activations_per_layer = []

    base_model = DynamicNN(
        input_dim=input_dim,
        neurons_per_layer=neurons_per_layer,
        activations_per_layer=activations_per_layer,
        output_dim=output_dim,
        output_activation=base_config.get('output_activation', 'linear'),
    ).double()

    model = PhysicsInformedNN(
        base_model,
        equation_type=model_arch,
        normalization_config=norm_c,
        config_key=physics_c,
    ).double()

    model.to(device)

    criterion_name = base_config.get('criterion', 'MSELoss')
    criterion = get_criterion(criterion_name)

    # --------------------------------------------------------------------- #
    #  Run SLSQP  (with optional Optuna outer loop for GM weight search)     #
    # --------------------------------------------------------------------- #
    opt_start_time = time.time()
    _best_gm_weights = {}   # filled when GM search is active

    if _has_gm_search:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        print(f"\n[SLSQP] GM weight search enabled via Optuna ({n_restarts} trial(s)).")
        for _k, _v in _gm_ranges.items():
            print(f"  {_k}: {_v}")

        _all_gm_trials = []   # list of (optuna_val, train_loss, model, gm_weights)
        _trial_timeout  = max(60, global_timeout // max(n_restarts, 1))
        _init_model     = copy.deepcopy(model)   # pristine copy for each trial
        _target_metric  = os.environ.get("OPTUNA_TARGET_METRIC", "ids").lower().strip()
        _use_physics_target = (_target_metric != "ids")
        if _use_physics_target:
            print(f"  Optuna target metric : {_target_metric} (RMSE)")
        else:
            print(f"  Optuna target metric : ids (training loss)")

        def _gm_objective(trial):
            # --- sample / apply GM weights for this trial ---
            _trial_gms = {}
            for _k, _v in _gm_ranges.items():
                _gm_num = _k[2]   # '1', '2', or '3'
                if isinstance(_v, tuple):
                    _w = trial.suggest_float(_k, _v[0], _v[1])
                else:
                    _w = _v
                _trial_gms[_k] = _w
                os.environ[f'GM{_gm_num}_WEIGHT'] = str(_w)

            _trial_model = copy.deepcopy(_init_model)
            _loss_val = float('inf')
            try:
                _trial_model, _loss_val, _ = train_model_slsqp(
                    model=_trial_model,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=Y_val,
                    criterion=criterion,
                    n_restarts=1,
                    timeout=_trial_timeout,
                    verbose=False,
                    gm1_true_arr=gm1_train,
                    gm2_true_arr=gm2_train,
                    gm3_true_arr=gm3_train,
                )
            except Exception as _e:
                print(f"[SLSQP GM trial {trial.number}] Error: {_e}")
                return float('inf')

            # --- choose Optuna objective ---
            if _use_physics_target:
                _optuna_val = float('inf')
                try:
                    _rm = evaluate_physics(
                        model=_trial_model,
                        T_val=T_train,
                        vgs_col=train_data_mode,
                        vds_col='Vds',
                        ids_col=current2train,
                        target_vds_list=target_vds_list,
                        device=device,
                    )
                    _optuna_val = _rm.get(f"rmse_{_target_metric}", float('inf'))
                    if _optuna_val is None:
                        _optuna_val = float('inf')
                except Exception as _ep:
                    print(f"[SLSQP GM trial {trial.number}] Physics eval error: {_ep}")
            else:
                _optuna_val = _loss_val

            if _loss_val < float('inf'):
                _current_optimized_real_params = {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in _trial_model.get_real_params().items()}
                _merged_optimized_params = dict(_trial_gms)
                _merged_optimized_params.update(_current_optimized_real_params)
                _all_gm_trials.append((_optuna_val, _loss_val, _trial_model, _merged_optimized_params))
            return _optuna_val

        _gm_study = _optuna.create_study(direction='minimize')
        _gm_study.optimize(_gm_objective, n_trials=n_restarts)

        if not _all_gm_trials:
            print("[SLSQP] All GM trials failed — aborting.")
            return None, None

        # Sort by Optuna objective, keep top n_best_results
        _ranked_gm = sorted(_all_gm_trials, key=lambda x: x[0])[:n_best_results]
        # _results_to_save uses (train_loss, model, gms) — compatible with save section
        _results_to_save = [(_tl, _m, _g) for (_ov, _tl, _m, _g) in _ranked_gm]

        model            = _results_to_save[0][1]
        best_val_loss    = _results_to_save[0][0]
        _best_gm_weights = _results_to_save[0][2]

        # restore best GM weights as env vars for downstream use
        for _k, _w in _best_gm_weights.items():
            os.environ[f'GM{_k[2]}_WEIGHT'] = str(_w)

        print(f"\n[SLSQP] GM search done. Keeping top {len(_ranked_gm)}/{len(_all_gm_trials)} trial(s).")
        for _r, (_ov, _tl, _, _g) in enumerate(_ranked_gm):
            if _use_physics_target:
                print(f"  Rank {_r}: {_target_metric}_rmse={_ov:.8g}  train_loss={_tl:.8g}  GM={_g}")
            else:
                print(f"  Rank {_r}: loss={_ov:.8g}  GM={_g}")

    else:
        print(f"\n[SLSQP] Starting noNN optimisation: {n_restarts} restart(s), "
              f"timeout={global_timeout}s, arch={model_arch}")

        model, best_val_loss, history = train_model_slsqp(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=Y_val,
            criterion=criterion,
            n_restarts=n_restarts,
            timeout=global_timeout,
            verbose=True,
            gm1_true_arr=gm1_train,
            gm2_true_arr=gm2_train,
            gm3_true_arr=gm3_train,
        )

        print(f"\n[SLSQP] Done.  Best val_loss = {best_val_loss:.8g}")
        _optimized_real_params = {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in model.get_real_params().items()}
        _results_to_save = [(best_val_loss, model, _optimized_real_params)]

    opt_end_time = time.time()
    current_execution_time = opt_end_time - opt_start_time
    print(f"[SLSQP] Wall time: {current_execution_time:.1f}s")

    # --------------------------------------------------------------------- #
    #  Early-exit gate (mirrors SAVE_JSON_MIN_TARGET_LOSS check)             #
    # --------------------------------------------------------------------- #
    min_target_loss = float(os.environ.get('SAVE_JSON_MIN_TARGET_LOSS', '1e9'))
    if _results_to_save[0][0] > min_target_loss:
        print(f"[SLSQP] Best loss {_results_to_save[0][0]:.3e} did not meet target "
              f"{min_target_loss:.3e}.  Skipping save.")
        return None, None

    # --------------------------------------------------------------------- #
    #  Shared metadata for all ranked results                                #
    # --------------------------------------------------------------------- #
    mode, eq_name, is_static, nn_wrapper, aug_dim = PhysicsInformedNN.get_eq_details(model_arch)
    now = datetime.now()

    # --------------------------------------------------------------------- #
    #  Save each ranked model to temp dir + build PKL trial list             #
    # --------------------------------------------------------------------- #
    final_filepath = os.path.join(
        results_dir, f'best_base{config_id}_opt{optimizer_space_id}_models.pkl',
    )
    final_trials_list = []
    for _rank, (_loss, _m, _gms) in enumerate(_results_to_save):
        _tf = os.path.join(TEMP_MODEL_DIR, f'trial_{_rank}.pt')
        torch.save(_m, _tf)
        _mock = types.SimpleNamespace(
            number=_rank,
            value=_loss,
            params=dict(_gms),
            user_attrs={
                'neurons_per_layer':     neurons_per_layer,
                'activations_per_layer': activations_per_layer,
                'normalization_config':  norm_c,
                'model_arch':            model_arch,
                'physics_c':             physics_c,
            },
            datetime_start=now - timedelta(seconds=current_execution_time),
            datetime_complete=now,
        )
        final_trials_list.append(extract_single_trial_data(_m, _mock, rank=_rank))

    print(f"[SLSQP] {len(final_trials_list)} model(s) saved to temp dir")

    save_data = {
        'trials':            final_trials_list,
        'optimizer_space_id': optimizer_space_id,
        'num_trials':        len(final_trials_list),
        'best_trial_idx':    0,
        'saved_timestamp':   str(np.datetime64('now')),
    }
    with open(final_filepath, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"[SLSQP] PKL saved → {final_filepath}")

    # --------------------------------------------------------------------- #
    #  Physics RMSE — evaluated for each ranked model                        #
    # --------------------------------------------------------------------- #
    _rmse_per_rank = []
    for _rank, (_loss, _m, _gms) in enumerate(_results_to_save):
        _rm = {'rmse_ids': None, 'rmse_gm1': None, 'rmse_gm2': None, 'rmse_gm3': None}
        try:
            _rm = evaluate_physics(
                model=_m,
                T_val=T_train,
                vgs_col=train_data_mode,
                vds_col='Vds',
                ids_col=current2train,
                target_vds_list=target_vds_list,
                device=device,
            )
        except Exception as e:
            print(f"[SLSQP] Could not evaluate physics RMSE (rank {_rank}): {e}")
        _rmse_per_rank.append(_rm)

    # --------------------------------------------------------------------- #
    #  JSON — one entry per ranked result                                    #
    # --------------------------------------------------------------------- #
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_results_list = []
    for _rank, (_loss, _m, _gms) in enumerate(_results_to_save):
        _rm = _rmse_per_rank[_rank]
        new_results_list.append({
            'environment':                  environment,
            'user_metadata':                user_metadata,
            'meas_load_kwargs':             meas_load_kwargs,
            'trials_executed':              n_restarts,
            'config_id':                    config_id,
            'trial_number':                 _rank,
            'trial_value':                  _loss,
            'optimizer_space_id':           optimizer_space_id,
            'optimized_params':             dict(_gms),
            'optuna_bound_hits':            [],
            'physics_bound_hits':           [],
            'fitness':                      _loss,
            'rmse_ids':                     _rm.get('rmse_ids'),
            'rmse_gm1':                     _rm.get('rmse_gm1'),
            'rmse_gm2':                     _rm.get('rmse_gm2'),
            'rmse_gm3':                     _rm.get('rmse_gm3'),
            'tpe_db_path':                  None,
            'full_csv_path':                path_to_csv,
            'timestamp':                    timestamp,
            'csv_basename':                 csv_basename,
            'weights_filepath':             final_filepath,
            'time_to_reach_seconds':        current_execution_time,
            'trial_execution_time_seconds': current_execution_time,
            'model_eq_name':                eq_name,
            'model_space_key':              optimizer_space_id,
            'physics_c':                    physics_c,
            'model_arch':                   model_arch,
            'model_aug_dim':                aug_dim,
            'model_mode':                   mode,
            'model_nn_wrapper':             nn_wrapper,
        })

    # Load + merge with any existing JSON (same atomic-write pattern)
    existing_data = {}
    existing_results = []
    if os.path.exists(out_json):
        try:
            with open(out_json, 'r') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_data    = loaded
                    existing_results = loaded.get('results', [])
                elif isinstance(loaded, list):
                    existing_results = loaded
        except json.JSONDecodeError:
            pass

    exec_times = existing_data.get('execution_times', [])
    exec_times.append(current_execution_time)
    total_time_sum = sum(exec_times)

    # Merge: new ranked results replace existing entries with the same trial_number
    merged = {item['trial_number']: item
              for item in existing_results
              if 'trial_number' in item}
    for entry in new_results_list:
        merged[entry['trial_number']] = entry
    final_ordered_list = [merged[k] for k in sorted(merged)]

    final_output_data = {
        'execution_times':      exec_times,
        'total_execution_time': total_time_sum,
        'results':              final_ordered_list,
    }

    temp_json = out_json + '.tmp'
    try:
        with open(temp_json, 'w') as f:
            json.dump(final_output_data, f, indent=4)
        os.replace(temp_json, out_json)
    except Exception as e:
        print(f"[SLSQP] Failed to write JSON: {e}")
        if os.path.exists(temp_json):
            os.remove(temp_json)

    print(f"[SLSQP] Results JSON → {out_json}")

    # --------------------------------------------------------------------- #
    #  Post-processing — mirrors optimize_config() RUN_OPTIMIZE_CONFIG_AND_PLOT_PROCESS
    # --------------------------------------------------------------------- #
    if os.environ.get("RUN_OPTIMIZE_CONFIG_AND_PLOT_PROCESS", "False") == "True":
        print("\n[SLSQP Global Trigger] 'RUN_OPTIMIZE_CONFIG_AND_PLOT_PROCESS' is True.")
        print("Starting automated plotting and equation creation...")
        try:
            plot_create_eq_best_predictions(
                base_json_path=os.path.dirname(os.path.dirname(out_json)),
                base_env=environment,
                base_config_no=out_json,
                space_config_no=optimizer_space_id,
                env_to_train_info=current_env_info,
                early_stopping_metric="mean",
                n_best=n_best_results,
                save_plot=True,
                save_equations=True,
                val_interval=val_interval,
            )
            print("[SLSQP Global Trigger] Post-processing complete.")
        except Exception as e:
            print(f"[SLSQP Global Trigger] Error in post-processing: {e}")
            import traceback
            traceback.print_exc()

    return None, out_json
