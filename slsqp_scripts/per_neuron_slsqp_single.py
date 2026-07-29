"""
Single-run SLSQP fit for a noNN PhysicsInformedNN model.

Bypasses `per_neuron_trainer_8` orchestration / Optuna / solver-spaces table.
Builds the model directly and invokes `train_model_slsqp` (pure scipy SLSQP).

Mirrors the spirit of `per_neuron_simple_angelov_nn_test.py` from the
physics+NN pipeline: argparse CLI -> data load -> model build -> fit ->
write a `slsqp_seed.json` consumable by `plot_saved_state.py --seed` and
`multi_experiment_runner.py --opt_params_path`.

Output JSON shape (matches the legacy trainer output):
    {
        "execution_times": [<float>],
        "total_execution_time": <float>,
        "results": [
            {
                "trial_number": 0,
                "trial_value": <float>,
                "optimized_params": {<name>: <float>, ...},
                "environment": "experimental",
                "model_eq_name": <eq_name>,
                "config_id": <config_key>,
                "n_restarts": <int>,
                "use_gm": <str>,
                "gm1_weight": <float>,
                "gm2_weight": <float>,
                "gm3_weight": <float>,
                "full_csv_path": <str>,
                "csv_basename": <str>,
                "timestamp": <str>
            }
        ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Portable PYTHONPATH bootstrap -> sibling optim_utils/ package
# slsqp_scripts/ -> physics_nn_pipeline/ -> parallel_optimization/optim_utils/
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PARALLEL_OPT_DIR = _HERE.parents[1]
_REPO_ROOT = _HERE.parents[2]
_OPTIM_UTILS = Path(os.environ.get(
    "OPTIM_UTILS_DIR",
    str(_PARALLEL_OPT_DIR / "optim_utils"),
))
if str(_OPTIM_UTILS) not in sys.path:
    sys.path.insert(0, str(_OPTIM_UTILS))

from measurements_load import meas_load  # noqa: E402
from per_neuron_models import PhysicsInformedNN  # noqa: E402
from per_neuron_normalization import USED_NORMALIZATIONS  # noqa: E402
from per_neuron_plotting import smooth_derivative  # noqa: E402
from per_neuron_slsqp import train_model_slsqp  # noqa: E402


# ------------------------------------------------------------------
# Local GM-derivative builder (train + val) — extracted from
# per_neuron_trainer_8.create_gms_for_train, with no module-level deps.
# ------------------------------------------------------------------
def _create_gms_for_train(T_train, T_test, vgs_col="Vgs_meas", ids_col="Ids"):
    T_train = T_train.copy()
    T_train["Step_Index"] = T_train.groupby("TN").cumcount()
    T_train = T_train.sort_values(by=["Step_Index", "TN"]).reset_index(drop=True)
    X_train = np.column_stack((T_train[vgs_col], T_train["Vds"]))
    y_train = np.array(T_train[ids_col]).reshape(-1, 1)

    T_test = T_test.copy()
    T_test["Step_Index"] = T_test.groupby("TN").cumcount()
    T_test = T_test.sort_values(by=["Step_Index", "TN"]).reset_index(drop=True)
    X_val = np.column_stack((T_test[vgs_col], T_test["Vds"]))
    Y_val = np.array(T_test[ids_col]).reshape(-1, 1)

    gm1_list, gm2_list, gm3_list = [], [], []
    for _, group in T_train.groupby("Step_Index"):
        v_pts = group[vgs_col].values
        i_pts = group[ids_col].values
        n_pts = len(v_pts)
        w_pre = min(11, n_pts if n_pts % 2 != 0 else n_pts - 1)
        w_post = min(13, n_pts if n_pts % 2 != 0 else n_pts - 1)
        if n_pts >= 4 and w_pre >= 3 and w_post >= 3:
            # gm-truth = dⁿIds/dVgsⁿ via np.gradient(Ids, Vgs); only valid if Vgs
            # is a strictly monotonic sweep within the group. Fail loudly on a
            # different layout rather than silently producing garbage gm-truth.
            dv = np.diff(v_pts)
            if not (np.all(dv > 0) or np.all(dv < 0)):
                raise ValueError(
                    f"_create_gms_for_train: Step_Index group is not strictly "
                    f"monotonic in {vgs_col} (n={n_pts}, "
                    f"span={v_pts.max() - v_pts.min():.4f}, "
                    f"min|d{vgs_col}|={np.abs(dv).min():.3e}). gm-truth requires a "
                    "sorted, monotonic Vgs sweep per group (each TN = a Vds sweep "
                    "at fixed Vgs, TNs ordered by Vgs)."
                )
            gm1 = smooth_derivative(v_pts, i_pts, order=1, win_pre=w_pre, win_post=w_post)
            gm2 = smooth_derivative(v_pts, i_pts, order=2, win_pre=w_pre, win_post=w_post)
            gm3 = smooth_derivative(v_pts, i_pts, order=3, win_pre=w_pre, win_post=w_post)
        else:
            gm1 = np.zeros(n_pts); gm2 = np.zeros(n_pts); gm3 = np.zeros(n_pts)
        gm1_list.extend(gm1); gm2_list.extend(gm2); gm3_list.extend(gm3)

    return (np.array(gm1_list), np.array(gm2_list), np.array(gm3_list),
            X_train, y_train, X_val, Y_val)


# ------------------------------------------------------------------
# Post-fit metric helpers (IDS RMSE + GM1/2/3 RMSE on training data).
# Mirrors `get_gm_rmse_metrics` from per_neuron_simple_angelov_nn_test.py
# but uses the model's dtype/device to avoid float32/float64 mismatches.
# ------------------------------------------------------------------
def _compute_post_fit_rmses(model, X_train_np, y_train_np,
                            gm1_true, gm2_true, gm3_true):
    p0 = next(model.parameters())
    dtype = p0.dtype
    device = p0.device

    X_t = torch.tensor(X_train_np, dtype=dtype, device=device)
    y_t = torch.tensor(y_train_np, dtype=dtype, device=device)

    # IDS RMSE on training data (no autograd needed).
    with torch.no_grad():
        preds = model(X_t)
        ids_rmse = float(torch.sqrt(torch.mean((preds - y_t) ** 2)).item())

    # GM1/2/3 RMSE via autograd against smoothed truth vectors.
    def _rmse(pred_t, truth_arr):
        if truth_arr is None or len(truth_arr) == 0:
            return 0.0
        t = torch.tensor(truth_arr, dtype=dtype, device=device)
        return float(torch.sqrt(torch.mean((pred_t.detach() - t) ** 2)).item())

    X_req = X_t.clone().detach().requires_grad_(True)
    out = model(X_req)
    try:
        gm1_pred = torch.autograd.grad(
            outputs=out, inputs=X_req,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
        )[0][:, 0]
        gm2_pred = torch.autograd.grad(
            outputs=gm1_pred, inputs=X_req,
            grad_outputs=torch.ones_like(gm1_pred),
            create_graph=True,
        )[0][:, 0]
        gm3_pred = torch.autograd.grad(
            outputs=gm2_pred, inputs=X_req,
            grad_outputs=torch.ones_like(gm2_pred),
        )[0][:, 0]
        gm1_rmse = _rmse(gm1_pred, gm1_true)
        gm2_rmse = _rmse(gm2_pred, gm2_true)
        gm3_rmse = _rmse(gm3_pred, gm3_true)
    except RuntimeError as e:
        print(f"[slsqp_single] WARN: GM RMSE autograd failed ({e}); reporting 0.0.")
        gm1_rmse = gm2_rmse = gm3_rmse = 0.0

    return ids_rmse, gm1_rmse, gm2_rmse, gm3_rmse


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, type=str,
                   help="Path to measurement CSV (auriga format).")
    p.add_argument("--eq_name", required=True,
                   choices=["classic_angelov", "mod1_angelov",
                            "angelov_6_term", "angelov_9_term"],
                   help="noNN equation flavour.")
    p.add_argument("--config_key", type=int, default=9,
                   help="Physics config key (PHYSICS_PARAM_CONFIG / nonn defaults).")
    p.add_argument("--n_restarts", type=int, default=5,
                   help="Number of random SLSQP restarts (best loss kept).")
    p.add_argument("--use_gm", type=str, default="0",
                   help='GM term spec: "0" (off) | "1" | "1,2" | "1,2,3".')
    p.add_argument("--gm1_weight", type=float, default=0.0)
    p.add_argument("--gm2_weight", type=float, default=0.0)
    p.add_argument("--gm3_weight", type=float, default=0.0)
    p.add_argument("--save_dir", required=True, type=str,
                   help="Output directory; slsqp_seed.json is written here.")
    p.add_argument("--seed_filename", type=str, default="slsqp_seed.json")
    p.add_argument("--timeout", type=float, default=None,
                   help="Optional per-restart wall-time budget (seconds).")
    p.add_argument("--test_percent", type=float, default=0.0,
                   help="meas_load test_percent (train/val split).")
    p.add_argument("--min_vgs", type=float, default=None,
                   help="If set (e.g. -4.0), meas_load auto-extrapolates extra start-groups so the most-negative Vgs reaches this floor.")
    p.add_argument("--meas_file_type", type=str, default="auriga")
    p.add_argument("--normalization_key", type=str, default="cgh40010f_vgs4_vds45",
                   help="Key into USED_NORMALIZATIONS for the model bounds.")
    p.add_argument("--torch_seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_json = save_dir / args.seed_filename

    csv_path = str(Path(args.csv).resolve())
    print(f"[slsqp_single] csv          = {csv_path}")
    print(f"[slsqp_single] eq_name      = {args.eq_name}")
    print(f"[slsqp_single] config_key   = {args.config_key}")
    print(f"[slsqp_single] n_restarts   = {args.n_restarts}")
    print(f"[slsqp_single] use_gm       = {args.use_gm}")
    print(f"[slsqp_single] gm_weights   = ({args.gm1_weight}, {args.gm2_weight}, {args.gm3_weight})")
    print(f"[slsqp_single] save_dir     = {save_dir}")

    # GM env vars are read at call time by train_model_slsqp.
    os.environ["USE_GM"] = str(args.use_gm)
    os.environ["GM1_WEIGHT"] = str(args.gm1_weight)
    os.environ["GM2_WEIGHT"] = str(args.gm2_weight)
    os.environ["GM3_WEIGHT"] = str(args.gm3_weight)

    torch.manual_seed(args.torch_seed)
    np.random.seed(args.torch_seed)

    # --- Data ---
    T_train, T_test = meas_load(
        csv_path,
        file_type=args.meas_file_type,
        test_percent=args.test_percent,
        min_vgs=args.min_vgs,
    )
    gm1_arr, gm2_arr, gm3_arr, X_train, y_train, X_val, Y_val = \
        _create_gms_for_train(T_train, T_test)

    # --- Model ---
    norm_c = dict(USED_NORMALIZATIONS.get(args.normalization_key, {}))
    norm_c["hybrid"] = True
    model = PhysicsInformedNN(
        base_nn=None,
        equation_type=f"noNN:{args.eq_name}",
        normalization_config=norm_c,
        config_key=args.config_key,
    ).double()

    assert getattr(model, "mode", None) == "noNN", \
        f"Expected noNN model, got mode={getattr(model, 'mode', None)!r}"

    criterion = nn.MSELoss()

    # --- Fit ---
    t0 = time.time()
    model, best_loss, history = train_model_slsqp(
        model,
        X_train, y_train,
        X_val, Y_val,
        criterion,
        n_restarts=args.n_restarts,
        timeout=args.timeout,
        verbose=True,
        gm1_true_arr=gm1_arr,
        gm2_true_arr=gm2_arr,
        gm3_true_arr=gm3_arr,
    )
    elapsed = time.time() - t0
    print(f"[slsqp_single] best_loss = {best_loss:.6e}  ({elapsed:.1f}s, {len(history)} restarts)")

    # --- Post-fit metrics (so compile_results_csv has populated columns) ---
    ids_rmse, gm1_rmse, gm2_rmse, gm3_rmse = _compute_post_fit_rmses(
        model, X_train, y_train, gm1_arr, gm2_arr, gm3_arr,
    )
    print(f"[slsqp_single] ids_rmse  = {ids_rmse:.6e}")
    print(f"[slsqp_single] gm_rmse   = ({gm1_rmse:.4e}, {gm2_rmse:.4e}, {gm3_rmse:.4e})")

    # --- Extract real params (dict of floats) ---
    real_params_raw = model.get_real_params()
    real_params = {}
    for k, v in real_params_raw.items():
        if hasattr(v, "item"):
            try:
                real_params[k] = float(v.item())
                continue
            except Exception:
                pass
        try:
            real_params[k] = float(v)
        except Exception:
            real_params[k] = v  # last-resort: keep as-is

    # --- Write JSON ---
    entry = {
        "trial_number":          0,
        "trial_value":           float(best_loss),
        "fitness":               float(best_loss),
        "ids_rmse":              float(ids_rmse),
        "gm1_rmse":              float(gm1_rmse),
        "gm2_rmse":              float(gm2_rmse),
        "gm3_rmse":              float(gm3_rmse),
        "optimized_params":      real_params,
        "environment":           "experimental",
        "model_eq_name":         args.eq_name,
        "config_id":             args.config_key,
        "n_restarts":            args.n_restarts,
        "use_gm":                args.use_gm,
        "gm1_weight":            args.gm1_weight,
        "gm2_weight":            args.gm2_weight,
        "gm3_weight":            args.gm3_weight,
        "full_csv_path":         csv_path,
        "csv_basename":          os.path.basename(csv_path),
        "timestamp":             datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    out_data = {
        "execution_times":       [elapsed],
        "total_execution_time":  elapsed,
        "results":               [entry],
    }

    tmp = out_json.with_suffix(out_json.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)
    os.replace(tmp, out_json)
    print(f"[slsqp_single] wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
