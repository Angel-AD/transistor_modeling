"""
Smoke test for slsqp_experiment_runner (config-driven, the production path).

Invokes the runner exactly the way a user would:

    python slsqp_experiment_runner.py --config slsqp_configs/opt_configs_smoke.json \
        --master_root_path <project_root>/tests/slsqp_runner_smoke ...

The smoke config (slsqp_configs/opt_configs_smoke.json) defines two cheap
experiments (mod1_angelov + classic_angelov, 1 SLSQP restart each) with
test_percent=0.0 — which deliberately exercises the empty-validation-set
fallback in train_model_slsqp.

For every produced experiment it asserts:
  1. a valid slsqp_seed.json was written and the runner's auto-plot
     (plot_saved_state.py --seed) produced plot_slsqp_seed.png;
  2. test_percent=0 fallback worked: trial_value is finite (with the old bug it
     was nan and the model was left at its initial, unoptimized params);
  3. RMSE consistency: plot_saved_state's --seed reconstruction reproduces the
     seed's logged ids_rmse.

It then persists the seeds as fixtures at
`<project_root>/tests/slsqp_{classic,mod1}_physics_seed_cfg9.json` for any
downstream tooling (e.g. the production physics+NN config references them).

All outputs land under <project_root>/tests/slsqp_runner_smoke/ and are wiped at
the start of each invocation.

Usage:
    python physics_nn_pipeline/slsqp_scripts/test_slsqp_runner_smoke.py
"""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))  # physics_nn_pipeline/ for smoke_paths + plot_saved_state
import smoke_paths  # noqa: E402  (all smoke outputs live under <project_root>/tests/)

# All smoke outputs live under <project_root>/tests/.
SMOKE_ROOT = smoke_paths.SLSQP_SMOKE_ROOT
SMOKE_CONFIG = smoke_paths.SLSQP_SMOKE_CONFIG
SMOKE_CSV = smoke_paths.SMOKE_CSV
SMOKE_MIN_VGS = smoke_paths.SMOKE_MIN_VGS
CLASSIC_FIXTURE = smoke_paths.CLASSIC_FIXTURE
MOD1_FIXTURE = smoke_paths.MOD1_FIXTURE
RUNNER = _HERE / "slsqp_experiment_runner.py"

_RMSE_RTOL = 1e-3
# Map equation name -> persisted fixture path.
_FIXTURE_BY_EQ = {"mod1_angelov": MOD1_FIXTURE, "classic_angelov": CLASSIC_FIXTURE}


def _rmtree_robust(path: Path, retries: int = 5, delay: float = 0.5) -> None:
    """rmtree that tolerates transient Windows file locks (e.g. PIL/Photos
    still holding the auto-plot PNG, OneDrive sync handle, etc.)."""
    def _onerror(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for attempt in range(retries):
        if not path.exists():
            return
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _validate_seed(seed_path: Path) -> tuple[bool, str, str | None]:
    """Validate one slsqp_seed.json. Returns (ok, message, eq_name)."""
    if not seed_path.is_file():
        return False, f"seed file missing at {seed_path}", None
    try:
        data = json.loads(seed_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"seed JSON unreadable: {e}", None
    results = data.get("results") if isinstance(data, dict) else None
    if not (isinstance(results, list) and results and isinstance(results[0], dict)):
        return False, "seed JSON missing results[0]", None
    entry = results[0]
    params = entry.get("optimized_params")
    if not isinstance(params, dict) or not params:
        return False, "seed missing results[0].optimized_params", None
    eq_name = entry.get("model_eq_name")
    cfg_key = entry.get("config_id", 9)

    # test_percent=0 fallback guard: with the old bug, every restart's val_loss
    # was nan, so trial_value was nan and the model kept its initial params.
    best = entry.get("trial_value")
    if best is None or (isinstance(best, float) and math.isnan(best)) or best == float("inf"):
        return False, f"SLSQP did not optimize (trial_value={best!r}); test_percent=0 fallback broken", eq_name

    # Auto-plot must have produced the PNG beside the seed.
    if not (seed_path.parent / "plot_slsqp_seed.png").is_file():
        return False, "auto-plot plot_slsqp_seed.png missing", eq_name

    # RMSE consistency: --seed reconstruction reproduces the logged ids_rmse.
    logged = entry.get("ids_rmse")
    if logged is None:
        return False, "seed missing 'ids_rmse'", eq_name
    import plot_saved_state as pss
    recomputed = pss.evaluate_seed_ids_rmse(
        str(seed_path), eq_name, int(cfg_key), str(SMOKE_CSV), min_vgs=SMOKE_MIN_VGS)
    rel = abs(recomputed - logged) / max(abs(logged), 1e-12)
    if rel > _RMSE_RTOL:
        return (False, f"ids_rmse mismatch: logged={logged:.6e} recomputed={recomputed:.6e} "
                f"rel_err={rel:.2e} > {_RMSE_RTOL:.0e}", eq_name)
    return True, (f"trial_value={best:.6e} ids_rmse logged={logged:.6e} "
                  f"recomputed={recomputed:.6e} rel_err={rel:.2e}"), eq_name


def main() -> int:
    if not SMOKE_CONFIG.exists():
        print(f"[smoke] FAIL: smoke config not found: {SMOKE_CONFIG}")
        return 1
    if not SMOKE_CSV.exists():
        print(f"[smoke] FAIL: measurement CSV not found: {SMOKE_CSV}")
        return 1

    # Always start from a clean slate so the runner's skip-if-finished
    # short-circuit (existing slsqp_seed.json) does not hide regressions.
    if SMOKE_ROOT.exists():
        _rmtree_robust(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] smoke_root  = {SMOKE_ROOT}")
    print(f"[smoke] config      = {SMOKE_CONFIG}")
    print(f"[smoke] csv         = {SMOKE_CSV}")

    # ---- Run the runner via its config-driven CLI (the production path) ----
    # Use the --opt=value form for args whose values start with '-' (negative
    # numbers / comma lists), otherwise argparse treats them as option flags.
    cmd = [
        sys.executable, str(RUNNER),
        "--config", str(SMOKE_CONFIG),
        "--master_root_path", str(SMOKE_ROOT),
        f"--min_vgs={SMOKE_MIN_VGS}",
        "--plot_vds_list=0,5,10,15,20,28",
        "--plot_vgs_list=-3.5,-3,-2.5,-2,-1.5,-1,-.5,0",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"[smoke] running: {' '.join(cmd)}")
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    print(f"[smoke] runner rc={rc} ({time.time() - t0:.1f}s)")
    if rc != 0:
        print(f"[smoke] FAIL: runner returned rc={rc}")
        return 1

    # ---- Validate every produced seed ----
    seeds = sorted(SMOKE_ROOT.rglob("slsqp_seed.json"))
    print(f"[smoke] found {len(seeds)} seed(s)")
    if len(seeds) < 2:
        print(f"[smoke] FAIL: expected >=2 seeds, got {len(seeds)}")
        return 1

    fail = False
    persisted: dict[str, Path] = {}
    for seed_path in seeds:
        ok, msg, eq_name = _validate_seed(seed_path)
        tag = seed_path.parent.name
        if ok:
            print(f"[smoke]   seed OK [{tag}]: {msg}")
            fx = _FIXTURE_BY_EQ.get(eq_name)
            if fx is not None:
                fx.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(seed_path, fx)
                persisted[eq_name] = fx
        else:
            print(f"[smoke]   !! seed FAILED [{tag}]: {msg}")
            fail = True

    for eq, fx in persisted.items():
        print(f"[smoke] fixture    = {fx} (updated from {eq})")

    print(f"\n[smoke] artifacts   = {SMOKE_ROOT}")
    if fail:
        print("[smoke] FAIL")
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
