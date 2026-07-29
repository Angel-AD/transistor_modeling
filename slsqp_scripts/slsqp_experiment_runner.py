"""
Parallel SLSQP Multi-Experiment Runner
======================================

This script orchestrates large-scale configuration sweeps for the SLSQP optimization 
pipeline (`per_neuron_slsqp_single.py`). It parses a JSON configuration file, computes 
the Cartesian product of all parameter lists (equations, keys, restarts, and Gm modes), 
and distributes the individual training tasks across multiple CPU cores using a 
ProcessPoolExecutor.

Unlike standard runners, this script also features a built-in post-processing pipeline: 
upon the successful completion of any individual optimization run, it automatically 
triggers `plot_saved_state.py` to generate visual metrics for that specific seed.

Accepted CLI Arguments
----------------------
* --config 
    Path to the JSON experiments configuration file. Defaults to `config.json` in the script directory.
* --master_root_path 
    Root directory where experiment subfolders will be generated.
* --max_workers 
    Global override for the maximum number of parallel CPU workers. Vetoes JSON presets.
* --only 
    Space-separated list of specific experiment names to execute, ignoring the rest.
* --min_vgs, --plot_vds_list, --plot_vgs_list 
    Data boundary and targeting parameters forwarded to the post-processing plot script.

JSON Configuration Structure
----------------------------
The JSON file should contain a `GLOBAL_DEFAULTS` dictionary for baseline settings, 
followed by individual experiment dictionaries.

Example `config.json`:
```json
{
  "GLOBAL_DEFAULTS": {
    "max_workers": 4,
    "timeout": 14400,
    "test_percent": 0.5
  },
  "baseline_sweep": {
    "eq_names": ["mod1_angelov"],
    "config_keys": [9, 10],
    "n_restarts_list": [5, 10],
    "gm_modes": [
      [0, 0.0, 0.0, 0.0],
      [1, 1.0, 0.0, 0.0]
    ]
  }
}
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _compress_log(log_path: Path) -> Path:
    """Gzip `log_path` in place -> `log_path.gz`, removing the original.
    Returns the path actually on disk (the .gz on success, else the original).
    The log is streamed live during the run; this compresses it once finished."""
    if not log_path.is_file():
        return log_path
    gz_path = log_path.with_suffix(log_path.suffix + ".gz")
    try:
        with open(log_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        log_path.unlink()
        return gz_path
    except Exception:
        return log_path  # keep the plain log if compression fails


def _read_log_text(path) -> str:
    """Read a per-config log that may be plain text or gzip-compressed."""
    p = Path(path)
    try:
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                return f.read()
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"[runner] could not read log: {e}"

# ------------------------------------------------------------------
# Portable paths
# ------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent              # .../physics_nn_pipeline/slsqp_scripts
_PHYS_NN_DIR = _HERE.parent                           # .../physics_nn_pipeline


def _find_project_root(start: Path) -> Path:
    """Walk up from `start` to the directory that holds both optim_utils/ and
    physics_nn_pipeline/. Resilient to layout changes (flattened repo vs the
    older parallel_optimization/ nesting)."""
    for d in (start, *start.parents):
        if (d / "optim_utils").is_dir() and (d / "physics_nn_pipeline").is_dir():
            return d
    return start.parents[1]  # best-effort fallback


_PROJECT_ROOT = _find_project_root(_HERE)

# Default to the current interpreter (works in any venv/conda env); override
# with $PYTHON_EXE only if a specific interpreter is required. Do NOT hard-fail
# on a hardcoded venv path that may not exist on this machine.
PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)

SCRIPT_PATH = _HERE / "per_neuron_slsqp_single.py"
if not SCRIPT_PATH.exists():
    raise FileNotFoundError(f"Single-run script not found: {SCRIPT_PATH}")

PLOT_SCRIPT_PATH = _PHYS_NN_DIR / "plot_saved_state.py"
if not PLOT_SCRIPT_PATH.exists():
    raise FileNotFoundError(f"Plot script not found: {PLOT_SCRIPT_PATH}")


def _find_default_csv() -> Path:
    """Locate the default measurement CSV across known data-dir names."""
    name = "cg2h40010_new_2.4_5_2_70W_center9.csv"
    for sub in ("csvs", "meas"):
        c = _PROJECT_ROOT / sub / name
        if c.exists():
            return c
    return _PROJECT_ROOT / "csvs" / name  # best-effort default (may be overridden by config)


DEFAULT_CSV = _find_default_csv()
DEFAULT_MASTER_ROOT = str(_PROJECT_ROOT / "master_slsqp_experiments_dirs")
DEFAULT_CONFIG_PATH = _HERE / "config.json"

# ------------------------------------------------------------------
# Hardcoded Base Fallbacks (Overridden dynamically by JSON GLOBAL_DEFAULTS)
# ------------------------------------------------------------------
_GM_OFF = ("0", 0.0, 0.0, 0.0)

DEFAULTS = {
    "csv":              str(DEFAULT_CSV),
    "eq_names":         ["mod1_angelov"],
    "config_keys":      [9],
    "n_restarts_list":  [5],
    "gm_modes":         [_GM_OFF],
    "test_percent":     0.0,
    "timeout":          4 * 3600,  # 14400 seconds
    "max_workers":      2,
}

# ------------------------------------------------------------------
# Worker Implementation
# ------------------------------------------------------------------
def run_experiment(task: dict) -> dict:
    try:
        return _run_experiment_impl(task)
    except Exception:
        import traceback
        tb = traceback.format_exc()
        name = task.get("name", "?")
        exp_dir = Path(task.get("exp_dir", "."))
        try:
            exp_dir.mkdir(parents=True, exist_ok=True)
            (exp_dir / "log.txt").write_text(
                f"# Experiment: {name}\n# Task: {task}\n\n"
                f"[runner] WORKER EXCEPTION:\n{tb}",
                encoding="utf-8",
            )
        except Exception:
            pass
        print(f"\n[runner] WORKER EXCEPTION for {name}:\n{tb}"
              f"--- end worker exception ---", flush=True)
        return {"name": name, "status": "failed", "elapsed": 0.0, "rc": -99,
                "seed": "", "log": str(exp_dir / "log.txt")}


def _run_experiment_impl(task: dict) -> dict:
    name      = task["name"]
    exp_dir   = Path(task["exp_dir"])
    csv       = task["csv"]
    eq_name   = task["eq_name"]
    cfg_key   = int(task["config_key"])
    n_rest    = int(task["n_restarts"])
    use_gm, gm1_w, gm2_w, gm3_w = task["gm_mode"]
    timeout_s = int(task["timeout"])
    min_vgs   = task.get("min_vgs")
    plot_vds_list = task.get("plot_vds_list")
    plot_vgs_list = task.get("plot_vgs_list")

    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "log.txt"
    seed_dst = exp_dir / "slsqp_seed.json"

    if seed_dst.exists():
        return {"name": name, "status": "skipped", "elapsed": 0.0,
                "seed": str(seed_dst), "log": str(log_path)}

    cmd = [
        PYTHON_EXE, str(SCRIPT_PATH),
        "--csv",         str(csv),
        "--eq_name",     eq_name,
        "--config_key",  str(cfg_key),
        "--n_restarts",  str(n_rest),
        "--use_gm",      str(use_gm),
        "--gm1_weight",  str(gm1_w),
        "--gm2_weight",  str(gm2_w),
        "--gm3_weight",  str(gm3_w),
        "--save_dir",    str(exp_dir),
    ]
    if min_vgs is not None:
        cmd.append(f"--min_vgs={min_vgs}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"]     = "utf-8"
    env["OMP_NUM_THREADS"]      = "1"
    env["MKL_NUM_THREADS"]      = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# Experiment: {name}\n")
        log.write(f"# CMD: {' '.join(cmd)}\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    proc.kill()
                rc = -1
                log.write("\n[runner] TIMEOUT — killed\n")
        except Exception as e:
            log.write(f"\n[runner] EXCEPTION: {e}\n")
            rc = -2

    elapsed = time.time() - start
    status  = "ok" if (rc == 0 and seed_dst.exists()) else "failed"

    if status == "ok":
        plot_path = exp_dir / "plot_slsqp_seed.png"
        plot_cmd = [
            PYTHON_EXE, str(PLOT_SCRIPT_PATH),
            "--seed",       str(seed_dst),
            "--eq_name",     eq_name,
            "--config_key", str(cfg_key),
            "--csv",         str(csv),
        ]
        if min_vgs is not None:
            plot_cmd.append(f"--min_vgs={min_vgs}")
        if plot_vds_list:
            plot_cmd.append(f"--plot_vds_list={plot_vds_list}")
        if plot_vgs_list:
            plot_cmd.append(f"--plot_vgs_list={plot_vgs_list}")

        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n[runner] Auto-plot: " + " ".join(plot_cmd) + "\n")
            log.flush()
            prc = subprocess.run(plot_cmd, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=600)
            log.write(f"[runner] Auto-plot rc={prc.returncode}\n")
        if prc.returncode != 0 or not plot_path.exists():
            status = "failed"
            rc = prc.returncode if prc.returncode != 0 else -3

    # Compress the per-config log now that all writes (run + auto-plot) are done.
    log_path = _compress_log(log_path)

    return {"name": name, "status": status, "elapsed": elapsed,
            "rc": rc, "seed": str(seed_dst), "log": str(log_path)}


# ------------------------------------------------------------------
# Tasks Factory
# ------------------------------------------------------------------
def _tag(eq_name, cfg_key, n_rest, gm_mode):
    use_gm, w1, w2, w3 = gm_mode
    return (f"{eq_name}_cfg{cfg_key}_n{n_rest}_gm{str(use_gm).replace(',', '-')}"
            f"_W{w1}-{w2}-{w3}")


def build_tasks(exp_name: str, cfg: dict, master_root: Path, min_vgs=None,
                plot_vds_list=None, plot_vgs_list=None):
    merged = {**DEFAULTS, **cfg}
    
    # Map raw JSON nested lists back to expected clean immutable tuples
    merged["gm_modes"] = [tuple(mode) for mode in merged["gm_modes"]]

    master_dir = master_root / exp_name
    master_dir.mkdir(parents=True, exist_ok=True)

    sweep = list(itertools.product(
        merged["eq_names"],
        merged["config_keys"],
        merged["n_restarts_list"],
        merged["gm_modes"],
    ))

    tasks = []
    for idx, (eq_name, cfg_key, n_rest, gm_mode) in enumerate(sweep, start=1):
        tag = _tag(eq_name, cfg_key, n_rest, gm_mode)
        tasks.append({
            "name":         f"{exp_name}/{tag}",
            "exp_dir":      str(master_dir / f"exp_{idx:03d}_{tag}"),
            "csv":          merged["csv"],
            "eq_name":      eq_name,
            "config_key":   cfg_key,
            "n_restarts":   n_rest,
            "gm_mode":      gm_mode,
            "timeout":      merged["timeout"],
            "min_vgs":      min_vgs,
            "plot_vds_list": plot_vds_list,
            "plot_vgs_list": plot_vgs_list,
        })
    return tasks, master_dir, merged["max_workers"]


# ------------------------------------------------------------------
# Main Execution Flow
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help=f"Path to the JSON experiments file (Default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--master_root_path", default=os.environ.get("MASTER_ROOT_PATH", DEFAULT_MASTER_ROOT),
                        help=f"Default: {DEFAULT_MASTER_ROOT}")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Override default/JSON max parallel workers globally if provided.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these experiment names (default: all).")
    parser.add_argument("--min_vgs", type=float, default=-4.0,
                        help="Forwarded floor limit value for auto-extrapolations.")
    parser.add_argument("--plot_vds_list", type=str, default="0,5,10,15,20,28",
                        help="Comma-separated Vds targets.")
    parser.add_argument("--plot_vgs_list", type=str, default="-3.5,-3,-2.5,-2,-1.5,-1,-.5,0",
                        help="Comma-separated Vgs targets.")
    args = parser.parse_args()

    # Convert relative paths smoothly to complete absolute paths
    config_path = Path(args.config).resolve()
    master_root = Path(args.master_root_path).resolve()
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Requested configuration JSON file not found: {config_path}")

    print(f"Loading Configuration: {config_path}")
    print(f"Master Output Root:    {master_root}")
    master_root.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    # Pop and apply global overrides from JSON into our runtime map context
    json_defaults = config_data.pop("GLOBAL_DEFAULTS", {})
    DEFAULTS.update(json_defaults)
    experiments = config_data

    # Allow CLI --max_workers to have absolute ultimate veto power over base and JSON presets
    if args.max_workers is not None:
        DEFAULTS["max_workers"] = args.max_workers

    if args.only:
        wanted = set(args.only)
        experiments = {k: v for k, v in experiments.items() if k in wanted}
        if not experiments:
            print(f"No valid experiments matched filtering criteria --only {args.only}")
            return

    all_results = []
    for exp_name, cfg in experiments.items():
        tasks, master_dir, max_workers_cfg = build_tasks(
            exp_name, cfg, master_root,
            min_vgs=args.min_vgs,
            plot_vds_list=args.plot_vds_list,
            plot_vgs_list=args.plot_vgs_list,
        )
        
        shutil.copy2(__file__, master_dir / Path(__file__).name)
        shutil.copy2(SCRIPT_PATH, master_dir / SCRIPT_PATH.name)

        print("=" * 70)
        print(f"Experiment: {exp_name}")
        print(f"  Output:   {master_dir}")
        print(f"  Tasks:    {len(tasks)}")
        print(f"  Workers:  {min(DEFAULTS['max_workers'], max_workers_cfg, len(tasks))}")
        print("=" * 70)

        workers = max(1, min(DEFAULTS['max_workers'], max_workers_cfg, len(tasks)))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(run_experiment, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                except Exception:
                    import traceback
                    t = futures[fut]
                    print(f"[runner] future raised for {t.get('name','?')}:\n"
                          f"{traceback.format_exc()}", flush=True)
                    r = {"name": t.get("name", "?"), "status": "failed",
                         "elapsed": 0.0, "rc": -100, "seed": "",
                         "log": t.get("exp_dir", "")}
                
                all_results.append(r)
                print(f"  [{r['status'].upper():7s}] {r['name']:60s} "
                      f"({r['elapsed']:.0f}s)", flush=True)
                
                if r["status"] != "ok":
                    log_path = Path(r.get("log", ""))
                    if log_path.is_file():
                        txt = _read_log_text(log_path)  # handles plain or .gz
                        tail = "\n".join(txt.splitlines()[-200:])
                        print(f"  --- log tail ({log_path}) ---\n{tail}\n"
                              f"  --- end log tail ---", flush=True)
                    else:
                        print(f"  [runner] no log file found at {log_path}", flush=True)

    print("\nSummary:")
    for r in all_results:
        print(f"  {r['status']:7s}  rc={r.get('rc','?'):>3}  "
              f"{r['elapsed']:6.0f}s  {r['name']}")
    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()