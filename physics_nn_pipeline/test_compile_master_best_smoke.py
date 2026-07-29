"""
Smoke test for compile_master_best.py.

Depends on the compile_results_csv smoke (which itself depends on the
physics+NN and SLSQP runner smokes). If any per-experiment
compiled_results.csv is missing, runs the upstream compile smoke first.

Validates that compile_master_best.py produces a master CSV per root
with the correct row count (top_n * number of experiments) and an
`experiment` column populated for each row.

Usage:
    python parallel_optimization/physics_nn_pipeline/test_compile_master_best_smoke.py
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import smoke_paths  # noqa: E402  (all smoke outputs live under <project_root>/tests/)

MASTER_SCRIPT = _HERE / "compile_master_best.py"
COMPILE_SMOKE = _HERE / "test_compile_results_csv_smoke.py"
PHYS_SMOKE_ROOT = smoke_paths.PHYS_SMOKE_ROOT
SLSQP_SMOKE_ROOT = smoke_paths.SLSQP_SMOKE_ROOT

# (root, metric, top_n)
CASES = [
    (PHYS_SMOKE_ROOT,  "ids_rmse",  2),
    (SLSQP_SMOKE_ROOT, "best_loss", 1),
]


def _all_compiled_present(root: Path) -> bool:
    if not root.exists():
        return False
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        return False
    return all(smoke_paths.find_compiled_csv(p) is not None for p in subdirs)


def _ensure_compiled_csvs() -> bool:
    if all(_all_compiled_present(r) for r, *_ in CASES):
        print("[smoke] compiled_results.csv present in all experiments")
        return True
    print(f"[smoke] running upstream compile smoke -> {COMPILE_SMOKE.name}")
    t0 = time.time()
    rc = subprocess.call([sys.executable, str(COMPILE_SMOKE)])
    print(f"[smoke] upstream compile smoke rc={rc} ({time.time()-t0:.1f}s)")
    return rc == 0


def _run_master(root: Path, metric: str, top_n: int) -> tuple[int, Path]:
    # compile_master_best.py appends the root's basename: master_best_<metric>_top<N>_<root>.csv
    out_path = root / f"master_best_{metric}_top{top_n}_{root.name}.csv"
    if out_path.exists():
        out_path.unlink()
    print(f"\n[smoke] --- compile_master_best --root {root.name} "
          f"--metric {metric} --top_n {top_n} ---")
    t0 = time.time()
    rc = subprocess.call([
        sys.executable, str(MASTER_SCRIPT),
        "--root", str(root),
        "--metric", metric,
        "--top_n", str(top_n),
    ])
    print(f"[smoke] master rc={rc} ({time.time()-t0:.1f}s)")
    return rc, out_path


def _validate(out_csv: Path, root: Path, top_n: int) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not out_csv.is_file():
        return False, [f"missing master CSV {out_csv}"]

    with out_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)

    if "experiment" not in header:
        errors.append("master CSV missing 'experiment' column")

    experiments = sorted(p.name for p in root.iterdir() if p.is_dir())
    seen = {r.get("experiment") for r in rows}
    missing_exp = [e for e in experiments if e not in seen]
    if missing_exp:
        errors.append(f"master CSV missing rows for experiments: {missing_exp}")

    # Expected: top_n rows per experiment, bounded by the per-experiment row count.
    expected = 0
    for exp in experiments:
        sub_csv = smoke_paths.find_compiled_csv(root / exp)
        if sub_csv is None:
            continue
        with sub_csv.open("r", newline="") as f:
            n = sum(1 for _ in csv.DictReader(f))
        expected += min(top_n, n)

    if len(rows) != expected:
        errors.append(f"master CSV has {len(rows)} rows, expected {expected}")
    else:
        print(f"[smoke]   {out_csv.name}: {len(rows)} rows OK")

    return not errors, errors


def main() -> int:
    if not MASTER_SCRIPT.is_file():
        print(f"[smoke] FAIL: missing {MASTER_SCRIPT}")
        return 1

    t_total = time.time()
    if not _ensure_compiled_csvs():
        return 1

    fail = False
    for root, metric, top_n in CASES:
        rc, out_csv = _run_master(root, metric, top_n)
        if rc != 0:
            print(f"[smoke] FAIL: compile_master_best returned non-zero for {root.name}")
            fail = True
            continue
        ok, errors = _validate(out_csv, root, top_n)
        if not ok:
            print(f"[smoke] FAIL ({root.name}):")
            for e in errors:
                print(f"          - {e}")
            fail = True

    print(f"\n[smoke] total       = {time.time()-t_total:.1f}s")
    if fail:
        print("[smoke] FAIL")
        return 1
    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
