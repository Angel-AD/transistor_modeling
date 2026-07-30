"""
Compile run_*.json files under a directory tree into a single sorted CSV.

Recursively scans <target_dir>/**/ for JSON outputs from either:
  * per_neuron_simple_angelov_nn_test.py (one run = one run_loss_*.json), or
  * the SLSQP seed schema produced by the slsqp runners
    ({"results": [{"fitness": ..., "optimized_params": {...}, ...}, ...]}).

Writes <target_dir>/compiled_results.csv with the same column set as
compile_ranked.py (plus config_name / lr):
    category, combo, phase, eq, knee_combiner, best_loss, combined_gm,
    ids_rmse, gm1_rmse, gm2_rmse, gm3_rmse, use_gm,
    gm1_weight, gm2_weight, gm3_weight, gm_surgery_mode,
    output_activation, config_name, lr, architecture, file_path
(eq / knee_combiner are "NA" for pure-NN rows; combined_gm =
 mean(gm1/0.04661, gm2/0.20352, gm3/0.94910).)

Rows are sorted ascending by --sort_by (multi-key tuple sort; non-numeric
or missing values are pushed to the bottom).

CLI args (see --help):
    --dir       target directory to scan (default: 'fast_pure_nn_tests_softplus')
    --root      master root containing multiple experiment subdirs; if set, a
                CSV is generated for each immediate subdirectory
    --sort_by   comma-separated columns to rank by; default 'best_loss'.
                First key is primary; the rest break ties.
                Valid keys: best_loss, ids_rmse, gm1_rmse, gm2_rmse, gm3_rmse, lr

Examples
--------
# Compile one experiment, sorted by ids_rmse then gm1_rmse:
python parallel_optimization/physics_nn_pipeline/compile_results_csv.py \
    --dir master_experiments_dirs/pure_nn_no_gm --sort_by ids_rmse,gm1_rmse

# Compile every experiment under a master root (one CSV per subdir):
python parallel_optimization/physics_nn_pipeline/compile_results_csv.py \
    --root master_experiments_dirs
"""

import os
import json
import csv
import glob
import argparse
import run_artifacts

# Baselines for the normalized combined_gm score (matches compile_ranked.py):
# best single-gm RMSEs from the base architecture search.
GM_BASELINES = (0.04661, 0.20352, 0.94910)


def _combined_gm(g1, g2, g3):
    """mean(gm1/b1, gm2/b2, gm3/b3); '' if any gm value is missing/non-numeric."""
    try:
        b1, b2, b3 = GM_BASELINES
        return round((float(g1) / b1 + float(g2) / b2 + float(g3) / b3) / 3.0, 4)
    except (TypeError, ValueError):
        return ""


def _row_sort_key(row, keys):
    """Build a tuple of floats for sorting a row by `keys`.

    Missing/blank/non-numeric values are pushed to the end (treated as +inf).
    """
    out = []
    for k in keys:
        v = row.get(k, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(float("inf"))
    return tuple(out)


def compile_results(target_dir, output_csv="compiled_results.csv", sort_keys=None,
                    base_csv=None, base_exp=None):
    json_pattern = os.path.join(target_dir, "**", "*.json")
    json_files = glob.glob(json_pattern, recursive=True)
    # Skip the best_n_configs/ collection dir: plot_best_configs.py copies winners'
    # run_loss_*.json there, and re-ingesting those would create duplicate rows.
    json_files = [j for j in json_files
                  if "best_n_configs" not in os.path.normpath(j).split(os.sep)]

    # Experiment folder name -> category/combo/phase (folders named
    # <category>__<combo>__<phase>; missing parts left blank), matching compile_ranked.py.
    _exp = os.path.basename(os.path.normpath(target_dir))
    _eb = _exp.split("__")
    _category = _eb[0]
    _combo = _eb[1] if len(_eb) > 1 else ""
    _phase = _eb[2] if len(_eb) > 2 else ""

    results = []
    
    for jpath in json_files:
        try:
            with open(jpath, 'r') as f:
                data = json.load(f)

            # ------------------------------------------------------------------
            # SLSQP seed schema: {"results": [{"fitness": ..., "optimized_params":
            #   {...}, "model_eq_name": ..., "config_id": ..., "use_gm": ...,
            #   "gm1_weight": ..., ...}, ...]}
            # Emit one row per result entry, picking the best by fitness for the
            # `best_loss` column.
            # ------------------------------------------------------------------
            if (isinstance(data, dict) and isinstance(data.get("results"), list)
                    and data["results"]
                    and isinstance(data["results"][0], dict)
                    and "optimized_params" in data["results"][0]
                    and ("fitness" in data["results"][0]
                         or "trial_value" in data["results"][0])):
                best = min(
                    data["results"],
                    key=lambda r: r.get("fitness",
                                        r.get("trial_value", float("inf"))),
                )
                loss = best.get("fitness", best.get("trial_value", float("inf")))
                eq = best.get("model_eq_name", "slsqp")
                cfg = best.get("config_id", "?")
                results.append({
                    "id": None,
                    "file_path": jpath,
                    "best_loss": loss,  # SLSQP fitness (objective value)
                    "ids_rmse": best.get("ids_rmse", ""),
                    "gm1_rmse": best.get("gm1_rmse", ""),
                    "gm2_rmse": best.get("gm2_rmse", ""),
                    "gm3_rmse": best.get("gm3_rmse", ""),
                    "config_name": f"slsqp_{eq}_cfg{cfg}",
                    "architecture": "slsqp",
                    "lr": "",
                    "output_activation": "",
                    "gm_surgery_mode": str(best.get("use_gm", "none")),
                })
                continue

            # ------------------------------------------------------------------
            # Physics+NN run_*.json schema (default).
            # ------------------------------------------------------------------
            loss = data.get("best_loss", data.get("mse_loss", float('inf')))
            config_name = data.get("config_name", "Unknown")
            architecture = data.get("architecture", "")
            lr = data.get("lr", data.get("learning_rate", ""))
            out_act = data.get("output_activation", "")
            surgery = data.get("gm_surgery_mode", "none")

            gm1_rmse = data.get("gm1_rmse", "")
            gm2_rmse = data.get("gm2_rmse", "")
            gm3_rmse = data.get("gm3_rmse", "")
            ids_rmse = data.get("ids_rmse", "")

            # The script logic sometimes places values nested in 'args' dictionary
            if "args" in data:
                args = data["args"]
                if not architecture: architecture = args.get("architecture", "")
                if not lr: lr = args.get("lr", args.get("learning_rate", ""))
                if not out_act: out_act = args.get("output_activation", "")
                if surgery == "none": surgery = args.get("gm_surgery_mode", "none")

            use_gm = data.get("use_gm", "")
            # gm weights: prefer the run JSON (records all three since the gmN_weight fix);
            # fall back to the run-dir name (exp_NNN_<mode>_W<g1>-<g2>-<g3>_s<seed>) for older
            # runs whose JSON stored only gm1_weight. Coerce to float for a clean column.
            _dn = ["", "", ""]
            _b = os.path.basename(os.path.dirname(jpath))
            if "_W" in _b and "_s" in _b:
                _p = _b.split("_W", 1)[1].split("_s", 1)[0].split("-")
                if len(_p) == 3:
                    _dn = _p
            def _gw(key, i):
                v = data.get(key)
                if v is None:
                    v = _dn[i]
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return v
            gm1_w, gm2_w, gm3_w = _gw("gm1_weight", 0), _gw("gm2_weight", 1), _gw("gm3_weight", 2)

            eq_type = data.get("equation_type", "")
            knee_combiner = data.get("knee_combiner", "")
            _pure = (eq_type == "pure")

            row = {
                "id": None,
                "run_id": None,    # readable 1..N per distinct run; same in every ranked file
                "run_hash": run_artifacts.run_hash(jpath),   # globally-unique stable id for this run
                "file_path": jpath,
                "category": _category,
                "combo": _combo,
                "phase": _phase,
                "arch_id": None,   # small readable 1..N per distinct arch; assigned before write
                "arch_hash": run_artifacts.arch_id(architecture),   # stable id (same arch -> same hash)
                "base_exp": "NA", "base_id": "NA", "base_csv": "NA",   # lineage; filled before write
                "eq": "NA" if _pure else eq_type,
                "knee_combiner": "NA" if _pure else knee_combiner,
                "best_loss": loss,
                "combined_gm": _combined_gm(gm1_rmse, gm2_rmse, gm3_rmse),
                "ids_rmse": ids_rmse,
                "gm1_rmse": gm1_rmse,
                "gm2_rmse": gm2_rmse,
                "gm3_rmse": gm3_rmse,
                "config_name": config_name,
                "architecture": architecture,
                "lr": lr,
                "output_activation": out_act,
                "use_gm": use_gm,
                "gm1_weight": gm1_w,
                "gm2_weight": gm2_w,
                "gm3_weight": gm3_w,
                "gm_surgery_mode": surgery,
                # --- full physics / knee-window / Ids-guard config ---
                "freeze_physics": data.get("freeze_physics", ""),
                "use_opt_params": data.get("use_opt_params", ""),
                "knee_alpha_scale": data.get("knee_alpha_scale", ""),
                "knee_vgs_thr": data.get("knee_vgs_thr", ""),
                "knee_vgs_tau": data.get("knee_vgs_tau", ""),
                "knee_max_correction": data.get("knee_max_correction", ""),
                "ids_constraint": data.get("ids_constraint", ""),
                "ids_target": data.get("ids_target", ""),
                "ids_lambda": data.get("ids_lambda", ""),
                "gm_warmup_epochs": data.get("gm_warmup_epochs", ""),
                "gm_warmup_lr": data.get("gm_warmup_lr", ""),
                "gm_vds_min": data.get("gm_vds_min", ""),
                "gm_vgs_min": data.get("gm_vgs_min", ""),
                "ids_region_weight": data.get("ids_region_weight", ""),
                "ids_region_center": data.get("ids_region_center", ""),
                "ids_region_width": data.get("ids_region_width", ""),
                "ids_region_lo": data.get("ids_region_lo", ""),
                "ids_region_hi": data.get("ids_region_hi", ""),
                "deterministic": data.get("deterministic", ""),
                "mixed_init": data.get("mixed_init", ""),
                "vds_loss": data.get("vds_loss", 0),   # absent (pre-feature runs) = penalty off = 0
                # Config/DEFAULTS knobs: newer JSONs record them; older runs fall back to the
                # exact invocation in run_log.txt.gz (CMD line).
                "gm_max_ratio": run_artifacts.field(data, os.path.dirname(jpath), "gm_max_ratio"),
                "lbfgs_epochs": run_artifacts.field(data, os.path.dirname(jpath), "lbfgs_epochs"),
                "lbfgs_max_iter": run_artifacts.field(data, os.path.dirname(jpath), "lbfgs_max_iter"),
                "lbfgs_gm_aware": run_artifacts.field(data, os.path.dirname(jpath), "lbfgs_gm_aware"),
                "loss_norm": run_artifacts.field(data, os.path.dirname(jpath), "loss_norm"),
                "adamw_avoid_localmin": run_artifacts.field(data, os.path.dirname(jpath), "adamw_avoid_localmin"),
                "csv": run_artifacts.field(data, os.path.dirname(jpath), "csv"),
                "epochs": data.get("epochs", ""),
                "seed": data.get("seed", ""),
            }
            # Region-localized metrics (compute_region_metrics.py) -> flat region_<name>_<metric>
            # keys. Auto-included as columns; absent in runs that weren't post-processed.
            row.update({k: v for k, v in data.items()
                        if k.startswith("region_") and k != "region_metrics"})
            # Region combined_gm: same baseline-normalized mean of the three region gm RMSEs
            # as the global combined_gm, derived per region.
            for gk in [k for k in row if k.startswith("region_") and k.endswith("_gm1_rmse")]:
                rname = gk[len("region_"):-len("_gm1_rmse")]
                cg = _combined_gm(row.get(f"region_{rname}_gm1_rmse"),
                                  row.get(f"region_{rname}_gm2_rmse"),
                                  row.get(f"region_{rname}_gm3_rmse"))
                if cg != "":
                    row[f"region_{rname}_combined_gm"] = cg
            results.append(row)

        except Exception as e:
            print(f"Failed to read {jpath}: {e}")

    # Sort results from best (lowest) to worst by the requested key(s).
    # Default = best_loss only (preserves historical behaviour).
    if not sort_keys:
        sort_keys = ["best_loss"]
    results.sort(key=lambda x: _row_sort_key(x, sort_keys))
    for _i, _r in enumerate(results, 1):     # 1-based id in this file (1 = best by sort_by)
        _r["id"] = _i
    # Small readable arch_id (1..N) per distinct architecture (keyed by arch_hash), stable order.
    _num = {h: i for i, h in enumerate(sorted({r.get("arch_hash") for r in results}), 1)}
    for _r in results:
        _r["arch_id"] = _num.get(_r.get("arch_hash"))
    # Global run_id: 1..N per distinct run (keyed by run_hash), stable order.
    _rnum = {h: i for i, h in enumerate(sorted({r.get("run_hash") for r in results}), 1)}
    for _r in results:
        _r["run_id"] = _rnum.get(_r.get("run_hash"))
    # Lineage: base_exp / base_id / base_csv, from _provenance.json (A) or base_csv (B).
    _bidx = run_artifacts.base_csv_index(base_csv, base_exp) if base_csv else None
    for _r in results:
        be, bid, bc = run_artifacts.base_lineage(_r.get("file_path", ""), _r.get("arch_hash", ""), _bidx)
        _r["base_exp"], _r["base_id"], _r["base_csv"] = be, bid, bc

    if not results:
        print(f"No JSON results found in directory: {target_dir}")
        return

    # Write to CSV
    keys = ["id", "run_id", "run_hash", "category", "combo", "phase", "arch_id", "arch_hash",
            "base_exp", "base_id", "base_csv",
            "eq", "knee_combiner", "best_loss", "combined_gm",
            "ids_rmse", "gm1_rmse", "gm2_rmse", "gm3_rmse", "use_gm",
            "gm1_weight", "gm2_weight", "gm3_weight", "gm_surgery_mode",
            "output_activation", "config_name", "lr",
            "freeze_physics", "use_opt_params", "knee_alpha_scale", "knee_vgs_thr",
            "knee_vgs_tau", "knee_max_correction", "ids_constraint", "ids_target",
            "ids_lambda", "gm_warmup_epochs", "gm_warmup_lr", "gm_vds_min", "gm_vgs_min",
            "ids_region_weight", "ids_region_center", "ids_region_width", "ids_region_lo", "ids_region_hi",
            "deterministic", "mixed_init", "vds_loss",
            "gm_max_ratio", "lbfgs_epochs", "lbfgs_max_iter", "lbfgs_gm_aware", "loss_norm",
            "adamw_avoid_localmin", "csv", "epochs", "seed"]
    # Region metrics (compute_region_metrics.py) vary by run; collect the union and
    # slot them in before architecture/file_path so they rank/sort like any column.
    region_cols = sorted({k for r in results for k in r if k.startswith("region_")})
    keys += region_cols + ["architecture", "file_path"]
    # SLSQP rows lack the extra config fields; let DictWriter leave them blank.
    writer_extras = "ignore"
    
    output_path = os.path.join(target_dir, output_csv)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction=writer_extras)
        writer.writeheader()
        writer.writerows(results)
        
    print(f"Compiled {len(results)} results into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile JSON results to CSV")
    parser.add_argument("--dir", type=str, default="fast_pure_nn_tests_softplus",
                        help="Target directory to parse")
    parser.add_argument("--root", type=str, default=None,
                        help="Master root path containing multiple experiment subdirs. "
                             "If set, a CSV is generated for each immediate subdirectory.")
    parser.add_argument("--sort_by", type=str, default="best_loss",
                        help="Comma-separated list of columns to sort by (ascending, "
                             "lowest first). Valid keys: best_loss, ids_rmse, gm1_rmse, "
                             "gm2_rmse, gm3_rmse, lr. Missing/non-numeric values are "
                             "pushed to the bottom. Default: 'best_loss'. "
                             "Example: --sort_by ids_rmse,gm1_rmse")
    parser.add_argument("--base_csv", default=None,
                        help="Base ranked CSV this sweep was generated from (Option B): fills "
                             "base_exp/base_id/base_csv by matching arch_hash. Not needed when the "
                             "runs carry a _provenance.json (from a gen_config_from_rows config).")
    parser.add_argument("--base_exp", default=None,
                        help="Override the base experiment name (default: parsed from --base_csv path).")
    args = parser.parse_args()

    sort_keys = [k.strip() for k in args.sort_by.split(",") if k.strip()]

    if not args.base_csv:   # auto-detect the base ranked CSV from the results folder
        _root = args.root or args.dir
        _meta = run_artifacts.run_meta(_root) if _root else {}
        _bc = _meta.get("base_csv")
        if _bc and os.path.isfile(_bc):
            args.base_csv = _bc
            # base CSV lives in base_files/, so take the true base_exp from the manifest.
            args.base_exp = args.base_exp or _meta.get("base_exp")
            print(f"[auto] base CSV: {_bc}"
                  + (f"  (base_exp={args.base_exp})" if args.base_exp else ""))

    if args.root:
        if not os.path.isdir(args.root):
            raise SystemExit(f"Root path does not exist: {args.root}")
        # Skip non-experiment output dirs: plotted_configs, base_files (the stashed input
        # config/CSV copies), and compile_ranked's grouped CSV folders (ranked_global /
        # ranked_region_*) -- none hold run JSONs, and base_files holds config JSONs that
        # would otherwise be mis-ingested as runs.
        _skip = lambda d: d in ("plotted_configs", "base_files") or d.startswith("ranked_")
        subdirs = [os.path.join(args.root, d) for d in os.listdir(args.root)
                   if os.path.isdir(os.path.join(args.root, d)) and not _skip(d)]
        if not subdirs:
            print(f"No subdirectories found in {args.root}")
        for sub in subdirs:
            print(f"\n--- Compiling: {sub} ---")
            exp_name = os.path.basename(sub)  # ← Extract experiment name
            output_csv = f"compiled_results_{exp_name}.csv"  # ← Append to filename
            compile_results(sub, output_csv=output_csv, sort_keys=sort_keys,
                            base_csv=args.base_csv, base_exp=args.base_exp)
    else:
        compile_results(args.dir, sort_keys=sort_keys,
                        base_csv=args.base_csv, base_exp=args.base_exp)
