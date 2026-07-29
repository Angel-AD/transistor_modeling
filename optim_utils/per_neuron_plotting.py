# ============================================================================
# per_neuron_plotting.py -- Plotting and evaluation functions
# Extracted from per_neuron_trainer_8.py to reduce file size.
# ============================================================================

import os
import ast
import json
import pickle
import shutil
import time
import types as _types
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg') # Strictly disable GUI to prevent thread hang on Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import optuna

from per_neuron_solver_spaces import optimizer_spaces_second_run_params
from per_neuron_models import check_physics_boundaries

PLOT_FORMAT = os.environ.get('PLOT_FORMAT', 'jpg').lower().lstrip('.')
PLOT_DPI = int(os.environ.get('PLOT_DPI', '150'))


def _ensure_trainer_deps():
    global _trainer_deps_loaded
    if globals().get('_trainer_deps_loaded'):
        return
    import new_models.per_neuron_optim.per_neuron_trainer_8 as _m
    global meas_load, get_best_n_trials, reconstruct_model_from_trial, \
           evaluate_physics, generate_equation_from_model, handle_equation_comparison, \
           generate_parameter_markdown_table, check_optuna_boundaries, \
           train_and_evaluate_model, update_execution_metrics, save_best_trials_weights, \
           get_space_config_from_results_json, load_trial_weights, \
           target_vds_list, target_vgs_list
    meas_load = _m.meas_load
    get_best_n_trials = _m.get_best_n_trials
    reconstruct_model_from_trial = _m.reconstruct_model_from_trial
    evaluate_physics = _m.evaluate_physics
    generate_equation_from_model = _m.generate_equation_from_model
    handle_equation_comparison = _m.handle_equation_comparison
    generate_parameter_markdown_table = _m.generate_parameter_markdown_table
    check_optuna_boundaries = _m.check_optuna_boundaries
    train_and_evaluate_model = _m.train_and_evaluate_model
    update_execution_metrics = _m.update_execution_metrics
    save_best_trials_weights = _m.save_best_trials_weights
    get_space_config_from_results_json = _m.get_space_config_from_results_json
    load_trial_weights = _m.load_trial_weights
    target_vds_list = _m.target_vds_list
    target_vgs_list = _m.target_vgs_list
    globals()['_trainer_deps_loaded'] = True



def load_evaluate_and_plot_weights(base_json_path, base_env, base_config_no, space_config_no, 
                                   env_to_train_info=None, n_best=3, save_plot=True, 
                                   save_equations=True, device=None, tol_vds=.1, tol_vgs=.03 
                                   ):
    """
    Load the best N predictions from Optuna results, apply saved weights (both base 
    and optimized), generate equations, and plot predictions without retraining.
    """
    _ensure_trainer_deps()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if isinstance(base_config_no, int):
        results_json_path = os.path.join(base_json_path, f"{base_env}_results", f"results_base{base_config_no}_opt{space_config_no}.json")
    else:
        print(f"Using passed results json {base_config_no}...")
        results_json_path = base_config_no
    
    with open(results_json_path, 'r') as f:
        loaded_data = json.load(f)
        results_data = loaded_data.get("results", loaded_data) if isinstance(loaded_data, dict) else loaded_data
    
    if not results_data:
        print("No results found in the JSON file.")
        return

    first_result = results_data[0]
    db_path = first_result['tpe_db_path']
    csv_path = first_result['full_csv_path']
    meas_load_kwargs = (env_to_train_info or {}).get('meas_load_kwargs') or first_result['meas_load_kwargs']
    base_env = first_result.get('environment', (env_to_train_info or {}).get('environment', base_env))
    config_id = first_result['config_id']
    optimizer_space_id = first_result['optimizer_space_id']
    user_metadata = first_result.get('user_metadata', {})
    current2train = user_metadata.get('current2train', 'Ids')  
    train_data_mode = user_metadata.get('train_data_mode', 'Vgs_meas')  
    results_dir = os.path.dirname(results_json_path)
    
    study = optuna.load_study(storage=f"sqlite:///{db_path}", study_name=f"study_env_{base_env}_base{config_id}_opt{optimizer_space_id}")
    best_trials = get_best_n_trials(study, n_best)
    
    if env_to_train_info is None or env_to_train_info.get("csv_path") is None:
        print("Loading environment data...")
        T_train, T_test = meas_load(csv_path, **meas_load_kwargs)

        if meas_load_kwargs["test_percent"] == 0:
            T_test = T_train
        
        X_val = np.column_stack((T_test[train_data_mode], T_test['Vds']))
        Y_val = np.array(T_test[current2train]).reshape(-1, 1)
        Y_val_flat = Y_val.flatten() if Y_val.ndim > 1 else Y_val
        XX = torch.as_tensor(X_val, dtype=torch.float64, device=device)

        results_lookup = {entry.get('trial_number'): entry for entry in results_data if entry.get('trial_number') is not None}
        # --- 1. GENERATE RANK MAPS ---
        valid_base = sorted([r for r in results_data if r.get('rmse_ids') is not None], key=lambda x: x['rmse_ids'])
        valid_opt = sorted([r for r in results_data if r.get('new_train_rmse_ids') is not None], key=lambda x: x['new_train_rmse_ids'])
        base_rank_map = {r.get('trial_number'): rank + 1 for rank, r in enumerate(valid_base)}
        opt_rank_map = {r.get('trial_number'): rank + 1 for rank, r in enumerate(valid_opt)}

        # --- 2. ITERATE OVER TRIALS ---
        for idx, trial in enumerate(best_trials):
            matched_result = results_lookup.get(trial.number)
            if matched_result is None:
                continue

            base_weights_filepath = matched_result.get("weights_filepath")
            optimized_weights_filepath = matched_result.get("optimized_weights_filepath")
            
            paths_to_evaluate = [
                ("Base Weights", base_weights_filepath),
                ("Optimized Weights", optimized_weights_filepath)
            ]

            print(f"\n[DEBUG] =========================================")
            print(f"[DEBUG] Checking JSON paths for Trial {trial.number}")
            print(f"[DEBUG] Base JSON path: {base_weights_filepath}")
            print(f"[DEBUG] Opt  JSON path: {optimized_weights_filepath}")
            print(f"[DEBUG] Are paths exactly identical? {base_weights_filepath == optimized_weights_filepath}")
            print(f"[DEBUG] =========================================\n")

            # --- 3. ITERATE OVER WEIGHT TYPES ---
            for weight_type, filepath in paths_to_evaluate:
                if not filepath or not os.path.exists(filepath):
                    print(f"Skipping {weight_type} for Trial {trial.number}: File not found ({filepath})")
                    continue

                print(f"\n--- Processing Trial {trial.number} | {weight_type} ---")
                
                # Fetch True Rank
                is_optimized = "Optimized" in weight_type
                current_rank = opt_rank_map.get(trial.number) if is_optimized else base_rank_map.get(trial.number)
                
                if current_rank is None:
                    print(f"WARNING: Could not find rank for Trial {trial.number} [{weight_type}]. Skipping.")
                    continue
                
                # Reconstruct model
                model = reconstruct_model_from_trial(trial)
                model.double()
                model.to(device)

                # Load Specific Weights into Memory
                try:
                    with open(filepath, 'rb') as f:
                        saved_data = pickle.load(f)
                    
                    trials_list = saved_data.get('trials', []) if isinstance(saved_data, dict) else saved_data
                    found_data = next((t_data for t_data in trials_list if t_data.get('trial_number') == trial.number), None)
                    
                    if found_data:
                        state_dict = found_data["model_state_dict"]
                        new_state_dict = {}
                        is_target_pinn = hasattr(model, 'base_nn')
                        for k, v in state_dict.items():
                            if k.startswith("net.") and is_target_pinn:
                                new_key = f"base_nn.{k}"
                            elif k.startswith("base_nn.") and not is_target_pinn:
                                new_key = k.replace("base_nn.", "", 1)
                            else:
                                new_key = k
                            new_state_dict[new_key] = v

                        model.load_state_dict(new_state_dict, strict=False)
                        
                        # [DEBUG] Calculate total L2 norm to mathematically verify if weights are different!
                        total_norm = sum(p.norm().item() for p in model.parameters())
                        print(f"[DEBUG] Successfully loaded {weight_type} for Trial {trial.number}.")
                        print(f"[DEBUG] --> MODEL L2 NORM: {total_norm:.8f} <-- (If base and opt are identical, your weights are exactly the same)")

                    else:
                        print(f"WARNING: Trial {trial.number} not found in {os.path.basename(filepath)}.")
                        continue
                except Exception as e:
                    print(f"Error reading {weight_type} file: {e}")
                    continue

                # Ensure model is in eval mode before passing to plotters
                model.eval()

                # Evaluate and Plot
                try:
                    evaluate_and_plot_single_pass(
                        model=model, 
                        T_val=T_train, # Ensure this matches where JSON metrics were measured!
                        trial_data=matched_result, 
                        weight_type=weight_type.capitalize(),
                        current_rank=current_rank,
                        vgs_col=train_data_mode,
                        vds_col='Vds',
                        ids_col=current2train,
                        target_vds_list=target_vds_list, 
                        target_vgs_list=target_vgs_list, 
                        save_dir=os.path.join(results_dir, "eqs_plots"),
                        config_id=config_id,
                        opt_space_id=optimizer_space_id,
                        tol_vds=tol_vds, tol_vgs=tol_vgs,
                        optuna_params=trial.params
                    )
                    plt.close('all')
                except Exception as e:
                    print(f"Error during plotting/saving for Trial {trial.number}: {e}")

    return


def evaluate_and_plot_single_pass(model, T_val, trial_data, weight_type, current_rank, 
                                  vgs_col, vds_col, ids_col, target_vds_list, target_vgs_list, 
                                  save_dir, config_id, opt_space_id, tol_vds=0.1, tol_vgs=0.03,
                                  optuna_params=None): # <-- Add parameter here
    _ensure_trainer_deps()
    plotting_mode = os.environ.get("PLOTTING_MODE", "grid").lower()
    device = next(model.parameters()).device

    trial_num = trial_data.get('trial_number')
    is_optimized = "Optimized" in weight_type
    prefix = "opt_" if is_optimized else "base_"
    rank_str = f"_rank{current_rank}"
    
    final_save_dir = os.path.join(save_dir, f"b_{config_id}_s_{opt_space_id}")
    os.makedirs(final_save_dir, exist_ok=True)

    # --- A. CALCULATE MATH METRICS ---
    calc_metrics = evaluate_physics(
        model, T_val, vgs_col, vds_col, ids_col, target_vds_list, device, tol_vds
    )
    
    # Grab JSON metrics for plotting comparisons
    if is_optimized:
        saved_metrics = {
            'new_ids': trial_data.get('new_train_rmse_ids'),
            'new_gm1': trial_data.get('new_train_rmse_gm1'),
            'new_gm2': trial_data.get('new_train_rmse_gm2'),
            'new_gm3': trial_data.get('new_train_rmse_gm3'),
            'base_ids': trial_data.get('rmse_ids'),
            'base_gm1': trial_data.get('rmse_gm1'),
            'base_gm2': trial_data.get('rmse_gm2'),
            'base_gm3': trial_data.get('rmse_gm3'),
        }
    else:
        saved_metrics = {
            'base_ids': trial_data.get('rmse_ids'),
            'base_gm1': trial_data.get('rmse_gm1'),
            'base_gm2': trial_data.get('rmse_gm2'),
            'base_gm3': trial_data.get('rmse_gm3'),
        }


    # print(f"\n[DEBUG] --- METRICS CHECK FOR {weight_type.upper()} ---")
    # print(f"[DEBUG] JSON Expected Ids RMSE:   {saved_metrics['ids']}")
    # print(f"[DEBUG] Evaluator Calculated Ids: {calc_metrics.get('rmse_ids')}")
    # print(f"[DEBUG] Match? {'YES' if str(saved_metrics['ids'])[:8] == str(calc_metrics.get('rmse_ids'))[:8] else 'NO'}\n")

    # --- B. EQUATION GENERATION ---
    eq_metrics = {}
    try:
        equation = generate_equation_from_model(model, trial_data)
        
        X_val_0 = T_val[vgs_col].values.flatten()
        X_val_1 = T_val[vds_col].values.flatten()
        Y_val_flat = T_val[ids_col].values.flatten()
        
        vgs_t = torch.tensor(X_val_0.reshape(-1, 1), dtype=torch.float64, device=device)
        vds_t = torch.tensor(X_val_1.reshape(-1, 1), dtype=torch.float64, device=device)
        with torch.no_grad():
            predictions = model(torch.cat([vgs_t, vds_t], dim=1)).cpu().numpy().flatten()
        
        eq_preds, eq_residuals, eq_r2, eq_rmse = handle_equation_comparison(
            X_val_0, X_val_1, Y_val_flat, equation, title=None
        )
        
        eq_vs_model_diff = eq_preds - predictions
        eq_vs_model_rmse = float(np.sqrt(np.mean(eq_vs_model_diff**2)))
        eq_vs_model_mae = float(np.mean(np.abs(eq_vs_model_diff)))
        eq_metrics = {'mae': eq_vs_model_mae, 'rmse': eq_vs_model_rmse}

        # Save Text File
        # eq_filename = f"{prefix}neural_network_equations_trial{trial_num}{rank_str}.txt"
        # with open(os.path.join(final_save_dir, eq_filename), 'w') as f:
        #     f.write(f"[{weight_type}] Trial {trial_num} (Rank {current_rank})\n")
        #     f.write(f"Eq vs NN MAE: {eq_vs_model_mae:.4e}\n")
        #     f.write(f"Eq vs NN RMSE: {eq_vs_model_rmse:.4e}\n")
        #     f.write(f"Eq vs Data RMSE: {eq_rmse:.4e}\n")
        #     f.write(f"Eq vs Data R2: {eq_r2:.4f}\n\n")
        #     f.write(equation)
        eq_filename = f"{prefix}model_details_trial{trial_num}{rank_str}.md"
        
        # 2. Generate the parameter table string
        param_table = generate_parameter_markdown_table(model, saved_metrics, optuna_params, True, is_optimized)

        # 3. Write out formatted Markdown
        with open(os.path.join(final_save_dir, eq_filename), 'w') as f:
            f.write(f"# Model Details: Trial {trial_num} (Rank {current_rank})\n")
            f.write(f"**Weight Type:** {weight_type}\n\n")
            
            f.write("## 1. Performance Metrics\n")
            f.write(f"- **Eq vs NN MAE:** `{eq_vs_model_mae:.4e}`\n")
            f.write(f"- **Eq vs NN RMSE:** `{eq_vs_model_rmse:.4e}`\n")
            f.write(f"- **Eq vs Data RMSE:** `{eq_rmse:.4e}`\n")
            f.write(f"- **Eq vs Data R2:** `{eq_r2:.4f}`\n\n")
            
            f.write("## 2. Optimized Physics Parameters & Bounds\n")
            f.write(param_table + "\n\n")
            
            f.write("## 3. Symbolic Equation\n")
            f.write("```text\n") # Code block formatting for the equation
            f.write(equation)
            f.write("\n```\n")

    except Exception as e:
        print(f"Skipping equation generation for Trial {trial_num} [{weight_type}]: {e}")

    # --- C. EXTRACT PLOT DATA ---
    plot_data = generate_physics_plot_data(
        model, T_val, vgs_col, vds_col, ids_col, target_vds_list, target_vgs_list, device, tol_vds, tol_vgs
    )

    # --- D. ROUTE TO SELECTED PLOTTER ---
    if plotting_mode == "grid":
        plot_grid(
            plot_data, calc_metrics, saved_metrics, eq_metrics,
            save_path=os.path.join(final_save_dir, f"{prefix}physics_curves_trial{trial_num}{rank_str}.{PLOT_FORMAT}"),
            title=f"Trial {trial_num} [{weight_type}] - Rank {current_rank}",
            target_vds_list=target_vds_list, target_vgs_list=target_vgs_list
        )
    elif plotting_mode == "separate":
        plot_separate(
            plot_data, eq_metrics,
            save_dir=final_save_dir, 
            base_name=f"{prefix}trial{trial_num}{rank_str}",
            title_prefix=f"Trial {trial_num} [{weight_type}] (Rank {current_rank})",
            target_vds_list=target_vds_list, target_vgs_list=target_vgs_list
        )


import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import optuna


def evaluate_and_collect_single_pass(
    model, T_val, trial_data, weight_type, current_rank, 
    vgs_col, vds_col, ids_col, target_vds_list, target_vgs_list, 
    save_dir, config_id, opt_space_id, tol_vds=0.1, tol_vgs=0.03, optuna_params=None
): 
    _ensure_trainer_deps()
    device = next(model.parameters()).device
    trial_num = trial_data.get('trial_number')
    is_optimized = "Optimized" in weight_type
    prefix = "opt_" if is_optimized else "base_"
    rank_str = f"_rank{current_rank}"
    
    final_save_dir = os.path.join(save_dir, f"b_{config_id}_s_{opt_space_id}")
    os.makedirs(final_save_dir, exist_ok=True)

    # --- A. CALCULATE MATH METRICS ---
    calc_metrics = evaluate_physics(
        model, T_val, vgs_col, vds_col, ids_col, target_vds_list, device, tol_vds
    )
    
    if is_optimized:
        saved_metrics = {
            'new_ids': trial_data.get('new_train_rmse_ids'),
            'new_gm1': trial_data.get('new_train_rmse_gm1'),
            'new_gm2': trial_data.get('new_train_rmse_gm2'),
            'new_gm3': trial_data.get('new_train_rmse_gm3'),
            'base_ids': trial_data.get('rmse_ids'),
        }
    else:
        saved_metrics = {
            'base_ids': trial_data.get('rmse_ids'),
            'base_gm1': trial_data.get('rmse_gm1'),
            'base_gm2': trial_data.get('rmse_gm2'),
            'base_gm3': trial_data.get('rmse_gm3'),
        }

    # --- B. EQUATION GENERATION ---
    try:
        equation = generate_equation_from_model(model, trial_data)
        X_val_0 = T_val[vgs_col].values.flatten()
        X_val_1 = T_val[vds_col].values.flatten()
        Y_val_flat = T_val[ids_col].values.flatten()
        
        vgs_t = torch.tensor(X_val_0.reshape(-1, 1), dtype=torch.float64, device=device)
        vds_t = torch.tensor(X_val_1.reshape(-1, 1), dtype=torch.float64, device=device)
        with torch.no_grad():
            predictions = model(torch.cat([vgs_t, vds_t], dim=1)).cpu().numpy().flatten()
        
        eq_preds, eq_residuals, eq_r2, eq_rmse = handle_equation_comparison(
            X_val_0, X_val_1, Y_val_flat, equation, title=None
        )
        
        eq_vs_model_diff = eq_preds - predictions
        eq_vs_model_rmse = float(np.sqrt(np.mean(eq_vs_model_diff**2)))
        eq_vs_model_mae = float(np.mean(np.abs(eq_vs_model_diff)))

        eq_filename = f"{prefix}model_details_trial{trial_num}{rank_str}.md"
        param_table = generate_parameter_markdown_table(model, saved_metrics, optuna_params, True, is_optimized)

        with open(os.path.join(final_save_dir, eq_filename), 'w') as f:
            f.write(f"# Model Details: Trial {trial_num} (Rank {current_rank})\n")
            f.write(f"**Weight Type:** {weight_type}\n\n")
            f.write("## 1. Performance Metrics\n")
            f.write(f"- **Eq vs NN MAE:** `{eq_vs_model_mae:.4e}`\n")
            f.write(f"- **Eq vs NN RMSE:** `{eq_vs_model_rmse:.4e}`\n")
            f.write(f"- **Eq vs Data RMSE:** `{eq_rmse:.4e}`\n")
            f.write(f"- **Eq vs Data R2:** `{eq_r2:.4f}`\n\n")
            f.write("## 2. Optimized Physics Parameters & Bounds\n")
            f.write(param_table + "\n\n")
            f.write("## 3. Symbolic Equation\n")
            f.write("```text\n")
            f.write(equation)
            f.write("\n```\n")

    except Exception as e:
        print(f"Skipping equation generation for Trial {trial_num} [{weight_type}]: {e}")

    # --- C. EXTRACT PLOT DATA (RETURN IT INSTEAD OF PLOTTING) ---
    plot_data = generate_physics_plot_data(
        model, T_val, vgs_col, vds_col, ids_col, target_vds_list, target_vgs_list, device, tol_vds, tol_vgs
    )

    # Return the pure data so the main loop can aggregate it
    return plot_data


import os
import matplotlib.pyplot as plt

def plot_aggregated_models(rank, models_list, save_path, target_vds_list, target_vgs_list):
    """
    Splits the models into Base and Optimized, then creates separate figures.
    Colors represent target voltages, line styles represent different model configs.
    """
    if not models_list:
        return

    # 1. Separate Base and Optimized models based on their labels
    base_models = [m for m in models_list if "Base" in m['label']]
    opt_models = [m for m in models_list if "Opt" in m['label']] # Catches 'Opt' or 'Optimized'

    # 2. Prepare the save paths
    base_dir, file_name = os.path.split(save_path)
    name, ext = os.path.splitext(file_name)
    
    # 3. Render figures if the lists are not empty
    if base_models:
        save_base = os.path.join(base_dir, f"{name}_base{ext}")
        _render_figure(rank, base_models, "Base Weights", save_base)
        
    if opt_models:
        save_opt = os.path.join(base_dir, f"{name}_opt{ext}")
        _render_figure(rank, opt_models, "Optimized Weights", save_opt)


def _render_figure(rank, models_subset, title_suffix, save_path):
    """Helper function to do the actual plotting."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Standard matplotlib distinct colors 
    colors = plt.cm.tab10.colors 
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', '<', '>']
    
    # Line styles: Solid (1st), Dashed (2nd), Dotted (3rd), Dash-dot (4th)
    linestyles = ['-', '--', ':', '-.', (0, (5, 1)), (0, (3, 1, 1, 1))]
    
    base_data = models_subset[0]['plot_data']
    
    # --- PLOT 1-4: VDS SWEEPS (Ids, Gm1, Gm2, Gm3 vs Vgs) ---
    for i, sweep in enumerate(base_data.get('vds_sweeps', [])):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        t_vds = sweep['target_vds']
        
        # Plot Measured Data (Colored + distinct marker)
        axes[0].plot(sweep['true_vgs'], sweep['true_ids'], color=c, marker=m, linestyle='', markersize=5, label=f"Data Vds={t_vds}V")
        axes[1].plot(sweep['true_vgs'], sweep['true_gm1'], color=c, marker=m, linestyle='', markersize=5)
        axes[2].plot(sweep['true_vgs'], sweep['true_gm2'], color=c, marker=m, linestyle='', markersize=5)
        axes[3].plot(sweep['true_vgs'], sweep['true_gm3'], color=c, marker=m, linestyle='', markersize=5)

    for model_idx, model_info in enumerate(models_subset):
        label = model_info['label']
        ls = linestyles[model_idx % len(linestyles)]  # Model config determines line style
        plot_data = model_info['plot_data']
        
        for i, sweep in enumerate(plot_data.get('vds_sweeps', [])):
            c = colors[i % len(colors)]               # Target sweep determines color
            line_label = label if i == 0 else ""      # Only add to legend once
            
            axes[0].plot(sweep['pred_vgs'], sweep['pred_ids'], color=c, linestyle=ls, label=line_label)
            axes[1].plot(sweep['pred_vgs'], sweep['pred_gm1'], color=c, linestyle=ls)
            axes[2].plot(sweep['pred_vgs'], sweep['pred_gm2'], color=c, linestyle=ls)
            axes[3].plot(sweep['pred_vgs'], sweep['pred_gm3'], color=c, linestyle=ls)

    # --- PLOT 5: VGS SWEEPS (Ids vs Vds) ---
    for i, sweep in enumerate(base_data.get('vgs_sweeps', [])):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]
        t_vgs = sweep['target_vgs']
        axes[4].plot(sweep['true_vds'], sweep['true_ids'], color=c, marker=m, linestyle='', markersize=5, label=f"Data Vgs={t_vgs}V")

    for model_idx, model_info in enumerate(models_subset):
        label = model_info['label']
        ls = linestyles[model_idx % len(linestyles)]
        plot_data = model_info['plot_data']
        
        for i, sweep in enumerate(plot_data.get('vgs_sweeps', [])):
            c = colors[i % len(colors)]
            line_label = label if i == 0 else ""
            axes[4].plot(sweep['pred_vds'], sweep['pred_ids'], color=c, linestyle=ls, label=line_label)

    # --- FORMATTING ---
    titles = ["Ids vs Vgs", "Gm1 vs Vgs", "Gm2 vs Vgs", "Gm3 vs Vgs", "Ids vs Vds"]
    y_labels = ["Ids (A)", "Gm1 (S)", "Gm2 (S/V)", "Gm3 (S/V^2)", "Ids (A)"]
    x_labels = ["Vgs (V)", "Vgs (V)", "Vgs (V)", "Vgs (V)", "Vds (V)"]

    for i in range(5):
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(y_labels[i])
        axes[i].set_xlabel(x_labels[i])
        axes[i].grid(True, linestyle='--', alpha=0.6)

    # Add legends
    axes[0].legend(loc='best', fontsize=8)
    axes[4].legend(loc='best', fontsize=8) 
    
    # Hide the unused 6th subplot
    axes[5].set_visible(False)

    plt.suptitle(f"Rank {rank} Model Comparisons - {title_suffix}", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close('all')
    print(f"Saved aggregated plot: {save_path}")



def load_evaluate_and_plot_multi_weights(
    base_json_path_list, base_env, base_config_no_list, space_config_no_list, 
    env_to_train_info=None, n_best=3, save_plot=True, 
    save_equations=True, device=None, tol_vds=.1, tol_vgs=.03,
    target_vds_list=None, target_vgs_list=None, custom_save_dir=None
):
    """
    Load predictions from lists of Optuna results, apply saved weights, 
    and aggregate them by rank to plot on the same figure.
    Accepts either a single string or a list of strings for base_json_path_list.
    """
    _ensure_trainer_deps()
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- HANDLE STRING VS LIST FOR PATHS ---
    if isinstance(base_json_path_list, str):
        # If it's a single string, duplicate it to match the length of the config list
        base_json_path_list = [base_json_path_list] * len(base_config_no_list)
    elif len(base_json_path_list) != len(base_config_no_list):
        print("WARNING: The length of base_json_path_list does not match base_config_no_list. The loop will stop at the shortest list.")

    # Structure: { rank_id: {'measured_data': None, 'models': []} }
    rank_grouped_data = {i: {'measured_data': None, 'models': []} for i in range(1, n_best + 1)}
    if custom_save_dir is not None:
        common_save_dir = custom_save_dir
        os.makedirs(common_save_dir, exist_ok=True)
    else:
        common_save_dir = None

    # --- 1. ITERATE OVER CONFIG LISTS ---
    # Because of the check above, this zip() works flawlessly for both strings and lists
    for json_path, config_no, space_no in zip(base_json_path_list, base_config_no_list, space_config_no_list):
        
        if isinstance(config_no, int):
            results_json_path = os.path.join(json_path, f"{base_env}_results", f"results_base{config_no}_opt{space_no}.json")
        else:
            print(f"Using passed results json {config_no}...")
            results_json_path = config_no
        
        with open(results_json_path, 'r') as f:
            loaded_data = json.load(f)
            results_data = loaded_data.get("results", loaded_data) if isinstance(loaded_data, dict) else loaded_data
        
        if not results_data:
            print(f"No results found in {results_json_path}. Skipping.")
            continue

        first_result = results_data[0]
        db_path = first_result['tpe_db_path']
        csv_path = first_result['full_csv_path']
        meas_load_kwargs = (env_to_train_info or {}).get('meas_load_kwargs') or first_result['meas_load_kwargs']
        env_val = first_result.get('environment', (env_to_train_info or {}).get('environment', base_env))
        config_id = first_result['config_id']
        optimizer_space_id = first_result['optimizer_space_id']
        
        user_metadata = first_result.get('user_metadata', {})
        current2train = user_metadata.get('current2train', 'Ids')  
        train_data_mode = user_metadata.get('train_data_mode', 'Vgs_meas')  
        
        if common_save_dir is None:
            common_save_dir = os.path.join(os.path.dirname(results_json_path), "eqs_plots_multi")
            os.makedirs(common_save_dir, exist_ok=True)
        
        study = optuna.load_study(storage=f"sqlite:///{db_path}", study_name=f"study_env_{env_val}_base{config_id}_opt{optimizer_space_id}")
        best_trials = get_best_n_trials(study, n_best)
        
        print(f"\nLoading environment data for Config Base {config_id} | Opt {optimizer_space_id}...")
        T_train, T_test = meas_load(csv_path, **meas_load_kwargs)
        if meas_load_kwargs["test_percent"] == 0:
            T_test = T_train

        results_lookup = {entry.get('trial_number'): entry for entry in results_data if entry.get('trial_number') is not None}
        
        valid_base = sorted([r for r in results_data if r.get('rmse_ids') is not None], key=lambda x: x['rmse_ids'])
        valid_opt = sorted([r for r in results_data if r.get('new_train_rmse_ids') is not None], key=lambda x: x['new_train_rmse_ids'])
        base_rank_map = {r.get('trial_number'): rank + 1 for rank, r in enumerate(valid_base)}
        opt_rank_map = {r.get('trial_number'): rank + 1 for rank, r in enumerate(valid_opt)}

        # --- 2. ITERATE OVER TRIALS ---
        for idx, trial in enumerate(best_trials):
            matched_result = results_lookup.get(trial.number)
            if matched_result is None:
                continue

            paths_to_evaluate = [
                ("Base Weights", matched_result.get("weights_filepath")),
                ("Optimized Weights", matched_result.get("optimized_weights_filepath"))
            ]

            # --- 3. ITERATE OVER WEIGHT TYPES ---
            for weight_type, filepath in paths_to_evaluate:
                if not filepath or not os.path.exists(filepath):
                    continue
                
                is_optimized = "Optimized" in weight_type
                current_rank = opt_rank_map.get(trial.number) if is_optimized else base_rank_map.get(trial.number)
                
                if current_rank is None or current_rank > n_best:
                    continue
                
                # Reconstruct and load model
                model = reconstruct_model_from_trial(trial)
                model.double().to(device)

                try:
                    with open(filepath, 'rb') as f:
                        saved_data = pickle.load(f)
                    trials_list = saved_data.get('trials', []) if isinstance(saved_data, dict) else saved_data
                    found_data = next((t_data for t_data in trials_list if t_data.get('trial_number') == trial.number), None)
                    
                    if found_data:
                        state_dict = found_data["model_state_dict"]
                        new_state_dict = {}
                        is_target_pinn = hasattr(model, 'base_nn')
                        for k, v in state_dict.items():
                            if k.startswith("net.") and is_target_pinn:
                                new_key = f"base_nn.{k}"
                            elif k.startswith("base_nn.") and not is_target_pinn:
                                new_key = k.replace("base_nn.", "", 1)
                            else:
                                new_key = k
                            new_state_dict[new_key] = v
                        model.load_state_dict(new_state_dict, strict=False)
                    else:
                        continue
                except Exception as e:
                    print(f"Error reading {weight_type} file: {e}")
                    continue

                model.eval()

                # Evaluate and collect data (DO NOT PLOT YET)
                # Evaluate and collect data
                try:
                    plot_data = evaluate_and_collect_single_pass(
                        model=model, T_val=T_train, trial_data=matched_result, 
                        weight_type=weight_type.capitalize(), current_rank=current_rank,
                        vgs_col=train_data_mode, vds_col='Vds', ids_col=current2train,
                        target_vds_list=target_vds_list, target_vgs_list=target_vgs_list, 
                        save_dir=common_save_dir, config_id=config_id,
                        opt_space_id=optimizer_space_id, tol_vds=tol_vds, tol_vgs=tol_vgs,
                        optuna_params=trial.params
                    )
                    
                    # Just store the whole dictionary! The plotter will handle the rest.
                    label = f"B{config_id}-S{optimizer_space_id} {weight_type}"
                    rank_grouped_data[current_rank]['models'].append({
                        'label': label,
                        'plot_data': plot_data 
                    })

                except Exception as e:
                    print(f"Error during data extraction for Trial {trial.number}: {e}")

    # --- 4. FINALLY: PLOT THE AGGREGATED DATA BY RANK ---
    print("\n--- Generating Aggregated Plots by Rank ---")
    for rank, data in rank_grouped_data.items():
        if not data['models']:
            continue
        
        save_path = os.path.join(common_save_dir, f"aggregated_physics_curves_rank{rank}.{PLOT_FORMAT}")
        plot_aggregated_models(rank, data['models'], save_path, target_vds_list, target_vgs_list)

    return


import numpy as np


def evaluate_and_plot_all_physics(model, T_val, results_json_path, target_vds_list, target_vgs_list, save_dir, weight_type, n_best=3, tol_vds=0.1, tol_vgs=0.03):
    """
    Evaluates and plots physics parameters for a specific model state, 
    using Optuna's best trials.
    
    Args:
        weight_type (str): E.g., "Base" or "Optimized".
    """
    _ensure_trainer_deps()
    # --- 1. PARSE JSON & SETUP ---
    with open(results_json_path, 'r') as f:
        data = json.load(f)
        results_data = data.get("results", data) if isinstance(data, dict) else data

    if not results_data: 
        print("No results found in the JSON file.")
        return

    first_result = results_data[0]
    config_id = first_result.get('config_id', 'unknown')
    opt_space_id = first_result.get('optimizer_space_id', 'unknown')
    vgs_col = first_result.get('user_metadata', {}).get('train_data_mode', 'Vgs_meas')
    ids_col = first_result.get('user_metadata', {}).get('current2train', 'Ids')
    vds_col = 'Vds'

    # --- 2. LOAD OPTUNA STUDY ---
    db_path = first_result.get('tpe_db_path')
    base_env = first_result.get('environment', 'unknown')
    
    study_name = f"study_env_{base_env}_base{config_id}_opt{opt_space_id}"
    try:
        study = optuna.load_study(storage=f"sqlite:///{db_path}", study_name=study_name)
        best_trials = get_best_n_trials(study, n_best)
    except Exception as e:
        print(f"Error loading Optuna study: {e}")
        return

    # --- 3. GENERATE MAPS & LOOKUPS ---
    valid_base = sorted([r for r in results_data if r.get('rmse_ids') is not None], key=lambda x: x['rmse_ids'])
    valid_opt = sorted([r for r in results_data if r.get('new_train_rmse_ids') is not None], key=lambda x: x['new_train_rmse_ids'])
    
    base_rank_map = {r.get('trial_number'): rank + 1 for rank, r in enumerate(valid_base)}
    opt_rank_map = {r.get('trial_number'): rank + 1 for rank, r in enumerate(valid_opt)}
    
    # Lookup dictionary to match Optuna trial number back to JSON metrics
    results_lookup = {entry.get('trial_number'): entry for entry in results_data if entry.get('trial_number') is not None}

    is_optimized = "optimized" in weight_type.lower()

    # --- 4. ITERATE OVER OPTUNA TRIALS ---
    for trial in best_trials:
        trial_num = trial.number
        
        # Match Optuna trial with JSON data
        matched_result = results_lookup.get(trial_num)
        if matched_result is None:
            continue
        
        # Skip if optimized weights were requested but don't exist for this trial
        if is_optimized and trial_num not in opt_rank_map: 
            continue
        
        current_rank = opt_rank_map[trial_num] if is_optimized else base_rank_map[trial_num]
        
        # Route to the single pass worker using the matched JSON data
        try:
            evaluate_and_plot_single_pass(
                model=model,
                T_val=T_val,
                trial_data=matched_result, 
                weight_type=weight_type.capitalize(),
                current_rank=current_rank,
                vgs_col=vgs_col,
                vds_col=vds_col,
                ids_col=ids_col,
                target_vds_list=target_vds_list,
                target_vgs_list=target_vgs_list,
                save_dir=save_dir,
                config_id=config_id,
                opt_space_id=opt_space_id,
                tol_vds=tol_vds,
                tol_vgs=tol_vgs,
                optuna_params=trial.params
            )
        except Exception as e:
            print(f"Error evaluating trial {trial_num} [{weight_type}]: {e}")


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def plot_grid(plot_data, calc_metrics, saved_metrics, eq_metrics, save_path, title,
              target_vds_list, target_vgs_list, use_real_voltages=False, eq_plot_data=None,
              legend_decimals=1):
    """2x3 Grid layout using a unified, high-contrast color palette for both Vds and Vgs.

    legend_decimals: rounding for the Vds/Vgs legend value labels (both "Actual"/"Target"
    variants), e.g. 1 -> "1.1", "2.3". Default 1.

    When eq_plot_data is provided the equation-string curves are overlaid as dashed
    lines on every panel (same colour as the NN curve, dashed style).
    """
    fig, axs = plt.subplots(2, 3, figsize=(32, 18))
    fig.suptitle(title, fontsize=20, fontweight='bold')

    cmap = plt.get_cmap('tab10')
    colors_vds = [cmap(i % 10) for i in range(len(target_vds_list))]
    colors_vgs = [cmap(i % 10) for i in range(len(target_vgs_list))]

    real_vds_map = {}
    real_vgs_map = {}

    # Build fast lookup for eq sweeps
    eq_vds_map = {sw['target_vds']: sw for sw in eq_plot_data.get('vds_sweeps', [])} if eq_plot_data else {}
    eq_vgs_map = {sw['target_vgs']: sw for sw in eq_plot_data.get('vgs_sweeps', [])} if eq_plot_data else {}

    # Ids vs Vgs & Gm
    for i, sweep in enumerate(plot_data['vds_sweeps']):
        nominal_vds = sweep['target_vds']
        c = colors_vds[target_vds_list.index(nominal_vds)]

        real_vds_map[nominal_vds] = sweep.get('actual_vds', nominal_vds)

        axs[0,0].plot(sweep['true_vgs'], sweep['true_ids'], 'o', color=c, alpha=0.5, markersize=8)
        axs[0,0].plot(sweep['pred_vgs'], sweep['pred_ids'], '-', color=c, lw=2.5)

        axs[0,1].plot(sweep['true_vgs'], sweep['true_gm1'], 'o', color=c, alpha=0.5, markersize=8)
        axs[0,1].plot(sweep['pred_vgs'], sweep['pred_gm1'], '-', color=c, lw=2.5)

        axs[0,2].plot(sweep['true_vgs'], sweep['true_gm2'], 'o', color=c, alpha=0.5, markersize=8)
        axs[0,2].plot(sweep['pred_vgs'], sweep['pred_gm2'], '-', color=c, lw=2.5)

        axs[1,2].plot(sweep['true_vgs'], sweep['true_gm3'], 'o', color=c, alpha=0.5, markersize=8)
        axs[1,2].plot(sweep['pred_vgs'], sweep['pred_gm3'], '-', color=c, lw=2.5)

        # Overlay equation string curves (dashed)
        if nominal_vds in eq_vds_map:
            eq_sw = eq_vds_map[nominal_vds]
            axs[0,0].plot(eq_sw['pred_vgs'], eq_sw['pred_ids'],  '--', color=c, lw=1.8, alpha=0.9)
            axs[0,1].plot(eq_sw['pred_vgs'], eq_sw['pred_gm1'], '--', color=c, lw=1.8, alpha=0.9)
            axs[0,2].plot(eq_sw['pred_vgs'], eq_sw['pred_gm2'], '--', color=c, lw=1.8, alpha=0.9)
            axs[1,2].plot(eq_sw['pred_vgs'], eq_sw['pred_gm3'], '--', color=c, lw=1.8, alpha=0.9)

    # Ids vs Vds
    for i, sweep in enumerate(plot_data['vgs_sweeps']):
        nominal_vgs = sweep['target_vgs']
        c = colors_vgs[target_vgs_list.index(nominal_vgs)]

        real_vgs_map[nominal_vgs] = sweep.get('actual_vgs', nominal_vgs)

        axs[1,0].plot(sweep['true_vds'], sweep['true_ids'], 'o', color=c, alpha=0.5, markersize=8)
        axs[1,0].plot(sweep['pred_vds'], sweep['pred_ids'], '-', color=c, lw=2.5)

        if nominal_vgs in eq_vgs_map:
            eq_sw = eq_vgs_map[nominal_vgs]
            axs[1,0].plot(eq_sw['pred_vds'], eq_sw['pred_ids'], '--', color=c, lw=1.8, alpha=0.9)

    # Styling
    for ax, y, x in [(axs[0,0], "Ids", "Vgs"), (axs[0,1], "Gm1", "Vgs"), (axs[0,2], "Gm2", "Vgs"), (axs[1,0], "Ids", "Vds"), (axs[1,2], "Gm3", "Vgs")]:
        ax.set_title(f"{y} vs {x}", fontsize=18, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)

    # Metrics Box
    axs[1,1].axis('off')
    def fmt(val): return f"{val:.3e}" if val is not None else "N/A"
    if "new_ids" in saved_metrics:
        save_text = (
            f"Saved JSON base RMSE:\n"
            f"Ids: {fmt(saved_metrics['base_ids']):>10} | Gm1: {fmt(saved_metrics['base_gm1']):>10}\n"
            f"Gm2: {fmt(saved_metrics['base_gm2']):>10} | Gm3: {fmt(saved_metrics['base_gm3']):>10}\n\n"
            f"Saved JSON optimized RMSE:\n"
            f"Ids: {fmt(saved_metrics['new_ids']):>10} | Gm1: {fmt(saved_metrics['new_gm1']):>10}\n"
            f"Gm2: {fmt(saved_metrics['new_gm2']):>10} | Gm3: {fmt(saved_metrics['new_gm3']):>10}\n\n"
        )
    else:
        save_text = (
            f"Saved JSON base RMSE:\n"
            f"Ids: {fmt(saved_metrics['base_ids']):>10} | Gm1: {fmt(saved_metrics['base_gm1']):>10}\n"
            f"Gm2: {fmt(saved_metrics['base_gm2']):>10} | Gm3: {fmt(saved_metrics['base_gm3']):>10}\n\n"
        )

    val_text = ""
    if calc_metrics.get('val_rmse_ids') is not None:
        _vsrc = calc_metrics.get('val_source', '')
        _vsrc_short = (_vsrc[:20] + '…') if len(_vsrc) > 20 else _vsrc
        val_text = (
            f"Validation RMSE [{_vsrc_short}]:\n"
            f"Ids: {fmt(calc_metrics.get('val_rmse_ids')):>10} | Gm1: {fmt(calc_metrics.get('val_rmse_gm1')):>10}\n"
            f"Gm2: {fmt(calc_metrics.get('val_rmse_gm2')):>10} | Gm3: {fmt(calc_metrics.get('val_rmse_gm3')):>10}\n\n"
        )

    text = (
        f"--- Model Performance ---\n\n"
        f"Calculated RMSE:\n"
        f"Ids: {fmt(calc_metrics.get('rmse_ids')):>10} | Gm1: {fmt(calc_metrics.get('rmse_gm1')):>10}\n"
        f"Gm2: {fmt(calc_metrics.get('rmse_gm2')):>10} | Gm3: {fmt(calc_metrics.get('rmse_gm3')):>10}\n\n"
        f"{save_text}"
        f"{val_text}"
        f"Symbolic Equation vs NN:\n"
        f"MAE: {fmt(eq_metrics.get('mae')):>10} | RMSE: {fmt(eq_metrics.get('rmse')):>10}"
    )

    axs[1,1].text(0.5, 0.78, text, transform=axs[1,1].transAxes, fontsize=13, family='monospace',
                  ha='center', va='center', bbox=dict(boxstyle='round', facecolor='#f8f9fa'))

    # --- THREE SEPARATE LEGEND BOXES ---

    # 1. Styles Legend
    style_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555555', alpha=0.6, markersize=9, label='Real Data'),
        Line2D([0], [0], color='#555555', lw=2.5, label='NN Model'),
    ]
    if eq_plot_data:
        style_elements.append(
            Line2D([0], [0], color='#555555', lw=1.8, ls='--', label='Eq String')
        )
    leg_style = axs[1,1].legend(handles=style_elements, loc='lower left', fontsize=12,
                                title="Curve Styles", title_fontsize=14,
                                frameon=True, bbox_to_anchor=(0.0, 0.0), facecolor='#f8f9fa')
    axs[1,1].add_artist(leg_style)

    # 2. Vds Legend
    if use_real_voltages:
        vds_elements = [Line2D([0], [0], color=colors_vds[i], lw=3, label=f"Vds ≈ {real_vds_map.get(vds, vds):.{legend_decimals}f}V") for i, vds in enumerate(target_vds_list)]
        vds_title = "Actual Vds"
    else:
        vds_elements = [Line2D([0], [0], color=colors_vds[i], lw=3, label=f"Vds = {vds:.{legend_decimals}f}V") for i, vds in enumerate(target_vds_list)]
        vds_title = "Target Vds"

    leg_vds = axs[1,1].legend(handles=vds_elements, loc='lower center', fontsize=12,
                              title=vds_title, title_fontsize=14,
                              frameon=True, bbox_to_anchor=(0.5, 0.0), facecolor='#f8f9fa')
    axs[1,1].add_artist(leg_vds)

    # 3. Vgs Legend
    if use_real_voltages:
        vgs_elements = [Line2D([0], [0], color=colors_vgs[i], lw=3, label=f"Vgs ≈ {real_vgs_map.get(vgs, vgs):.{legend_decimals}f}V") for i, vgs in enumerate(target_vgs_list)]
        vgs_title = "Actual Vgs"
    else:
        vgs_elements = [Line2D([0], [0], color=colors_vgs[i], lw=3, label=f"Vgs = {vgs:.{legend_decimals}f}V") for i, vgs in enumerate(target_vgs_list)]
        vgs_title = "Target Vgs"

    leg_vgs = axs[1,1].legend(handles=vgs_elements, loc='lower right', fontsize=12,
                              title=vgs_title, title_fontsize=14,
                              frameon=True, bbox_to_anchor=(1.0, 0.0), facecolor='#f8f9fa')

    plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)


from scipy.signal import savgol_filter
from scipy.optimize import minimize as scipy_minimize



def smooth_derivative(x, y, order=1, win_pre=11, win_post=13, poly=2):
    """
    Replicates MATLAB sgolay + gradient + sgolay pipeline.
    
    x: sorted x
    y: sorted y
    order: derivative order (1 = gm, 2 = gm2)
    win_pre: window for initial smoothing (must be odd)
    win_post: window for smoothing after each gradient (must be odd)
    poly: polynomial order
    """

    # Ensure numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Make windows odd
    if win_pre % 2 == 0:
        win_pre += 1
    if win_post % 2 == 0:
        win_post += 1

    # --- 1. Pre-smooth ---
    y_s = savgol_filter(y, win_pre, poly)

    # --- 2. Iterative differentiation ---
    for _ in range(order):
        # Gradient
        y_s = np.gradient(y_s, x)

        # Smooth after derivative
        y_s = savgol_filter(y_s, win_post, poly)

    return y_s



def generate_physics_plot_data(model, T_val, vgs_col, vds_col, ids_col, target_vds_list, target_vgs_list, device, tol_vds=0.1, tol_vgs=0.03, force_target_range=False):
    """Generates a clean dictionary of true data points and model predictions."""
    plot_data = {'vds_sweeps': [], 'vgs_sweeps': []}
    has_tn = 'TN' in T_val.columns
    unique_tns = sorted(T_val['TN'].unique()) if has_tn else []

    # --- 1. DATA FOR Ids vs Vgs & Gm ---
    for t_vds in target_vds_list:
        v_pts, i_pts = [], []
        # Actual Vds fed to the model below (vds_synth_np). Real Vds sweeps are irregularly
        # spaced (unlike Vgs, spacing here can be >0.5V, growing with Vds), so rather than
        # fabricating an interpolated true-Ids value at the exact nominal t_vds, each trace's
        # CLOSEST real (Vds, Ids) point is used directly -- no synthetic detail invented. The
        # model curve's fixed Vds is then the actual MEAN of those matched real points (not the
        # nominal target), so the model is compared against the same real values the true
        # scatter points actually are, same principle as the Vgs-sweep fix above.
        matched_vds = []

        if has_tn:
            for tn in unique_tns:
                df_tn = T_val[T_val['TN'] == tn].sort_values(vds_col)
                if df_tn.empty: continue
                if df_tn[vds_col].min() - tol_vds <= t_vds <= df_tn[vds_col].max() + tol_vds:
                    idx = (df_tn[vds_col] - t_vds).abs().idxmin()
                    v_pts.append(df_tn[vgs_col].mean())
                    i_pts.append(df_tn.loc[idx, ids_col])
                    matched_vds.append(df_tn.loc[idx, vds_col])
            model_vds = float(np.mean(matched_vds)) if matched_vds else t_vds
        else:
            for tn in T_val['TN'].unique() if 'TN' in T_val.columns else [0]:
                df_tn = T_val[T_val['TN'] == tn] if 'TN' in T_val.columns else T_val
                if df_tn.empty: continue
                idx = (df_tn[vds_col] - t_vds).abs().idxmin()
                if abs(df_tn.loc[idx, vds_col] - t_vds) <= tol_vds:
                    v_pts.append(df_tn[vgs_col].mean())
                    i_pts.append(df_tn.loc[idx, ids_col])
                    matched_vds.append(df_tn.loc[idx, vds_col])
            model_vds = float(np.mean(matched_vds)) if matched_vds else t_vds

        if len(v_pts) < 5 and not force_target_range:
            continue

        if len(v_pts) > 0:
            s = np.argsort(v_pts)
            v_pts, i_pts = np.array(v_pts)[s], np.array(i_pts)[s]
            v_pts, unique_idx = np.unique(v_pts, return_index=True)
            i_pts = i_pts[unique_idx]
        else:
            v_pts, i_pts = np.array([]), np.array([])

        # Calculate True Derivatives (only when enough measured points available)
        true_gm1 = true_gm2 = true_gm3 = np.array([])
        n_pts = len(v_pts)
        if n_pts >= 3:
            w_pre = min(11, n_pts if n_pts % 2 != 0 else n_pts - 1)
            w_post = min(13, n_pts if n_pts % 2 != 0 else n_pts - 1)
            if w_pre >= 3 and w_post >= 3:
                try:
                    true_gm1 = smooth_derivative(v_pts, i_pts, order=1, win_pre=w_pre, win_post=w_post)
                    true_gm2 = smooth_derivative(v_pts, i_pts, order=2, win_pre=w_pre, win_post=w_post)
                    true_gm3 = smooth_derivative(v_pts, i_pts, order=3, win_pre=w_pre, win_post=w_post)
                except NameError:
                    true_gm1 = np.gradient(i_pts, v_pts); true_gm2 = np.gradient(true_gm1, v_pts); true_gm3 = np.gradient(true_gm2, v_pts)
            else:
                true_gm1 = np.gradient(i_pts, v_pts); true_gm2 = np.gradient(true_gm1, v_pts); true_gm3 = np.gradient(true_gm2, v_pts)

        # Model prediction: use target_vgs_list range in extrapolation mode, else measured range
        if force_target_range and target_vgs_list:
            vgs_lo = min(target_vgs_list)
            vgs_hi = max(target_vgs_list)
        else:
            vgs_lo = v_pts.min()
            vgs_hi = v_pts.max()
        vgs_synth_np = np.linspace(vgs_lo, vgs_hi, 250).reshape(-1, 1)
        vds_synth_np = np.full_like(vgs_synth_np, model_vds)
        vgs_synth_t = torch.tensor(vgs_synth_np, dtype=torch.float64, device=device).requires_grad_(True)
        vds_synth_t = torch.tensor(vds_synth_np, dtype=torch.float64, device=device)
        
        try:
            ids_pred = model(torch.cat([vgs_synth_t, vds_synth_t], dim=1))
            gm1_pred = torch.autograd.grad(ids_pred, vgs_synth_t, torch.ones_like(ids_pred), create_graph=True)[0]
            gm2_pred = torch.autograd.grad(gm1_pred, vgs_synth_t, torch.ones_like(gm1_pred), create_graph=True)[0]
            gm3_pred = torch.autograd.grad(gm2_pred, vgs_synth_t, torch.ones_like(gm2_pred))[0]
            
            plot_data['vds_sweeps'].append({
                'target_vds': t_vds, 'actual_vds': model_vds,
                'true_vgs': v_pts, 'true_ids': i_pts, 'true_gm1': true_gm1, 'true_gm2': true_gm2, 'true_gm3': true_gm3,
                'pred_vgs': vgs_synth_np.flatten(), 'pred_ids': ids_pred.detach().cpu().numpy().flatten(),
                'pred_gm1': gm1_pred.detach().cpu().numpy().flatten(), 'pred_gm2': gm2_pred.detach().cpu().numpy().flatten(), 'pred_gm3': gm3_pred.detach().cpu().numpy().flatten(),
            })
        except Exception: pass

    # --- 2. DATA FOR Ids vs Vds ---
    for t_vgs in target_vgs_list:
        vd_pts, id_pts = [], []
        # Vgs actually fed to the model below when synthesizing pred_vds/pred_ids. Real sweeps
        # rarely land exactly on a target_vgs_list grid value (e.g. mean measured Vgs=-1.85 for a
        # nominal target of -2), so this is overridden with the matched trace's OWN actual mean
        # Vgs whenever real data is found -- otherwise the model curve would be evaluated at a
        # slightly different Vgs than the true points it's plotted against, offsetting them from
        # each other even for a perfect model. Falls back to the nominal t_vgs only when no real
        # trace matched (pure extrapolation, force_target_range).
        model_vgs = t_vgs

        if has_tn:
            best_tn, min_vgs_diff = None, float('inf')
            for tn in unique_tns:
                df_tn = T_val[T_val['TN'] == tn]
                if df_tn.empty: continue
                diff = abs(df_tn[vgs_col].mean() - t_vgs)
                if diff < min_vgs_diff: min_vgs_diff, best_tn = diff, tn
            if best_tn is not None and min_vgs_diff <= max(tol_vgs, 0.1):
                df_best = T_val[T_val['TN'] == best_tn].sort_values(vds_col)
                vd_pts, id_pts = df_best[vds_col].values, df_best[ids_col].values
                model_vgs = df_best[vgs_col].mean()
            elif not force_target_range:
                continue
        else:
            matched_vgs = []
            for tn in T_val['TN'].unique() if 'TN' in T_val.columns else [0]:
                df_tn = T_val[T_val['TN'] == tn] if 'TN' in T_val.columns else T_val
                if df_tn.empty: continue
                idx = (df_tn[vgs_col] - t_vgs).abs().idxmin()
                if abs(df_tn.loc[idx, vgs_col] - t_vgs) <= tol_vgs:
                    vd_pts.append(df_tn.loc[idx, vds_col])
                    id_pts.append(df_tn.loc[idx, ids_col])
                    matched_vgs.append(df_tn.loc[idx, vgs_col])
            if matched_vgs:
                model_vgs = float(np.mean(matched_vgs))

        if len(vd_pts) > 1:
            s = np.argsort(vd_pts)
            vd_pts, id_pts = np.array(vd_pts)[s], np.array(id_pts)[s]
        elif not force_target_range:
            continue
        else:
            vd_pts, id_pts = np.array([]), np.array([])

        # Model prediction: use target_vds_list range in extrapolation mode, else measured range
        if force_target_range and target_vds_list:
            vds_lo = min(target_vds_list)
            vds_hi = max(target_vds_list)
        else:
            vds_lo = vd_pts.min()
            vds_hi = vd_pts.max()
        vd_synth_np = np.linspace(vds_lo, vds_hi, 250).reshape(-1, 1)
        vg_synth_np = np.full_like(vd_synth_np, model_vgs)
        vg_synth_t, vd_synth_t = torch.tensor(vg_synth_np, dtype=torch.float64, device=device), torch.tensor(vd_synth_np, dtype=torch.float64, device=device)
        
        try:
            with torch.no_grad(): id_pred = model(torch.cat([vg_synth_t, vd_synth_t], dim=1)).cpu().numpy().flatten()
            plot_data['vgs_sweeps'].append({
                'target_vgs': t_vgs, 'actual_vgs': model_vgs, 'true_vds': vd_pts, 'true_ids': id_pts,
                'pred_vds': vd_synth_np.flatten(), 'pred_ids': id_pred
            })
        except Exception: pass
            
    return plot_data

def plot_separate(plot_data, eq_metrics, save_dir, base_name, title_prefix, target_vds_list, target_vgs_list):
    """Saves entirely separate plots, placing the MAE in the Ids vs Vgs plot."""
    os.makedirs(save_dir, exist_ok=True)
    colors_vds = plt.cm.magma(np.linspace(0.1, 0.9, len(target_vds_list)))
    colors_vgs = plt.cm.viridis(np.linspace(0.1, 0.9, len(target_vgs_list)))

    metrics_to_plot = [
        ('Ids vs Vgs', 'true_vgs', 'true_ids', 'pred_vgs', 'pred_ids', 'vds_sweeps', 'target_vds', colors_vds, target_vds_list, 'o'),
        ('Gm1 vs Vgs', 'true_vgs', 'true_gm1', 'pred_vgs', 'pred_gm1', 'vds_sweeps', 'target_vds', colors_vds, target_vds_list, '^'),
        ('Gm2 vs Vgs', 'true_vgs', 'true_gm2', 'pred_vgs', 'pred_gm2', 'vds_sweeps', 'target_vds', colors_vds, target_vds_list, 's'),
        ('Gm3 vs Vgs', 'true_vgs', 'true_gm3', 'pred_vgs', 'pred_gm3', 'vds_sweeps', 'target_vds', colors_vds, target_vds_list, 'd'),
        ('Ids vs Vds', 'true_vds', 'true_ids', 'pred_vds', 'pred_ids', 'vgs_sweeps', 'target_vgs', colors_vgs, target_vgs_list, 'o')
    ]

    def fmt(val): return f"{val:.3e}" if val is not None else "N/A"

    for title, tx, ty, px, py, sweep_key, target_key, colors, target_list, marker in metrics_to_plot:
        plt.figure(figsize=(10, 8))
        plt.title(f"{title_prefix} - {title}", fontsize=16)
        
        for sweep in plot_data[sweep_key]:
            c = colors[target_list.index(sweep[target_key])]
            plt.plot(sweep[tx], sweep[ty], marker, color=c, alpha=0.5, label=f"True {sweep[target_key]}V")
            plt.plot(sweep[px], sweep[py], '-', color=c, lw=2.5, label=f"Pred {sweep[target_key]}V")
            
        plt.xlabel(tx.split('_')[1].capitalize()); plt.ylabel(ty.split('_')[1].capitalize())
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add equation MAE only to the Ids plots
        if 'Ids' in title and eq_metrics.get('mae') is not None:
             plt.text(0.05, 0.95, f"Eq vs NN MAE: {fmt(eq_metrics['mae'])}", 
                      transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Deduplicate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        
        clean_title = title.replace(' ', '_').lower()
        plt.savefig(os.path.join(save_dir, f"{base_name}_{clean_title}.{PLOT_FORMAT}"), dpi=PLOT_DPI, bbox_inches='tight')
        plt.close()



def plot_best_predictions(base_json_path, base_env, base_config_no, space_config_no, n_best=3, save_plot=False):
    """
    Plot the best N predictions from Optuna results using results.json file.
    
    Args:
        results_json_path: Path to the results.json file
        n_best: Number of best trials to plot
        save_plot: Whether to save the plot to results directory
    """
    _ensure_trainer_deps()
    # Load results.json
    results_json_path = os.path.join(base_json_path, f"{base_env}_results", f"results_base{base_config_no}_opt{space_config_no}.json")
    results_dir = os.path.dirname(results_json_path)
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)
    
    if not results_data:
        print("No results found in the JSON file.")
        return
    
    # Use the first result entry to get paths and parameters
    first_result = results_data[0]
    if "env_to_train_info" in first_result and first_result["env_to_train_info"] is not None:
        csv_path = first_result["env_to_train_info"]['csv_path']
        meas_load_kwargs = first_result["env_to_train_info"]['meas_load_kwargs']
    else:
        csv_path = first_result['full_csv_path']
        meas_load_kwargs = first_result['meas_load_kwargs']

    db_path = first_result['tpe_db_path']
    environment = first_result['environment']
    config_id = first_result['config_id']
    optimizer_space_id = first_result['optimizer_space_id']

    weights_filepath = os.path.join(results_dir, f"optimized_weights_base{config_id}_opt{optimizer_space_id}.pkl")
    results_dir = os.path.join(base_json_path, f"{base_env}_results")
    
    # Load training and validation data using meas_load
    T_train, T_test = meas_load(csv_path, **meas_load_kwargs)
    
    # Extract variables from user_metadata if available, otherwise use defaults
    user_metadata = first_result['user_metadata']
    current2train = user_metadata['current2train']  # default fallback
    train_data_mode = user_metadata['train_data_mode']  # default fallback
    
    # Prepare training and validation data
    X_train = np.column_stack((T_train[train_data_mode], T_train['Vds']))
    y_train = np.array(T_train[current2train]).reshape(-1, 1)
    X_val = np.column_stack((T_test[train_data_mode], T_test['Vds']))
    Y_val = np.array(T_test[current2train]).reshape(-1, 1)
    
        # Load the study
    study = optuna.load_study(storage=f"sqlite:///{db_path}", study_name=f"study_env_simulation_base{config_id}_opt{optimizer_space_id}")
    best_trials = get_best_n_trials(study, n_best)
    # Get best trials
    # best_trials = study.best_trials[:n_best]
    
    if len(best_trials) == 0:
        print("No completed trials found in the study.")
        return
    
    # Flatten Y_val for consistent plotting
    Y_val_flat = Y_val.flatten() if Y_val.ndim > 1 else Y_val

    # Convert training and validation data to tensors
    
    plot_results = []
    figures = []

    os.path.join(results_dir, f"optimized_weights_base{config_id}_opt{optimizer_space_id}.pkl")
    # Train and evaluate each trial's model - create separate figure for each
    for idx, trial in enumerate(best_trials):
        # Reconstruct and train model from trial parameters
        try:
            print(f"Loading model for Trial {trial.number}...")

            params = trial.params
    
            # Reconstruct model architecture
            model = reconstruct_model_from_trial(trial)

            optimizer_name = params['optimizer']

            trial_data = load_trial_weights(weights_filepath, idx)
            model.load_state_dict(trial_data["model_state_dict"])
            params.update(optimizer_spaces_second_run_params[1][optimizer_name])
            
            # equation = generate_equation_from_model(model, trial)

            # predictions, eq_residuals, eq_r2, eq_rmse = handle_equation_comparison(
            #     X_val[:, 0], X_val[:, 1], Y_val_flat, equation, 
            #     title=None
            # )
            # Check if X_val is already a tensor
            if isinstance(X_val, torch.Tensor):
                X_val = X_val.detach().clone().to(dtype=torch.float64)
            else:
                # If it's a numpy array or other type
                X_val = torch.tensor(X_val, dtype=torch.float64)

            model.eval()
            with torch.no_grad():
                predictions_tensor = model(X_val)
                predictions = predictions_tensor.detach().numpy().flatten()

            _handle_plots(X_val, Y_val_flat, predictions, trial,
                            figures, plot_results, save_plot, results_dir,
                            config_id, optimizer_space_id, idx, None,
                            None)
                            # {"eq_residuals": eq_residuals, "eq_r2": eq_r2, "eq_rmse": eq_rmse})


            
        except Exception as e:
            error_msg = f"Error plotting for trial {trial.number}: {e}"
            print(error_msg)
            

    # Show all figures at once (non-blocking)
    return plot_results


def _handle_plots(X_val, Y_val_flat, predictions, trial, figures, plot_results, save_plot, results_dir, config_id, optimizer_space_id, idx, best_loss, eq_stats=None):

    residuals = Y_val_flat - predictions
    label = f'Trial {trial.number} (Loss: {trial.value:.4e})'
    
    # Store results
    plot_results.append((label, trial.params, trial.value, predictions))
    
    # ==========================================
    # 1. DATA EXTRACTION & ENVIRONMENT VARS
    # ==========================================
    Vgs_data = X_val[:, 0]
    Vds_data = X_val[:, 1]
    Ids_meas = Y_val_flat
    Ids_pred = predictions
    
    r2 = 1 - np.sum(residuals**2) / np.sum((Y_val_flat - np.mean(Y_val_flat))**2)
    rmse = np.sqrt(np.mean(residuals**2))

    # Fetch and parse environment variables (with fallback defaults)
    vgs_env_str = os.getenv("TARGET_VGS_TO_PLOT", "[-3.0, -2.0, -1.0, 0.0]")
    vds_env_str = os.getenv("TARGET_VDS_TO_PLOT", "[5.0, 10.0, 15.0, 20.0]")
    
    try:
        target_vgs_list = [float(x) for x in ast.literal_eval(vgs_env_str)]
        target_vds_list = [float(x) for x in ast.literal_eval(vds_env_str)]
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Could not parse environment variables for plotting. Using defaults. Error: {e}")
        target_vgs_list = [-3.0, -2.0, -1.0, 0.0]
        target_vds_list = [5.0, 10.0, 15.0, 20.0]

    # ==========================================
    # 2. MATPLOTLIB BLOCK (2x3 Grid)
    # ==========================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f'Trial {trial.number} - Loss: {trial.value:.4e} - Rank: {idx}', fontsize=20, fontweight='bold')
    
    # Unpack the axes
    ax_trans, ax_out, ax_pred = axes[0]
    ax_gm, ax_gm1, ax_gm2 = axes[1]
    
    if eq_stats is not None:
        fig.text(0.02, 0.02, f"Eq stats: RÂ² {eq_stats['eq_r2']:.4e}, RMSE: {eq_stats['eq_rmse']:.4e}", fontsize=12, 
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                 verticalalignment='bottom')

    # ---------------------------------------------------------
    # ROW 1, COL 1: Transfer & ROW 2: Gm, Gm1, Gm2
    # ---------------------------------------------------------
    colors_trans = plt.cm.plasma(np.linspace(0.1, 0.9, len(target_vds_list)))
    
    for i, vds_val in enumerate(target_vds_list):
        mask = np.isclose(Vds_data, vds_val, atol=1e-2)
        if not np.any(mask): continue
            
        vgs_sub, ids_m_sub, ids_p_sub = Vgs_data[mask], Ids_meas[mask], Ids_pred[mask]
        
        # Sort values by Vgs to ensure lines and gradients calculate correctly
        sort_idx = np.argsort(vgs_sub)
        vgs_sub, ids_m_sub, ids_p_sub = vgs_sub[sort_idx], ids_m_sub[sort_idx], ids_p_sub[sort_idx]
        
        # Plot Transfer (Ids vs Vgs)
        ax_trans.plot(vgs_sub, ids_m_sub, 's', color=colors_trans[i], markersize=5, alpha=0.6, label=f'Meas Vds={vds_val}V')
        ax_trans.plot(vgs_sub, ids_p_sub, '--', color=colors_trans[i], linewidth=2, label=f'Model Vds={vds_val}V')
        
        # Calculate & Plot Derivatives (Gm, Gm1, Gm2)
        if len(vgs_sub) > 1:
            # First Derivative (Gm)
            gm_m = np.gradient(ids_m_sub, vgs_sub)
            gm_p = np.gradient(ids_p_sub, vgs_sub)
            ax_gm.plot(vgs_sub, gm_m, 's', color=colors_trans[i], markersize=5, alpha=0.6)
            ax_gm.plot(vgs_sub, gm_p, '--', color=colors_trans[i], linewidth=2)
            
            # Second Derivative (Gm1)
            gm1_m = np.gradient(gm_m, vgs_sub)
            gm1_p = np.gradient(gm_p, vgs_sub)
            ax_gm1.plot(vgs_sub, gm1_m, 's', color=colors_trans[i], markersize=5, alpha=0.6)
            ax_gm1.plot(vgs_sub, gm1_p, '--', color=colors_trans[i], linewidth=2)
            
            # Third Derivative (Gm2)
            gm2_m = np.gradient(gm1_m, vgs_sub)
            gm2_p = np.gradient(gm1_p, vgs_sub)
            ax_gm2.plot(vgs_sub, gm2_m, 's', color=colors_trans[i], markersize=5, alpha=0.6)
            ax_gm2.plot(vgs_sub, gm2_p, '--', color=colors_trans[i], linewidth=2)

    # Styling for Transfer and Transconductance plots
    ax_trans.set_title('Transfer Characteristics', fontsize=16)
    ax_trans.set_xlabel('Gate-Source Voltage, Vgs (V)', fontsize=14)
    ax_trans.set_ylabel('Drain Current, Ids (A)', fontsize=14)
    ax_trans.grid(True, linestyle='--', alpha=0.7)
    
    ax_gm.set_title('$G_m$ (1st Deriv)', fontsize=16)
    ax_gm.set_xlabel('Gate-Source Voltage, Vgs (V)', fontsize=14)
    ax_gm.set_ylabel('$G_m$ (S)', fontsize=14)
    ax_gm.grid(True, linestyle='--', alpha=0.7)

    ax_gm1.set_title('$G_{m1}$ (2nd Deriv)', fontsize=16)
    ax_gm1.set_xlabel('Gate-Source Voltage, Vgs (V)', fontsize=14)
    ax_gm1.set_ylabel('$G_{m1}$ (S/V)', fontsize=14)
    ax_gm1.grid(True, linestyle='--', alpha=0.7)

    ax_gm2.set_title('$G_{m2}$ (3rd Deriv)', fontsize=16)
    ax_gm2.set_xlabel('Gate-Source Voltage, Vgs (V)', fontsize=14)
    ax_gm2.set_ylabel('$G_{m2}$ (S/V$^2$)', fontsize=14)
    ax_gm2.grid(True, linestyle='--', alpha=0.7)

    # ---------------------------------------------------------
    # ROW 1, COL 2: Output (Ids vs Vds)
    # ---------------------------------------------------------
    colors_out = plt.cm.viridis(np.linspace(0.1, 0.9, len(target_vgs_list)))
    
    for i, vgs_val in enumerate(target_vgs_list):
        mask = np.isclose(Vgs_data, vgs_val, atol=1e-2)
        if not np.any(mask): continue 
            
        vds_sub, ids_m_sub, ids_p_sub = Vds_data[mask], Ids_meas[mask], Ids_pred[mask]
        
        # Sort by Vds
        sort_idx = np.argsort(vds_sub)
        vds_sub, ids_m_sub, ids_p_sub = vds_sub[sort_idx], ids_m_sub[sort_idx], ids_p_sub[sort_idx]

        ax_out.plot(vds_sub, ids_m_sub, 'o', color=colors_out[i], markersize=5, alpha=0.6, label=f'Meas Vgs={vgs_val}V')
        ax_out.plot(vds_sub, ids_p_sub, '-', color=colors_out[i], linewidth=2, label=f'Model Vgs={vgs_val}V')

    ax_out.set_title('Output Characteristics', fontsize=16)
    ax_out.set_xlabel('Drain-Source Voltage, Vds (V)', fontsize=14)
    ax_out.set_ylabel('Drain Current, Ids (A)', fontsize=14)
    ax_out.grid(True, linestyle='--', alpha=0.7)

    # ---------------------------------------------------------
    # ROW 1, COL 3: Predictions vs True
    # ---------------------------------------------------------
    ax_pred.scatter(Y_val_flat, predictions, alpha=0.6, c='red', marker='^')
    
    # Add an ideal 1:1 line
    line_min, line_max = Y_val_flat.min(), Y_val_flat.max()
    ax_pred.plot([line_min, line_max], [line_min, line_max], 'k--', lw=2)
    
    ax_pred.set_title('Predictions vs True', fontsize=16)
    ax_pred.set_xlabel('True Ids (A)', fontsize=14)
    ax_pred.set_ylabel('Predicted Ids (A)', fontsize=14)
    ax_pred.grid(True, linestyle='--', alpha=0.7)
    
    # Text box for global metrics
    textstr = f'RÂ²: {r2:.4f}\nRMSE: {rmse:.3e} A'
    ax_pred.text(0.05, 0.95, textstr, transform=ax_pred.transAxes, fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Enable legends for top row (safeguard to avoid errors if missing handles)
    handles_trans, _ = ax_trans.get_legend_handles_labels()
    if handles_trans: ax_trans.legend(fontsize=9, loc='best')
    
    handles_out, _ = ax_out.get_legend_handles_labels()
    if handles_out: ax_out.legend(fontsize=9, loc='best')

    # Apply layout constraints
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Give room for the global title
    figures.append(fig)
    
    # ==========================================
    # 3. SAVE BLOCK
    # ==========================================
    if save_plot:
        figs_path = os.path.join(results_dir, f"figures_base{config_id}_opt{optimizer_space_id}")
        if os.environ["OVERRIDE_FILES"] == "True" and os.path.exists(figs_path):
            shutil.rmtree(figs_path)
        os.makedirs(figs_path, exist_ok=True)
        plot_path = os.path.join(results_dir, f"figures_base{config_id}_opt{optimizer_space_id}", f"trial_{trial.number}_bestno_{idx}_rmse_{rmse:.4e}_max_{np.max(np.abs(residuals)):.4e}.{PLOT_FORMAT}")
        plt.savefig(plot_path, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")




def plot_create_eq_best_predictions(base_json_path, base_env, base_config_no, space_config_no, 
                                    env_to_train_info=None, early_stopping_metric="mean", 
                                    n_best=3, save_plot=True, save_equations=True, val_interval=10, 
                                    device=None, physics_c=None, temp_model_dir=None):
    """
    Plot the best N predictions from Optuna results using results.json file, 
    perform L-BFGS fine-tuning, and calculate updated physics boundary hits.
    """
    _ensure_trainer_deps()
    print("XXXXXXXXXXXXXXXXX DEBUG XXXXXXXX")
    if isinstance(base_config_no, int):
        results_json_path = os.path.join(base_json_path, f"{base_env}_results", f"results_base{base_config_no}_opt{space_config_no}.json")
    else:
        print(f"Using passed results json {base_config_no}...")
        results_json_path = base_config_no
    
    with open(results_json_path, 'r') as f:
        loaded_data = json.load(f)
        results_data = loaded_data.get("results", loaded_data) if isinstance(loaded_data, dict) else loaded_data
    
    if not results_data:
        print("No results found in the JSON file.")
        return

    first_result = results_data[0]
    db_path = first_result['tpe_db_path']
    csv_path = first_result['full_csv_path']
    meas_load_kwargs = (env_to_train_info or {}).get('meas_load_kwargs') or first_result['meas_load_kwargs']
    base_env = first_result.get('environment', env_to_train_info.get('environment', base_env))
    config_id = first_result['config_id']
    optimizer_space_id = first_result['optimizer_space_id']
    user_metadata = first_result.get('user_metadata', {})
    current2train = user_metadata.get('current2train', 'Ids')  
    train_data_mode = user_metadata.get('train_data_mode', 'Vgs_meas')  
    results_dir = os.path.dirname(results_json_path)
    
    weights_filepath = os.path.join(results_dir, f"optimized_weights_base{config_id}_opt{optimizer_space_id}.pkl")
    print(f"Weights filepath: {weights_filepath}")

    _is_slsqp = db_path is None
    if _is_slsqp:
        # SLSQP path: no Optuna study â€” build mock trials directly from JSON results.
        # user_attrs (architecture info) are loaded from the SLSQP PKL file.
        import types as _types
        _slsqp_user_attrs = {}
        _slsqp_pkl_path = first_result.get('weights_filepath')
        if _slsqp_pkl_path and os.path.exists(_slsqp_pkl_path):
            try:
                with open(_slsqp_pkl_path, 'rb') as _f:
                    _slsqp_pkl = pickle.load(_f)
                _slsqp_tlist = _slsqp_pkl.get('trials', []) if isinstance(_slsqp_pkl, dict) else _slsqp_pkl
                if _slsqp_tlist:
                    _slsqp_user_attrs = _slsqp_tlist[0].get('user_attrs', {})
            except Exception as _e:
                print(f"[SLSQP post-proc] Could not load user_attrs from PKL: {_e}")
        sorted_results = sorted(
            (e for e in results_data if e.get('trial_number') is not None),
            key=lambda x: x.get('trial_value', float('inf')),
        )
        best_trials = [
            _types.SimpleNamespace(
                number=entry['trial_number'],
                value=entry['trial_value'],
                params=entry.get('optimized_params', {}),
                user_attrs=_slsqp_user_attrs,
            )
            for entry in sorted_results[:n_best]
        ]
        study = None
        optuna_boundary_hits = {}
        print(f"[SLSQP post-proc] Built {len(best_trials)} mock trial(s) from JSON.")
    else:
        study = optuna.load_study(storage=f"sqlite:///{db_path}", study_name=f"study_env_{base_env}_base{config_id}_opt{optimizer_space_id}")
        best_trials = get_best_n_trials(study, n_best)

        # Check Optuna boundaries
        optuna_boundary_hits = {}
        try:
            optuna_boundary_hits = check_optuna_boundaries(study, top_n=n_best, tolerance=0.05)
        except Exception as e:
            print(f"Warning: Could not perform Optuna boundary checks: {e}")

    plot_results = []
    figures = []
    equations = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_trial_list = []

    # =========================================================================
    # BRANCH 1: SAME ENVIRONMENT
    # =========================================================================
    if env_to_train_info is None or env_to_train_info.get("csv_path") is None:
        print("Running train in same env...")
        T_train, T_test = meas_load(csv_path, **meas_load_kwargs)

        if meas_load_kwargs["test_percent"] == 0:
            T_test = T_train
        
        X_train = np.column_stack((T_train[train_data_mode], T_train['Vds']))
        y_train = np.array(T_train[current2train]).reshape(-1, 1)
        X_val = np.column_stack((T_test[train_data_mode], T_test['Vds']))
        Y_val = np.array(T_test[current2train]).reshape(-1, 1)
        
        if len(best_trials) == 0:
            print("No completed trials found in the study.")
            return
            
        Y_val_flat = Y_val.flatten() if Y_val.ndim > 1 else Y_val

        results_lookup = {entry.get('trial_number'): entry for entry in results_data if entry.get('trial_number') is not None}

        for idx, trial in enumerate(best_trials):
            matched_result = results_lookup.get(trial.number)
            if matched_result is None:
                continue
            
            # --- Check if this specific trial already has the metrics ---
            is_faulty_run = os.environ.get("RUN_POST_ON_FAULTY_JSONS_ONLY") == "True"
            if is_faulty_run:
                keys_to_check = ['new_train_rmse_ids', 'new_train_rmse_gm1', 'new_train_rmse_gm2', 'new_train_rmse_gm3']
                if all(key in matched_result and matched_result[key] is not None for key in keys_to_check):
                    print(f"Skipping Trial {trial.number}: 'RUN_POST_ON_FAULTY_JSONS_ONLY' is True and all metrics already exist.")
                    continue
            # ------------------------------------------------------------

            if matched_result and "weights_filepath" in matched_result:
                original_path = matched_result.get("optimized_weights_filepath", matched_result["weights_filepath"])
                if env_to_train_info is None: env_to_train_info = {}
                env_to_train_info["weights_filepath"] = original_path  
                env_to_train_info["trial_rank"] = matched_result.get("rank", idx)
            else:
                continue

            if not isinstance(base_config_no, int):
                space_config = get_space_config_from_results_json(results_json_path, idx, base_env)
            else:
                space_config = None 

            print(f"Training model for Trial {trial.number}...")
            lbfgs_start_time = time.time()
            
            model, predictions, best_loss, optimizer = train_and_evaluate_model(
                trial, X_train, y_train, X_val, Y_val, 
                env_to_train_info, space_config, 
                early_stopping_metric=early_stopping_metric, 
                val_interval=val_interval, config_id=config_id, space_config_no=space_config_no, T_train=T_train, T_test=T_test
            )
            
            lbfgs_exec_time = time.time() - lbfgs_start_time
            model_trial_list.append((model, trial, optimizer))
            equation = generate_equation_from_model(model, trial)

            residuals = Y_val_flat - predictions
            r2 = 1 - np.sum(residuals**2) / np.sum((Y_val_flat - np.mean(Y_val_flat))**2)
            rmse = np.sqrt(np.mean(residuals**2))

            # matched_result.setdefault("time_lbfgs_execution_list", []).append(lbfgs_exec_time)
            # total_lbfgs_time = sum(matched_result["time_lbfgs_execution_list"])
            # matched_result["lbfgs_execution_time"] = total_lbfgs_time
            
            # opt_exec_time = matched_result.get("trial_execution_time_seconds", 0)
            # time_to_reach = matched_result.get("time_to_reach_seconds", 0)
            # final_total_exec = opt_exec_time + total_lbfgs_time
            # matched_result["time_total_execution_2"] = final_total_exec
            # matched_result["time_total_training_2"] = time_to_reach + final_total_exec

            if meas_load_kwargs["test_percent"] == 0:
                matched_result["fitness_post_train"] = best_loss
                matched_result["fit_train_rmse"] = rmse
                matched_result["fit_train_r2"] = r2
                matched_result.setdefault("fitness_post_train_list", []).append(best_loss)
                matched_result.setdefault("fit_train_rmse_list", []).append(rmse)
                matched_result.setdefault("fit_train_r2_list", []).append(r2)
            else:
                matched_result["fitness_post_val"] = best_loss
                matched_result["fit_val_rmse"] = rmse
                matched_result["fit_val_r2"] = r2
                matched_result.setdefault("fitness_post_val_list", []).append(best_loss)
                matched_result.setdefault("fit_val_rmse_list", []).append(rmse)
                matched_result.setdefault("fit_val_r2_list", []).append(r2)
            
            matched_result["optimized_weights_filepath"] = weights_filepath

            rmse_metrics = evaluate_physics(
                model=model, 
                T_val=T_train,
                vgs_col=train_data_mode, 
                vds_col='Vds', 
                ids_col=current2train, 
                target_vds_list=target_vds_list,
                device=device,          # <-- Added missing device (CPU/CUDA)!
            )
            # if rmse_metrics:
            #     for key, value in rmse_metrics.items():
            #         latest_key = f"new_train_{key}"
            #         list_key = f"{latest_key}_list"
                    
            #         matched_result[latest_key] = value
                    
            #         if list_key not in matched_result:
            #             matched_result[list_key] = []
            #         matched_result[list_key].append(value)
            matched_result = update_execution_metrics(matched_result, rmse_metrics, lbfgs_exec_time)
        # --- OPTIMIZED BOUNDARY LOGIC (IN-MEMORY) ---
        physics_boundary_hits = {}
        if physics_c is not None:
            print("Running post-LBFGS physics boundary checks directly from memory...")
            try:
                physics_boundary_hits = check_physics_boundaries(
                    memory_models=model_trial_list, 
                    device=device, 
                    physics_c=physics_c, 
                    tolerance=0.01
                )
            except Exception as e:
                print(f"Warning: Could not perform optimized physics boundary checks: {e}")

        for trial in best_trials:
            matched_result = results_lookup.get(trial.number)
            if matched_result:
                matched_result["optimized_optuna_bound_hits"] = optuna_boundary_hits.get(trial.number, [])
                matched_result["optimized_physics_bound_hits"] = physics_boundary_hits.get(trial.number, [])
        # --------------------------------------------

        if model_trial_list:
            save_best_trials_weights(model_trial_list, weights_filepath)
        
        # (The old equation saving block has been completely removed from here)

        temp_json = results_json_path + ".tmp"
        with open(temp_json, "w") as f:
             if isinstance(loaded_data, dict) and "results" in loaded_data:
                 loaded_data["results"] = results_data
                 json.dump(loaded_data, f, indent=4)
             else:
                 json.dump(results_data, f, indent=4)
        os.replace(temp_json, results_json_path)
        try:
            evaluate_and_plot_all_physics(model,T_val=T_train,results_json_path=results_json_path,
                                      target_vds_list=target_vds_list,target_vgs_list=target_vgs_list,
                                      save_dir=os.path.join(results_dir, "eqs_plots"),
                                      weight_type="Optimized", n_best=n_best)
        except Exception as e:
            print(f"Error during plotting/saving for Trial {trial.number}: {e}")

        return plot_results, equations
    # =========================================================================
    # BRANCH 2: TRANSFER LEARNING / DIFFERENT ENVIRONMENT
    # =========================================================================
    else:
        csv_path = env_to_train_info["csv_path"]
        environment = env_to_train_info["environment"]
        env_to_train_info["weights_filepath"] = weights_filepath

        if "from_env" in env_to_train_info and env_to_train_info["from_env"]:
            results_dir = os.path.join(base_json_path, f"{environment}_from_{base_env}_results")
        else:
            results_dir = os.path.join(base_json_path, f"{environment}_conf_modyof_{base_env}_results")

        os.makedirs(results_dir, exist_ok=True)
        weights_filepath_new = os.path.join(results_dir, f"optimized_weights_base{config_id}_opt{optimizer_space_id}.pkl")
        
        T_train, T_test = meas_load(csv_path, **env_to_train_info["meas_load_kwargs"])
        X_train = np.column_stack((T_train[train_data_mode], T_train['Vds']))
        y_train = np.array(T_train[current2train]).reshape(-1, 1)
        X_val = np.column_stack((T_test[train_data_mode], T_test['Vds']))
        Y_val = np.array(T_test[current2train]).reshape(-1, 1)
        Y_val_flat = Y_val.flatten() if Y_val.ndim > 1 else Y_val

        model_trial_list = []
        val_metrics = []
        lbfgs_metrics = [] 
        
        for idx, trial in enumerate(best_trials):
            env_to_train_info["trial_rank"] = idx
            lbfgs_start_time = time.time()
            
            model, predictions, best_loss, optimizer = train_and_evaluate_model(
                trial, X_train, y_train, X_val, Y_val, 
                env_to_train_info, space_config_no, 
                early_stopping_metric=early_stopping_metric,
                val_interval=val_interval
            )
            
            lbfgs_exec_time = time.time() - lbfgs_start_time
            lbfgs_metrics.append(lbfgs_exec_time)
            
            residuals = Y_val_flat - predictions
            r2 = 1 - np.sum(residuals**2) / np.sum((Y_val_flat - np.mean(Y_val_flat))**2)
            rmse = np.sqrt(np.mean(residuals**2))
            
            val_metrics.append((rmse,r2))
            model_trial_list.append((model, trial, optimizer))
            equation = generate_equation_from_model(model, trial)
            
            try:
                predictions_eq, eq_residuals, eq_r2, eq_rmse = handle_equation_comparison(X_val[:, 0], X_val[:, 1], Y_val_flat, equation, title=None)
                # _handle_plots(X_val, Y_val_flat, predictions_eq, trial, figures, plot_results, save_plot, results_dir, config_id, optimizer_space_id, idx, best_loss, {"eq_residuals": eq_residuals, "eq_r2": eq_r2, "eq_rmse": eq_rmse})
            except Exception as e:
                print(f"Skipping plotting due to error: {e}")
            
            equations.append(f"Trial {trial.number} (Loss: {trial.value:.6e}):\n{equation}\n")

        # --- NEW OPTIMIZED BOUNDARY LOGIC (IN-MEMORY) ---
        physics_boundary_hits = {}
        if physics_c is not None:
            print("Running post-LBFGS physics boundary checks directly from memory for Transfer Learning...")
            try:
                physics_boundary_hits = check_physics_boundaries(
                    memory_models=model_trial_list, 
                    device=device, 
                    physics_c=physics_c, 
                    tolerance=0.01
                )
            except Exception as e:
                print(f"Warning: Could not perform optimized physics boundary checks: {e}")
        # ------------------------------------------------

        save_best_trials_weights(model_trial_list, weights_filepath_new)
        
        if save_equations:
            equations_path = os.path.join(results_dir, f"neural_network_equations_{config_id}_opt{optimizer_space_id}.txt")
            with open(equations_path, 'w') as f:
                for equation in equations: f.write(equation + "\n")

        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, t in enumerate(best_trials):
            
            orig_data = next((item for item in results_data if item.get("trial_number") == t.number), {})
            opt_exec_time = orig_data.get("trial_execution_time_seconds", 0)
            time_to_reach = orig_data.get("time_to_reach_seconds", 0)
            
            results.append({
                "config_id": config_id,
                "optimizer_space_id": optimizer_space_id,
                "optimized_params": t.params,
                
                # Carry over base hits, but add the optimized hits
                "optuna_bound_hits": orig_data.get("optuna_bound_hits", []),
                "physics_bound_hits": orig_data.get("physics_bound_hits", []),
                "optimized_optuna_bound_hits": optuna_boundary_hits.get(t.number, []),
                "optimized_physics_bound_hits": physics_boundary_hits.get(t.number, []),
                
                "fitness": t.value,
                "val_rmse": val_metrics[idx][0],
                "val_r2": val_metrics[idx][1],
                "tpe_db_path": db_path,
                "full_csv_path": env_to_train_info["csv_path"],
                "timestamp": timestamp,
                "csv_basename": os.path.basename(env_to_train_info["csv_path"]),
                "train_data_mode": train_data_mode,
                "base_config": config_id,
                "optimizer_spaces": optimizer_space_id,
                "environment": environment,
                "from_environment": base_env,
                "meas_load_kwargs": meas_load_kwargs,
                "user_metadata": user_metadata or {},
                "env_to_train_info": env_to_train_info,
                "weights_filepath_new": weights_filepath_new,
                "time_to_reach_seconds": time_to_reach,
                "trial_execution_time_seconds": opt_exec_time,
                "time_lbfgs_execution_list": [lbfgs_metrics[idx]],
                "time_lbfgs_execution": lbfgs_metrics[idx],
                "time_total_execution_2": opt_exec_time + lbfgs_metrics[idx],
                "time_total_training_2": time_to_reach + opt_exec_time + lbfgs_metrics[idx]
            })

        out_json = os.path.join(results_dir, f"results_base{config_id}_opt{optimizer_space_id}.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        
        return plot_results, equations


