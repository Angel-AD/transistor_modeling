import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


import pandas as pd
import numpy as np


def extrapolate_groups_old(df: pd.DataFrame, num_extra_start: int = 0, num_extra_end: int = 0, load_only: bool = False, use_noise: bool = False, smooth_window: int = 5, noise_limit: float = None) -> pd.DataFrame:
    """
    Extrapolates groups at the beginning (using first 3 groups) and/or 
    at the end (using last 5 groups).
    Bases voltage calculations on the Vgs_meas array, and sets Vgs as the mean.
    """
    if num_extra_start <= 0 and num_extra_end <= 0:
        return df

    # Map standard column names
    col_tn = "GroupID" if load_only else "TN"
    col_vgs = "V1_t" if load_only else "Vgs"
    col_vds = "V2_t" if load_only else "Vds"
    col_ids = "I2_t" if load_only else "Ids"
    col_igs = "I1_t" if load_only else "Igs"

    # Safely map the Vgs_meas column
    col_vgs_meas = "V1_meas_t" if load_only else "Vgs_meas"
    if col_vgs_meas not in df.columns:
        col_vgs_meas = "Vgs_meas" if "Vgs_meas" in df.columns else col_vgs

    unique_tns = df[col_tn].unique()
    global_vds_min = df[col_vds].min()
    
    new_rows_start = []
    new_rows_end = []

    # Helper function to smooth data and strictly limit noise spikes
    def clean_signal(data_array):
        if use_noise:
            return data_array 
            
        smoothed = pd.Series(data_array).rolling(window=smooth_window, center=True, min_periods=1).mean().values
        if noise_limit is not None:
            return np.clip(data_array, smoothed - noise_limit, smoothed + noise_limit)
        return smoothed

    # ==========================================
    # 1. EXTRAPOLATE START (First 3 groups)
    # ==========================================
    if num_extra_start > 0 and len(unique_tns) >= 3:
        g0 = df[df[col_tn] == unique_tns[0]]
        g2 = df[df[col_tn] == unique_tns[2]]

        # Calculate VGS step using Vgs_meas
        vgs_meas_0_med = g0[col_vgs_meas].median()
        vgs_meas_2_med = g2[col_vgs_meas].median()
        delta_vgs_start = (vgs_meas_2_med - vgs_meas_0_med) / 2.0
        
        # Grab the base Vgs_meas array (cleaned if use_noise=False)
        base_vgs_meas_0 = clean_signal(g0[col_vgs_meas].values)
        
        vds_count = len(g0)
        vds_min_orig = g0[col_vds].min()
        vds_max_orig = g0[col_vds].max()
        vds_step = (vds_max_orig - vds_min_orig) / max(1, vds_count - 1)
        vds_max_new = global_vds_min + (vds_step * (vds_count - 1))
        vds_profile_ideal = np.linspace(global_vds_min, vds_max_new, vds_count)

        vds_0 = g0[col_vds].values
        ids_0 = clean_signal(g0[col_ids].values)
        
        sort_idx_g2 = np.argsort(g2[col_vds].values)
        xp_g2 = g2[col_vds].values[sort_idx_g2]
        fp_ids_g2 = clean_signal(g2[col_ids].values[sort_idx_g2])
        
        ids_g2_interp = np.interp(vds_0, xp_g2, fp_ids_g2)
        delta_ids_start = (ids_g2_interp - ids_0) / 2.0
        
        if col_igs in df.columns:
            igs_0 = clean_signal(g0[col_igs].values)
            fp_igs_g2 = clean_signal(g2[col_igs].values[sort_idx_g2])
            igs_g2_interp = np.interp(vds_0, xp_g2, fp_igs_g2)
            delta_igs_start = (igs_g2_interp - igs_0) / 2.0

        if use_noise:
            vds_ideal_g0 = np.linspace(vds_min_orig, vds_max_orig, vds_count)
            max_vds_noise_start = np.max(np.abs(g0[col_vds].values - vds_ideal_g0))

        for i in range(num_extra_start, 0, -1):
            # Shift the measured array, then calculate Vgs as the mean
            final_vgs_meas_array = base_vgs_meas_0 - ( i * delta_vgs_start)
            final_vgs_mean = np.mean(final_vgs_meas_array)
            
            final_ids_array = np.clip(ids_0 - (i * delta_ids_start), 0.0, None)
            
            if use_noise:
                final_vds_array = np.clip(vds_profile_ideal + np.random.uniform(-max_vds_noise_start, max_vds_noise_start, vds_count), global_vds_min, None)
            else:
                final_vds_array = vds_profile_ideal
            
            temp_dict = {
                col_tn: -i, 
                col_vgs_meas: final_vgs_meas_array,
                col_vds: final_vds_array, 
                col_ids: final_ids_array
            }
            # Only assign the mean to col_vgs if it's uniquely defined
            if col_vgs != col_vgs_meas:
                temp_dict[col_vgs] = final_vgs_mean
                
            temp_df = pd.DataFrame(temp_dict)
            
            if col_igs in df.columns: 
                temp_df[col_igs] = igs_0 - (i * delta_igs_start)
                
            for col in df.columns:
                if col not in temp_df.columns: temp_df[col] = np.nan
            new_rows_start.append(temp_df)

    # ==========================================
    # 2. EXTRAPOLATE END (Last 5 groups)
    # ==========================================
    if num_extra_end > 0 and len(unique_tns) >= 5:
        last_5_tns = unique_tns[-5:]
        g_first_of_last = df[df[col_tn] == last_5_tns[0]]
        g_last = df[df[col_tn] == last_5_tns[-1]]

        # Calculate VGS step using Vgs_meas
        vgs_meas_start_of_last_med = g_first_of_last[col_vgs_meas].median()
        vgs_meas_final_med = g_last[col_vgs_meas].median()
        delta_vgs_end = (vgs_meas_final_med - vgs_meas_start_of_last_med) / 4.0
        
        # Grab the base Vgs_meas array (cleaned if use_noise=False)
        base_vgs_meas_last = clean_signal(g_last[col_vgs_meas].values)

        vds_count = len(g_last)
        vds_min_orig = g_last[col_vds].min()
        vds_max_orig = g_last[col_vds].max()
        vds_step = (vds_max_orig - vds_min_orig) / max(1, vds_count - 1)
        vds_max_new = global_vds_min + (vds_step * (vds_count - 1))
        vds_profile_ideal = np.linspace(global_vds_min, vds_max_new, vds_count)

        vds_last = g_last[col_vds].values
        ids_last = clean_signal(g_last[col_ids].values)
        
        sort_idx = np.argsort(g_first_of_last[col_vds].values)
        xp = g_first_of_last[col_vds].values[sort_idx]
        fp_ids = clean_signal(g_first_of_last[col_ids].values[sort_idx])
        
        ids_first_of_last_interp = np.interp(vds_last, xp, fp_ids)
        delta_ids_end = (ids_last - ids_first_of_last_interp) / 4.0
        
        if col_igs in df.columns:
            igs_last = clean_signal(g_last[col_igs].values)
            fp_igs = clean_signal(g_first_of_last[col_igs].values[sort_idx])
            igs_first_of_last_interp = np.interp(vds_last, xp, fp_igs)
            delta_igs_end = (igs_last - igs_first_of_last_interp) / 4.0

        if use_noise:
            vds_ideal_glast = np.linspace(vds_min_orig, vds_max_orig, vds_count)
            max_vds_noise_end = np.max(np.abs(g_last[col_vds].values - vds_ideal_glast))

        max_tn = unique_tns.max()

        for i in range(1, num_extra_end + 1):
            # Shift the measured array, then calculate Vgs as the mean
            final_vgs_meas_array = base_vgs_meas_last + (i * delta_vgs_end)
            final_vgs_mean = np.mean(final_vgs_meas_array)
            
            final_ids_array = ids_last + (i * delta_ids_end)
            
            if use_noise:
                final_vds_array = np.clip(vds_profile_ideal + np.random.uniform(-max_vds_noise_end, max_vds_noise_end, vds_count), global_vds_min, None)
            else:
                final_vds_array = vds_profile_ideal
            
            temp_dict = {
                col_tn: max_tn + i, 
                col_vgs_meas: final_vgs_meas_array,
                col_vds: final_vds_array, 
                col_ids: final_ids_array
            }
            if col_vgs != col_vgs_meas:
                temp_dict[col_vgs] = final_vgs_mean
                
            temp_df = pd.DataFrame(temp_dict)
            
            if col_igs in df.columns: 
                temp_df[col_igs] = igs_last + (i * delta_igs_end)
                
            for col in df.columns:
                if col not in temp_df.columns: temp_df[col] = np.nan
            new_rows_end.append(temp_df)

    # ==========================================
    # 3. COMBINE AND RENUMBER
    # ==========================================
    if new_rows_start or new_rows_end:
        dfs_to_concat = new_rows_start + [df] + new_rows_end
        df_combined = pd.concat(dfs_to_concat, ignore_index=True)
        df_combined = df_combined[df.columns] 
        
        new_unique = df_combined[col_tn].unique()
        tn_mapping = {old_tn: new_tn for new_tn, old_tn in enumerate(new_unique, start=1)}
        df_combined[col_tn] = df_combined[col_tn].map(tn_mapping)
        
        return df_combined

    return df


def extrapolate_groups(df: pd.DataFrame, num_extra_start: int = 0, num_extra_end: int = 0, load_only: bool = False, use_noise: bool = False, smooth_window: int = 5, noise_limit: float = None, min_vgs: float = None) -> pd.DataFrame:
    """
    Extrapolates groups at the beginning and/or at the end.
    Calculates the Vgs step from adjacent groups, shifts the Vgs_meas array
    by that step, and sets Vgs to the mean.

    If `min_vgs` is given, `num_extra_start` is auto-computed so that the
    new most-negative group's mean Vgs is <= `min_vgs` (e.g. min_vgs=-4.0).
    Any explicit `num_extra_start` is ignored in that case.
    """
    # Map standard column names
    col_tn = "GroupID" if load_only else "TN"
    col_vgs = "V1_t" if load_only else "Vgs"
    col_vds = "V2_t" if load_only else "Vds"
    col_ids = "I2_t" if load_only else "Ids"
    col_igs = "I1_t" if load_only else "Igs"

    # Safely map the Vgs_meas column
    col_vgs_meas = "V1_meas_t" if load_only else "Vgs_meas"
    if col_vgs_meas not in df.columns:
        col_vgs_meas = "Vgs_meas" if "Vgs_meas" in df.columns else col_vgs

    unique_tns = df[col_tn].unique()
    global_vds_min = df[col_vds].min()

    # --- Auto-compute num_extra_start from min_vgs target ---
    if min_vgs is not None and len(unique_tns) >= 2:
        g0_vgs = df[df[col_tn] == unique_tns[0]][col_vgs_meas].mean()
        g1_vgs = df[df[col_tn] == unique_tns[1]][col_vgs_meas].mean()
        step = g1_vgs - g0_vgs  # >0 if data starts at most-negative Vgs
        print(f"[extrapolate_groups] min_vgs={min_vgs}, col_vgs_meas='{col_vgs_meas}', "
              f"g0_vgs={g0_vgs:.3f}, g1_vgs={g1_vgs:.3f}, step={step:.3f}")
        if step > 0 and g0_vgs > min_vgs:
            # Need n such that g0_vgs - n*step <= min_vgs  =>  n >= (g0_vgs - min_vgs) / step
            num_extra_start = int(np.ceil((g0_vgs - min_vgs) / step))
            print(f"[extrapolate_groups] min_vgs={min_vgs}: g0_vgs={g0_vgs:.3f}, step={step:.3f} -> num_extra_start={num_extra_start}")
        else:
            print(f"[extrapolate_groups] SKIP: step<=0 or g0_vgs<=min_vgs (no need to extrapolate)")
            num_extra_start = 0

    if num_extra_start <= 0 and num_extra_end <= 0:
        return df
    
    new_rows_start = []
    new_rows_end = []

    def clean_signal(data_array):
        if use_noise:
            return data_array 
            
        smoothed = pd.Series(data_array).rolling(window=smooth_window, center=True, min_periods=1).mean().values
        if noise_limit is not None:
            return np.clip(data_array, smoothed - noise_limit, smoothed + noise_limit)
        return smoothed

    # ==========================================
    # 1. EXTRAPOLATE START
    # ==========================================
    if num_extra_start > 0 and len(unique_tns) >= 2:
        g0 = df[df[col_tn] == unique_tns[0]]
        g1 = df[df[col_tn] == unique_tns[1]]

        # 1. Calculate the step from the first two existing groups
        vgs_step_start = g1[col_vgs_meas].mean() - g0[col_vgs_meas].mean()
        
        # Grab the base Vgs_meas array
        base_vgs_meas_0 = clean_signal(g0[col_vgs_meas].values)
        
        vds_count = len(g0)
        vds_min_orig = g0[col_vds].min()
        vds_max_orig = g0[col_vds].max()
        vds_step = (vds_max_orig - vds_min_orig) / max(1, vds_count - 1)
        vds_max_new = global_vds_min + (vds_step * (vds_count - 1))
        vds_profile_ideal = np.linspace(global_vds_min, vds_max_new, vds_count)

        vds_0 = g0[col_vds].values
        ids_0 = clean_signal(g0[col_ids].values)
        
        sort_idx_g1 = np.argsort(g1[col_vds].values)
        xp_g1 = g1[col_vds].values[sort_idx_g1]
        fp_ids_g1 = clean_signal(g1[col_ids].values[sort_idx_g1])
        
        ids_g1_interp = np.interp(vds_0, xp_g1, fp_ids_g1)
        delta_ids_start = ids_g1_interp - ids_0
        
        if col_igs in df.columns:
            igs_0 = clean_signal(g0[col_igs].values)
            fp_igs_g1 = clean_signal(g1[col_igs].values[sort_idx_g1])
            igs_g1_interp = np.interp(vds_0, xp_g1, fp_igs_g1)
            delta_igs_start = igs_g1_interp - igs_0

        if use_noise:
            vds_ideal_g0 = np.linspace(vds_min_orig, vds_max_orig, vds_count)
            max_vds_noise_start = np.max(np.abs(g0[col_vds].values - vds_ideal_g0))

        for i in range(num_extra_start, 0, -1):
            # 2. Subtract the step to calculate new Vgs_meas values
            final_vgs_meas_array = base_vgs_meas_0 - (i * vgs_step_start) + .02
            
            # 3. Vgs is the mean
            final_vgs_mean = np.mean(final_vgs_meas_array)
            
            final_ids_array = np.clip(ids_0 - (i * delta_ids_start), 0.0, None)
            
            if use_noise:
                final_vds_array = np.clip(vds_profile_ideal + np.random.uniform(-max_vds_noise_start, max_vds_noise_start, vds_count), global_vds_min, None)
            else:
                final_vds_array = vds_profile_ideal
            
            temp_dict = {
                col_tn: -i, 
                col_vgs_meas: final_vgs_meas_array,
                col_vds: final_vds_array, 
                col_ids: final_ids_array
            }
            if col_vgs != col_vgs_meas:
                temp_dict[col_vgs] = final_vgs_mean
                
            temp_df = pd.DataFrame(temp_dict)
            
            if col_igs in df.columns: 
                temp_df[col_igs] = igs_0 - (i * delta_igs_start)
                
            for col in df.columns:
                if col not in temp_df.columns: temp_df[col] = np.nan
            new_rows_start.append(temp_df)

    # ==========================================
    # 2. EXTRAPOLATE END
    # ==========================================
    if num_extra_end > 0 and len(unique_tns) >= 2:
        g_last = df[df[col_tn] == unique_tns[-1]]
        g_prev = df[df[col_tn] == unique_tns[-2]]

        # 1. Calculate the step from the last two existing groups
        vgs_step_end = g_last[col_vgs_meas].mean() - g_prev[col_vgs_meas].mean()
        
        # Grab the base Vgs_meas array
        base_vgs_meas_last = clean_signal(g_last[col_vgs_meas].values)

        vds_count = len(g_last)
        vds_min_orig = g_last[col_vds].min()
        vds_max_orig = g_last[col_vds].max()
        vds_step = (vds_max_orig - vds_min_orig) / max(1, vds_count - 1)
        vds_max_new = global_vds_min + (vds_step * (vds_count - 1))
        vds_profile_ideal = np.linspace(global_vds_min, vds_max_new, vds_count)

        vds_last = g_last[col_vds].values
        ids_last = clean_signal(g_last[col_ids].values)
        
        sort_idx = np.argsort(g_prev[col_vds].values)
        xp = g_prev[col_vds].values[sort_idx]
        fp_ids = clean_signal(g_prev[col_ids].values[sort_idx])
        
        ids_prev_interp = np.interp(vds_last, xp, fp_ids)
        delta_ids_end = ids_last - ids_prev_interp
        
        if col_igs in df.columns:
            igs_last = clean_signal(g_last[col_igs].values)
            fp_igs = clean_signal(g_prev[col_igs].values[sort_idx])
            igs_prev_interp = np.interp(vds_last, xp, fp_igs)
            delta_igs_end = igs_last - igs_prev_interp

        if use_noise:
            vds_ideal_glast = np.linspace(vds_min_orig, vds_max_orig, vds_count)
            max_vds_noise_end = np.max(np.abs(g_last[col_vds].values - vds_ideal_glast))

        max_tn = unique_tns.max()

        for i in range(1, num_extra_end + 1):
            # 2. Add the step to calculate new Vgs_meas values
            final_vgs_meas_array = base_vgs_meas_last + (i * vgs_step_end)
            
            # 3. Vgs is the mean
            final_vgs_mean = np.mean(final_vgs_meas_array)
            
            final_ids_array = ids_last + (i * delta_ids_end)
            
            if use_noise:
                final_vds_array = np.clip(vds_profile_ideal + np.random.uniform(-max_vds_noise_end, max_vds_noise_end, vds_count), global_vds_min, None)
            else:
                final_vds_array = vds_profile_ideal
            
            temp_dict = {
                col_tn: max_tn + i, 
                col_vgs_meas: final_vgs_meas_array,
                col_vds: final_vds_array, 
                col_ids: final_ids_array
            }
            if col_vgs != col_vgs_meas:
                temp_dict[col_vgs] = final_vgs_mean
                
            temp_df = pd.DataFrame(temp_dict)
            
            if col_igs in df.columns: 
                temp_df[col_igs] = igs_last + (i * delta_igs_end)
                
            for col in df.columns:
                if col not in temp_df.columns: temp_df[col] = np.nan
            new_rows_end.append(temp_df)

    # ==========================================
    # 3. COMBINE AND RENUMBER
    # ==========================================
    if new_rows_start or new_rows_end:
        dfs_to_concat = new_rows_start + [df] + new_rows_end
        df_combined = pd.concat(dfs_to_concat, ignore_index=True)
        df_combined = df_combined[df.columns] 
        
        new_unique = df_combined[col_tn].unique()
        tn_mapping = {old_tn: new_tn for new_tn, old_tn in enumerate(new_unique, start=1)}
        df_combined[col_tn] = df_combined[col_tn].map(tn_mapping)
        
        return df_combined

    return df


def meas_load_full(path_to_csv, file_type: str = "auriga", keep_every_N_group: int = 0, remove_negative_vds: int = 0, remove_negative_ids: int = 0, load_only: bool = False, clip_vds: float = None, num_extrapolate_groups_start: int = 0, num_extrapolate_groups_end: int = 0, use_noise: bool = True, smooth_window=5, noise_limit=0.005, min_vgs: float = None, add_zero_vds: bool = False):
    """
        keep_every_N_group: keep every N-th TN (N given by keep_every_N_group), starting from the first
        num_extrapolate_groups_start: add N artificially extrapolated groups at the beginning based on the first 3 groups
        num_extrapolate_groups_end: add N artificially extrapolated groups at the end based on the last 5 groups
        add_zero_vds: if True, overwrites the first (lowest-Vds) row of every TN group, forcing Vds=0 and Ids=0 to pin each curve to the origin. Done in place for every group unconditionally — no row is added, and that group's original first sample is discarded.
    """
    if file_type == "auriga":
        dcvi_data = pd.read_csv(path_to_csv, skiprows=106)
        var_names = ["Item", "TN", "PT", "Vout Q", "Iout Q", "Vds", "Ids", "Vin Q", "Iin Q", "Vgs", "Igs"]
    elif file_type == "ads":
        dcvi_data = pd.read_csv(path_to_csv, skiprows=1)
        var_names = ["Item", "Vgs", "Igs", "Vds", "Ids", "TN"]
        if load_only:
            var_names = ["index", "V1_t","I1_t","V2_t","I2_t","GroupID"]
    elif file_type == "auriga_custom":
        dcvi_data = pd.read_csv(path_to_csv, skiprows=1)
        var_names=["TN","Vds","Ids","Vgs","Igs"]
        if load_only:
            var_names = ["GroupID", "V2_t","I2_t","V1_t","I1_t"]
        dcvi_data = dcvi_data.iloc[:, :len(var_names)]

    add_fake_vals = False
    number_of_points = 50
    fake_vgs = np.arange(-7, -3.2, 0.2)
    max_vds = 30
    
    T = pd.DataFrame(dcvi_data.values, columns=var_names)

    # 🚨 Inject extrapolation logic here before any other processing happens
    if num_extrapolate_groups_start > 0 or num_extrapolate_groups_end > 0 or min_vgs is not None:
        _vgs_col_dbg = "V1_t" if load_only else "Vgs"
        _tn_col_dbg = "GroupID" if load_only else "TN"
        print(f"[meas_load_full] BEFORE extrapolate: min_vgs={min_vgs}, "
              f"Vgs range = [{T[_vgs_col_dbg].min():.3f}, {T[_vgs_col_dbg].max():.3f}], "
              f"n_groups = {T[_tn_col_dbg].nunique()}")
        T = extrapolate_groups(T, num_extra_start=num_extrapolate_groups_start, num_extra_end=num_extrapolate_groups_end, load_only=load_only, use_noise=use_noise, smooth_window=smooth_window, noise_limit=noise_limit, min_vgs=min_vgs)
        print(f"[meas_load_full] AFTER  extrapolate: "
              f"Vgs range = [{T[_vgs_col_dbg].min():.3f}, {T[_vgs_col_dbg].max():.3f}], "
              f"n_groups = {T[_tn_col_dbg].nunique()}")
    
    if file_type=="auriga_custom":
        if load_only:
            T["index"] = T.index
        else:
            T["Item"] = T.index
            
    if keep_every_N_group > 0:
        T = filter_table_by_group(T, keep_every_N_group, load_only)
        
    # --- EARLY RETURN PATH ---
    if load_only:
        T["index"] = np.arange(1, len(T) + 1)
        if remove_negative_vds is not None:
            T = T[T["V2_t"] >= remove_negative_vds]
            
        # 🚨 ADDED CLIP LOGIC HERE (V2_t is Vds)
        if clip_vds is not None:
            T["V2_t"] = T["V2_t"].clip(lower=clip_vds)
            
        if remove_negative_ids is not None:
            T.loc[T["I2_t"] < remove_negative_ids, "I2_t"] = 0
        return T

    # Prepare containers for data
    max_index = int(T["TN"].max() - T["TN"].min() + 1)
    Ids, Vds, Igs, Vgs = [], [], [], []
    Vgs_cleaned = np.ones(len(T))

    counter = 0
    counter_clean = 0
    for n in range(int(T["TN"].min()), int(T["TN"].max()) + 1):
        mask = T["TN"] == n
        if not mask.any(): 
            continue

        Ids_m = T["Ids"][mask]
        Vgs_m = T["Vgs"][mask]
        Vds_m = T["Vds"][mask]
        Vgs_l = len(Vgs_m)
        
        Ids.append(Ids_m.values)
        Vds.append(Vds_m.values)
        Vgs.append(Vgs_m.values)
        
        Vgs_cleaned_m = np.ones(Vgs_l) * np.median(Vgs_m)
        Vgs_cleaned[counter_clean:counter_clean + Vgs_l] = Vgs_cleaned_m
        
        Igs.append(T["Igs"][T["TN"] == n].values)
        
        counter += 1
        counter_clean += Vgs_l

    T_cleaned = pd.DataFrame({
        "TN": T["TN"],
        "Vgs": Vgs_cleaned,
        "Vgs_meas": T["Vgs"],
        "Vds": T["Vds"],
        "Ids": T["Ids"],
        "Igs": T["Igs"]
    })

    if add_fake_vals:
        repeated_vector_vgs = np.repeat(fake_vgs, number_of_points)
        repeated_vector_vds = np.tile(np.linspace(0, max_vds, number_of_points), len(fake_vgs))
        repeated_vector_ids_igs = np.zeros(len(repeated_vector_vds))
        numbers_TN = np.arange(T_cleaned["TN"].max() + 1, T_cleaned["TN"].max() + len(fake_vgs) + 1)
        repeated_vector_TN = np.repeat(numbers_TN, number_of_points)
        
        T_cleaned_2 = pd.DataFrame({
            "TN": repeated_vector_TN,
            "Vgs": repeated_vector_vgs,
            "Vgs_meas": repeated_vector_vgs,
            "Vds": repeated_vector_vds,
            "Ids": repeated_vector_ids_igs,
            "Igs": repeated_vector_ids_igs
        })
        T_cleaned = pd.concat([T_cleaned, T_cleaned_2], ignore_index=True)

    if add_zero_vds:
        modified = 0
        for tn, grp in T_cleaned.groupby('TN'):
            first_idx = grp.index[0]
            T_cleaned.at[first_idx, 'Vds'] = 0.0
            T_cleaned.at[first_idx, 'Ids'] = 0.0
            modified += 1
        T_cleaned = T_cleaned.sort_values(['TN', 'Vds']).reset_index(drop=True)
        print(f"[meas_load] add_zero_vds: set Vds=0, Ids=0 on first row of {modified} groups")

    T_cleaned["index"] = np.arange(1, len(T_cleaned) + 1)

    if remove_negative_vds is not None:
        T_cleaned = T_cleaned[T_cleaned["Vds"] >= remove_negative_vds]

    # 🚨 ADDED CLIP LOGIC HERE (Vds)
    if clip_vds is not None:
        T_cleaned["Vds"] = T_cleaned["Vds"].clip(lower=clip_vds)

    if remove_negative_ids is not None:
        T_cleaned.loc[T_cleaned["Ids"] < remove_negative_ids, "Ids"] = 0

    return T_cleaned


def meas_load(path_to_csv, file_type: str = "auriga", keep_every_N_group: int = 0, remove_negative_vds: int = 0, remove_negative_ids: int = 0, test_percent: float = 0.3, clip_vds: float = None, num_extrapolate_groups_start: int = 0, num_extrapolate_groups_end: int = 0, use_noise: bool = False, smooth_window=5, noise_limit=0.005, min_vgs: float = None, add_zero_vds: bool = False):
    """
        keep_every_N_group: keep every N-th TN (N given by keep_every_N_group), starting from the first, it additionally splits the data into Train and Test sets.
        min_vgs: if given (e.g. -4.0), auto-extrapolates extra start-groups so the most-negative Vgs reaches this floor (overrides num_extrapolate_groups_start).
        add_zero_vds: if True, overwrites the first (lowest-Vds) row of every TN group, forcing Vds=0 and Ids=0 to pin each curve to the origin. Done in place for every group unconditionally — no row is added, and that group's original first sample is discarded.
    """
    # 🚨 Passed down extrapolation arguments to meas_load_full
    T_cleaned = meas_load_full(path_to_csv, file_type, keep_every_N_group, remove_negative_vds, remove_negative_ids, False, clip_vds=clip_vds, num_extrapolate_groups_start=num_extrapolate_groups_start, num_extrapolate_groups_end=num_extrapolate_groups_end, use_noise=use_noise, smooth_window=smooth_window, noise_limit=noise_limit, min_vgs=min_vgs, add_zero_vds=add_zero_vds)

    T_train, T_test = split_test_train_data(T_cleaned, test_percent)

    return T_train, T_test


def split_test_train_data(T_data, test_percent):
    n = len(T_data["Ids"])
    
    # Handle the case when no test set is requested
    if test_percent == 0:
        T_train = T_data
        T_test = T_data[:0]   # same structure, but empty
        return T_train, T_test
    
    test_split = np.zeros(n, dtype=bool)
    nth_point = 1 / test_percent
    one_indices = np.arange(2, n, int(nth_point))
    test_split[one_indices.astype(int)] = True
    train_split = ~test_split

    T_train = T_data[train_split]
    T_test = T_data[test_split]
    return T_train, T_test


def filter_table_by_group(df: pd.DataFrame, N: int, load_only: bool = False) -> pd.DataFrame:
    """
    Filters a DataFrame to include every N-th group by 'TN' value.
    Always includes the first and last TN groups.
    """
    # Get unique TN values (assumed to define the groups)
    if load_only:
        tn_values = df['GroupID'].unique()
    else:
        tn_values = df['TN'].unique()
    
    # Always keep every N-th TN, starting from the first
    selected_tn = tn_values[::N]

    # Ensure the last TN is included
    if tn_values[-1] not in selected_tn:
        selected_tn = list(selected_tn) + [tn_values[-1]]

    # Filter the DataFrame to include only selected TN values
    if load_only:
        filtered_df = df[df['GroupID'].isin(selected_tn)]
    else:
        filtered_df = df[df['TN'].isin(selected_tn)]

    return filtered_df


if __name__ == "__main__":
    path_to_csv = r"C:\Users\Angel\Documents\project1\cgh40010-finer-low-vg.csv"
    T_train, T_test = meas_load(path_to_csv)
    # Plot Test data
    plt.figure(1)
    for n in range(int(T_test["TN"].min()), int(T_test["TN"].max()) + 1):
        Ids_m = T_test["Ids"][T_test["TN"] == n]
        Vgs_m = T_test["Vgs"][T_test["TN"] == n]
        Vds_m = T_test["Vds"][T_test["TN"] == n]
        Vgs_med = np.median(Vgs_m)
        plt.plot(Vds_m, Ids_m, label=f"Vgs={Vgs_med}")
    plt.title("Test Data")
    plt.xlabel("Vds")
    plt.ylabel("Ids")
    plt.legend()
    plt.show(block=False)

    # Plot Train data
    plt.figure(2)
    for n in range(int(T_train["TN"].min()), int(T_train["TN"].max()) + 1):
        Ids_m = T_train["Ids"][T_train["TN"] == n]
        Vgs_m = T_train["Vgs"][T_train["TN"] == n]
        Vds_m = T_train["Vds"][T_train["TN"] == n]
        Vgs_med = np.median(Vgs_m)
        plt.plot(Vds_m, Ids_m, label=f"Vgs={Vgs_med}")
    plt.title("Train Data")
    plt.xlabel("Vds")
    plt.ylabel("Ids")
    plt.legend()
    plt.show(block=False)
    print("Outside not ps1")
    if not hasattr(sys, 'ps1'):
        print("Inside not ps1")
        input("Press enter to continue...")