import os
import glob
import numpy as np
import pandas as pd

def load_and_preprocess_sequences(
    master_meals_path="data/processed/master_meals_features.csv",
    processed_dir="data/processed",
    pre_window_mins=60,
    post_window_mins=120,
    downsample_factor=5
):
    """
    Extracts multi-horizon continuous time-series trajectories and static metadata for all meals.
    
    Returns:
    - pre_sequences: np.ndarray of shape (N, 60, 3) -> [Unified GL, HR, METs]
    - static_features: np.ndarray of shape (N, D_static) -> Meal + Clinical + Gut Microbiome features
    - target_trajectories: np.ndarray of shape (N, 24) -> Continuous post-meal glucose curve (sampled every 5 mins)
    - meal_metadata: pd.DataFrame with subject IDs and timestamps for Group K-Fold splitting.
    - feature_names: list of static feature names.
    """
    master_df = pd.read_csv(master_meals_path)
    master_df['timestamp'] = pd.to_datetime(master_df['timestamp'])
    
    # Identify static feature columns (exclude target variables, metadata, raw timestamps)
    exclude_cols = [
        'subject', 'timestamp', 'meal_type', 'target_peak_glucose',
        'target_glucose_rise', 'target_time_to_peak', 'target_iauc', 'Self-identify'
    ]
    
    # One-hot encode Gender if present
    df_static = master_df.copy()
    if 'Gender' in df_static.columns:
        df_static['Gender_M'] = (df_static['Gender'] == 'M').astype(float)
        exclude_cols.append('Gender')
        
    static_cols = [col for col in df_static.columns if col not in exclude_cols]
    
    # Fill remaining NaNs in static features with median
    for col in static_cols:
        if df_static[col].isna().sum() > 0:
            df_static[col] = df_static[col].fillna(df_static[col].median())
            
    pre_sequences = []
    static_features = []
    target_trajectories = []
    valid_records = []
    
    # Group master df by subject for fast lookup
    for subject_id, group in master_df.groupby('subject'):
        formatted_subj = f"CGMacros-{int(subject_id):03d}"
        subj_file = os.path.join(processed_dir, f"{formatted_subj}_cleaned.csv")
        
        if not os.path.exists(subj_file):
            print(f"Warning: Cleaned file for {formatted_subj} not found. Skipping.")
            continue
            
        subj_ts_df = pd.read_csv(subj_file)
        subj_ts_df['Timestamp'] = pd.to_datetime(subj_ts_df['Timestamp'])
        subj_ts_df = subj_ts_df.sort_values('Timestamp').reset_index(drop=True)
        
        # Pre-fill missing HR or METs in time-series
        subj_ts_df['HR'] = subj_ts_df['HR'].ffill().bfill().fillna(70.0)
        subj_ts_df['METs'] = subj_ts_df['METs'].fillna(10.0)
        subj_ts_df['Unified GL'] = subj_ts_df['Unified GL'].interpolate(method='linear', limit=15).ffill().bfill()
        
        ts_times = subj_ts_df['Timestamp'].values
        gl_vals = subj_ts_df['Unified GL'].values
        hr_vals = subj_ts_df['HR'].values
        mets_vals = subj_ts_df['METs'].values
        
        for idx, row in group.iterrows():
            meal_time = row['timestamp']
            
            # Find nearest index in time-series
            time_diffs = np.abs((ts_times - np.datetime64(meal_time)) / np.timedelta64(1, 'm'))
            nearest_idx = np.argmin(time_diffs)
            
            if time_diffs[nearest_idx] > 5:
                # Meal timestamp too far from sensor reading
                continue
                
            # Extract 60-minute pre-meal window (nearest_idx - 60 to nearest_idx)
            start_pre = nearest_idx - pre_window_mins
            if start_pre < 0:
                continue
                
            pre_gl = gl_vals[start_pre:nearest_idx]
            pre_hr = hr_vals[start_pre:nearest_idx]
            pre_mets = mets_vals[start_pre:nearest_idx]
            
            if len(pre_gl) != pre_window_mins:
                continue
                
            # Extract 120-minute post-meal window (nearest_idx to nearest_idx + 120)
            end_post = nearest_idx + post_window_mins
            if end_post > len(gl_vals):
                continue
                
            post_gl = gl_vals[nearest_idx:end_post]
            if len(post_gl) != post_window_mins:
                continue
                
            # Downsample post-meal curve to 5-minute sampling rate (24 steps)
            target_seq = post_gl[::downsample_factor]
            if len(target_seq) != (post_window_mins // downsample_factor):
                continue
                
            # Dynamic pre-sequence (60, 3) -> [Unified GL, HR, METs]
            seq_matrix = np.column_stack([pre_gl, pre_hr, pre_mets])
            
            # Baseline glucose G_0 at meal onset (last minute of pre_gl)
            baseline_g0 = pre_gl[-1]
            
            # Static feature vector
            static_vec = df_static.loc[idx, static_cols].values.astype(np.float32)
            
            pre_sequences.append(seq_matrix)
            static_features.append(static_vec)
            target_trajectories.append(target_seq.astype(np.float32))
            
            valid_records.append({
                'subject': subject_id,
                'meal_idx': idx,
                'timestamp': meal_time,
                'baseline_g0': baseline_g0
            })
            
    pre_sequences = np.array(pre_sequences, dtype=np.float32)
    static_features = np.array(static_features, dtype=np.float32)
    target_trajectories = np.array(target_trajectories, dtype=np.float32)
    metadata_df = pd.DataFrame(valid_records)
    
    baseline_g0s = metadata_df['baseline_g0'].values.astype(np.float32)
    target_deltas = target_trajectories - baseline_g0s[:, None]
    
    print(f"Extracted dataset: {len(valid_records)} sequences across {metadata_df['subject'].nunique()} subjects.")
    print(f"Pre-sequence shape: {pre_sequences.shape}")
    print(f"Static features shape: {static_features.shape}")
    print(f"Target trajectory shape: {target_trajectories.shape}")
    print(f"Target deltas shape: {target_deltas.shape}")
    
    return pre_sequences, static_features, target_trajectories, target_deltas, baseline_g0s, metadata_df, static_cols

if __name__ == "__main__":
    X_seq, X_stat, Y_traj, Y_delta, G0, meta, cols = load_and_preprocess_sequences()

