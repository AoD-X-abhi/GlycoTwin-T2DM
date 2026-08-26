import os
import glob
import pandas as pd
import numpy as np

def clean_time_series(df):
    """
    Cleans the raw 1-minute time-series data:
    1. Standardizes/normalizes varying column names.
    2. Parses Timestamps.
    3. Combines Libre and Dexcom glucose columns into a Unified Glucose Level.
    4. Imputes short gaps in glucose and heart rate.
    5. Fills activity gaps with baseline values.
    """
    df = df.copy()
    
    if 'Intensity' in df.columns and 'METs' not in df.columns:
        df['METs'] = df['Intensity']
    if 'Steps' in df.columns and 'Calories (Activity)' not in df.columns:
        df['Calories (Activity)'] = df['Steps']
    if 'Amount Consumed' not in df.columns:
        df['Amount Consumed'] = 1.0
        
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    
    
    df['Unified GL'] = df[['Libre GL', 'Dexcom GL']].mean(axis=1)
    
    df['Unified GL'] = df['Unified GL'].fillna(df['Libre GL']).fillna(df['Dexcom GL'])
    
    df['Unified GL'] = df['Unified GL'].interpolate(method='linear', limit=15)
    
    df['HR'] = df['HR'].interpolate(method='linear', limit=10)
    df['HR'] = df['HR'].ffill(limit=30) # Allow up to 30 mins forward fill for minor watch disconnections
    
    # 5. Clean Physical Activity
    # METs baseline resting value is 10 (which is 1.0 MET). If missing, set to 10.
    df['METs'] = df['METs'].fillna(10.0)
    df['Calories (Activity)'] = df['Calories (Activity)'].fillna(0.0)
    
    return df

def calculate_slope(series):
    """
    Calculates the linear slope of a series using simple regression.
    Returns 0.0 if there are insufficient non-null values.
    """
    y = series.dropna()
    if len(y) < 5:  # Require at least 5 points to compute a trend
        return 0.0
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

def calculate_iauc(y, dx=1.0):
    """
    Calculates the incremental Area Under the Curve (iAUC) using the trapezoidal rule.
    Handles NumPy 2.0 compatibility by avoiding np.trapz.
    """
    if len(y) < 2:
        return 0.0
    return dx * (np.sum(y) - 0.5 * (y[0] + y[-1]))

def extract_meal_features_and_targets(df, subject_id, window_pre_mins=60, window_post_mins=120):
    """
    Identifies meal events and extracts:
    - Pre-meal features (historical glucose and activity trend).
    - Meal macronutrient features.
    - Post-meal targets (Peak, Time-to-Peak, iAUC).
    """
    # Identify rows where a meal was logged (macronutrients are not null)
    meal_indices = df[df['Carbs'].notna() & (df['Carbs'] > 0)].index
    
    meal_records = []
    
    for idx in meal_indices:
        meal_time = df.loc[idx, 'Timestamp']
        
        # 1. Scale macronutrients by Amount Consumed if available
        # It can be logged as a percentage (e.g. 100.0) or fraction (e.g. 1.0)
        amt_consumed = df.loc[idx, 'Amount Consumed']
        if pd.isna(amt_consumed) or amt_consumed <= 0:
            amt_consumed = 1.0  # Default to full consumption if missing
        elif amt_consumed > 1.0:
            amt_consumed = amt_consumed / 100.0
            
        carbs = df.loc[idx, 'Carbs'] * amt_consumed
        protein = df.loc[idx, 'Protein'] * amt_consumed if pd.notna(df.loc[idx, 'Protein']) else 0.0
        fat = df.loc[idx, 'Fat'] * amt_consumed if pd.notna(df.loc[idx, 'Fat']) else 0.0
        fiber = df.loc[idx, 'Fiber'] * amt_consumed if pd.notna(df.loc[idx, 'Fiber']) else 0.0
        calories = df.loc[idx, 'Calories'] * amt_consumed if pd.notna(df.loc[idx, 'Calories']) else 0.0
        meal_type = df.loc[idx, 'Meal Type']
        
        # 2. Extract Pre-Meal History Window (e.g., idx - window_pre_mins to idx)
        start_pre_idx = max(0, idx - window_pre_mins)
        pre_df = df.loc[start_pre_idx:idx]
        
        # Pre-meal features
        baseline_glucose = df.loc[idx, 'Unified GL']
        if pd.isna(baseline_glucose):
            continue  # Cannot compute PPGR if we don't have baseline glucose
            
        pre_mean_glucose = pre_df['Unified GL'].mean()
        pre_sd_glucose = pre_df['Unified GL'].std()
        pre_glucose_slope = calculate_slope(pre_df['Unified GL'])
        
        pre_mean_hr = pre_df['HR'].mean()
        pre_sum_mets = pre_df['METs'].sum()
        pre_sum_calories = pre_df['Calories (Activity)'].sum()
        
        # 3. Extract Post-Meal Excursion Window (e.g., idx to idx + window_post_mins)
        end_post_idx = min(len(df) - 1, idx + window_post_mins)
        post_df = df.loc[idx:end_post_idx]
        
        # Check if we have sufficient glucose data post-meal
        post_glucose = post_df['Unified GL']
        if post_glucose.isna().mean() > 0.3 or len(post_glucose) < (window_post_mins * 0.7):
            # Skip meals with more than 30% missing glucose in the post-meal window
            continue
            
        # Target variables
        peak_glucose = post_glucose.max()
        glucose_rise = peak_glucose - baseline_glucose
        
        # Time-to-peak in minutes
        peak_idx = post_glucose.idxmax()
        time_to_peak = int((df.loc[peak_idx, 'Timestamp'] - meal_time).total_seconds() / 60.0)
        
        # Incremental Area Under the Curve (iAUC) using trapezoidal rule
        # Subtract baseline, clip negative values to 0
        incremental_glucose = (post_glucose - baseline_glucose).clip(lower=0).fillna(0).values
        iauc = calculate_iauc(incremental_glucose, dx=1.0)
        
        # Temporal cyclic encodings
        hour = meal_time.hour
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        
        record = {
            'subject': subject_id,
            'timestamp': meal_time,
            'meal_type': meal_type,
            'carbs': carbs,
            'protein': protein,
            'fat': fat,
            'fiber': fiber,
            'calories': calories,
            'sin_hour': sin_hour,
            'cos_hour': cos_hour,
            'baseline_glucose': baseline_glucose,
            'pre_mean_glucose': pre_mean_glucose,
            'pre_sd_glucose': pre_sd_glucose,
            'pre_glucose_slope': pre_glucose_slope,
            'pre_mean_hr': pre_mean_hr,
            'pre_sum_mets': pre_sum_mets,
            'pre_sum_calories': pre_sum_calories,
            'target_peak_glucose': peak_glucose,
            'target_glucose_rise': glucose_rise,
            'target_time_to_peak': time_to_peak,
            'target_iauc': iauc
        }
        
        meal_records.append(record)
        
    return pd.DataFrame(meal_records)

def process_subject(subject_dir, output_dir):
    """
    Processes a single subject directory:
    - Reads the raw subject CSV.
    - Cleans the continuous time series.
    - Saves the cleaned continuous time-series CSV.
    - Returns the extracted meal dataframe.
    """
    subject_name = os.path.basename(subject_dir)
    # Find the CSV file inside the directory
    csv_files = glob.glob(os.path.join(subject_dir, f"{subject_name}.csv"))
    if not csv_files:
        print(f"Warning: No CSV found for {subject_name}")
        return pd.DataFrame()
        
    csv_file = csv_files[0]
    df_raw = pd.read_csv(csv_file)
    # Strip leading and trailing whitespace from column headers to prevent KeyErrors
    df_raw.columns = df_raw.columns.str.strip()
    
    # Extract subject ID (integer)
    try:
        subject_id = int(subject_name.split('-')[-1])
    except ValueError:
        subject_id = subject_name
        
    # Clean continuous time series
    df_cleaned = clean_time_series(df_raw)
    
    # Save cleaned time-series data
    os.makedirs(output_dir, exist_ok=True)
    cleaned_file_path = os.path.join(output_dir, f"{subject_name}_cleaned.csv")
    df_cleaned.to_csv(cleaned_file_path, index=False)
    
    # Extract meal events
    df_meals = extract_meal_features_and_targets(df_cleaned, subject_id)
    
    print(f"Processed subject {subject_name}: {len(df_raw)} raw rows -> {len(df_cleaned)} cleaned rows | {len(df_meals)} valid meals extracted.")
    return df_meals

def run_pipeline():
    """
    Main runner script to process all participants and save final outcomes.
    """
    cgmacros_dir = "CGMacros"
    output_dir = "data/processed"
    
    # Find all subject folders matching CGMacros-0YY
    subject_dirs = sorted([d for d in glob.glob(os.path.join(cgmacros_dir, "CGMacros-*")) if os.path.isdir(d)])
    
    print(f"Found {len(subject_dirs)} subjects to process.")
    
    all_meals = []
    for s_dir in subject_dirs:
        df_meals = process_subject(s_dir, output_dir)
        if not df_meals.empty:
            all_meals.append(df_meals)
            
    if all_meals:
        master_meals_df = pd.concat(all_meals, ignore_index=True)
        master_meals_df.to_csv(os.path.join(output_dir, "master_meals_raw.csv"), index=False)
        print(f"\nPipeline execution finished. Saved master meals dataset with {len(master_meals_df)} total records.")
    else:
        print("No meals were extracted.")

if __name__ == "__main__":
    run_pipeline()
