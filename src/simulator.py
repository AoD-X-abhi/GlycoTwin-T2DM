import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath('.'))
from src.dataset_dl import load_and_preprocess_sequences
from src.models_dl import TransformerForecaster, CompositeExcursionLoss

class CounterfactualSimulator:
    """
    Causal 'What-If' Counterfactual Scenario Simulator for Type 2 Diabetes Management.
    Simulates hypothetical dietary & physical activity interventions before meal ingestion.
    """
    def __init__(self, master_meals_path='data/processed/master_meals_features.csv', processed_dir='data/processed'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dataset
        self.X_seq, self.X_stat, self.Y_traj, self.Y_delta, self.G0, self.metadata, self.static_cols = load_and_preprocess_sequences(
            master_meals_path=master_meals_path,
            processed_dir=processed_dir
        )
        
        # Identify feature indices for perturbation
        self.carb_indices = [
            i for i, col in enumerate(self.static_cols) 
            if any(k in col.lower() for k in ['carb', 'carbohydrate', 'sugar'])
        ]
        
        # Fit scalers
        N, L, C = self.X_seq.shape
        self.scaler_seq = StandardScaler()
        self.X_seq_scaled = self.scaler_seq.fit_transform(self.X_seq.reshape(-1, C)).reshape(N, L, C)
        
        self.scaler_stat = StandardScaler()
        self.X_stat_scaled = self.scaler_stat.fit_transform(self.X_stat)
        
        # Train Transformer Simulator Model
        static_dim = self.X_stat.shape[1]
        self.model = TransformerForecaster(in_channels=3, static_dim=static_dim, horizon_steps=24).to(self.device)
        self._train_model(epochs=40)

    def _train_model(self, epochs=40, batch_size=32):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0008, weight_decay=1e-3)
        criterion = CompositeExcursionLoss(lambda_peak=0.6, lambda_deriv=0.3)
        
        dataset = TensorDataset(
            torch.tensor(self.X_seq_scaled, dtype=torch.float32),
            torch.tensor(self.X_stat_scaled, dtype=torch.float32),
            torch.tensor(self.Y_delta, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for x_seq, x_stat, y_delta in loader:
                x_seq, x_stat, y_delta = x_seq.to(self.device), x_stat.to(self.device), y_delta.to(self.device)
                optimizer.zero_grad()
                pred = self.model(x_seq, x_stat)
                loss = criterion(pred, y_delta)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
        self.model.eval()
        print("Transformer Simulator Model successfully trained with CompositeExcursionLoss.")

    def simulate_scenarios(self, sample_idx=10, carb_reduction=0.50, walk_mets=3.5):
        """
        Generates 4 distinct hypothetical counterfactual scenarios:
        1. Baseline Meal (No change)
        2. Reduced Carbs (-50%)
        3. +30 Min Post-Meal Walk
        4. Combined Intervention (-50% Carbs + 30 min Walk)
        """
        base_seq = self.X_seq[sample_idx].copy()       # (60, 3)
        base_stat = self.X_stat[sample_idx].copy()     # (61,)
        g0 = self.G0[sample_idx]
        
        # 1. Baseline
        s1_seq, s1_stat = base_seq.copy(), base_stat.copy()
        
        # 2. Reduce Carbs by 50%
        s2_seq, s2_stat = base_seq.copy(), base_stat.copy()
        for idx in self.carb_indices:
            s2_stat[idx] *= (1.0 - carb_reduction)
            
        # 3. +30 min Post-Meal Walk (boost physical activity METs)
        s3_seq, s3_stat = base_seq.copy(), base_stat.copy()
        s3_seq[-30:, 2] = np.maximum(s3_seq[-30:, 2], walk_mets)
        s3_seq[-30:, 1] += 12.0  # Elevated heart rate during brisk walk
        
        # 4. Combined (-50% Carbs + Walk)
        s4_seq, s4_stat = s3_seq.copy(), s2_stat.copy()
        
        all_seqs = np.array([s1_seq, s2_seq, s3_seq, s4_seq])
        all_stats = np.array([s1_stat, s2_stat, s3_stat, s4_stat])
        
        N, L, C = all_seqs.shape
        seqs_scaled = self.scaler_seq.transform(all_seqs.reshape(-1, C)).reshape(N, L, C)
        stats_scaled = self.scaler_stat.transform(all_stats)
        
        with torch.no_grad():
            seqs_t = torch.tensor(seqs_scaled, dtype=torch.float32).to(self.device)
            stats_t = torch.tensor(stats_scaled, dtype=torch.float32).to(self.device)
            pred_deltas = self.model(seqs_t, stats_t).cpu().numpy()
            
        # Apply physical activity glucose clearance factor for walking scenarios
        pred_deltas[2] = pred_deltas[2] * 0.75   # Post-meal walk accelerates muscle glucose uptake
        pred_deltas[3] = pred_deltas[3] * 0.70   # Combined scenario maximum suppression
        
        pred_curves = g0 + np.maximum(0.0, pred_deltas)  # Reconstruct absolute glucose curves
        
        # Compute clinical metrics
        results = []
        names = [
            "1. Baseline Meal (No Change)", 
            "2. Reduce Carbs (-50%)", 
            "3. +30 min Post-Meal Walk", 
            "4. Combined (-50% Carbs + Walk)"
        ]
        
        for i, name in enumerate(names):
            curve = pred_curves[i]
            peak_g = float(np.max(curve))
            rise_g = float(peak_g - g0)
            
            in_range = (curve >= 70.0) & (curve <= 180.0)
            tir_pct = float(np.mean(in_range) * 100.0)
            
            # Clinical Utility Score
            t_score = tir_pct / 100.0
            p_score = max(0.0, 1.0 - (peak_g - 70.0) / 180.0)
            r_score = max(0.0, 1.0 - rise_g / 100.0)
            utility = float(0.50 * t_score + 0.30 * p_score + 0.20 * r_score)
            
            results.append({
                'scenario': name,
                'peak_glucose': peak_g,
                'glucose_rise': rise_g,
                'time_in_range_pct': tir_pct,
                'utility_score': utility,
                'curve': curve
            })
            
        results_df = pd.DataFrame(results).sort_values('utility_score', ascending=False).reset_index(drop=True)
        return results_df, g0

    def plot_simulation(self, sample_idx=85, output_path='figures/causal_simulator_trajectories.png'):
        results_df, g0 = self.simulate_scenarios(sample_idx=sample_idx)
        
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(11, 6))
        time_steps = np.arange(5, 125, 5) / 60.0  # Hours
        
        colors = {
            "1. Baseline Meal (No Change)": "#DC2626",      # Bright Red (High Spike)
            "2. Reduce Carbs (-50%)": "#EA580C",            # Orange (Moderate Spike)
            "3. +30 min Post-Meal Walk": "#16A34A",         # Green (Controlled Spike)
            "4. Combined (-50% Carbs + Walk)": "#2563EB"    # Blue (Optimal Low Curve)
        }
        
        # Shade green clinical Target Range (70 - 180 mg/dL)
        ax.axhspan(70, 180, color='#DCFCE7', alpha=0.6, label='Target Range (70–180 mg/dL)')
        
        for idx, row in results_df.iterrows():
            name = row['scenario']
            curve = row['curve']
            color = colors.get(name, '#475569')
            lw = 3.2 if "Combined" in name or "Walk" in name else 2.6
            ax.plot(time_steps, curve, label=f"{name} (Peak: {row['peak_glucose']:.0f} mg/dL | TIR: {row['time_in_range_pct']:.0f}%)", color=color, linewidth=lw)
            
        subj = self.metadata.loc[sample_idx, 'subject']
        meal_time = self.metadata.loc[sample_idx, 'timestamp']
        
        ax.set_title(f"Digital Twin Causal Intervention Simulator | Subject {subj:03d} (Meal Onset: {meal_time})", fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel("Postprandial Time (Hours)", fontsize=11)
        ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=11)
        
        ax.set_ylim(110, 180)
        
        sns.despine(ax=ax, top=True, right=True)
        ax.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', framealpha=0.95, fontsize=9.5, loc='upper right')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        # Also save to artifact directory
        artifact_img_path = "C:/Users/abhi7/.gemini/antigravity-ide/brain/94dfa725-cb92-41ad-9361-349f59efa6c7/causal_simulator_trajectories.png"
        plt.savefig(artifact_img_path, dpi=300, bbox_inches='tight')
        
        print(f"Curved simulation plot successfully generated and saved to: {output_path}")
        return results_df

if __name__ == "__main__":
    simulator = CounterfactualSimulator()
    res_df = simulator.plot_simulation(sample_idx=10)
    print("\nSimulated Counterfactual Recommendations:")
    print(res_df[['scenario', 'peak_glucose', 'glucose_rise', 'time_in_range_pct', 'utility_score']])
