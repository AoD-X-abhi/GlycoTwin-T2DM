import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

class DenoisingTaxaDataset(Dataset):
    """
    Dataset wrapper with dynamic stochastic taxa masking.
    To prevent overfitting on clinical cohort sizes (N ~ 45), randomly masks
    a subset of positive taxa (1 -> 0) to enforce learning co-occurrence patterns.
    """
    def __init__(self, data, mask_prob=0.15, augment_factor=25):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.mask_prob = mask_prob
        self.augment_factor = augment_factor
        self.n_samples = len(data) * augment_factor

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        real_idx = idx % len(self.data)
        x_target = self.data[real_idx]
        
        # Stochastic taxa masking
        x_corrupt = x_target.clone()
        if self.mask_prob > 0:
            mask = (torch.rand_like(x_corrupt) < self.mask_prob) & (x_corrupt == 1.0)
            x_corrupt[mask] = 0.0
            
        return x_corrupt, x_target


class MicrobiomeVAE(nn.Module):
    """
    Deep Variational Autoencoder (VAE) for Sparse Binary Gut Microbiome Taxonomic Profiles.
    - Encodes 1,979 binary bacterial species into a smooth, continuous latent manifold (z ∈ ℝ^8).
    - Uses LayerNorm, LeakyReLU, and Dropout to ensure regularized embeddings for clinical cohorts.
    - Decodes latent coordinates back into Bernoulli occurrence logits.
    """
    def __init__(self, in_features=1979, hidden_dims=[256, 64], latent_dim=8, dropout=0.2):
        super(MicrobiomeVAE, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        
        # 1. Encoder Backbone
        enc_layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            curr_dim = h_dim
        self.encoder = nn.Sequential(*enc_layers)
        
        # Latent heads: Mean μ and Log-Variance log(σ²)
        self.fc_mu = nn.Linear(curr_dim, latent_dim)
        self.fc_logvar = nn.Linear(curr_dim, latent_dim)
        
        # 2. Decoder Backbone
        dec_layers = []
        curr_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            curr_dim = h_dim
        dec_layers.append(nn.Linear(curr_dim, in_features))
        self.decoder = nn.Sequential(*dec_layers)
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        return mu

    def decode(self, z):
        return self.decoder(z) # Unnormalized logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def get_latent_embeddings(self, x):
        """Returns deterministic posterior mean μ for patient encoding."""
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

    def reconstruct_probabilities(self, x):
        """Returns reconstructed taxonomic presence probabilities."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            logits = self.decode(mu)
            probs = torch.sigmoid(logits)
        return probs


def vae_loss_function(logits, targets, mu, logvar, beta=0.02, pos_weight=None):
    """
    Beta-VAE multi-objective loss for sparse binary vectors:
    Reconstruction: BCEWithLogitsLoss
    Regularization: KL Divergence to Standard Normal N(0, I)
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='sum'
    ) / targets.size(0)
    
    # KL Divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / targets.size(0)
    
    total_loss = bce + (beta * kld)
    return total_loss, bce, kld


def train_microbiome_vae(
    microbes_csv_path='CGMacros/microbes.csv',
    latent_dim=8,
    epochs=120,
    batch_size=32,
    lr=1e-3,
    beta_max=0.03,
    mask_prob=0.15,
    save_model_path='src/models/microbiome_vae.pt',
    device=None
):
    """
    Trains the MicrobiomeVAE model on CGMacros taxonomic profiles with beta-annealing
    and denoising taxa-masking augmentation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # 1. Load data
    df_microbes = pd.read_csv(microbes_csv_path)
    taxa_cols = [c for c in df_microbes.columns if c != 'subject']
    df_clean = df_microbes.dropna(subset=[taxa_cols[0]]).reset_index(drop=True)
    
    clean_subjects = df_clean['subject'].values
    X_clean = df_clean[taxa_cols].values.astype(np.float32)
    in_features = len(taxa_cols)
    
    print(f"Loaded {len(clean_subjects)} subjects with {in_features} bacterial species.")
    
    # Compute positive class weight to balance ~13% presence sparsity
    pos_frac = (X_clean == 1.0).mean()
    pos_weight_val = (1.0 - pos_frac) / max(pos_frac, 1e-4)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"Taxa sparsity: {pos_frac*100:.2f}% positive presence | pos_weight: {pos_weight_val:.2f}")

    # 2. Setup Dataset & Dataloader
    dataset = DenoisingTaxaDataset(X_clean, mask_prob=mask_prob, augment_factor=30)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model & Optimizer
    model = MicrobiomeVAE(in_features=in_features, hidden_dims=[256, 64], latent_dim=latent_dim, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    history = {'epoch': [], 'total_loss': [], 'bce_loss': [], 'kld_loss': [], 'beta': []}
    
    # 4. Training loop with beta-annealing
    print("\n--- Training MicrobiomeVAE with Beta-Annealing ---")
    model.train()
    for epoch in range(1, epochs + 1):
        # Beta linear annealing over first 50% of epochs
        beta = min(beta_max, beta_max * (epoch / (epochs * 0.5)))
        
        epoch_total = 0.0
        epoch_bce = 0.0
        epoch_kld = 0.0
        n_batches = 0
        
        for x_corrupt, x_target in loader:
            x_corrupt = x_corrupt.to(device)
            x_target = x_target.to(device)
            
            optimizer.zero_grad()
            logits, mu, logvar = model(x_corrupt)
            loss, bce, kld = vae_loss_function(logits, x_target, mu, logvar, beta=beta, pos_weight=pos_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            epoch_total += loss.item()
            epoch_bce += bce.item()
            epoch_kld += kld.item()
            n_batches += 1
            
        scheduler.step()
        
        avg_total = epoch_total / n_batches
        avg_bce = epoch_bce / n_batches
        avg_kld = epoch_kld / n_batches
        
        history['epoch'].append(epoch)
        history['total_loss'].append(avg_total)
        history['bce_loss'].append(avg_bce)
        history['kld_loss'].append(avg_kld)
        history['beta'].append(beta)
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Total: {avg_total:.2f} | BCE: {avg_bce:.2f} | KLD: {avg_kld:.2f} | Beta: {beta:.4f}")
            
    # 5. Evaluate Reconstruction Fidelity on original uncorrupted matrix
    model.eval()
    with torch.no_grad():
        x_orig_t = torch.tensor(X_clean, dtype=torch.float32, device=device)
        probs_clean = model.reconstruct_probabilities(x_orig_t).cpu().numpy()
        latent_clean = model.get_latent_embeddings(x_orig_t).cpu().numpy()
        
    y_true_flat = X_clean.flatten()
    y_pred_flat = probs_clean.flatten()
    recon_roc_auc = roc_auc_score(y_true_flat, y_pred_flat)
    recon_pr_auc = average_precision_score(y_true_flat, y_pred_flat)
    print(f"\n[MicrobiomeVAE Evaluation]")
    print(f"Taxa Reconstruction ROC-AUC: {recon_roc_auc:.4f}")
    print(f"Taxa Reconstruction PR-AUC (Average Precision): {recon_pr_auc:.4f}")
    
    # 6. Build Patient Latent Feature DataFrame
    latent_cols = [f'Microbiome_VAE{i}' for i in range(1, latent_dim + 1)]
    df_latent_clean = pd.DataFrame(latent_clean, columns=latent_cols)
    df_latent_clean['subject'] = clean_subjects
    
    # Impute missing Subject 48 via cohort latent prior (mean μ = 0)
    all_subjects = df_microbes['subject'].unique()
    missing_subjects = [s for s in all_subjects if s not in clean_subjects]
    
    impute_records = []
    cohort_mean_latents = df_latent_clean[latent_cols].mean().to_dict()
    for s in missing_subjects:
        impute_records.append({'subject': s, **cohort_mean_latents})
        print(f"Imputed missing Subject {s} using cohort latent mean coordinates.")
        
    if impute_records:
        df_latent_full = pd.concat([df_latent_clean, pd.DataFrame(impute_records)], ignore_index=True)
    else:
        df_latent_full = df_latent_clean
        
    df_latent_full = df_latent_full.sort_values('subject').reset_index(drop=True)
    
    # 7. Save Model
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'latent_dim': latent_dim,
            'in_features': in_features,
            'taxa_cols': taxa_cols,
            'roc_auc': recon_roc_auc,
            'pr_auc': recon_pr_auc
        }, save_model_path)
        print(f"Trained MicrobiomeVAE serialized to: {save_model_path}")
        
    return model, df_latent_full, history, (recon_roc_auc, recon_pr_auc)


def enrich_master_features_with_vae(
    df_latent,
    master_path='data/processed/master_meals_features.csv',
    output_path='data/processed/master_meals_features_vae.csv'
):
    """
    Enriches master meal dataset with non-linear MicrobiomeVAE latent dimensions.
    """
    df_master = pd.read_csv(master_path)
    print(f"\nOriginal master dataset shape: {df_master.shape}")
    
    latent_cols = [c for c in df_latent.columns if c.startswith('Microbiome_VAE')]
    
    # Drop existing VAE cols if present to allow idempotent re-runs
    existing_vae = [c for c in df_master.columns if c in latent_cols]
    if existing_vae:
        df_master = df_master.drop(columns=existing_vae)
        
    df_enriched = pd.merge(df_master, df_latent[['subject'] + latent_cols], on='subject', how='left')
    print(f"Enriched master dataset with {len(latent_cols)} VAE latent features: {df_enriched.shape}")
    print(f"NaN count in VAE columns: {df_enriched[latent_cols].isnull().sum().sum()}")
    
    df_enriched.to_csv(output_path, index=False)
    # Also update master_meals_features.csv so all downstream modules can use either PCA or VAE
    df_enriched.to_csv(master_path, index=False)
    print(f"Successfully saved enriched dataset to: {output_path} and updated: {master_path}")
    return df_enriched


def plot_vae_diagnostics(
    history,
    df_latent,
    bio_path='CGMacros/bio.csv',
    output_path='figures/microbiome_vae_diagnostics.png'
):
    """
    Generates publication-quality diagnostic plots:
    1. Training loss curves (BCE, KLD, Total)
    2. 2D PCA/t-SNE latent manifold colored by Clinical Group (Healthy, Pre-DM, T2DM)
    3. Latent coordinate variance & distribution
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Training Convergence
    ax = axes[0]
    ax.plot(history['epoch'], history['bce_loss'], label='Reconstruction BCE', color='#2563EB', lw=2)
    ax.plot(history['epoch'], history['kld_loss'], label='KL Divergence', color='#DC2626', lw=2)
    ax.set_title('MicrobiomeVAE Multi-Objective Convergence', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('Loss Component', fontsize=10)
    ax.legend()
    
    # 2. Latent Distribution (Boxplot of the 8 latent variables)
    ax = axes[1]
    latent_cols = [c for c in df_latent.columns if c.startswith('Microbiome_VAE')]
    sns.boxplot(data=df_latent[latent_cols], ax=ax, palette='Blues_r')
    ax.set_title('Latent Distribution Across Cohort (z ∈ ℝ⁸)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Latent Dimension', fontsize=10)
    ax.set_ylabel('Latent Coordinate Value', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    
    # 3. Clinical Group Manifold Projection
    ax = axes[2]
    if os.path.exists(bio_path):
        df_bio = pd.read_csv(bio_path)
        df_bio.columns = df_bio.columns.str.strip()
        if 'Clinical Group' not in df_bio.columns and 'A1c PDL (Lab)' in df_bio.columns:
            def classify_group(hba1c):
                if hba1c < 5.7:
                    return 'Healthy Control'
                elif hba1c <= 6.4:
                    return 'Pre-diabetes'
                else:
                    return 'Type 2 Diabetes'
            df_bio['Clinical Group'] = df_bio['A1c PDL (Lab)'].apply(classify_group)
            
        merged = pd.merge(df_latent, df_bio[['subject', 'Clinical Group']], on='subject', how='inner')
        palette = {'Healthy Control': '#16A34A', 'Pre-diabetes': '#EA580C', 'Type 2 Diabetes': '#DC2626'}
        sns.scatterplot(
            data=merged,
            x='Microbiome_VAE1',
            y='Microbiome_VAE2',
            hue='Clinical Group',
            palette=palette,
            s=90,
            alpha=0.9,
            edgecolor='black',
            ax=ax
        )
        ax.set_title('Microbiome Latent Space vs. Clinical Status', fontsize=12, fontweight='bold')
        ax.set_xlabel('Microbiome_VAE1', fontsize=10)
        ax.set_ylabel('Microbiome_VAE2', fontsize=10)
        ax.legend(title='Clinical Group', loc='best')
    else:
        sns.scatterplot(data=df_latent, x='Microbiome_VAE1', y='Microbiome_VAE2', s=80, ax=ax)
        ax.set_title('Latent Projection (VAE1 vs VAE2)', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Also save to artifact directory for display
    artifact_path = "C:/Users/abhi7/.gemini/antigravity-ide/brain/bb2c8bd4-742e-4a56-ad82-d5664480b2f8/microbiome_vae_diagnostics.png"
    plt.savefig(artifact_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Diagnostic plot saved to: {output_path}")


if __name__ == "__main__":
    model, df_latent, history, (auc, prauc) = train_microbiome_vae(epochs=120, latent_dim=8)
    enrich_master_features_with_vae(df_latent)
    plot_vae_diagnostics(history, df_latent)
    print("\nMicrobiomeVAE training and integration completed successfully.")
