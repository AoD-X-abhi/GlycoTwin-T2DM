import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for 1-minute time-series tokens.
    """
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) for static feature gating and non-linear transformation.
    Inspired by Temporal Fusion Transformer (TFT).
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        g = torch.sigmoid(self.gate(h))
        gated_out = g * out
        return self.layer_norm(residual + gated_out)


class TransformerForecaster(nn.Module):
    """
    Multi-Modal Transformer Encoder + Gated Feature Fusion + Residual Multi-Horizon Decoder.
    - Encodes dynamic pre-meal time series (Glucose, HR, METs) via Multi-Head Self-Attention.
    - Encodes static features (Macronutrients, Clinical Baselines, Microbiome PCA) via GRN.
    - Decodes relative glucose rise trajectories (ΔG_t).
    """
    def __init__(self, in_channels=3, static_dim=61, horizon_steps=24, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerForecaster, self).__init__()
        
        self.seq_proj = nn.Linear(in_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.static_grn = GatedResidualNetwork(static_dim, d_model, d_model, dropout=dropout)
        
        fused_dim = d_model * 2
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon_steps)
        )

    def forward(self, x_seq, x_static):
        # x_seq: (batch_size, seq_len, in_channels)
        # x_static: (batch_size, static_dim)
        
        # 1. Temporal sequence encoding
        h_seq = self.seq_proj(x_seq)
        h_seq = self.pos_encoder(h_seq)
        tf_out = self.transformer_encoder(h_seq)
        h_temp = tf_out.mean(dim=1)  # Mean pool across time tokens
        
        # 2. Static feature encoding
        h_stat = self.static_grn(x_static)
        
        # 3. Multi-modal fusion
        fused = torch.cat([h_temp, h_stat], dim=1)
        
        # 4. Predict relative excursion trajectory delta
        delta_trajectory = self.decoder(fused)
        return delta_trajectory


class CompositeExcursionLoss(nn.Module):
    """
    Physiological Loss Function combining:
    1. MSE Loss (Overall curve fit)
    2. Peak Height Loss (Matches maximum glucose rise amplitude)
    3. Derivative/Velocity Loss (Matches curve rate of change and smoothness)
    """
    def __init__(self, lambda_peak=0.5, lambda_deriv=0.2):
        super(CompositeExcursionLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lambda_peak = lambda_peak
        self.lambda_deriv = lambda_deriv

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        
        # Peak height error
        peak_pred = torch.max(y_pred, dim=1)[0]
        peak_true = torch.max(y_true, dim=1)[0]
        peak_loss = self.l1(peak_pred, peak_true)
        
        # Velocity / trajectory derivative error
        diff_pred = y_pred[:, 1:] - y_pred[:, :-1]
        diff_true = y_true[:, 1:] - y_true[:, :-1]
        deriv_loss = self.mse(diff_pred, diff_true)
        
        return mse_loss + (self.lambda_peak * peak_loss) + (self.lambda_deriv * deriv_loss)


class MLPForecaster(nn.Module):
    def __init__(self, seq_len=60, in_channels=3, static_dim=61, horizon_steps=24, hidden_dim=128, dropout=0.2):
        super(MLPForecaster, self).__init__()
        input_dim = (seq_len * in_channels) + static_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon_steps)
        )

    def forward(self, x_seq, x_static):
        batch_size = x_seq.size(0)
        x_flat = x_seq.view(batch_size, -1)
        x_in = torch.cat([x_flat, x_static], dim=1)
        return self.net(x_in)


class LSTMForecaster(nn.Module):
    def __init__(self, in_channels=3, static_dim=61, horizon_steps=24, rnn_hidden_dim=64, rnn_layers=2, fc_hidden_dim=128, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        fused_dim = rnn_hidden_dim + static_dim
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, horizon_steps)
        )

    def forward(self, x_seq, x_static):
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        h_last = h_n[-1]
        fused = torch.cat([h_last, x_static], dim=1)
        return self.decoder(fused)


class GRUForecaster(nn.Module):
    def __init__(self, in_channels=3, static_dim=61, horizon_steps=24, rnn_hidden_dim=64, rnn_layers=2, fc_hidden_dim=128, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        fused_dim = rnn_hidden_dim + static_dim
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fc_hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, horizon_steps)
        )

    def forward(self, x_seq, x_static):
        gru_out, h_n = self.gru(x_seq)
        h_last = h_n[-1]
        fused = torch.cat([h_last, x_static], dim=1)
        return self.decoder(fused)



if __name__ == "__main__":
    # Test forward pass with dummy tensors
    bsz = 16
    x_seq = torch.randn(bsz, 60, 3)
    x_stat = torch.randn(bsz, 61)
    
    model = TransformerForecaster(in_channels=3, static_dim=61, horizon_steps=24)
    out = model(x_seq, x_stat)
    
    loss_fn = CompositeExcursionLoss()
    target_dummy = torch.randn(bsz, 24)
    loss = loss_fn(out, target_dummy)
    
    print(f"TransformerForecaster output shape: {out.shape}")
    print(f"CompositeExcursionLoss test value: {loss.item():.4f}")
