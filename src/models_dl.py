import torch
import torch.nn as nn

class MLPForecaster(nn.Module):
    """
    Direct Multi-Layer Perceptron baseline.
    Flattens pre-meal time-series window (60 x 3) and concatenates static features.
    """
    def __init__(self, seq_len=60, in_channels=3, static_dim=60, horizon_steps=24, hidden_dim=128, dropout=0.2):
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
    """
    LSTM Encoder + Static Context Fuser + Multi-Horizon Decoder.
    Recurrently encodes the 60-minute pre-meal dynamic sequence (Glucose, HR, METs).
    Fuses the final hidden representation with static clinical & microbiome features.
    """
    def __init__(self, in_channels=3, static_dim=60, horizon_steps=24, rnn_hidden_dim=64, rnn_layers=2, fc_hidden_dim=128, dropout=0.2):
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
        # x_seq: (batch_size, seq_len, in_channels)
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        
        # Take last layer's final hidden state: (batch_size, rnn_hidden_dim)
        h_last = h_n[-1]
        
        # Fuse with static features
        fused = torch.cat([h_last, x_static], dim=1)
        
        # Forecast trajectory
        trajectory = self.decoder(fused)
        return trajectory


class GRUForecaster(nn.Module):
    """
    GRU Encoder + Static Context Fuser + Multi-Horizon Decoder.
    Gated Recurrent Unit variant for efficient continuous sequence modeling.
    """
    def __init__(self, in_channels=3, static_dim=60, horizon_steps=24, rnn_hidden_dim=64, rnn_layers=2, fc_hidden_dim=128, dropout=0.2):
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
        # x_seq: (batch_size, seq_len, in_channels)
        gru_out, h_n = self.gru(x_seq)
        
        # Take last layer's final hidden state
        h_last = h_n[-1]
        
        # Fuse with static features
        fused = torch.cat([h_last, x_static], dim=1)
        
        # Forecast trajectory
        trajectory = self.decoder(fused)
        return trajectory
