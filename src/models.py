import torch
from torch import nn

class CNNDenoiser(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, 9, padding=4), nn.ReLU(),
            nn.Conv1d(channels, channels, 9, padding=4), nn.ReLU(),
            nn.Conv1d(channels, channels, 9, padding=4), nn.ReLU(),
            nn.Conv1d(channels, 1, 9, padding=4), nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class LSTMDenoiser(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden*2, 1)
    def forward(self, x):
        y = x.transpose(1, 2)
        y, _ = self.lstm(y)
        y = self.out(y)
        return torch.tanh(y.transpose(1, 2))

class TransformerDenoiser(nn.Module):
    def __init__(self, d_model=32, nhead=4):
        super().__init__()
        self.in_proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=96, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=1)
        self.out = nn.Linear(d_model, 1)
    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.in_proj(y)
        y = self.enc(y)
        return torch.tanh(self.out(y).transpose(1, 2))

class AudioCodecAutoencoder(nn.Module):
    def __init__(self, latent_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16, latent_channels, 9, stride=2, padding=4), nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 16, 8, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 8, stride=2, padding=3), nn.Tanh()
        )
    def forward(self, x):
        z = self.encoder(x)
        zq = torch.round(z*32)/32
        y = self.decoder(zq)
        return y[..., :x.shape[-1]], zq

def create_model(name):
    if name == "CNN Denoiser": return CNNDenoiser()
    if name == "LSTM Denoiser": return LSTMDenoiser()
    if name == "Transformer Denoiser": return TransformerDenoiser()
    if name == "Audio Codec Autoencoder": return AudioCodecAutoencoder()
    raise ValueError(name)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_mb(model):
    return count_parameters(model)*4/(1024**2)
