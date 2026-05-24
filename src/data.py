from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .audio_utils import read_wav, write_wav, add_noise, fix_length

def synth_speech_like(sr=16000, duration=1.0):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    f0 = np.random.uniform(90, 180)
    sig = np.zeros_like(t, dtype=np.float32)
    for k in range(1, 6):
        sig += (1/k)*np.sin(2*np.pi*f0*k*t + np.random.rand()*np.pi)
    sig += 0.35*np.sin(2*np.pi*np.random.uniform(600, 900)*t)
    sig += 0.2*np.sin(2*np.pi*np.random.uniform(1200, 1900)*t)
    env = 0.5*(1-np.cos(2*np.pi*np.minimum(t/duration, 1)))
    sig = sig * env
    return (sig/(np.max(np.abs(sig))+1e-8)).astype(np.float32)

def generate_demo_dataset(root, n=80, sr=16000, duration=1.0, snr_db=4.0):
    root = Path(root)
    clean_dir = root/"clean"; noisy_dir = root/"noisy"
    clean_dir.mkdir(parents=True, exist_ok=True); noisy_dir.mkdir(parents=True, exist_ok=True)
    for p in list(clean_dir.glob("*.wav")) + list(noisy_dir.glob("*.wav")):
        p.unlink()
    for i in range(n):
        clean = synth_speech_like(sr, duration)
        noisy = add_noise(clean, snr_db, np.random.choice(["white", "pink", "hum"]))
        write_wav(clean_dir/f"sample_{i:03d}.wav", sr, clean)
        write_wav(noisy_dir/f"sample_{i:03d}.wav", sr, noisy)

class PairedAudioDataset(Dataset):
    def __init__(self, root, length=16000):
        root = Path(root)
        clean = {p.name:p for p in (root/"clean").glob("*.wav")}
        noisy = {p.name:p for p in (root/"noisy").glob("*.wav")}
        names = sorted(set(clean).intersection(noisy))
        self.pairs = [(noisy[n], clean[n], n) for n in names]
        if not self.pairs:
            raise FileNotFoundError(f"No paired wav files found in {root}/clean and {root}/noisy")
        self.length = length
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        noisy_path, clean_path, name = self.pairs[idx]
        _, noisy = read_wav(noisy_path)
        _, clean = read_wav(clean_path)
        return torch.tensor(fix_length(noisy, self.length)).float().unsqueeze(0), torch.tensor(fix_length(clean, self.length)).float().unsqueeze(0), name
