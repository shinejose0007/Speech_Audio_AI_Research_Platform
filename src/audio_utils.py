from pathlib import Path
import io, wave
import numpy as np
from scipy.io import wavfile
from scipy import signal

def normalize_audio(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        return x
    if np.max(np.abs(x)) > 1:
        x = x / 32768.0
    m = np.max(np.abs(x)) + 1e-8
    if m > 1:
        x = x / m
    return np.clip(x, -1, 1)

def read_wav(path):
    sr, x = wavfile.read(str(path))
    return int(sr), normalize_audio(x)

def write_wav(path, sr, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = normalize_audio(audio)
    wavfile.write(str(path), int(sr), (audio * 32767).astype(np.int16))

def wav_bytes(sr, audio):
    audio_i16 = (normalize_audio(audio) * 32767).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio_i16.tobytes())
    return bio.getvalue()

def fix_length(audio, length=16000):
    audio = normalize_audio(audio)
    if len(audio) >= length:
        return audio[:length]
    return np.pad(audio, (0, length-len(audio)))

def add_noise(clean, snr_db=5.0, noise_type="white"):
    clean = normalize_audio(clean)
    if noise_type == "pink":
        white = np.random.randn(len(clean)).astype(np.float32)
        b, a = signal.butter(1, 0.05, btype="low")
        noise = signal.lfilter(b, a, white)
    elif noise_type == "hum":
        t = np.linspace(0, len(clean)/16000, len(clean), endpoint=False)
        noise = np.sin(2*np.pi*50*t).astype(np.float32) + 0.5*np.sin(2*np.pi*100*t).astype(np.float32)
    else:
        noise = np.random.randn(len(clean)).astype(np.float32)
    clean_power = np.mean(clean**2) + 1e-8
    noise_power = np.mean(noise**2) + 1e-8
    target_noise_power = clean_power / (10 ** (snr_db/10))
    noise = noise * np.sqrt(target_noise_power/noise_power)
    return normalize_audio(clean + noise)

def apply_augmentation(audio, sr, aug_name):
    x = normalize_audio(audio)
    if aug_name == "None":
        return x
    if aug_name == "White noise":
        return add_noise(x, 8, "white")
    if aug_name == "Pink noise":
        return add_noise(x, 8, "pink")
    if aug_name == "50 Hz hum":
        return add_noise(x, 12, "hum")
    if aug_name == "Low-pass filter":
        b, a = signal.butter(5, min(3000/(sr/2), 0.99), btype="low")
        return normalize_audio(signal.lfilter(b, a, x))
    if aug_name == "High-pass filter":
        b, a = signal.butter(5, min(300/(sr/2), 0.99), btype="high")
        return normalize_audio(signal.lfilter(b, a, x))
    if aug_name == "Telephone band 300-3400 Hz":
        low = min(300/(sr/2), 0.95)
        high = min(3400/(sr/2), 0.99)
        b, a = signal.butter(5, [low, high], btype="band")
        return normalize_audio(signal.lfilter(b, a, x))
    if aug_name == "Random volume":
        return normalize_audio(np.random.uniform(0.4, 1.3) * x)
    if aug_name == "Small time shift":
        return np.roll(x, int(0.08 * sr))
    return x

def log_spectrogram(audio, sr=16000, n_fft=512, hop=128):
    f, t, Z = signal.stft(normalize_audio(audio), fs=sr, nperseg=n_fft, noverlap=n_fft-hop, boundary=None)
    return f, t, 20*np.log10(np.abs(Z)+1e-6)

def mel_filterbank(sr=16000, n_fft=512, n_mels=64):
    def hz_to_mel(hz): return 2595*np.log10(1+hz/700)
    def mel_to_hz(mel): return 700*(10**(mel/2595)-1)
    mels = np.linspace(hz_to_mel(0), hz_to_mel(sr/2), n_mels+2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft+1)*hz/sr).astype(int)
    fb = np.zeros((n_mels, n_fft//2+1), dtype=np.float32)
    for m in range(1, n_mels+1):
        left, center, right = bins[m-1], bins[m], bins[m+1]
        if center > left:
            fb[m-1, left:center] = (np.arange(left, center)-left)/(center-left)
        if right > center:
            fb[m-1, center:right] = (right-np.arange(center, right))/(right-center)
    return fb

def log_mel_spectrogram(audio, sr=16000, n_fft=512, hop=128, n_mels=64):
    f, t, spec = log_spectrogram(audio, sr, n_fft, hop)
    mag = 10 ** (spec/20)
    mel = mel_filterbank(sr, n_fft, n_mels) @ mag
    return t, 20*np.log10(mel+1e-6)
