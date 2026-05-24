from pathlib import Path
import time
import torch
from .audio_utils import read_wav, write_wav, fix_length
from .training import load_trained_model, safe_name
from .models import model_size_mb

def run_inference(input_path, model_name, model_dir, output_dir):
    sr, audio = read_wav(input_path)
    audio = fix_length(audio, 16000)
    model, device, _ = load_trained_model(model_name, model_dir)
    x = torch.tensor(audio).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(x)
    start = time.time()
    with torch.no_grad():
        if model_name == "Audio Codec Autoencoder":
            y, z = model(x)
        else:
            y = model(x); z = None
    elapsed = time.time()-start
    out = y.squeeze().cpu().numpy()
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir/f"{Path(input_path).stem}_{safe_name(model_name)}_output.wav"
    write_wav(out_path, sr, out)
    duration = len(audio)/sr
    info = {
        "sr": sr, "input_audio": audio, "output_audio": out, "output_path": str(out_path),
        "inference_ms": round(elapsed*1000, 3),
        "real_time_factor": round(elapsed/duration, 4),
        "model_size_mb": round(model_size_mb(model), 4),
    }
    if z is not None:
        raw_bits = len(audio)*16
        latent_bits = z.numel()*6
        info["estimated_compression_ratio"] = round(raw_bits/max(1, latent_bits), 2)
    return info

def export_onnx(model_name, model_dir, output_dir):
    from .training import load_trained_model, safe_name
    model, device, _ = load_trained_model(model_name, model_dir)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir/f"{safe_name(model_name)}.onnx"
    dummy = torch.randn(1,1,16000, device=device)
    if model_name == "Audio Codec Autoencoder":
        class Wrapper(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m=m
            def forward(self, x): return self.m(x)[0]
        model_to_export = Wrapper(model)
    else:
        model_to_export = model
    torch.onnx.export(model_to_export, dummy, str(out_path), input_names=["audio"], output_names=["output"], opset_version=12)
    return out_path
