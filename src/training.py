from pathlib import Path
import json, time
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from .data import PairedAudioDataset
from .models import create_model, count_parameters, model_size_mb

def safe_name(name):
    return name.lower().replace(" ", "_")

def train_model(data_root, model_name, out_dir, epochs=5, batch_size=8, lr=1e-3, callback=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PairedAudioDataset(data_root)
    train_size = max(1, int(0.8*len(ds)))
    val_size = len(ds)-train_size
    if val_size == 0:
        train_ds, val_ds = ds, ds
    else:
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    model = create_model(model_name).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()
    logs = []
    start = time.time()
    for epoch in range(1, epochs+1):
        model.train(); tr = 0
        for noisy, clean, _ in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)[0] if model_name == "Audio Codec Autoencoder" else model(noisy)
            loss = loss_fn(pred, clean)
            opt.zero_grad(); loss.backward(); opt.step()
            tr += loss.item()*noisy.size(0)
        tr /= len(train_loader.dataset)
        model.eval(); va = 0
        with torch.no_grad():
            for noisy, clean, _ in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)[0] if model_name == "Audio Codec Autoencoder" else model(noisy)
                va += loss_fn(pred, clean).item()*noisy.size(0)
        va /= max(1, len(val_loader.dataset))
        item = {"epoch": epoch, "train_L1": round(tr, 6), "val_L1": round(va, 6)}
        logs.append(item)
        if callback: callback(epoch, epochs, item)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{safe_name(model_name)}.pt"
    torch.save(model.state_dict(), model_path)
    meta = {
        "model_name": model_name,
        "model_path": str(model_path),
        "device": device,
        "epochs": epochs,
        "parameters": count_parameters(model),
        "estimated_model_size_mb": round(model_size_mb(model), 4),
        "training_seconds": round(time.time()-start, 2),
        "logs": logs,
    }
    (out_dir/f"{safe_name(model_name)}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def load_trained_model(model_name, model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name).to(device)
    path = Path(model_dir)/f"{safe_name(model_name)}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Train this model first: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device, path
