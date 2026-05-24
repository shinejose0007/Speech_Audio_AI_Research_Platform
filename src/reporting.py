from pathlib import Path
from datetime import datetime

def export_report(path, title, metadata, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "", "## Metadata"]
    for k, v in metadata.items():
        if k != "logs":
            lines.append(f"- **{k}:** {v}")
    if metadata.get("logs"):
        lines += ["", "## Training logs"]
        for item in metadata["logs"]:
            lines.append(f"- Epoch {item['epoch']}: train L1={item['train_L1']}, val L1={item['val_L1']}")
    lines += ["", "## Evaluation metrics"]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"]*len(headers)) + " |")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    else:
        lines.append("No metrics available.")
    lines += ["", "## Portfolio note", "This project demonstrates speech/audio preprocessing, neural denoising, experimental audio coding, model comparison, ONNX export, and deployment awareness."]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
