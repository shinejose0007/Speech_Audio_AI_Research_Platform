import numpy as np

def align(a, b):
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float32), np.asarray(b[:n], dtype=np.float32)

def snr_db(clean, estimate):
    clean, estimate = align(clean, estimate)
    noise = clean - estimate
    return float(10*np.log10((np.sum(clean**2)+1e-8)/(np.sum(noise**2)+1e-8)))

def si_snr_db(clean, estimate):
    clean, estimate = align(clean, estimate)
    clean = clean - np.mean(clean)
    estimate = estimate - np.mean(estimate)
    proj = np.sum(estimate*clean) * clean / (np.sum(clean**2)+1e-8)
    noise = estimate - proj
    return float(10*np.log10((np.sum(proj**2)+1e-8)/(np.sum(noise**2)+1e-8)))

def all_metrics(clean, noisy, enhanced, sr=16000):
    out = {
        "SNR noisy (dB)": round(snr_db(clean, noisy), 2),
        "SNR enhanced (dB)": round(snr_db(clean, enhanced), 2),
        "SNR improvement (dB)": round(snr_db(clean, enhanced)-snr_db(clean, noisy), 2),
        "SI-SNR enhanced (dB)": round(si_snr_db(clean, enhanced), 2),
        "MSE enhanced": round(float(np.mean((align(clean, enhanced)[0]-align(clean, enhanced)[1])**2)), 6),
    }
    try:
        from pystoi import stoi
        c, e = align(clean, enhanced)
        out["STOI enhanced"] = round(float(stoi(c, e, sr, extended=False)), 3)
    except Exception:
        out["STOI enhanced"] = "optional"
    try:
        from pesq import pesq
        c, e = align(clean, enhanced)
        out["PESQ enhanced"] = round(float(pesq(sr, c, e, "wb" if sr >= 16000 else "nb")), 3)
    except Exception:
        out["PESQ enhanced"] = "optional"
    return out
