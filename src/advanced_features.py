from pathlib import Path
import subprocess, shutil, time, json
import numpy as np
import pandas as pd
from scipy import signal
from .audio_utils import normalize_audio, write_wav
from .metrics import all_metrics

def pcm_quantize(audio, bits=8):
    x=normalize_audio(audio); levels=2**bits
    q=np.round((x+1)*(levels-1)/2)
    return normalize_audio((q*2/(levels-1))-1)

def lowrate_sim(audio, sr, target_sr=8000):
    x=normalize_audio(audio); n=max(1,int(len(x)*target_sr/sr)); down=signal.resample(x,n); up=signal.resample(down,len(x)); return normalize_audio(up)

def telephone_sim(audio, sr):
    x=normalize_audio(audio); low=min(300/(sr/2),0.95); high=min(3400/(sr/2),0.99)
    b,a=signal.butter(5,[low,high],btype='band'); return pcm_quantize(signal.lfilter(b,a,x),8)

def codec_benchmark(clean, noisy, sr, enhanced=None):
    raw=len(noisy)*2
    variants={'Noisy 16-bit WAV':(noisy,raw),'PCM 8-bit simulation':(pcm_quantize(noisy,8),len(noisy)),'8 kHz low-rate simulation':(lowrate_sim(noisy,sr,8000),int(len(noisy)*8000/sr)*2),'Telephone-band 8-bit simulation':(telephone_sim(noisy,sr),len(noisy))}
    rows=[]
    for name,(audio,size) in variants.items():
        m=all_metrics(clean,noisy,audio,sr)
        rows.append({'method':name,'estimated_size_bytes':size,'compression_ratio':round(raw/max(1,size),2),'SNR enhanced (dB)':m['SNR enhanced (dB)'],'SNR improvement (dB)':m['SNR improvement (dB)'],'SI-SNR enhanced (dB)':m['SI-SNR enhanced (dB)']})
    if enhanced is not None:
        m=all_metrics(clean,noisy,enhanced,sr)
        rows.append({'method':'Neural model output','estimated_size_bytes':'model-dependent','compression_ratio':'model-dependent','SNR enhanced (dB)':m['SNR enhanced (dB)'],'SNR improvement (dB)':m['SNR improvement (dB)'],'SI-SNR enhanced (dB)':m['SI-SNR enhanced (dB)']})
    return rows, {k:v[0] for k,v in variants.items()}

def ffmpeg_available(): return shutil.which('ffmpeg') is not None

def ffmpeg_mp3(input_wav, out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); out=out_dir/(Path(input_wav).stem+'_48k.mp3')
    if not ffmpeg_available(): return None, 'FFmpeg not found. Install FFmpeg and add it to PATH.'
    try:
        subprocess.run(['ffmpeg','-y','-i',str(input_wav),'-b:a','48k',str(out)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out, 'OK'
    except Exception as e: return None, str(e)

def quantize_onnx(fp32_path, out_dir):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    fp32_path=Path(fp32_path); out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    int8=out_dir/fp32_path.name.replace('.onnx','_int8.onnx')
    quantize_dynamic(str(fp32_path), str(int8), weight_type=QuantType.QInt8)
    a=fp32_path.stat().st_size; b=int8.stat().st_size
    return {'fp32_path':str(fp32_path),'int8_path':str(int8),'fp32_size_bytes':a,'int8_size_bytes':b,'size_reduction_percent':round(100*(1-b/max(1,a)),2)}

def log_run(exp_dir, run_name, params=None, metrics=None):
    exp_dir=Path(exp_dir); exp_dir.mkdir(parents=True, exist_ok=True); ts=time.strftime('%Y%m%d_%H%M%S')
    path=exp_dir/f"{ts}_{run_name.lower().replace(' ','_')}.json"
    path.write_text(json.dumps({'run_name':run_name,'timestamp':ts,'params':params or {},'metrics':metrics or {}}, indent=2), encoding='utf-8')
    return path

def list_runs(exp_dir):
    rows=[]
    for p in sorted(Path(exp_dir).glob('*.json'), reverse=True):
        try:
            d=json.loads(p.read_text(encoding='utf-8')); r={'file':p.name,'run_name':d.get('run_name'),'timestamp':d.get('timestamp')}
            for k,v in d.get('params',{}).items(): r['param_'+k]=v
            for k,v in d.get('metrics',{}).items(): r['metric_'+k]=v
            rows.append(r)
        except Exception: pass
    return pd.DataFrame(rows)

def try_mlflow(mlruns_dir, name, params=None, metrics=None):
    try:
        import mlflow
        mlruns_dir=Path(mlruns_dir); mlruns_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(str(mlruns_dir.resolve())); mlflow.set_experiment('speech_audio_pro_dashboard')
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params or {})
            mlflow.log_metrics({k:float(v) for k,v in (metrics or {}).items() if isinstance(v,(int,float))})
        return True, 'Logged to MLflow.'
    except Exception as e: return False, str(e)
