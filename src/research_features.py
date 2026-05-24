from pathlib import Path
import json, time, shutil
import numpy as np, pandas as pd
from .audio_utils import read_wav
from .training import train_model, safe_name

def audit_dataset(root, expected_sr=16000):
    root=Path(root); clean={p.name:p for p in (root/'clean').glob('*.wav')} if (root/'clean').exists() else {}; noisy={p.name:p for p in (root/'noisy').glob('*.wav')} if (root/'noisy').exists() else {}
    rows=[]
    for name in sorted(set(clean)|set(noisy)):
        issues=[]; row={'file':name}
        for label, mp in [('clean',clean),('noisy',noisy)]:
            if name not in mp: issues.append(f'missing {label} pair'); continue
            try:
                sr,x=read_wav(mp[name]); dur=len(x)/sr; silence=float(np.mean(np.abs(x)<1e-4)); clipping=bool(np.mean(np.abs(x)>0.98)>0.01)
                row[f'{label}_sr']=sr; row[f'{label}_duration_s']=round(dur,3); row[f'{label}_silence_ratio']=round(silence,3); row[f'{label}_clipping']=clipping
                if sr!=expected_sr: issues.append(f'{label} sample rate {sr}, expected {expected_sr}')
                if dur<0.3: issues.append(f'{label} too short')
                if silence>0.95: issues.append(f'{label} mostly silent')
                if clipping: issues.append(f'{label} clipping')
            except Exception as e: issues.append(f'{label} unreadable: {e}')
        row['status']='CHECK' if issues else 'OK'; row['issues']='; '.join(issues); rows.append(row)
    return rows

def wer(ref, hyp):
    r=ref.lower().split(); h=hyp.lower().split(); dp=[[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0]=i
    for j in range(len(h)+1): dp[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1): dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(r[i-1]!=h[j-1]))
    return dp[-1][-1]/max(1,len(r))

def cer(ref, hyp):
    r=list(ref.lower()); h=list(hyp.lower()); dp=[[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): dp[i][0]=i
    for j in range(len(h)+1): dp[0][j]=j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1): dp[i][j]=min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(r[i-1]!=h[j-1]))
    return dp[-1][-1]/max(1,len(r))

def run_asr_optional(wav_path, model_id='openai/whisper-tiny'):
    try:
        from transformers import pipeline
        pipe=pipeline('automatic-speech-recognition', model=model_id)
        out=pipe(str(wav_path)); return True, out.get('text', str(out))
    except Exception as e:
        return False, 'ASR unavailable. Install optional_requirements_research.txt and allow model download. Details: '+str(e)

def extract_embedding_optional(wav_path, model_id='facebook/wav2vec2-base-960h'):
    try:
        import torch
        from transformers import AutoProcessor, AutoModel
        from .audio_utils import normalize_audio, resample_to
        sr,x=read_wav(wav_path); sr,x=resample_to(x,sr,16000)
        proc=AutoProcessor.from_pretrained(model_id); model=AutoModel.from_pretrained(model_id)
        inp=proc(x, sampling_rate=16000, return_tensors='pt', padding=True)
        with torch.no_grad(): out=model(**inp)
        emb=out.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return True, {'shape':tuple(emb.shape), 'mean':float(np.mean(emb)), 'std':float(np.std(emb)), 'preview':emb[:10].round(4).tolist()}
    except Exception as e:
        return False, 'Embedding extraction unavailable. Install optional_requirements_research.txt. Details: '+str(e)

def run_hparam_search(data_root, model_dir, exp_dir, model_names, lrs, batch_sizes, epochs=2, callback=None):
    rows=[]; total=max(1,len(model_names)*len(lrs)*len(batch_sizes)); step=0
    for m in model_names:
        for lr in lrs:
            for bs in batch_sizes:
                step+=1
                if callback: callback(step,total,f'Training {m} lr={lr} batch={bs}')
                try:
                    meta=train_model(data_root,m,model_dir,epochs=epochs,batch_size=bs,lr=lr); last=meta['logs'][-1]
                    row={'model':m,'lr':lr,'batch_size':bs,'epochs':epochs,'final_train_L1':last['train_L1'],'final_val_L1':last['val_L1'],'model_size_mb':meta['estimated_model_size_mb']}
                except Exception as e: row={'model':m,'lr':lr,'batch_size':bs,'status':'failed: '+str(e)}
                rows.append(row)
    df=pd.DataFrame(rows)
    if 'final_val_L1' in df.columns: df=df.sort_values('final_val_L1', na_position='last')
    Path(exp_dir).mkdir(parents=True, exist_ok=True); df.to_csv(Path(exp_dir)/f'hparam_search_{int(time.time())}.csv', index=False)
    return df

def register_model(model_name, model_dir, registry_dir, alias='candidate', notes=''):
    registry_dir=Path(registry_dir); registry_dir.mkdir(parents=True, exist_ok=True)
    src=Path(model_dir)/(safe_name(model_name)+'.pt'); meta_src=Path(model_dir)/(safe_name(model_name)+'_metadata.json')
    if not src.exists(): raise FileNotFoundError('Train the model before registering it.')
    version=len(list(registry_dir.glob(safe_name(model_name)+'_v*.pt')))+1; dst=registry_dir/f'{safe_name(model_name)}_v{version}.pt'; shutil.copy2(src,dst)
    meta=json.loads(meta_src.read_text()) if meta_src.exists() else {}; meta.update({'version':version,'alias':alias,'notes':notes,'registry_model_path':str(dst)})
    (registry_dir/f'{safe_name(model_name)}_v{version}.json').write_text(json.dumps(meta,indent=2), encoding='utf-8'); return meta

def list_registry(registry_dir):
    rows=[]
    for p in sorted(Path(registry_dir).glob('*.json')):
        try: rows.append(json.loads(p.read_text(encoding='utf-8')))
        except Exception: pass
    return rows
