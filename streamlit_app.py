from pathlib import Path
import sys, json, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st, torch
ROOT=Path(__file__).parent; sys.path.insert(0,str(ROOT))
from src.auth import register_user, verify_user, list_users, list_login_events
from src.audio_utils import read_wav, write_wav, wav_bytes, log_spectrogram, log_mel_spectrogram, apply_augmentation
from src.data import generate_demo_dataset, PairedAudioDataset
from src.training import train_model, load_trained_model, safe_name
from src.inference import run_inference, export_onnx
from src.metrics import all_metrics, snr_db
from src.models import create_model, count_parameters, model_size_mb
from src.reporting import export_report
from src.advanced_features import codec_benchmark, ffmpeg_mp3, quantize_onnx, log_run, list_runs, try_mlflow
from src.research_features import audit_dataset, run_asr_optional, extract_embedding_optional, wer, cer, run_hparam_search, register_model, list_registry
DATA_DEMO=ROOT/'data'/'demo'; DATA_CUSTOM=ROOT/'data'/'custom'; MODEL_DIR=ROOT/'outputs'/'models'; AUDIO_OUT=ROOT/'outputs'/'audio'; REPORT_OUT=ROOT/'outputs'/'reports'; ONNX_OUT=ROOT/'outputs'/'onnx'; QUANT_OUT=ROOT/'outputs'/'quantized'; CODEC_OUT=ROOT/'outputs'/'codec'; MIC_OUT=ROOT/'outputs'/'microphone'; EXP_DIR=ROOT/'experiments'; MLRUNS=ROOT/'mlruns'; DB=ROOT/'app_data'/'users.db'; REGISTRY=ROOT/'model_registry'
MODELS=['CNN Denoiser','LSTM Denoiser','Transformer Denoiser','Audio Codec Autoencoder']
st.set_page_config(page_title='Speech/Audio AI Research Platform', page_icon='🎧', layout='wide')

def login_gate():
    if 'logged_in' not in st.session_state: st.session_state.logged_in=False; st.session_state.user=None
    if st.session_state.logged_in:
        with st.sidebar:
            role=st.session_state.user.get('role','user'); st.success(f"{st.session_state.user['username']} ({role})")
            if st.button('Logout'): st.session_state.logged_in=False; st.session_state.user=None; st.rerun()
        return True
    st.title('🎧 Speech/Audio AI Research Platform')
    st.write('Local login/register using SQLite. First registered user becomes admin.')
    a,b=st.tabs(['Login','Register'])
    with a:
        u=st.text_input('Username'); p=st.text_input('Password',type='password')
        if st.button('Login'):
            ok,user=verify_user(DB,u,p)
            if ok: st.session_state.logged_in=True; st.session_state.user=user; st.rerun()
            else: st.error('Invalid login')
    with b:
        u=st.text_input('New username'); e=st.text_input('Email'); p=st.text_input('New password',type='password'); p2=st.text_input('Confirm password',type='password')
        if st.button('Create account'):
            if p!=p2: st.error('Passwords do not match')
            else:
                ok,msg=register_user(DB,u,e,p); st.success(msg) if ok else st.error(msg)
    return False

def counts(root): return len(list((root/'clean').glob('*.wav'))) if (root/'clean').exists() else 0, len(list((root/'noisy').glob('*.wav'))) if (root/'noisy').exists() else 0

def plot_wave(audio,sr,title):
    fig,ax=plt.subplots(figsize=(8,2.4)); t=np.arange(len(audio))/sr; ax.plot(t,audio); ax.set(title=title,xlabel='Time (s)',ylabel='Amplitude'); ax.grid(True,alpha=.3); fig.tight_layout(); return fig

def plot_spec(audio,sr,title,mode):
    fig,ax=plt.subplots(figsize=(8,3.2))
    if mode=='Mel':
        t,mel=log_mel_spectrogram(audio,sr); im=ax.imshow(mel,origin='lower',aspect='auto',extent=[0,len(audio)/sr,0,mel.shape[0]]); ax.set_ylabel('Mel band')
    else:
        f,t,s=log_spectrogram(audio,sr); im=ax.imshow(s,origin='lower',aspect='auto',extent=[0,len(audio)/sr,0,sr/2]); ax.set_ylabel('Frequency (Hz)')
    ax.set(title=title,xlabel='Time (s)'); fig.colorbar(im,ax=ax,label='dB'); fig.tight_layout(); return fig

def metadata(model):
    p=MODEL_DIR/(safe_name(model)+'_metadata.json'); return json.loads(p.read_text()) if p.exists() else None

def eval_dataset(root, model_name, max_files=10):
    ds=PairedAudioDataset(root); model,device,_=load_trained_model(model_name,MODEL_DIR); rows=[]
    for i in range(min(max_files,len(ds))):
        noisy_t,clean_t,name=ds[i]; noisy=noisy_t.squeeze().numpy(); clean=clean_t.squeeze().numpy()
        with torch.no_grad(): y=model(noisy_t.unsqueeze(0).to(device)); y=y[0] if model_name=='Audio Codec Autoencoder' else y
        enh=y.squeeze().cpu().numpy(); row=all_metrics(clean,noisy,enh,16000); row['file']=name; rows.append(row)
    return rows

def model_output(audio, model_name):
    model,device,_=load_trained_model(model_name,MODEL_DIR); x=audio[:16000] if len(audio)>=16000 else np.pad(audio,(0,16000-len(audio)))
    with torch.no_grad(): y=model(torch.tensor(x).float().unsqueeze(0).unsqueeze(0).to(device)); y=y[0] if model_name=='Audio Codec Autoencoder' else y
    return y.squeeze().cpu().numpy()

if not login_gate(): st.stop()
st.title('🎧 Neural Speech Enhancement, Audio Coding, ASR Evaluation and Edge Deployment Dashboard')
st.caption('Complete local portfolio platform: dataset audit, pretrained speech models, hyperparameter search, MLflow/model registry, ASR evaluation, FastAPI, ONNX web, Docker, admin dashboard.')
with st.sidebar:
    ds_choice=st.radio('Dataset',['Demo synthetic dataset','Custom uploaded dataset']); data_root=DATA_DEMO if ds_choice.startswith('Demo') else DATA_CUSTOM
    model_name=st.selectbox('Model',MODELS); epochs=st.slider('Epochs',1,30,5); batch=st.selectbox('Batch size',[2,4,8,16],index=2); lr=st.selectbox('Learning rate',[1e-4,5e-4,1e-3,2e-3],index=2); spec_mode=st.radio('Feature view',['STFT','Mel'],horizontal=True)
tabs=st.tabs(['Dataset+Audit','DSP','Train','Evaluate','Denoise/Coding','Codec','ASR+Pretrained','Hyperparameter','Quantization','Tracking+Registry','Robustness','Deployment','Admin+Report'])
with tabs[0]:
    st.header('Dataset + quality audit'); c,n=counts(data_root); a,b,c3=st.columns(3); a.metric('Clean',c); b.metric('Noisy',n); c3.metric('Selected','Demo' if data_root==DATA_DEMO else 'Custom')
    nfiles=st.slider('Demo file pairs',20,200,80,10); dur=st.slider('Duration',0.5,3.0,1.0,.5); snr=st.slider('Demo SNR dB',-2.0,20.0,4.0)
    if st.button('Generate demo dataset'): generate_demo_dataset(DATA_DEMO,n=nfiles,duration=dur,snr_db=snr); st.success('Generated demo dataset.')
    uc=st.file_uploader('Clean WAV',type=['wav']); un=st.file_uploader('Noisy WAV',type=['wav'])
    if uc and un and st.button('Save clean/noisy pair'):
        (DATA_CUSTOM/'clean').mkdir(parents=True,exist_ok=True); (DATA_CUSTOM/'noisy').mkdir(parents=True,exist_ok=True); name=f'custom_{int(time.time())}.wav'; (DATA_CUSTOM/'clean'/name).write_bytes(uc.getbuffer()); (DATA_CUSTOM/'noisy'/name).write_bytes(un.getbuffer()); st.success('Saved.')
    if st.button('Run quality audit'):
        st.dataframe(pd.DataFrame(audit_dataset(data_root)), use_container_width=True)
with tabs[1]:
    files=sorted((data_root/'noisy').glob('*.wav')) if (data_root/'noisy').exists() else []; st.header('DSP and features')
    if not files: st.warning('Generate/upload data first.')
    else:
        sel=st.selectbox('Sample',files,format_func=lambda p:p.name); sr,noisy=read_wav(sel); _,clean=read_wav(data_root/'clean'/sel.name); aug=st.selectbox('Augmentation',['None','White noise','Pink noise','50 Hz hum','Low-pass filter','High-pass filter','Telephone band 300-3400 Hz','Random volume','Small time shift']); noisy2=apply_augmentation(noisy,sr,aug)
        l,r=st.columns(2)
        with l: st.audio(wav_bytes(sr,clean),format='audio/wav'); st.pyplot(plot_wave(clean,sr,'Clean')); st.pyplot(plot_spec(clean,sr,'Clean '+spec_mode,spec_mode))
        with r: st.audio(wav_bytes(sr,noisy2),format='audio/wav'); st.pyplot(plot_wave(noisy2,sr,'Noisy/augmented')); st.pyplot(plot_spec(noisy2,sr,'Noisy '+spec_mode,spec_mode))
        st.metric('SNR vs clean', f'{snr_db(clean,noisy2):.2f} dB')
with tabs[2]:
    st.header('Train'); m=create_model(model_name); x,y,z=st.columns(3); x.metric('Model',model_name); y.metric('Parameters',f'{count_parameters(m):,}'); z.metric('Size',f'{model_size_mb(m):.3f} MB')
    if st.button('Train selected model'):
        prog=st.progress(0); box=st.empty(); logs=[]
        def cb(e,t,item): prog.progress(e/t); logs.append(item); box.dataframe(pd.DataFrame(logs),use_container_width=True)
        try:
            meta=train_model(data_root,model_name,MODEL_DIR,epochs,batch,lr,cb); st.success('Training completed'); st.json(meta); metrics={'final_val_L1':meta['logs'][-1]['val_L1'],'model_size_mb':meta['estimated_model_size_mb']}; log_run(EXP_DIR,'train_'+model_name,{'model':model_name,'epochs':epochs,'lr':lr},metrics); try_mlflow(MLRUNS,'train_'+model_name,{'model':model_name,'epochs':epochs,'lr':lr},metrics)
        except Exception as e: st.error(str(e))
with tabs[3]:
    st.header('Evaluate + leaderboard'); limit=st.slider('Files to evaluate',1,30,10)
    if st.button('Evaluate selected model'):
        try:
            df=pd.DataFrame(eval_dataset(data_root,model_name,limit)); st.dataframe(df,use_container_width=True); REPORT_OUT.mkdir(parents=True,exist_ok=True); p=REPORT_OUT/(safe_name(model_name)+'_metrics.csv'); df.to_csv(p,index=False); st.download_button('Download CSV',p.read_bytes(),file_name=p.name)
        except Exception as e: st.error(str(e))
    if st.button('Build leaderboard'):
        rows=[]
        for mo in MODELS:
            try: df=pd.DataFrame(eval_dataset(data_root,mo,5)); rows.append({'model':mo,'mean_SNR_improvement':round(pd.to_numeric(df['SNR improvement (dB)']).mean(),2), **(metadata(mo) or {})})
            except Exception: rows.append({'model':mo,'status':'not trained'})
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
with tabs[4]:
    st.header('Denoise / neural audio coding'); files=sorted((data_root/'noisy').glob('*.wav')) if (data_root/'noisy').exists() else []; up=st.file_uploader('Upload WAV for inference',type=['wav'],key='infer'); path=None
    if up: AUDIO_OUT.mkdir(parents=True,exist_ok=True); path=AUDIO_OUT/'uploaded.wav'; path.write_bytes(up.getbuffer())
    elif files: path=st.selectbox('Or choose sample',files,format_func=lambda p:p.name,key='infs')
    if path:
        sr,x=read_wav(path); st.audio(wav_bytes(sr,x),format='audio/wav')
        if st.button('Run inference'):
            try:
                res=run_inference(path,model_name,MODEL_DIR,AUDIO_OUT); st.metric('Inference',str(res['inference_ms'])+' ms'); st.metric('Real-time factor',res['real_time_factor']); st.audio(wav_bytes(res['sr'],res['output_audio']),format='audio/wav'); st.download_button('Download output',Path(res['output_path']).read_bytes(),file_name=Path(res['output_path']).name)
            except Exception as e: st.error(str(e))
with tabs[5]:
    st.header('Classical codec vs neural codec'); files=sorted((data_root/'noisy').glob('*.wav')) if (data_root/'noisy').exists() else []
    if not files: st.warning('Generate/upload data first.')
    else:
        sel=st.selectbox('Codec sample',files,format_func=lambda p:p.name,key='codec'); sr,noisy=read_wav(sel); _,clean=read_wav(data_root/'clean'/sel.name)
        if st.button('Run codec benchmark'):
            try:
                enh=model_output(noisy,model_name); rows,vars=codec_benchmark(clean,noisy,sr,enh); st.dataframe(pd.DataFrame(rows),use_container_width=True); CODEC_OUT.mkdir(parents=True,exist_ok=True)
                for name,audio in vars.items(): write_wav(CODEC_OUT/(sel.stem+'_'+name.replace(' ','_')+'.wav'),sr,audio)
            except Exception as e: st.error(str(e))
        if st.button('Optional MP3 encode with FFmpeg'):
            out,msg=ffmpeg_mp3(sel,CODEC_OUT); st.success(str(out)) if out else st.error(msg)
with tabs[6]:
    st.header('ASR before/after + pretrained embeddings'); st.info('Install optional_requirements_research.txt for ASR/Wav2Vec2 features.')
    files=sorted((data_root/'noisy').glob('*.wav')) if (data_root/'noisy').exists() else []
    if files:
        sel=st.selectbox('ASR sample',files,format_func=lambda p:p.name,key='asr'); sr,noisy=read_wav(sel); st.audio(wav_bytes(sr,noisy),format='audio/wav'); ref=st.text_area('Optional reference transcript for WER/CER','')
        if st.button('Run ASR before/after denoising'):
            try:
                den=model_output(noisy,model_name); denp=AUDIO_OUT/(sel.stem+'_denoised_for_asr.wav'); write_wav(denp,sr,den); ok1,t1=run_asr_optional(sel); ok2,t2=run_asr_optional(denp); st.write('Noisy transcript:',t1); st.write('Denoised transcript:',t2)
                if ref and ok1 and ok2: st.metric('Noisy WER',round(wer(ref,t1),3)); st.metric('Denoised WER',round(wer(ref,t2),3)); st.metric('Denoised CER',round(cer(ref,t2),3))
            except Exception as e: st.error(str(e))
        if st.button('Extract Wav2Vec2 embedding'):
            ok,res=extract_embedding_optional(sel); st.json(res) if ok else st.warning(res)
with tabs[7]:
    st.header('Hyperparameter tuning'); mods=st.multiselect('Models',MODELS,default=['CNN Denoiser']); lrs=st.multiselect('Learning rates',[1e-4,5e-4,1e-3],default=[1e-3]); batches=st.multiselect('Batch sizes',[2,4,8],default=[4]); ep=st.slider('Epochs per trial',1,5,2)
    if st.button('Run grid search'):
        prog=st.progress(0); status=st.empty()
        def cb(i,t,msg): prog.progress(i/t); status.info(msg)
        try: st.dataframe(run_hparam_search(data_root,MODEL_DIR,EXP_DIR,mods,lrs,batches,ep,cb),use_container_width=True)
        except Exception as e: st.error(str(e))
with tabs[8]:
    st.header('ONNX export + INT8 quantization')
    if st.button('Export FP32 ONNX'):
        try: p=export_onnx(model_name,MODEL_DIR,ONNX_OUT); st.success(str(p)); st.download_button('Download ONNX',p.read_bytes(),file_name=p.name)
        except Exception as e: st.error(str(e))
    if st.button('Export + quantize INT8'):
        try: fp=export_onnx(model_name,MODEL_DIR,ONNX_OUT); res=quantize_onnx(fp,QUANT_OUT); st.json(res); p=Path(res['int8_path']); st.download_button('Download INT8 ONNX',p.read_bytes(),file_name=p.name)
        except Exception as e: st.error(str(e))
with tabs[9]:
    st.header('Tracking + model registry'); df=list_runs(EXP_DIR); st.dataframe(df,use_container_width=True) if not df.empty else st.info('No runs yet.'); st.code('mlflow ui --backend-store-uri ./mlruns')
    alias=st.selectbox('Registry alias',['candidate','production','archived']); notes=st.text_input('Notes','')
    if st.button('Register selected model'):
        try: st.json(register_model(model_name,MODEL_DIR,REGISTRY,alias,notes))
        except Exception as e: st.error(str(e))
    rows=list_registry(REGISTRY); st.dataframe(pd.DataFrame(rows),use_container_width=True) if rows else st.info('No registered models yet.')
with tabs[10]:
    st.header('Robustness evaluation'); files=sorted((data_root/'noisy').glob('*.wav')) if (data_root/'noisy').exists() else []
    if files:
        sel=st.selectbox('Robustness sample',files,format_func=lambda p:p.name,key='rob'); sr,_=read_wav(sel); _,clean=read_wav(data_root/'clean'/sel.name)
        if st.button('Run robustness test'):
            rows=[]
            for aug in ['White noise','Pink noise','50 Hz hum','Low-pass filter','High-pass filter','Telephone band 300-3400 Hz','Random volume','Small time shift']:
                corrupted=apply_augmentation(clean,sr,aug); enh=model_output(corrupted,model_name); rows.append({'condition':aug, **all_metrics(clean,corrupted,enh,sr)})
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
with tabs[11]:
    st.header('Deployment: FastAPI, C++, browser ONNX, Docker'); st.code('uvicorn api.main:app --reload'); st.write('FastAPI docs: http://127.0.0.1:8000/docs'); st.code('cd cpp_inference_demo\nmkdir build\ncd build\ncmake ..\ncmake --build .'); st.code('docker compose up --build'); st.write('Browser ONNX template: open onnx_web_demo/index.html after exporting a model.')
with tabs[12]:
    st.header('Admin dashboard + report')
    if st.session_state.user and st.session_state.user.get('role')=='admin':
        st.subheader('Users'); st.dataframe(pd.DataFrame(list_users(DB)),use_container_width=True); st.subheader('Login events'); st.dataframe(pd.DataFrame(list_login_events(DB)),use_container_width=True)
        a,b,c=st.columns(3); a.metric('Trained models',len(list(MODEL_DIR.glob('*.pt')))); b.metric('ONNX files',len(list(ONNX_OUT.glob('*.onnx')))); c.metric('Experiment logs',len(list(EXP_DIR.glob('*.json'))))
    else: st.info('Admin dashboard visible only for admin user.')
    if st.button('Create Markdown report'):
        meta=metadata(model_name) or {'model_name':model_name,'note':'No trained metadata'}
        try: rows=eval_dataset(data_root,model_name,5)
        except Exception: rows=[]
        p=export_report(REPORT_OUT/(safe_name(model_name)+'_complete_report.md'),'Speech/Audio AI Research Platform Report',meta,rows); st.download_button('Download report',p.read_bytes(),file_name=p.name)
    st.subheader('CV project line'); st.code('Neural Speech Enhancement, Audio Coding, ASR Evaluation and Edge Deployment Dashboard | PyTorch, Streamlit, ONNX, FastAPI, MLflow, SQLite\nDeveloped a self-directed speech/audio AI research platform with neural denoising, experimental audio coding, ASR before/after evaluation, pretrained Wav2Vec2 embedding extraction, dataset quality auditing, codec benchmarking, ONNX INT8 quantization, microphone inference, MLflow tracking/model registry, robustness testing, FastAPI backend, C++ and browser deployment demo structure.')
