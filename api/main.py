from pathlib import Path
import sys, shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.inference import run_inference
app=FastAPI(title='Speech Audio AI API')
MODEL_DIR=ROOT/'outputs'/'models'; AUDIO_OUT=ROOT/'outputs'/'audio'; UPLOADS=ROOT/'outputs'/'api_uploads'; UPLOADS.mkdir(parents=True,exist_ok=True)
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/models')
def models(): return {'trained_models':[p.stem for p in MODEL_DIR.glob('*.pt')]}
@app.post('/denoise')
async def denoise(file: UploadFile = File(...), model_name: str = Form('CNN Denoiser')):
    inp=UPLOADS/file.filename
    with open(inp,'wb') as f: shutil.copyfileobj(file.file,f)
    try:
        res=run_inference(inp,model_name,MODEL_DIR,AUDIO_OUT)
        return FileResponse(res['output_path'], media_type='audio/wav', filename=Path(res['output_path']).name)
    except Exception as e:
        return JSONResponse({'error':str(e)}, status_code=500)
