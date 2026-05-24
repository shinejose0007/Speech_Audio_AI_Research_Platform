@echo off
echo Starting Neural Speech Audio Coding Pro Dashboard...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
pause
