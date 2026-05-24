@echo off
call .venv\Scripts\activate
mlflow ui --backend-store-uri ./mlruns
pause
