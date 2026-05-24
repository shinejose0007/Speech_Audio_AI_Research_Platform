@echo off
call .venv\Scripts\activate
uvicorn api.main:app --reload
pause
