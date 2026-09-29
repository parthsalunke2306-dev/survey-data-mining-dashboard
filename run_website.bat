@echo off
title Student Financial Habits - Web Portal (Pure Python)
echo ======================================================================
echo   Launching FinPulse Field Survey & Data Mining Web Portal
echo   Pure Python: FastAPI + Plotly + ID3/J48 Algorithms
echo   Address: http://localhost:8000
echo ======================================================================
cd /d "D:\Survey_Data_Mining_Dashboard"
call .venv\Scripts\activate.bat
python web_app.py
pause
