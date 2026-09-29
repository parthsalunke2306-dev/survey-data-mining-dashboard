@echo off
title Student Financial Habits - Survey Analytics & Data Mining Dashboard
echo ======================================================================
echo   Launching Field Survey Analytics & Decision Tree Mining Dashboard
echo   Location: D:\Survey_Data_Mining_Dashboard
echo ======================================================================
cd /d "D:\Survey_Data_Mining_Dashboard"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
