@echo off
title Quantitative Market Intelligence & Double Edge
cd /d "%~dp0"
echo Launching Quantitative Market Dashboard...
py -3.12 -m streamlit run app.py
pause
