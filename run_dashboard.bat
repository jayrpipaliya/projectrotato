@echo off
title Quantitative Market Intelligence & Rotato
cd /d "%~dp0"
echo Launching Quantitative Market Dashboard...
py -3.12 -m streamlit run app.py
pause
