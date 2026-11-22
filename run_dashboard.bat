@echo off
title Double Edge // Stealth Quantitative Macro Desk
cd /d "%~dp0"
echo Launching Double Edge Quantitative Macro Desk...
start http://127.0.0.1:8000/
py -3.12 app.py
pause
