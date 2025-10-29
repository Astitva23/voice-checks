@echo off
REM create venv
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
REM install packages (using PyTorch index)
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
echo.
echo Done. To use the venv: 
echo    Windows PowerShell: .\.venv\Scripts\Activate.ps1
echo    CMD: .venv\Scripts\activate.bat
pause
