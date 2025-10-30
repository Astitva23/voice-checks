# setup_and_run_device2.ps1
# Run from project folder (E:\new-voice\voice-checks\accent_app)
# Usage: Open PowerShell as normal user and run: .\setup_and_run_device2.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# 1) Create venv2 if missing
$venv = Join-Path $root ".venv2"
if (-Not (Test-Path $venv)) {
    Write-Host "Creating venv at $venv"
    python -m venv $venv
} else { Write-Host "Found venv2 already." }

# 2) Activate venv for the script
$activate = Join-Path $venv "Scripts\Activate.ps1"
if (-Not (Test-Path $activate)) {
    Write-Error "Cannot find $activate. Make sure Python is on PATH and venv created."
    exit 1
}
& $activate

# 3) Upgrade pip & wheel
python -m pip install --upgrade pip setuptools wheel

# 4) Install from requirements.txt
$req = Join-Path $root "requirements-device2.txt"
if (-Not (Test-Path $req)) {
    Write-Host "requirements-device2.txt not found. Writing default requirements."
    @"
flask
soundfile
sounddevice
numpy
scipy
librosa
torch
torchaudio
tqdm
pandas
requests
git+https://github.com/speechbrain/speechbrain.git
"@ | Out-File -Encoding ascii $req
}
Write-Host "Installing packages (may take a while)..."
python -m pip install -r $req

# 5) HF env vars (avoid symlink errors)
setx HF_HUB_DISABLE_SYMLINKS 1
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
$env:HF_HUB_DISABLE_SYMLINKS = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

# 6) (Optional) Check ffmpeg presence, show hint
$ff = (Get-Command ffmpeg -ErrorAction SilentlyContinue)
if (-not $ff) {
    Write-Warning "ffmpeg not found in PATH. Browser recordings (webm/ogg) may fail to convert. Install ffmpeg and add to PATH."
} else {
    Write-Host "ffmpeg detected: $ff"
}

# 7) Run the Flask app (foreground). Change file name if you use different app file.
Write-Host "`nStarting app_flask.py..."
python app_flask.py
