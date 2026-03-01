$ErrorActionPreference = "Stop"

Write-Host "=== REPO TEK KAYNAK MODU ===" -ForegroundColor Cyan

# 1) Kökler
$repoRoot = "F:\REPO_GitHub\Cagatay_22.02\Project"
$mainVenvActivate = "F:\Root\venv\Scripts\Activate.ps1"

# 2) AudioBridge'i repo icindeki worker'a zorla (F:\Project devre disi)
$env:AUDIO_WORKER_SCRIPT = Join-Path $repoRoot "core\audio_worker.py"
$env:VENV_AUDIO_PYTHON   = "F:\Root\venv_audio\Scripts\python.exe"

Write-Host ("AUDIO_WORKER_SCRIPT = " + $env:AUDIO_WORKER_SCRIPT) -ForegroundColor Yellow
Write-Host ("VENV_AUDIO_PYTHON   = " + $env:VENV_AUDIO_PYTHON) -ForegroundColor Yellow

if (!(Test-Path $env:AUDIO_WORKER_SCRIPT)) {
  throw "Repo icinde audio worker bulunamadi: $env:AUDIO_WORKER_SCRIPT"
}
if (!(Test-Path $env:VENV_AUDIO_PYTHON)) {
  throw "venv_audio python bulunamadi: $env:VENV_AUDIO_PYTHON"
}

# 3) Ana venv'i aktif et
Write-Host "Ana sanal ortam aktif ediliyor..." -ForegroundColor Green
if (!(Test-Path $mainVenvActivate)) {
  throw "Ana venv activate bulunamadi: $mainVenvActivate"
}
. $mainVenvActivate

# 4) Repo'ya gir ve calistir
Write-Host "Repo'ya geciliyor ve main.py baslatiliyor..." -ForegroundColor Green
Set-Location $repoRoot

# (Opsiyonel) Python'un nereden geldigini gorelim
Write-Host ("Python: " + (Get-Command python).Source) -ForegroundColor DarkGray

python .\main.py

# Konsol acik kalsin
Write-Host "`nBitti. Pencere acik kalacak." -ForegroundColor Cyan
powershell.exe -NoExit