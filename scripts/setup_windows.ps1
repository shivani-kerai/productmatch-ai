# Session 1: create a Python 3.11 environment and install development dependencies.
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))

py -3.11 -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python 3.11 unavailable. Install Python 3.11 first." }

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }

& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
if ($LASTEXITCODE -ne 0) { throw "Failed to install ProductMatch AI." }

Write-Host "Setup complete. Run .\scripts\check_windows.ps1" -ForegroundColor Green
