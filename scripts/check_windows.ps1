# Session 1: run the same quality checks as GitHub Actions.
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Virtual environment missing. Run setup_windows.ps1." }

& $python -m ruff check .
if ($LASTEXITCODE -ne 0) { throw "Ruff lint failed." }

& $python -m ruff format --check .
if ($LASTEXITCODE -ne 0) { throw "Ruff format check failed." }

& $python -m pytest -q --cov=productmatch --cov-report=term-missing --cov-fail-under=85
if ($LASTEXITCODE -ne 0) { throw "Tests or coverage failed." }

& $python -m productmatch normalize "Dove Body-Wash 250ML"
if ($LASTEXITCODE -ne 0) { throw "CLI smoke test failed." }

Write-Host "All checks passed." -ForegroundColor Green
