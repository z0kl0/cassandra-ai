Write-Host ""
Write-Host "============================================================"
Write-Host " CASSANDRA - local-first environment setup (PowerShell)"
Write-Host "============================================================"
Write-Host ""

Write-Host "[CASSANDRA] Checking for Ollama..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "[CASSANDRA] Ollama is already installed." -ForegroundColor Green
} else {
    Write-Host "[CASSANDRA] Ollama not found. Attempting to install via winget..." -ForegroundColor Yellow
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[CASSANDRA] Ollama installed successfully. Please restart your terminal after setup." -ForegroundColor Green
    } else {
        Write-Warning "[CASSANDRA] Winget installation failed or was cancelled."
        Write-Host "Please install manually from https://ollama.com/download"
    }
}

Write-Host "[CASSANDRA] Creating virtual environment (.venv)..."
python -m venv .venv

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "[CASSANDRA] 'python' command failed to create files. Trying 'py' launcher..."
    py -m venv .venv
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Error "[ERROR] Could not create venv. Type 'python -m venv .venv' manually in your terminal to see the exact error."
    exit 1
}

Write-Host "[CASSANDRA] Activating venv..."
# In PowerShell, we dot-source the activation script
. .\.venv\Scripts\Activate.ps1

Write-Host "[CASSANDRA] Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "[CASSANDRA] Installing core dependencies..."
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "[ERROR] Dependency install failed."
    exit 1
}

Write-Host ""
Write-Host "[CASSANDRA] Installation complete. Verifying packages..."
python -m pip list

if (-not (Test-Path ".env")) {
    Write-Host "[CASSANDRA] Seeding .env from .env.example..."
    Copy-Item -Path ".env.example" -Destination ".env" -Force
} else {
    Write-Host "[CASSANDRA] .env already exists - leaving it untouched."
}

Write-Host ""
Write-Host "============================================================"
Write-Host " [CASSANDRA] Setup Environment Complete."
Write-Host ""
Write-Host " NEXT STEPS:"
Write-Host "  1. Restart Terminal       Required to refresh PATH if Ollama was just installed"
Write-Host "  2. Pull Model             Run: ollama pull llama3"
Write-Host "  3. Configure Secrets      Edit .env and set SEC_USER_AGENT"
Write-Host "  4. Launch Application     Run: streamlit run app.py"
Write-Host ""
Write-Host "============================================================"
