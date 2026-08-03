# ============================================================
# Hybrid RAG Development Environment Setup for Windows VM
# ============================================================
# Run this script as Administrator on the Azure VM.
#
# Installs: Python 3.13, Git, VS Code, Docker Desktop, Azure CLI,
#           Node.js, and clones + configures the project.
#
# Usage: Right-click PowerShell -> Run as Administrator
#        Then: .\vm-setup.ps1
# ============================================================

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Hybrid RAG Dev Environment Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Install Chocolatey (Windows package manager) ---
Write-Host "[1/8] Installing Chocolatey package manager..." -ForegroundColor Yellow
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Host "  Chocolatey installed." -ForegroundColor Green
} else {
    Write-Host "  Chocolatey already installed." -ForegroundColor Green
}

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# --- 2. Install Python 3.13 ---
Write-Host ""
Write-Host "[2/8] Installing Python 3.13..." -ForegroundColor Yellow
choco install python313 -y --no-progress
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
python --version

# --- 3. Install Git ---
Write-Host ""
Write-Host "[3/8] Installing Git..." -ForegroundColor Yellow
choco install git -y --no-progress
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
git --version

# --- 4. Install VS Code ---
Write-Host ""
Write-Host "[4/8] Installing Visual Studio Code..." -ForegroundColor Yellow
choco install vscode -y --no-progress
Write-Host "  VS Code installed." -ForegroundColor Green

# --- 5. Install Azure CLI ---
Write-Host ""
Write-Host "[5/8] Installing Azure CLI..." -ForegroundColor Yellow
choco install azure-cli -y --no-progress
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
Write-Host "  Azure CLI installed." -ForegroundColor Green

# --- 6. Install Docker Desktop ---
Write-Host ""
Write-Host "[6/8] Installing Docker Desktop..." -ForegroundColor Yellow
choco install docker-desktop -y --no-progress
Write-Host "  Docker Desktop installed (requires restart to complete)." -ForegroundColor Green

# --- 7. Clone the repository and set up the project ---
Write-Host ""
Write-Host "[7/8] Cloning repository and setting up project..." -ForegroundColor Yellow

$projectDir = "C:\Projects\hybrid-rag-capstone"
if (Test-Path $projectDir) {
    Write-Host "  Project directory already exists. Pulling latest..." -ForegroundColor Yellow
    Set-Location $projectDir
    git pull
} else {
    New-Item -ItemType Directory -Path "C:\Projects" -Force | Out-Null
    Set-Location "C:\Projects"
    git clone https://github.com/VinodPungle/hybrid-rag-capstone.git
    Set-Location $projectDir
}

# Create virtual environment
Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
& "$projectDir\venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "  Installing Python dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest

# --- 8. Create .env template ---
Write-Host ""
Write-Host "[8/8] Creating .env template..." -ForegroundColor Yellow

$envFile = "$projectDir\.env"
if (-not (Test-Path $envFile)) {
    @"
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_ENDPOINT=https://<your-embedding-resource>.cognitiveservices.azure.com/
AZURE_OPENAI_EMBEDDING_KEY=<your-embedding-key>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=embedding-model
AZURE_MODEL_NAME=gpt-4o-mini
AZURE_API_VERSION=2024-12-01-preview
AZURE_EMBEDDING_API_VERSION=2024-02-01

# Document
INPUT_FILE=data/raw/audit-committee-guide-2025.pdf

# Neo4j Configuration
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your-password>
"@ | Out-File -FilePath $envFile -Encoding utf8
    Write-Host "  .env template created at $envFile" -ForegroundColor Green
    Write-Host "  IMPORTANT: Edit .env with your actual credentials!" -ForegroundColor Red
} else {
    Write-Host "  .env already exists." -ForegroundColor Green
}

# --- Summary ---
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup COMPLETE!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Project location: $projectDir" -ForegroundColor White
Write-Host ""
Write-Host "  Installed:" -ForegroundColor White
Write-Host "    - Python 3.13" -ForegroundColor White
Write-Host "    - Git" -ForegroundColor White
Write-Host "    - Visual Studio Code" -ForegroundColor White
Write-Host "    - Azure CLI" -ForegroundColor White
Write-Host "    - Docker Desktop (restart required)" -ForegroundColor White
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "    1. RESTART the VM (required for Docker Desktop)" -ForegroundColor Yellow
Write-Host "    2. Edit C:\Projects\hybrid-rag-capstone\.env with your credentials" -ForegroundColor Yellow
Write-Host "    3. Open VS Code: code C:\Projects\hybrid-rag-capstone" -ForegroundColor Yellow
Write-Host "    4. Run tests: python -m pytest tests/ -v -m 'not integration'" -ForegroundColor Yellow
Write-Host "    5. Start API: uvicorn api.app:app --port 8000" -ForegroundColor Yellow
Write-Host "    6. Start UI:  streamlit run ui/app.py" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
