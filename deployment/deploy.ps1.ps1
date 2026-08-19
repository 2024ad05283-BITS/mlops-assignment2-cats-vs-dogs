<#
==============================================================
deploy.ps1

Windows Deployment Script

Responsibilities
----------------
1. Verify Docker is installed.
2. Verify Docker Compose is available.
3. Pull the latest Docker image.
4. Deploy using Docker Compose.
5. Verify container status.
6. Display service URL.
==============================================================
#>

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host " M4 Deployment Started"
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# ------------------------------------------------------------
# Check Docker
# ------------------------------------------------------------

Write-Host "[1/6] Checking Docker installation..."

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {

    Write-Host "ERROR: Docker is not installed." -ForegroundColor Red
    exit 1
}

docker --version

Write-Host ""

# ------------------------------------------------------------
# Check Docker Compose
# ------------------------------------------------------------

Write-Host "[2/6] Checking Docker Compose..."

docker compose version

if ($LASTEXITCODE -ne 0) {

    Write-Host "ERROR: Docker Compose not available." -ForegroundColor Red
    exit 1
}

Write-Host ""

# ------------------------------------------------------------
# Pull latest image
# ------------------------------------------------------------

Write-Host "[3/6] Pulling latest Docker image..."

docker compose pull

if ($LASTEXITCODE -ne 0) {

    Write-Host "ERROR: Failed to pull Docker image." -ForegroundColor Red
    exit 1
}

Write-Host ""

# ------------------------------------------------------------
# Deploy container
# ------------------------------------------------------------

Write-Host "[4/6] Starting deployment..."

docker compose up -d

if ($LASTEXITCODE -ne 0) {

    Write-Host "ERROR: Deployment failed." -ForegroundColor Red
    exit 1
}

Write-Host ""

# ------------------------------------------------------------
# Verify deployment
# ------------------------------------------------------------

Write-Host "[5/6] Checking running containers..."

docker compose ps

Write-Host ""

# ------------------------------------------------------------
# Deployment Complete
# ------------------------------------------------------------

Write-Host "==================================================" -ForegroundColor Green
Write-Host " Deployment Successful"
Write-Host "==================================================" -ForegroundColor Green

Write-Host ""
Write-Host "API URL:"
Write-Host "http://localhost:8000"
Write-Host ""

Write-Host "Swagger UI:"
Write-Host "http://localhost:8000/docs"
Write-Host ""

Write-Host "Health Endpoint:"
Write-Host "http://localhost:8000/health"
Write-Host ""

Write-Host "[6/6] Deployment Finished Successfully."

Write-Host ""