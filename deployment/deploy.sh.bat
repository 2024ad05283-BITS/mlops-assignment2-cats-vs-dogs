#!/bin/bash

# ==========================================================
# deploy.sh
#
# Linux/macOS Deployment Script
#
# Responsibilities
# ----------------
# 1. Verify Docker installation
# 2. Verify Docker Compose
# 3. Pull latest Docker image
# 4. Deploy application
# 5. Verify deployment
# ==========================================================

set -e

echo
echo "=================================================="
echo "        M4 Deployment Started"
echo "=================================================="
echo

# ----------------------------------------------------------
# Check Docker
# ----------------------------------------------------------

echo "[1/6] Checking Docker..."

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed."
    exit 1
fi

docker --version

echo

# ----------------------------------------------------------
# Check Docker Compose
# ----------------------------------------------------------

echo "[2/6] Checking Docker Compose..."

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: Docker Compose is not installed."
    exit 1
fi

docker compose version

echo

# ----------------------------------------------------------
# Pull Latest Image
# ----------------------------------------------------------

echo "[3/6] Pulling latest Docker image..."

docker compose pull

echo

# ----------------------------------------------------------
# Deploy Container
# ----------------------------------------------------------

echo "[4/6] Starting deployment..."

docker compose up -d

echo

# ----------------------------------------------------------
# Verify Deployment
# ----------------------------------------------------------

echo "[5/6] Checking running containers..."

docker compose ps

echo

# ----------------------------------------------------------
# Show Logs
# ----------------------------------------------------------

echo "[6/6] Recent container logs..."

docker compose logs --tail=20

echo
echo "=================================================="
echo "Deployment Successful"
echo "=================================================="
echo

echo "API URL"
echo "http://localhost:8000"

echo
echo "Swagger UI"
echo "http://localhost:8000/docs"

echo
echo "Health Endpoint"
echo "http://localhost:8000/health"

echo
echo "Deployment Finished Successfully."
echo