#!/bin/bash
# ============================================================
# Step 2: Build and Push Docker Images to Azure Container Registry
# ============================================================
# Builds separate images for API and UI, pushes to ACR.
#
# Prerequisites:
#   - Step 1 completed (Azure infrastructure created)
#   - Docker Desktop running
#
# Usage: bash deploy/02-build-push.sh
# ============================================================

set -e

# --- Configuration ---
RESOURCE_GROUP="hybrid-rag-rg"
ACR_NAME="hybridragacr"

echo "============================================"
echo "Step 2: Build and Push Docker Images"
echo "============================================"

# 1. Login to Azure Container Registry
echo ""
echo "[1/5] Logging in to ACR: $ACR_NAME..."
az acr login --name "$ACR_NAME"

ACR_SERVER="$ACR_NAME.azurecr.io"

# 2. Build API image
echo ""
echo "[2/5] Building API image..."
docker build -f Dockerfile.api -t "$ACR_SERVER/hybrid-rag-api:latest" .

# 3. Build UI image
echo ""
echo "[3/5] Building UI image..."
docker build -f Dockerfile.ui -t "$ACR_SERVER/hybrid-rag-ui:latest" .

# 4. Push API image
echo ""
echo "[4/5] Pushing API image to ACR..."
docker push "$ACR_SERVER/hybrid-rag-api:latest"

# 5. Push UI image
echo ""
echo "[5/5] Pushing UI image to ACR..."
docker push "$ACR_SERVER/hybrid-rag-ui:latest"

echo ""
echo "============================================"
echo "Step 2 COMPLETE!"
echo "============================================"
echo "  API image: $ACR_SERVER/hybrid-rag-api:latest"
echo "  UI image:  $ACR_SERVER/hybrid-rag-ui:latest"
echo ""
echo "Verify images in ACR:"
echo "  az acr repository list --name $ACR_NAME --output table"
echo ""
echo "NEXT: Run 'bash deploy/03-deploy-api.sh' to deploy the API."
echo "============================================"
