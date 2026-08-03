#!/bin/bash
# ============================================================
# Step 4: Deploy Streamlit UI to Azure Container Apps
# ============================================================
# Deploys the UI container and connects it to the API.
#
# Prerequisites:
#   - Steps 1, 2, 3 completed
#   - API is deployed and reachable
#
# Usage: bash deploy/04-deploy-ui.sh
# ============================================================

set -e

# --- Configuration ---
RESOURCE_GROUP="hybrid-rag-rg"
ACR_NAME="hybridragacr"
ENVIRONMENT_NAME="hybrid-rag-env"
API_APP_NAME="hybrid-rag-api"
UI_APP_NAME="hybrid-rag-ui"
ACR_SERVER="$ACR_NAME.azurecr.io"

echo "============================================"
echo "Step 4: Deploy Streamlit UI"
echo "============================================"

# 1. Get the API URL from the deployed API container
echo ""
echo "[1/3] Getting API URL..."
API_FQDN=$(az containerapp show \
  --name "$API_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

if [ -z "$API_FQDN" ]; then
    echo "ERROR: API container not found. Deploy the API first (Step 3)."
    exit 1
fi

API_URL="https://$API_FQDN"
echo "  API URL: $API_URL"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# 2. Deploy UI container
echo ""
echo "[2/3] Deploying UI container: $UI_APP_NAME..."
az containerapp create \
  --name "$UI_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENVIRONMENT_NAME" \
  --image "$ACR_SERVER/hybrid-rag-ui:latest" \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8501 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 2 \
  --cpu 0.5 --memory 1Gi \
  --env-vars \
    "API_BASE_URL=$API_URL" \
  --output table

# 3. Get the UI URL
echo ""
echo "[3/3] Getting UI URL..."
UI_FQDN=$(az containerapp show \
  --name "$UI_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

UI_URL="https://$UI_FQDN"

# Update API CORS to allow the UI origin
echo ""
echo "Updating API CORS to allow UI origin..."
az containerapp update \
  --name "$API_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --set-env-vars "ALLOWED_ORIGINS=$UI_URL,http://localhost:8501,http://localhost:8000" \
  --output table

echo ""
echo "============================================"
echo "Step 4 COMPLETE!"
echo "============================================"
echo "  UI URL:      $UI_URL"
echo "  API URL:     $API_URL"
echo "  Swagger UI:  $API_URL/docs"
echo ""
echo "TEST: Open $UI_URL in your browser."
echo ""
echo "NEXT (optional): Run 'bash deploy/05-verify.sh' to test the deployment."
echo "============================================"
