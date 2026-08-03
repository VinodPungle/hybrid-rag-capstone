#!/bin/bash
# ============================================================
# Step 3: Deploy FastAPI API to Azure Container Apps
# ============================================================
# Deploys the API container with all required environment variables.
#
# Prerequisites:
#   - Step 1 & 2 completed
#   - .env file exists with all secrets
#
# Usage: bash deploy/03-deploy-api.sh
# ============================================================

set -e

# --- Configuration ---
RESOURCE_GROUP="hybrid-rag-rg"
ACR_NAME="hybridragacr"
ENVIRONMENT_NAME="hybrid-rag-env"
API_APP_NAME="hybrid-rag-api"
ACR_SERVER="$ACR_NAME.azurecr.io"

echo "============================================"
echo "Step 3: Deploy FastAPI API"
echo "============================================"

# 1. Read secrets from .env file
echo ""
echo "[1/3] Reading secrets from .env file..."
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Create it with your Azure OpenAI and Neo4j credentials."
    exit 1
fi

# Source .env variables (handle spaces and quotes)
export $(grep -v '^#' .env | grep -v '^\s*$' | sed 's/ *= */=/g' | xargs)

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# 2. Deploy API container
echo ""
echo "[2/3] Deploying API container: $API_APP_NAME..."
az containerapp create \
  --name "$API_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENVIRONMENT_NAME" \
  --image "$ACR_SERVER/hybrid-rag-api:latest" \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 --memory 2Gi \
  --env-vars \
    "AZURE_OPENAI_ENDPOINT=$AZURE_OPENAI_ENDPOINT" \
    "AZURE_OPENAI_KEY=$AZURE_OPENAI_KEY" \
    "AZURE_OPENAI_DEPLOYMENT=$AZURE_OPENAI_DEPLOYMENT" \
    "AZURE_OPENAI_EMBEDDING_ENDPOINT=$AZURE_OPENAI_EMBEDDING_ENDPOINT" \
    "AZURE_OPENAI_EMBEDDING_KEY=$AZURE_OPENAI_EMBEDDING_KEY" \
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT=$AZURE_OPENAI_EMBEDDING_DEPLOYMENT" \
    "AZURE_API_VERSION=$AZURE_API_VERSION" \
    "AZURE_EMBEDDING_API_VERSION=$AZURE_EMBEDDING_API_VERSION" \
    "NEO4J_URI=$NEO4J_URI" \
    "NEO4J_USER=$NEO4J_USER" \
    "NEO4J_PASSWORD=$NEO4J_PASSWORD" \
    "ALLOWED_ORIGINS=*" \
  --output table

# 3. Get the API URL
echo ""
echo "[3/3] Getting API URL..."
API_FQDN=$(az containerapp show \
  --name "$API_APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

API_URL="https://$API_FQDN"

echo ""
echo "============================================"
echo "Step 3 COMPLETE!"
echo "============================================"
echo "  API URL:     $API_URL"
echo "  Health:      $API_URL/health"
echo "  Swagger UI:  $API_URL/docs"
echo "  Metrics:     $API_URL/metrics"
echo ""
echo "TEST: Run the following to verify the API is working:"
echo "  curl $API_URL/health"
echo ""
echo "NEXT: Run 'bash deploy/04-deploy-ui.sh' to deploy the UI."
echo "============================================"
