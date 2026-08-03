#!/bin/bash
# ============================================================
# Step 1: Azure Infrastructure Setup
# ============================================================
# Creates: Resource Group, Container Registry, Container Apps Environment
#
# Prerequisites:
#   - Azure CLI installed (https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
#   - Logged in: az login
#   - Subscription selected: az account set --subscription "<id>"
#
# Usage: bash deploy/01-setup-azure.sh
# ============================================================

set -e  # Exit on any error

# --- Configuration (modify these for your environment) ---
RESOURCE_GROUP="hybrid-rag-rg"
LOCATION="eastus2"
ACR_NAME="hybridragacr"
ENVIRONMENT_NAME="hybrid-rag-env"
LOG_WORKSPACE="hybrid-rag-logs"

echo "============================================"
echo "Step 1: Azure Infrastructure Setup"
echo "============================================"

# 1. Create Resource Group
echo ""
echo "[1/4] Creating Resource Group: $RESOURCE_GROUP in $LOCATION..."
az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output table

# 2. Create Azure Container Registry
echo ""
echo "[2/4] Creating Azure Container Registry: $ACR_NAME..."
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --output table

# 3. Create Log Analytics Workspace (required for Container Apps)
echo ""
echo "[3/4] Creating Log Analytics Workspace: $LOG_WORKSPACE..."
az monitor log-analytics workspace create \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$LOG_WORKSPACE" \
  --output table

# Get workspace credentials for Container Apps
LOG_WORKSPACE_ID=$(az monitor log-analytics workspace show \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$LOG_WORKSPACE" \
  --query customerId -o tsv)

LOG_WORKSPACE_KEY=$(az monitor log-analytics workspace get-shared-keys \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$LOG_WORKSPACE" \
  --query primarySharedKey -o tsv)

# 4. Create Container Apps Environment
echo ""
echo "[4/4] Creating Container Apps Environment: $ENVIRONMENT_NAME..."
az containerapp env create \
  --name "$ENVIRONMENT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --logs-workspace-id "$LOG_WORKSPACE_ID" \
  --logs-workspace-key "$LOG_WORKSPACE_KEY" \
  --output table

echo ""
echo "============================================"
echo "Step 1 COMPLETE!"
echo "============================================"
echo "  Resource Group:    $RESOURCE_GROUP"
echo "  Location:          $LOCATION"
echo "  Container Registry: $ACR_NAME.azurecr.io"
echo "  Environment:       $ENVIRONMENT_NAME"
echo ""
echo "NEXT: Run 'bash deploy/02-build-push.sh' to build and push Docker images."
echo "============================================"
