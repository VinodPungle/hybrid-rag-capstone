#!/bin/bash
# ============================================================
# Full Deployment — Runs all steps sequentially
# ============================================================
# Deploys the complete Hybrid RAG solution to Azure Container Apps.
#
# Prerequisites:
#   - Azure CLI installed and logged in (az login)
#   - Docker Desktop running
#   - .env file with all secrets
#
# Usage: bash deploy/deploy-all.sh
# ============================================================

set -e

echo "============================================================"
echo "  Hybrid RAG — Full Azure Deployment"
echo "============================================================"
echo ""
echo "This script will:"
echo "  1. Create Azure infrastructure (Resource Group, ACR, Container Apps)"
echo "  2. Build and push Docker images"
echo "  3. Deploy FastAPI API"
echo "  4. Deploy Streamlit UI"
echo "  5. Verify the deployment"
echo ""
read -p "Continue? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "========== Step 1/5: Azure Infrastructure =========="
bash deploy/01-setup-azure.sh

echo ""
echo "========== Step 2/5: Build & Push Images =========="
bash deploy/02-build-push.sh

echo ""
echo "========== Step 3/5: Deploy API =========="
bash deploy/03-deploy-api.sh

echo ""
echo "========== Step 4/5: Deploy UI =========="
bash deploy/04-deploy-ui.sh

echo ""
echo "========== Step 5/5: Verify =========="
bash deploy/05-verify.sh
