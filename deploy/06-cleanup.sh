#!/bin/bash
# ============================================================
# Step 6: Cleanup — Delete All Azure Resources
# ============================================================
# Deletes the entire resource group and all resources within it.
# WARNING: This is irreversible!
#
# Usage: bash deploy/06-cleanup.sh
# ============================================================

RESOURCE_GROUP="hybrid-rag-rg"

echo "============================================"
echo "WARNING: This will delete ALL resources in:"
echo "  Resource Group: $RESOURCE_GROUP"
echo ""
echo "This includes:"
echo "  - Container Apps (API + UI)"
echo "  - Container Registry (all images)"
echo "  - Container Apps Environment"
echo "  - Log Analytics Workspace"
echo "============================================"
echo ""
read -p "Are you sure? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Deleting resource group: $RESOURCE_GROUP..."
az group delete --name "$RESOURCE_GROUP" --yes --no-wait

echo ""
echo "============================================"
echo "Deletion initiated (runs in background)."
echo "Check status: az group show --name $RESOURCE_GROUP"
echo "============================================"
