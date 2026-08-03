#!/bin/bash
# ============================================================
# Step 5: Verify Azure Deployment
# ============================================================
# Tests all deployed endpoints and shows status.
#
# Usage: bash deploy/05-verify.sh
# ============================================================

set -e

# --- Configuration ---
RESOURCE_GROUP="hybrid-rag-rg"
API_APP_NAME="hybrid-rag-api"
UI_APP_NAME="hybrid-rag-ui"

echo "============================================"
echo "Step 5: Verify Azure Deployment"
echo "============================================"

# 1. Get URLs
API_FQDN=$(az containerapp show --name "$API_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)
UI_FQDN=$(az containerapp show --name "$UI_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)

API_URL="https://$API_FQDN"
UI_URL="https://$UI_FQDN"

echo ""
echo "API URL: $API_URL"
echo "UI URL:  $UI_URL"

# 2. Test Health Endpoint
echo ""
echo "--- Test 1: Health Check ---"
HEALTH_RESP=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "$API_URL/health" 2>/dev/null)
HTTP_STATUS=$(echo "$HEALTH_RESP" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$HEALTH_RESP" | grep -v "HTTP_STATUS")

if [ "$HTTP_STATUS" == "200" ]; then
    echo "  PASS: Health endpoint returned 200"
    echo "  Response: $BODY"
else
    echo "  FAIL: Health endpoint returned $HTTP_STATUS"
    echo "  Response: $BODY"
fi

# 3. Test Swagger UI
echo ""
echo "--- Test 2: Swagger UI ---"
SWAGGER_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs" 2>/dev/null)
if [ "$SWAGGER_STATUS" == "200" ]; then
    echo "  PASS: Swagger UI is accessible"
else
    echo "  FAIL: Swagger UI returned $SWAGGER_STATUS"
fi

# 4. Test Metrics Endpoint
echo ""
echo "--- Test 3: Metrics Endpoint ---"
METRICS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/metrics" 2>/dev/null)
if [ "$METRICS_STATUS" == "200" ]; then
    echo "  PASS: Metrics endpoint is accessible"
else
    echo "  FAIL: Metrics endpoint returned $METRICS_STATUS"
fi

# 5. Test UI
echo ""
echo "--- Test 4: Streamlit UI ---"
UI_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$UI_URL" 2>/dev/null)
if [ "$UI_STATUS" == "200" ]; then
    echo "  PASS: Streamlit UI is accessible"
else
    echo "  FAIL: Streamlit UI returned $UI_STATUS"
fi

# 6. Test Ingest (upload sample PDF)
echo ""
echo "--- Test 5: Document Ingestion ---"
if [ -f "data/raw/Renewable Energy Adoption in India.pdf" ]; then
    echo "  Uploading PDF to $API_URL/ingest..."
    INGEST_RESP=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL/ingest" \
      -F "file=@data/raw/Renewable Energy Adoption in India.pdf" 2>/dev/null)
    HTTP_STATUS=$(echo "$INGEST_RESP" | grep "HTTP_STATUS" | cut -d: -f2)
    BODY=$(echo "$INGEST_RESP" | grep -v "HTTP_STATUS")
    if [ "$HTTP_STATUS" == "200" ]; then
        echo "  PASS: Document ingested successfully"
        echo "  Response: $BODY"
    else
        echo "  FAIL: Ingestion returned $HTTP_STATUS"
        echo "  Response: $BODY"
    fi
else
    echo "  SKIP: Sample PDF not found locally"
fi

# 7. Test Ask
echo ""
echo "--- Test 6: Question Answering ---"
ASK_RESP=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$API_URL/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What are the key challenges in renewable energy adoption?"}' 2>/dev/null)
HTTP_STATUS=$(echo "$ASK_RESP" | grep "HTTP_STATUS" | cut -d: -f2)
BODY=$(echo "$ASK_RESP" | grep -v "HTTP_STATUS")
if [ "$HTTP_STATUS" == "200" ]; then
    echo "  PASS: Question answered successfully"
    echo "  Response (truncated): $(echo "$BODY" | python -c "import sys,json; d=json.load(sys.stdin); print(d['answer'][:200])" 2>/dev/null || echo "$BODY" | head -c 200)"
else
    echo "  INFO: Ask returned $HTTP_STATUS (expected if no document ingested via Test 5)"
fi

# 8. Container status
echo ""
echo "--- Container Status ---"
echo ""
echo "API Container:"
az containerapp show --name "$API_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "{name:name, status:properties.runningStatus, replicas:properties.template.scale}" \
  -o table 2>/dev/null || echo "  Not found"

echo ""
echo "UI Container:"
az containerapp show --name "$UI_APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query "{name:name, status:properties.runningStatus, replicas:properties.template.scale}" \
  -o table 2>/dev/null || echo "  Not found"

echo ""
echo "============================================"
echo "Verification COMPLETE!"
echo "============================================"
echo ""
echo "Access your deployed solution:"
echo "  Streamlit UI:  $UI_URL"
echo "  Swagger API:   $API_URL/docs"
echo "  Health Check:  $API_URL/health"
echo "  Metrics:       $API_URL/metrics"
echo "============================================"
