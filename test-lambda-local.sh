#!/bin/bash
set -e

IMAGE_NAME="${IMAGE_NAME:-panel-design-lambda}"
IMAGE_TAG="${IMAGE_TAG:-test}"
CONTAINER_NAME="${CONTAINER_NAME:-lambda-test}"
PORT="${PORT:-8080}"

echo "=========================================="
echo "Lambda Local Test Script"
echo "=========================================="
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Port: ${PORT}"
echo "=========================================="

# Step 1: Build image
echo ""
echo "Step 1: Building Lambda image..."
docker build -f Dockerfile.lambda -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Step 2: Stop existing container (if any)
echo ""
echo "Step 2: Cleaning up existing container..."
docker stop ${CONTAINER_NAME} 2>/dev/null || true
docker rm ${CONTAINER_NAME} 2>/dev/null || true

# Step 3: Run container
echo ""
echo "Step 3: Starting Lambda container..."
docker run -d \
  --name ${CONTAINER_NAME} \
  -p ${PORT}:8080 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e LOG_LEVEL=DEBUG \
  ${IMAGE_NAME}:${IMAGE_TAG}

# Step 4: Wait for container to be ready
echo ""
echo "Step 4: Waiting for container to be ready..."
sleep 3

# Step 5: Test endpoints
echo ""
echo "Step 5: Testing endpoints..."
echo ""

# Test root endpoint
echo "Testing root endpoint..."
curl -s http://localhost:${PORT}/ | jq . || echo "Root endpoint failed"
echo ""

# Test health endpoint (if exists)
echo "Testing health endpoint..."
curl -s http://localhost:${PORT}/health | jq . || echo "Health endpoint not found"
echo ""

# Test docs endpoint
echo "Testing docs endpoint..."
curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:${PORT}/docs
echo ""

# Step 6: Show logs
echo ""
echo "Step 6: Container logs (last 20 lines)..."
docker logs --tail 20 ${CONTAINER_NAME}

echo ""
echo "=========================================="
echo "✅ Lambda container is running!"
echo "=========================================="
echo "Container: ${CONTAINER_NAME}"
echo "URL: http://localhost:${PORT}"
echo "Docs: http://localhost:${PORT}/docs"
echo ""
echo "Commands:"
echo "  View logs: docker logs -f ${CONTAINER_NAME}"
echo "  Stop: docker stop ${CONTAINER_NAME}"
echo "  Remove: docker rm ${CONTAINER_NAME}"
echo "=========================================="
