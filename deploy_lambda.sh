#!/bin/bash
set -e

# === CONFIG ===
AWS_REGION="ap-southeast-1"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
LAMBDA_FUNCTION_NAME="extract-panel-api"
ECR_REPO="${LAMBDA_FUNCTION_NAME}"
IMAGE_TAG="latest"

# === BUILD & PUSH ===
echo "🛠 Building Lambda container image..."
docker build -t ${ECR_REPO}:${IMAGE_TAG} -f Dockerfile.lambda .

# Tạo repo ECR nếu chưa có
aws ecr describe-repositories --repository-names ${ECR_REPO} --region ${AWS_REGION} >/dev/null 2>&1 || \
aws ecr create-repository --repository-name ${ECR_REPO} --region ${AWS_REGION} >/dev/null

# Login & push
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
docker tag ${ECR_REPO}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
docker push ${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}

# === DEPLOY LAMBDA ===
echo "🚀 Deploying Lambda..."
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# Tạo function nếu chưa có
if ! aws lambda get-function --function-name ${LAMBDA_FUNCTION_NAME} --region ${AWS_REGION} >/dev/null 2>&1; then
    aws lambda create-function \
      --function-name ${LAMBDA_FUNCTION_NAME} \
      --package-type Image \
      --code ImageUri=${IMAGE_URI} \
      --region ${AWS_REGION} \
      --role arn:aws:iam::${ACCOUNT_ID}:role/lambda-basic-execution
else
    aws lambda update-function-code \
      --function-name ${LAMBDA_FUNCTION_NAME} \
      --image-uri ${IMAGE_URI} \
      --region ${AWS_REGION} >/dev/null
fi

# === API GATEWAY AUTO SETUP ===
echo "🌐 Checking API Gateway..."

EXISTING_API_ID=$(aws apigatewayv2 get-apis --query "Items[?Name=='${LAMBDA_FUNCTION_NAME}-api'].ApiId" --output text --region ${AWS_REGION})

if [ -z "$EXISTING_API_ID" ]; then
  echo "⚙️ Creating new API Gateway..."
  API_JSON=$(aws apigatewayv2 create-api \
    --name "${LAMBDA_FUNCTION_NAME}-api" \
    --protocol-type HTTP \
    --target "arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}" \
    --region "${AWS_REGION}")
  API_ID=$(echo $API_JSON | jq -r '.ApiId')
  ENDPOINT=$(echo $API_JSON | jq -r '.ApiEndpoint')
else
  API_ID=$EXISTING_API_ID
  ENDPOINT=$(aws apigatewayv2 get-api --api-id $API_ID --query "ApiEndpoint" --output text --region ${AWS_REGION})
fi

echo "✅ Done!"
echo "Lambda: ${LAMBDA_FUNCTION_NAME}"
echo "Endpoint: ${ENDPOINT}"
