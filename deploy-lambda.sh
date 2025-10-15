#!/bin/bash
set -e

# Configuration
AWS_REGION="${AWS_REGION:-ap-southeast-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REPO_NAME="${ECR_REPO_NAME:-panel-design-lambda}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-panel-design-api}"
LAMBDA_ROLE="${LAMBDA_ROLE:-arn:aws:iam::${AWS_ACCOUNT_ID}:role/lambda-execution-role}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=========================================="
echo "AWS Lambda Deployment Script"
echo "=========================================="
echo "Region: ${AWS_REGION}"
echo "Account ID: ${AWS_ACCOUNT_ID}"
echo "ECR Repo: ${ECR_REPO_NAME}"
echo "Lambda Function: ${LAMBDA_FUNCTION_NAME}"
echo "=========================================="

# Step 1: Create ECR repository (if not exists)
echo ""
echo "Step 1: Creating ECR repository..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --region ${AWS_REGION} 2>/dev/null || \
  aws ecr create-repository --repository-name ${ECR_REPO_NAME} --region ${AWS_REGION}

# Step 2: Login to ECR
echo ""
echo "Step 2: Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 3: Build Docker image
echo ""
echo "Step 3: Building Docker image..."
# Use buildx with docker output format (not OCI) for Lambda compatibility
docker buildx build \
  --platform linux/amd64 \
  --output type=docker \
  -f Dockerfile.lambda \
  -t ${ECR_REPO_NAME}:${IMAGE_TAG} \
  .

# Step 4: Tag image
echo ""
echo "Step 4: Tagging image..."
docker tag ${ECR_REPO_NAME}:${IMAGE_TAG} \
  ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}

# Step 5: Push to ECR
echo ""
echo "Step 5: Pushing image to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}

# Step 6: Create or update Lambda function
echo ""
echo "Step 6: Deploying Lambda function..."
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"

# Check if function exists
if aws lambda get-function --function-name ${LAMBDA_FUNCTION_NAME} --region ${AWS_REGION} 2>/dev/null; then
  echo "Updating existing Lambda function..."
  aws lambda update-function-code \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --image-uri ${IMAGE_URI} \
    --region ${AWS_REGION}
  
  # Wait for update to complete
  echo "Waiting for function update to complete..."
  aws lambda wait function-updated --function-name ${LAMBDA_FUNCTION_NAME} --region ${AWS_REGION}
  
  # Update configuration
  aws lambda update-function-configuration \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --timeout 300 \
    --memory-size 2048 \
    --region ${AWS_REGION}
else
  echo "Creating new Lambda function..."
  aws lambda create-function \
    --function-name ${LAMBDA_FUNCTION_NAME} \
    --package-type Image \
    --code ImageUri=${IMAGE_URI} \
    --role ${LAMBDA_ROLE} \
    --timeout 300 \
    --memory-size 2048 \
    --region ${AWS_REGION}
fi

# Step 7: Create Function URL (optional)
echo ""
echo "Step 7: Creating/updating Function URL..."
aws lambda create-function-url-config \
  --function-name ${LAMBDA_FUNCTION_NAME} \
  --auth-type NONE \
  --region ${AWS_REGION} 2>/dev/null || \
  echo "Function URL already exists"

# Add permission for public access
aws lambda add-permission \
  --function-name ${LAMBDA_FUNCTION_NAME} \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE \
  --region ${AWS_REGION} 2>/dev/null || \
  echo "Permission already exists"

# Get Function URL
FUNCTION_URL=$(aws lambda get-function-url-config \
  --function-name ${LAMBDA_FUNCTION_NAME} \
  --region ${AWS_REGION} \
  --query 'FunctionUrl' \
  --output text)

echo ""
echo "=========================================="
echo "✅ Deployment completed successfully!"
echo "=========================================="
echo "Function URL: ${FUNCTION_URL}"
echo "Test with: curl ${FUNCTION_URL}"
echo "=========================================="
