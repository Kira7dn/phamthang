# AWS Lambda Deployment Script for Windows PowerShell
# ====================================================

$ErrorActionPreference = "Stop"

# Configuration
$AWS_REGION = if ($env:AWS_REGION) { $env:AWS_REGION } else { "ap-southeast-1" }
$AWS_ACCOUNT_ID = if ($env:AWS_ACCOUNT_ID) { $env:AWS_ACCOUNT_ID } else { (aws sts get-caller-identity --query Account --output text) }
$ECR_REPO_NAME = if ($env:ECR_REPO_NAME) { $env:ECR_REPO_NAME } else { "panel-design-lambda" }
$LAMBDA_FUNCTION_NAME = if ($env:LAMBDA_FUNCTION_NAME) { $env:LAMBDA_FUNCTION_NAME } else { "panel-design-api" }
$LAMBDA_ROLE = if ($env:LAMBDA_ROLE) { $env:LAMBDA_ROLE } else { "arn:aws:iam::${AWS_ACCOUNT_ID}:role/lambda-execution-role" }
$IMAGE_TAG = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { "latest" }

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AWS Lambda Deployment Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Region: $AWS_REGION"
Write-Host "Account ID: $AWS_ACCOUNT_ID"
Write-Host "ECR Repo: $ECR_REPO_NAME"
Write-Host "Lambda Function: $LAMBDA_FUNCTION_NAME"
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Create ECR repository (if not exists)
Write-Host ""
Write-Host "Step 1: Creating ECR repository..." -ForegroundColor Yellow
try {
    aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION 2>$null
    Write-Host "Repository already exists" -ForegroundColor Green
} catch {
    aws ecr create-repository --repository-name $ECR_REPO_NAME --region $AWS_REGION
    Write-Host "Repository created" -ForegroundColor Green
}

# Step 2: Login to ECR
Write-Host ""
Write-Host "Step 2: Logging in to ECR..." -ForegroundColor Yellow
$loginPassword = aws ecr get-login-password --region $AWS_REGION
$loginPassword | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Step 3: Build Docker image with Docker format (not OCI) for Lambda compatibility
Write-Host ""
Write-Host "Step 3: Building Docker image with Docker format..." -ForegroundColor Yellow
docker buildx build `
    --platform linux/amd64 `
    --output type=docker `
    -f Dockerfile.lambda `
    -t "${ECR_REPO_NAME}:${IMAGE_TAG}" `
    .

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Step 4: Tag image
Write-Host ""
Write-Host "Step 4: Tagging image..." -ForegroundColor Yellow
$IMAGE_URI = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${ECR_REPO_NAME}:${IMAGE_TAG}"
docker tag "${ECR_REPO_NAME}:${IMAGE_TAG}" $IMAGE_URI

# Step 5: Push to ECR
Write-Host ""
Write-Host "Step 5: Pushing image to ECR..." -ForegroundColor Yellow
docker push $IMAGE_URI

if ($LASTEXITCODE -ne 0) {
    Write-Host "Push failed!" -ForegroundColor Red
    exit 1
}

# Step 6: Create or update Lambda function
Write-Host ""
Write-Host "Step 6: Deploying Lambda function..." -ForegroundColor Yellow

# Check if function exists
try {
    aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION 2>$null | Out-Null
    $functionExists = $true
} catch {
    $functionExists = $false
}

if ($functionExists) {
    Write-Host "Updating existing Lambda function..." -ForegroundColor Green
    aws lambda update-function-code `
        --function-name $LAMBDA_FUNCTION_NAME `
        --image-uri $IMAGE_URI `
        --region $AWS_REGION
    
    # Wait for update to complete
    Write-Host "Waiting for function update to complete..." -ForegroundColor Yellow
    aws lambda wait function-updated --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION
    
    # Update configuration
    aws lambda update-function-configuration `
        --function-name $LAMBDA_FUNCTION_NAME `
        --timeout 300 `
        --memory-size 2048 `
        --region $AWS_REGION
} else {
    Write-Host "Creating new Lambda function..." -ForegroundColor Green
    aws lambda create-function `
        --function-name $LAMBDA_FUNCTION_NAME `
        --package-type Image `
        --code ImageUri=$IMAGE_URI `
        --role $LAMBDA_ROLE `
        --timeout 300 `
        --memory-size 2048 `
        --region $AWS_REGION
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "Lambda deployment failed!" -ForegroundColor Red
    exit 1
}

# Step 7: Create Function URL (optional)
Write-Host ""
Write-Host "Step 7: Creating/updating Function URL..." -ForegroundColor Yellow
try {
    aws lambda create-function-url-config `
        --function-name $LAMBDA_FUNCTION_NAME `
        --auth-type NONE `
        --region $AWS_REGION 2>$null
} catch {
    Write-Host "Function URL already exists" -ForegroundColor Yellow
}

# Add permission for public access
try {
    aws lambda add-permission `
        --function-name $LAMBDA_FUNCTION_NAME `
        --statement-id FunctionURLAllowPublicAccess `
        --action lambda:InvokeFunctionUrl `
        --principal "*" `
        --function-url-auth-type NONE `
        --region $AWS_REGION 2>$null
} catch {
    Write-Host "Permission already exists" -ForegroundColor Yellow
}

# Get Function URL
$FUNCTION_URL = aws lambda get-function-url-config `
    --function-name $LAMBDA_FUNCTION_NAME `
    --region $AWS_REGION `
    --query 'FunctionUrl' `
    --output text

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "✅ Deployment completed successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Function URL: $FUNCTION_URL" -ForegroundColor Cyan
Write-Host "Test with: curl $FUNCTION_URL" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Green
