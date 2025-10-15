# AWS Lambda Deployment Guide

## 📋 Overview

Deploy FastAPI application to AWS Lambda using container images with AWS Lambda Web Adapter.

## 🏗️ Architecture

```
Client Request
    ↓
AWS Lambda Function URL / API Gateway
    ↓
Lambda Runtime (Container - Amazon Linux 2023)
    ↓
AWS Lambda Web Adapter (/opt/extensions/lambda-adapter)
    ↓
FastAPI (uvicorn on port 8080)
    ↓
Your Application Code
```

## ⚠️ Important Notes

- **Base Image**: Uses `public.ecr.aws/lambda/python:3.12` (Amazon Linux 2023)
- **Package Manager**: Uses `dnf` (not `yum` - AL2023 change)
- **Lambda Web Adapter**: Version v0.8.4 (direct binary, not zip)
- **No find command**: Use Python for file operations or rely on ENV vars

## 📦 Files

- **`Dockerfile.lambda`**: Optimized Dockerfile for AWS Lambda
- **`lambda-entrypoint.sh`**: Lambda entrypoint wrapper script
- **`deploy-lambda.sh`**: Automated deployment script
- **`test-lambda-local.sh`**: Local testing script
- **`app/requirements.prod.txt`**: Python dependencies

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install AWS CLI
pip install awscli

# Configure AWS credentials
aws configure
```

### 2. Set Environment Variables (Optional)

```bash
export AWS_REGION=ap-southeast-1
export ECR_REPO_NAME=panel-design-lambda
export LAMBDA_FUNCTION_NAME=panel-design-api
export LAMBDA_ROLE=arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role
```

### 3. Deploy

```bash
# Automated deployment
./deploy-lambda.sh

# Or manual steps below
```

## 🛠️ Manual Deployment Steps

### Step 1: Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name panel-design-lambda \
  --region ap-southeast-1
```

### Step 2: Login to ECR

```bash
aws ecr get-login-password --region ap-southeast-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com
```

### Step 3: Build Docker Image

```bash
docker build -f Dockerfile.lambda -t panel-design-lambda:latest .
```

### Step 4: Tag Image

```bash
docker tag panel-design-lambda:latest \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/panel-design-lambda:latest
```

### Step 5: Push to ECR

```bash
docker push YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/panel-design-lambda:latest
```

### Step 6: Create Lambda Function

```bash
aws lambda create-function \
  --function-name panel-design-api \
  --package-type Image \
  --code ImageUri=YOUR_ACCOUNT_ID.dkr.ecr.ap-southeast-1.amazonaws.com/panel-design-lambda:latest \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --timeout 300 \
  --memory-size 2048 \
  --region ap-southeast-1
```

### Step 7: Create Function URL (Optional)

```bash
aws lambda create-function-url-config \
  --function-name panel-design-api \
  --auth-type NONE \
  --region ap-southeast-1

# Add public access permission
aws lambda add-permission \
  --function-name panel-design-api \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE \
  --region ap-southeast-1
```

## 🧪 Local Testing

Test Lambda container locally before deployment:

```bash
# Build image
docker build -f Dockerfile.lambda -t panel-design-lambda:test .

# Run container
docker run -p 8080:8080 panel-design-lambda:test

# Or use the automated test script
./test-lambda-local.sh

# Test endpoint
curl http://localhost:8080
curl http://localhost:8080/health
curl http://localhost:8080/docs
```

## 🔧 Configuration

### Lambda Function Settings

| Setting | Recommended Value | Notes |
|---------|------------------|-------|
| Memory | 2048 MB | For OpenCV processing |
| Timeout | 300 seconds | Max for Lambda |
| Ephemeral Storage | 512 MB (default) | Increase if needed |
| Architecture | x86_64 | Default for Lambda base image |

### Environment Variables

Set in Lambda console or via CLI:

```bash
aws lambda update-function-configuration \
  --function-name panel-design-api \
  --environment Variables="{
    OPENAI_API_KEY=your-key,
    LOG_LEVEL=INFO
  }"
```

## 📊 Optimization Details

### Dockerfile.lambda Optimizations

1. **Multi-stage build**: Separate builder and runtime stages
2. **Minimal runtime libs**: Only essential libraries for opencv-headless
3. **Bytecode cleanup**: Remove `.pyc`, `__pycache__` to reduce size
4. **Prefix install**: Install to `/install` then copy to `/var/lang`
5. **AWS Lambda Web Adapter**: Enable FastAPI without code changes

### Expected Image Size

- **Base image**: ~600MB (public.ecr.aws/lambda/python:3.12)
- **Dependencies**: ~200-300MB (FastAPI, OpenCV, etc.)
- **Application code**: ~10-20MB
- **Total**: ~800-920MB

### Cost Optimization

- Use **x86_64** architecture (cheaper than arm64 for Lambda)
- Set appropriate **memory** (2048MB for CV workloads)
- Enable **Lambda SnapStart** if using Java (N/A for Python)
- Use **Provisioned Concurrency** only if needed

## 🔍 Monitoring

### CloudWatch Logs

```bash
# View logs
aws logs tail /aws/lambda/panel-design-api --follow

# Filter errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/panel-design-api \
  --filter-pattern "ERROR"
```

### Metrics

Monitor in CloudWatch:
- **Invocations**: Total requests
- **Duration**: Execution time
- **Errors**: Failed invocations
- **Throttles**: Rate limit hits

## 🐛 Troubleshooting

### Common Issues

#### 0. "entrypoint requires the handler name to be the first argument"
**Fixed**: Use `lambda-entrypoint.sh` wrapper script in Dockerfile.
```bash
# Should work now with simple command
docker run -p 8080:8080 panel-design-lambda:test

# Or use the test script
./test-lambda-local.sh
```

#### 1. "Task timed out after 3.00 seconds"
```bash
# Increase timeout
aws lambda update-function-configuration \
  --function-name panel-design-api \
  --timeout 300
```

#### 2. "Runtime exited with error: exit status 137" (OOM)
```bash
# Increase memory
aws lambda update-function-configuration \
  --function-name panel-design-api \
  --memory-size 3008
```

#### 3. "Unable to import module 'app.main'"
- Check `PYTHONPATH` is set to `${LAMBDA_TASK_ROOT}`
- Verify app code is copied to `${LAMBDA_TASK_ROOT}`

#### 4. Cold start too slow
- Use **Provisioned Concurrency**
- Optimize dependencies (remove unused packages)
- Consider **Lambda SnapStart** (Java only)

## 📚 References

- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter)
- [FastAPI on Lambda](https://fastapi.tiangolo.com/deployment/lambda/)

## 🔐 IAM Role Requirements

Lambda execution role needs:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

## 🎯 Next Steps

1. ✅ Deploy to Lambda
2. 🔧 Configure API Gateway (if needed)
3. 📊 Set up CloudWatch alarms
4. 🔐 Add authentication (API keys, Cognito, etc.)
5. 🚀 Set up CI/CD pipeline
