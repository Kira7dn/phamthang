# 🚀 Quick Start: Deploy to AWS Lambda

## ✅ Prerequisites đã hoàn thành

- [x] AWS CLI installed
- [x] Docker image tested locally
- [ ] AWS credentials configured
- [ ] Lambda execution role created

## 📋 Step-by-Step Deployment

### Step 1: Setup AWS Credentials

**Option A: Interactive Setup (Recommended)**
```bash
./setup-aws.sh
```

**Option B: Manual Setup**
```bash
aws configure
# Nhập: Access Key, Secret Key, Region (ap-southeast-1), Output (json)
```

**Option C: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="ap-southeast-1"
```

### Step 2: Verify Credentials

```bash
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

### Step 3: Create Lambda Execution Role (if needed)

```bash
# Check if role exists
aws iam get-role --role-name lambda-execution-role

# If not exists, create it
aws iam create-role \
  --role-name lambda-execution-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach basic execution policy
aws iam attach-role-policy \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

### Step 4: Deploy to Lambda

**Automated Deployment (Recommended)**
```bash
./deploy-lambda.sh
```

This script will:
1. ✅ Create ECR repository
2. ✅ Login to ECR
3. ✅ Build Docker image
4. ✅ Tag and push to ECR
5. ✅ Create/update Lambda function
6. ✅ Create Function URL
7. ✅ Display endpoint URL

**Manual Deployment**

See detailed steps in `AWS_LAMBDA_DEPLOYMENT.md`

### Step 5: Test Deployed Function

```bash
# Get Function URL
FUNCTION_URL=$(aws lambda get-function-url-config \
  --function-name panel-design-api \
  --query 'FunctionUrl' \
  --output text)

echo "Function URL: $FUNCTION_URL"

# Test endpoint
curl $FUNCTION_URL
curl $FUNCTION_URL/health
curl $FUNCTION_URL/docs
```

## 🎯 Configuration Options

### Environment Variables

Set environment variables for Lambda function:

```bash
aws lambda update-function-configuration \
  --function-name panel-design-api \
  --environment Variables="{
    OPENAI_API_KEY=sk-...,
    LOG_LEVEL=INFO
  }"
```

### Memory and Timeout

```bash
aws lambda update-function-configuration \
  --function-name panel-design-api \
  --memory-size 2048 \
  --timeout 300
```

### Custom Settings

Edit `deploy-lambda.sh` to customize:

```bash
# Configuration
AWS_REGION="ap-southeast-1"           # Your region
ECR_REPO_NAME="panel-design-lambda"   # ECR repository name
LAMBDA_FUNCTION_NAME="panel-design-api" # Lambda function name
IMAGE_TAG="latest"                     # Image tag
```

## 📊 Monitoring

### View Logs

```bash
# Tail logs
aws logs tail /aws/lambda/panel-design-api --follow

# Filter errors
aws logs filter-log-events \
  --log-group-name /aws/lambda/panel-design-api \
  --filter-pattern "ERROR"
```

### Check Function Status

```bash
# Get function info
aws lambda get-function --function-name panel-design-api

# Get function URL
aws lambda get-function-url-config --function-name panel-design-api
```

### Invoke Function

```bash
# Synchronous invoke
aws lambda invoke \
  --function-name panel-design-api \
  --payload '{"rawPath": "/", "requestContext": {"http": {"method": "GET"}}}' \
  response.json

cat response.json
```

## 🔄 Update Deployment

To update the Lambda function with new code:

```bash
# Rebuild and redeploy
./deploy-lambda.sh

# Or manually
docker build -f Dockerfile.lambda -t panel-design-lambda:latest .
docker tag panel-design-lambda:latest $ECR_URI:latest
docker push $ECR_URI:latest

aws lambda update-function-code \
  --function-name panel-design-api \
  --image-uri $ECR_URI:latest
```

## 🗑️ Cleanup

To remove all resources:

```bash
# Delete Lambda function
aws lambda delete-function --function-name panel-design-api

# Delete ECR repository
aws ecr delete-repository \
  --repository-name panel-design-lambda \
  --force

# Delete Lambda execution role
aws iam detach-role-policy \
  --role-name lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam delete-role --role-name lambda-execution-role
```

## 🐛 Troubleshooting

### Issue: "Unable to locate credentials"
```bash
# Check credentials
aws configure list

# Reconfigure
aws configure
```

### Issue: "AccessDeniedException"
- Verify IAM permissions (see `SETUP_AWS_CREDENTIALS.md`)
- Check if user has ECR and Lambda permissions

### Issue: "Role does not exist"
```bash
# Create Lambda execution role
./setup-aws.sh
# Or manually create (see Step 3 above)
```

### Issue: Cold start too slow
- Increase memory (more CPU allocated)
- Use Provisioned Concurrency
- Optimize dependencies

## 📚 Additional Resources

- **Full Documentation**: `AWS_LAMBDA_DEPLOYMENT.md`
- **Credentials Setup**: `SETUP_AWS_CREDENTIALS.md`
- **Deployment Options**: `DEPLOYMENT_OPTIONS.md`
- **Local Testing**: `./test-lambda-local.sh`

## 💡 Tips

1. **Use environment variables** for sensitive data (API keys)
2. **Monitor CloudWatch Logs** for errors
3. **Set appropriate memory** (2048MB for OpenCV)
4. **Enable X-Ray tracing** for debugging
5. **Use Provisioned Concurrency** for production (if needed)

## 🎉 Success!

Once deployed, your FastAPI app will be available at:
```
https://xxxxxxxxxx.lambda-url.ap-southeast-1.on.aws/
```

Test it:
```bash
curl https://your-function-url.lambda-url.ap-southeast-1.on.aws/
```
