# Setup AWS Credentials

## ✅ AWS CLI đã được cài đặt

```bash
aws-cli/1.42.50 Python/3.12.11 Linux
```

## 🔐 Configure AWS Credentials

### Option 1: AWS Configure (Interactive)

```bash
aws configure
```

Nhập thông tin:
```
AWS Access Key ID [None]: YOUR_ACCESS_KEY
AWS Secret Access Key [None]: YOUR_SECRET_KEY
Default region name [None]: ap-southeast-1
Default output format [None]: json
```

### Option 2: Environment Variables

```bash
export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
export AWS_DEFAULT_REGION="ap-southeast-1"
```

### Option 3: AWS Credentials File

Tạo file `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

Tạo file `~/.aws/config`:
```ini
[default]
region = ap-southeast-1
output = json
```

## 🧪 Verify Credentials

```bash
# Check configuration
aws configure list

# Test credentials
aws sts get-caller-identity
```

Output mong đợi:
```json
{
    "UserId": "AIDAXXXXXXXXXXXXXXXXX",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/your-username"
}
```

## 📋 Required IAM Permissions

User/Role cần có các permissions sau:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:CreateRepository",
        "ecr:DescribeRepositories",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:BatchCheckLayerAvailability"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "lambda:CreateFunction",
        "lambda:UpdateFunctionCode",
        "lambda:UpdateFunctionConfiguration",
        "lambda:GetFunction",
        "lambda:CreateFunctionUrlConfig",
        "lambda:GetFunctionUrlConfig",
        "lambda:AddPermission"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": "arn:aws:iam::*:role/lambda-execution-role"
    }
  ]
}
```

## 🎯 Next Steps

Sau khi configure credentials:

1. **Verify credentials**:
   ```bash
   aws sts get-caller-identity
   ```

2. **Create Lambda execution role** (nếu chưa có):
   ```bash
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
   
   aws iam attach-role-policy \
     --role-name lambda-execution-role \
     --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
   ```

3. **Deploy to Lambda**:
   ```bash
   ./deploy-lambda.sh
   ```

## 🔗 Useful Links

- [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)
- [IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Lambda Execution Role](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html)
