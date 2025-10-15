#!/bin/bash
set -e

echo "=========================================="
echo "AWS Credentials Setup"
echo "=========================================="
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip install awscli boto3 --quiet
fi

echo "AWS CLI version:"
aws --version
echo ""

# Check current credentials
echo "Current AWS configuration:"
aws configure list
echo ""

# Prompt for credentials
read -p "Do you want to configure AWS credentials now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Please enter your AWS credentials:"
    echo ""
    
    read -p "AWS Access Key ID: " AWS_ACCESS_KEY_ID
    read -sp "AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
    echo ""
    read -p "Default region (default: ap-southeast-1): " AWS_REGION
    AWS_REGION=${AWS_REGION:-ap-southeast-1}
    
    # Configure AWS CLI
    aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
    aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
    aws configure set region "$AWS_REGION"
    aws configure set output json
    
    echo ""
    echo "✅ AWS credentials configured!"
    echo ""
    
    # Verify credentials
    echo "Verifying credentials..."
    if aws sts get-caller-identity; then
        echo ""
        echo "✅ Credentials verified successfully!"
        
        # Get account ID
        ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
        echo ""
        echo "AWS Account ID: $ACCOUNT_ID"
        echo "Region: $AWS_REGION"
        
        # Check if Lambda execution role exists
        echo ""
        echo "Checking Lambda execution role..."
        if aws iam get-role --role-name lambda-execution-role 2>/dev/null; then
            echo "✅ Lambda execution role already exists"
        else
            echo "⚠️  Lambda execution role not found"
            read -p "Do you want to create it now? (y/n): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Creating Lambda execution role..."
                aws iam create-role \
                  --role-name lambda-execution-role \
                  --assume-role-policy-document '{
                    "Version": "2012-10-17",
                    "Statement": [{
                      "Effect": "Allow",
                      "Principal": {"Service": "lambda.amazonaws.com"},
                      "Action": "sts:AssumeRole"
                    }]
                  }' > /dev/null
                
                aws iam attach-role-policy \
                  --role-name lambda-execution-role \
                  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
                
                echo "✅ Lambda execution role created!"
            fi
        fi
        
        echo ""
        echo "=========================================="
        echo "✅ Setup completed!"
        echo "=========================================="
        echo ""
        echo "Next steps:"
        echo "  1. Review configuration: aws configure list"
        echo "  2. Deploy to Lambda: ./deploy-lambda.sh"
        echo ""
    else
        echo ""
        echo "❌ Failed to verify credentials"
        echo "Please check your Access Key and Secret Key"
        exit 1
    fi
else
    echo ""
    echo "Skipped credentials configuration"
    echo "You can configure later with: aws configure"
    echo ""
fi
