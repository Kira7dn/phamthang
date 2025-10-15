#!/usr/bin/env bash
set -euo pipefail

# ================= CONFIG =================
AWS_REGION="${AWS_REGION:-ap-southeast-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text --region "${AWS_REGION}")"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-extract-panel-api}"
ECR_REPO="${ECR_REPO:-${LAMBDA_FUNCTION_NAME}}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-lambda-basic-execution}"
LAMBDA_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.lambda}"
BUILD_CONTEXT="${BUILD_CONTEXT:-.}"
CACHE_DIR=".build_cache"

declare -A LAMBDA_ENV_MAP
LAMBDA_ENV_MAP[IS_PRODUCTION]="true"

ENV_FILE=""
if [ -n "${LAMBDA_ENV_FILE:-}" ] && [ -f "${LAMBDA_ENV_FILE}" ]; then
    ENV_FILE="${LAMBDA_ENV_FILE}"
elif [ -f ".env.lambda" ]; then
    ENV_FILE=".env.lambda"
elif [ -f ".env" ]; then
    ENV_FILE=".env"
fi

if [ -n "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%$'\r'}"
        if [ -z "$line" ]; then
            continue
        fi
        trimmed="$(echo "$line" | sed 's/^[[:space:]]*//')"
        case "$trimmed" in
            ''|\#*)
                continue
                ;;
        esac
        if [[ "$line" != *=* ]]; then
            continue
        fi
        key="${line%%=*}"
        value="${line#*=}"
        key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        if [ -z "$key" ]; then
            continue
        fi
        if { [ "${value:0:1}" = "'" ] && [ "${value: -1}" = "'" ]; } || { [ "${value:0:1}" = '"' ] && [ "${value: -1}" = '"' ]; }; then
            value="${value:1:${#value}-2}"
        fi
        LAMBDA_ENV_MAP["$key"]="$value"
    done < "$ENV_FILE"
fi

ENV_VARS_STRING=""
for key in "${!LAMBDA_ENV_MAP[@]}"; do
    value="${LAMBDA_ENV_MAP[$key]}"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    if [ -n "$ENV_VARS_STRING" ]; then
        ENV_VARS_STRING+=","
    fi
    ENV_VARS_STRING+="${key}=${value}"
done

if [ -z "$ENV_VARS_STRING" ]; then
    ENV_VARS_STRING="IS_PRODUCTION=true"
fi

echo "🛠 Deploy Lambda container image"
echo "Region: $AWS_REGION"
echo "Lambda: $LAMBDA_FUNCTION_NAME"
echo "ECR repo: $ECR_REPO"
echo "Image tag: $IMAGE_TAG"
echo

# ================= CHECK DEPENDENCIES =================
command -v aws >/dev/null 2>&1 || { echo "ERROR: aws CLI missing"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker missing"; exit 1; }
docker buildx version >/dev/null 2>&1 || { echo "ERROR: docker buildx missing"; exit 1; }

# ================= CREATE OR UPDATE IAM ROLE =================
if ! aws iam get-role --role-name "$LAMBDA_ROLE_NAME" >/dev/null 2>&1; then
    echo "Creating IAM role $LAMBDA_ROLE_NAME for Lambda..."
    aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": { "Service": "lambda.amazonaws.com" },
                "Action": "sts:AssumeRole"
            }]
        }' >/dev/null

    aws iam attach-role-policy \
        --role-name "$LAMBDA_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null

    echo "Waiting 10s for IAM role propagation..."
    sleep 10
fi

# ================= CREATE ECR REPO =================
if ! aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Creating ECR repository $ECR_REPO..."
    aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION" >/dev/null
fi

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# ================= LOGIN ECR =================
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ================= BUILD & PUSH IMAGE =================
echo "🔨 Building & pushing image $ECR_URI ..."
docker buildx build \
    --platform linux/amd64 \
    --file "$DOCKERFILE" \
    --build-arg TARGET_ENV=lambda \
    --cache-from=type=local,src="$CACHE_DIR" \
    --cache-to=type=local,dest="$CACHE_DIR",mode=max \
    -t "$ECR_URI" \
    --provenance=false \
    --sbom=false \
    --output type=registry,oci-mediatypes=false \
    "$BUILD_CONTEXT"

# ================= CREATE OR UPDATE LAMBDA =================
echo "🚀 Deploying Lambda function..."
if ! aws lambda get-function --function-name "$LAMBDA_FUNCTION_NAME" --region "$AWS_REGION" >/dev/null 2>&1; then
    echo "Creating Lambda function..."
    aws lambda create-function \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --package-type Image \
        --code ImageUri="$ECR_URI" \
        --role "$LAMBDA_ROLE_ARN" \
        --region "$AWS_REGION" \
        --memory-size 1024 \
        --timeout 60 \
        --ephemeral-storage '{"Size":512}' \
        --environment "Variables={${ENV_VARS_STRING}}" \
        >/dev/null

    echo "Waiting for function to become active..."
    aws lambda wait function-active \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --region "$AWS_REGION"
else
    echo "Updating Lambda code..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --image-uri "$ECR_URI" \
        --region "$AWS_REGION" >/dev/null

    # ensure environment vars are set
    echo "Waiting for function code update to complete..."
    aws lambda wait function-updated \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --region "$AWS_REGION"

    echo "Updating function configuration (env vars)..."
    set +e
    for i in {1..10}; do
        aws lambda update-function-configuration \
            --function-name "$LAMBDA_FUNCTION_NAME" \
            --environment "Variables={${ENV_VARS_STRING}}" \
            --region "$AWS_REGION" >/dev/null && break
        echo "Config update in progress, retrying in 6s ($i/10)..."
        sleep 6
    done
    set -e
fi

# ================= CREATE OR GET API GATEWAY =================
echo "🌐 Checking API Gateway..."
EXISTING_API_ID=$(aws apigatewayv2 get-apis --query "Items[?Name=='${LAMBDA_FUNCTION_NAME}-api'].ApiId" --output text --region "$AWS_REGION")

if [ -z "$EXISTING_API_ID" ]; then
    echo "⚙️ Creating new API Gateway HTTP..."
    API_ID=$(aws apigatewayv2 create-api \
        --name "${LAMBDA_FUNCTION_NAME}-api" \
        --protocol-type HTTP \
        --target "arn:aws:lambda:${AWS_REGION}:${ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}" \
        --region "$AWS_REGION" \
        --query "ApiId" --output text)

    ENDPOINT=$(aws apigatewayv2 get-api \
        --api-id "$API_ID" \
        --query "ApiEndpoint" --output text --region "$AWS_REGION")

    # Ensure integration uses payload format version 2.0
    INTEGRATION_ID=$(aws apigatewayv2 get-integrations \
        --api-id "$API_ID" \
        --region "$AWS_REGION" \
        --query 'Items[0].IntegrationId' --output text)
    if [ -n "$INTEGRATION_ID" ] && [ "$INTEGRATION_ID" != "None" ]; then
        aws apigatewayv2 update-integration \
            --api-id "$API_ID" \
            --integration-id "$INTEGRATION_ID" \
            --payload-format-version "2.0" \
            --region "$AWS_REGION" >/dev/null
    fi

    # Create explicit routes for FastAPI docs (idempotent)
    if [ -n "$INTEGRATION_ID" ] && [ "$INTEGRATION_ID" != "None" ]; then
        for ROUTE in \
            "GET /docs" \
            "GET /openapi.json" \
            "GET /docs/oauth2-redirect" \
            "GET /" \
            "GET /health" \
            "POST /extract" \
            "ANY /{proxy+}"; do
            aws apigatewayv2 create-route \
                --api-id "$API_ID" \
                --route-key "$ROUTE" \
                --target "integrations/$INTEGRATION_ID" \
                --region "$AWS_REGION" >/dev/null 2>&1 || echo "Route $ROUTE may already exist, continuing"
        done
    fi

    # Add permission for API Gateway to invoke Lambda (new API)
    SID="apigw-invoke-${API_ID}"
    aws lambda add-permission \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --statement-id "$SID" \
        --action "lambda:InvokeFunction" \
        --principal "apigateway.amazonaws.com" \
        --source-arn "arn:aws:execute-api:${AWS_REGION}:${ACCOUNT_ID}:${API_ID}/*/*/*" \
        --region "$AWS_REGION" || echo "Permission $SID may already exist, continuing"
else
    API_ID="$EXISTING_API_ID"
    ENDPOINT=$(aws apigatewayv2 get-api \
        --api-id "$API_ID" \
        --query "ApiEndpoint" --output text --region "$AWS_REGION")

    # Ensure permission exists for existing API as well
    SID="apigw-invoke-${API_ID}"
    aws lambda add-permission \
        --function-name "$LAMBDA_FUNCTION_NAME" \
        --statement-id "$SID" \
        --action "lambda:InvokeFunction" \
        --principal "apigateway.amazonaws.com" \
        --source-arn "arn:aws:execute-api:${AWS_REGION}:${ACCOUNT_ID}:${API_ID}/*/*/*" \
        --region "$AWS_REGION" >/dev/null 2>&1 || echo "Permission $SID may already exist, continuing"

    # Fetch integration id for existing API and ensure payload v2.0
    INTEGRATION_ID=$(aws apigatewayv2 get-integrations \
        --api-id "$API_ID" \
        --region "$AWS_REGION" \
        --query 'Items[0].IntegrationId' --output text)
    if [ -n "$INTEGRATION_ID" ] && [ "$INTEGRATION_ID" != "None" ]; then
        aws apigatewayv2 update-integration \
            --api-id "$API_ID" \
            --integration-id "$INTEGRATION_ID" \
            --payload-format-version "2.0" \
            --region "$AWS_REGION" >/dev/null

        # Create explicit routes for FastAPI docs and main endpoints (idempotent)
        for ROUTE in \
            "GET /docs" \
            "GET /openapi.json" \
            "GET /docs/oauth2-redirect" \
            "GET /" \
            "GET /health" \
            "POST /extract" \
            "ANY /{proxy+}"; do
            aws apigatewayv2 create-route \
                --api-id "$API_ID" \
                --route-key "$ROUTE" \
                --target "integrations/$INTEGRATION_ID" \
                --region "$AWS_REGION" >/dev/null 2>&1 || echo "Route $ROUTE may already exist, continuing"
        done
    fi
fi

echo
echo "✅ Deployment completed!"
echo "Lambda function: $LAMBDA_FUNCTION_NAME"
echo "API endpoint: $ENDPOINT"
