# Deployment Options

## 📦 Available Dockerfiles

### 1. **Dockerfile** (Standard Container)
- **Use case**: Docker, Kubernetes, ECS, Cloud Run
- **Base image**: `python:3.12-slim`
- **Size**: ~375-480MB (optimized)
- **Features**:
  - Multi-stage build
  - Non-root user (appuser)
  - Minimal runtime dependencies
  - Optimized layers

**Deploy:**
```bash
docker build -t panel-design:latest .
docker run -p 8000:8000 panel-design:latest
```

### 2. **Dockerfile.lambda** (AWS Lambda)
- **Use case**: AWS Lambda with container images
- **Base image**: `public.ecr.aws/lambda/python:3.12`
- **Size**: ~800-920MB
- **Features**:
  - AWS Lambda Web Adapter
  - Lambda-optimized runtime
  - FastAPI without code changes
  - Response streaming support

**Deploy:**
```bash
./deploy-lambda.sh
# Or
./test-lambda-local.sh  # Test locally first
```

## 🚀 Quick Comparison

| Feature | Standard Docker | AWS Lambda |
|---------|----------------|------------|
| **Base Image** | python:3.12-slim | lambda/python:3.12 |
| **Image Size** | ~400MB | ~900MB |
| **Cold Start** | N/A | 2-5 seconds |
| **Max Timeout** | Unlimited | 15 minutes |
| **Scaling** | Manual/K8s | Automatic |
| **Cost Model** | Per hour | Per request |
| **Best For** | Long-running, predictable load | Sporadic, variable load |

## 📋 Deployment Guides

- **Standard Docker**: See `DOCKER_OPTIMIZATION.md`
- **AWS Lambda**: See `AWS_LAMBDA_DEPLOYMENT.md`

## 🎯 Which to Choose?

### Choose **Standard Docker** if:
- ✅ Running on Kubernetes, ECS, or Cloud Run
- ✅ Need long-running processes (>15 min)
- ✅ Predictable, consistent traffic
- ✅ Want full control over infrastructure
- ✅ Need WebSocket or long-polling

### Choose **AWS Lambda** if:
- ✅ Sporadic or variable traffic
- ✅ Want automatic scaling (0 to thousands)
- ✅ Pay-per-use pricing model
- ✅ Don't want to manage servers
- ✅ Tasks complete in <15 minutes
- ✅ Can tolerate cold starts

## 🔧 Environment Variables

Both deployments support:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
PYTHONDONTWRITEBYTECODE=1
```

## 📊 Cost Estimation

### Standard Docker (ECS Fargate)
```
vCPU: 1, Memory: 2GB
Cost: ~$30/month (24/7 running)
```

### AWS Lambda
```
Memory: 2048MB, Avg duration: 3s
1M requests/month: ~$20-30
100K requests/month: ~$2-5
```

## 🧪 Testing

### Standard Docker
```bash
docker build -t panel-design:test .
docker run -p 8000:8000 panel-design:test
curl http://localhost:8000
```

### AWS Lambda
```bash
./test-lambda-local.sh
curl http://localhost:8080
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
