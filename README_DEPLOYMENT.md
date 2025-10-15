# Panel Design API - Deployment Guide

## 📦 Available Deployment Options

### 1. **AWS Lambda** (Serverless)
- ✅ Auto-scaling
- ✅ Pay-per-use
- ✅ No server management
- 📄 Guide: `QUICK_START_LAMBDA.md`

### 2. **Docker Container** (Standard)
- ✅ Full control
- ✅ Any cloud provider
- ✅ Kubernetes/ECS ready
- 📄 Guide: `DOCKER_OPTIMIZATION.md`

## 🚀 Quick Start

### Option A: Deploy to AWS Lambda

```bash
# 1. Setup AWS credentials
./setup-aws.sh

# 2. Deploy to Lambda
./deploy-lambda.sh

# 3. Test
curl https://your-function-url.lambda-url.ap-southeast-1.on.aws/
```

### Option B: Run with Docker

```bash
# 1. Build image
docker build -t panel-design:latest .

# 2. Run container
docker run -p 8000:8000 panel-design:latest

# 3. Test
curl http://localhost:8000
```

## 📋 Prerequisites

### For AWS Lambda:
- [x] AWS CLI installed ✅
- [ ] AWS credentials configured
- [ ] Lambda execution role

### For Docker:
- [x] Docker installed
- [ ] Docker daemon running

## 📚 Documentation

| File | Description |
|------|-------------|
| **QUICK_START_LAMBDA.md** | 🚀 Quick guide to deploy to AWS Lambda |
| **AWS_LAMBDA_DEPLOYMENT.md** | 📖 Detailed Lambda deployment guide |
| **SETUP_AWS_CREDENTIALS.md** | 🔐 AWS credentials setup |
| **DOCKER_OPTIMIZATION.md** | 🐳 Docker optimization guide |
| **DEPLOYMENT_OPTIONS.md** | 📊 Compare deployment options |

## 🛠️ Scripts

| Script | Purpose |
|--------|---------|
| `setup-aws.sh` | Setup AWS credentials and IAM role |
| `deploy-lambda.sh` | Deploy to AWS Lambda |
| `test-lambda-local.sh` | Test Lambda container locally |

## 🏗️ Architecture

### AWS Lambda
```
Client → API Gateway/Function URL → Lambda → FastAPI
```

### Docker
```
Client → Load Balancer → Container → FastAPI
```

## 📊 Cost Comparison

### AWS Lambda
- **Free Tier**: 1M requests/month
- **Typical Cost**: $2-5 for 100K requests
- **Best for**: Variable traffic

### Docker (ECS Fargate)
- **Cost**: ~$30/month (24/7)
- **Best for**: Consistent traffic

## 🎯 Recommended Setup

**For Development/Testing:**
```bash
docker run -p 8000:8000 panel-design:latest
```

**For Production (Low/Variable Traffic):**
```bash
./deploy-lambda.sh
```

**For Production (High/Consistent Traffic):**
- Deploy to ECS/Kubernetes
- Use Docker image from `Dockerfile`

## 🔧 Configuration

### Environment Variables

Both deployments support:

```bash
OPENAI_API_KEY=sk-...        # Required
LOG_LEVEL=INFO               # Optional
PYTHONUNBUFFERED=1           # Optional
```

### Lambda-specific

```bash
AWS_LWA_INVOKE_MODE=response_stream
AWS_LWA_READINESS_CHECK_PORT=8080
```

## 📈 Next Steps

1. ✅ Choose deployment option
2. ✅ Follow quick start guide
3. ✅ Configure environment variables
4. ✅ Test endpoints
5. ✅ Monitor logs
6. ✅ Set up CI/CD (optional)

## 🐛 Troubleshooting

### AWS Lambda Issues
See `AWS_LAMBDA_DEPLOYMENT.md` → Troubleshooting section

### Docker Issues
See `DOCKER_OPTIMIZATION.md`

### Common Problems

**"Unable to locate credentials"**
```bash
./setup-aws.sh
```

**"Docker daemon not running"**
```bash
# Start Docker Desktop or Docker service
```

**"Port already in use"**
```bash
# Change port
docker run -p 8001:8000 panel-design:latest
```

## 📞 Support

- **Documentation**: See guides above
- **Issues**: Check troubleshooting sections
- **Logs**: 
  - Lambda: `aws logs tail /aws/lambda/panel-design-api --follow`
  - Docker: `docker logs -f container-name`

## 🎉 Success Checklist

- [ ] AWS credentials configured (for Lambda)
- [ ] Docker image built and tested
- [ ] Deployment successful
- [ ] Endpoints responding
- [ ] Environment variables set
- [ ] Monitoring configured
- [ ] Documentation reviewed

---

**Ready to deploy?** Start with `QUICK_START_LAMBDA.md` or run `./setup-aws.sh`! 🚀
