# SELM Deployment Guide

This guide covers deploying the Smart Enhanced Language Model (SELM) system in different environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Production Considerations](#production-considerations)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 50GB storage
- **Recommended**: 16GB+ RAM, 8+ CPU cores, 100GB+ storage
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Software Requirements

- Python 3.9+
- Docker 20.10+ (for Docker/K8s deployments)
- Kubernetes 1.20+ (for K8s deployment)
- Redis 6.0+

## Local Development

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd SELM
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start Redis:**
   ```bash
   # Using Docker
   docker run -d -p 6379:6379 redis:7-alpine
   
   # Or install locally and start
   redis-server
   ```

6. **Initialize system:**
   ```bash
   python scripts/initialize_system.py
   ```

7. **Start the API:**
   ```bash
   python -m src.api.main
   ```

### Using Deployment Script

For automated local setup:

**Linux/macOS:**
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh --environment local
```

**Windows:**
```powershell
.\scripts\deploy.ps1 -Environment local
```

## Docker Deployment

### Using Docker Compose

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd SELM
   cp .env.example .env
   # Edit .env as needed
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

3. **Verify deployment:**
   ```bash
   curl http://localhost:8000/health
   ```

### Using Deployment Script

```bash
# Build and deploy
./scripts/deploy.sh --environment docker --build

# With custom image tag
./scripts/deploy.sh --environment docker --build --tag v1.0.0
```

### Service Configuration

The Docker Compose setup includes:

- **selm-api**: Main API service (port 8000)
- **redis**: Redis queue service (port 6379)
- **selm-trainer**: Background training worker
- **prometheus**: Metrics collection (port 9090)
- **grafana**: Monitoring dashboard (port 3000)

### Volumes

- `model_cache`: Stores downloaded models
- `chroma_data`: Vector database storage
- `redis_data`: Redis persistence
- `app_logs`: Application logs

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster with 4+ nodes
- kubectl configured
- Container registry access
- Storage class for persistent volumes

### Preparation

1. **Build and push image:**
   ```bash
   # Build image
   docker build -t your-registry/selm:v1.0.0 .
   
   # Push to registry
   docker push your-registry/selm:v1.0.0
   ```

2. **Update manifests:**
   Edit `kubernetes/03-api.yaml` and `kubernetes/04-training-worker.yaml` to use your image.

### Deployment Steps

1. **Deploy using script:**
   ```bash
   ./scripts/deploy.sh --environment k8s --push --registry your-registry.com --tag v1.0.0
   ```

2. **Manual deployment:**
   ```bash
   # Apply all manifests
   kubectl apply -f kubernetes/
   
   # Check deployment status
   kubectl get pods -n selm-system
   kubectl rollout status deployment/selm-api -n selm-system
   ```

3. **Access the API:**
   ```bash
   # Port forward for local access
   kubectl port-forward svc/selm-api-service 8000:8000 -n selm-system
   
   # Or configure ingress for external access
   ```

### Kubernetes Resources

The deployment creates:

- **Namespace**: `selm-system`
- **Deployments**: API server, Redis, training worker
- **Services**: Internal service networking
- **PVCs**: Persistent storage for models and data
- **ConfigMaps**: Configuration management
- **Secrets**: Sensitive configuration
- **Ingress**: External access (optional)
- **HPA**: Auto-scaling for API pods

### Scaling

```bash
# Scale API pods
kubectl scale deployment selm-api --replicas=5 -n selm-system

# Scale training workers
kubectl scale deployment selm-trainer --replicas=2 -n selm-system
```

## Production Considerations

### Security

1. **Authentication & Authorization:**
   - Implement JWT token authentication
   - Add rate limiting
   - Use HTTPS/TLS encryption
   - Configure CORS properly

2. **Secrets Management:**
   - Use Kubernetes secrets or external secret managers
   - Rotate secrets regularly
   - Don't commit secrets to version control

3. **Network Security:**
   - Use network policies in Kubernetes
   - Restrict Redis access
   - Configure firewall rules

### Performance

1. **Model Optimization:**
   - Use quantization (4-bit/8-bit)
   - Enable GPU acceleration
   - Optimize batch sizes
   - Consider model distillation

2. **Database Optimization:**
   - Configure ChromaDB for your workload
   - Use SSD storage for vector database
   - Optimize Redis memory usage

3. **Caching:**
   - Implement response caching
   - Cache model outputs for similar inputs
   - Use CDN for static assets

### Monitoring

1. **Metrics Collection:**
   - Enable Prometheus metrics
   - Monitor API response times
   - Track queue sizes
   - Monitor resource usage

2. **Logging:**
   - Centralized log aggregation
   - Structured logging
   - Error tracking and alerting

3. **Health Checks:**
   - Implement comprehensive health checks
   - Monitor external dependencies
   - Set up alerting for failures

### Backup & Recovery

1. **Data Backup:**
   - Regular backups of vector database
   - Model checkpoint backups
   - Redis data persistence

2. **Disaster Recovery:**
   - Multi-region deployment
   - Automated failover
   - Recovery procedures documentation

## Monitoring

### Prometheus Metrics

Key metrics to monitor:

- `http_requests_total`: API request count
- `http_request_duration_seconds`: Request latency
- `queue_size`: Feedback/training queue sizes
- `model_inference_duration`: Model response time
- `memory_usage_bytes`: Memory consumption
- `gpu_utilization`: GPU usage (if applicable)

### Grafana Dashboards

Import the provided dashboards from `docker/grafana/dashboards/`:

- **API Overview**: Request rates, latency, errors
- **Model Performance**: Inference times, queue status
- **System Resources**: CPU, memory, disk usage
- **Training Metrics**: Training progress, accuracy

### Alerting

Set up alerts for:

- High error rates (>5%)
- High latency (>5s P95)
- Queue backup (>100 items)
- High memory usage (>80%)
- Service downtime

## Troubleshooting

### Common Issues

1. **Model Loading Fails:**
   ```bash
   # Check available memory
   free -h
   
   # Check disk space
   df -h
   
   # Try with CPU-only mode
   export GPU_ENABLED=false
   ```

2. **Redis Connection Issues:**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Check network connectivity
   telnet redis-host 6379
   ```

3. **ChromaDB Initialization Fails:**
   ```bash
   # Check storage permissions
   ls -la data/chroma/
   
   # Clear and reinitialize
   rm -rf data/chroma/*
   python scripts/initialize_system.py
   ```

4. **Out of Memory:**
   ```bash
   # Monitor memory usage
   htop
   
   # Reduce batch size in .env
   TRAINING_BATCH_SIZE=2
   
   # Enable quantization
   PRECISION=4bit
   ```

### Log Analysis

Check logs for issues:

```bash
# Docker Compose
docker-compose logs selm-api
docker-compose logs redis

# Kubernetes
kubectl logs deployment/selm-api -n selm-system
kubectl logs deployment/selm-redis -n selm-system

# Local development
tail -f logs/selm.log
```

### Performance Tuning

1. **API Performance:**
   - Increase worker processes
   - Optimize database queries
   - Enable response compression
   - Use async endpoints

2. **Model Performance:**
   - Use smaller models for development
   - Enable mixed precision training
   - Optimize input tokenization
   - Batch similar requests

3. **Database Performance:**
   - Tune ChromaDB settings
   - Optimize embedding dimensions
   - Use appropriate distance metrics
   - Regular database maintenance

### Getting Help

1. **Check Documentation:**
   - API documentation: `docs/API.md`
   - Configuration reference: `src/config.py`
   - Architecture overview: `README.md`

2. **Debug Mode:**
   ```bash
   # Enable debug logging
   export LOG_LEVEL=DEBUG
   
   # Enable API debug mode
   export API_DEBUG=true
   ```

3. **Health Checks:**
   ```bash
   # Check system status
   curl http://localhost:8000/status
   
   # Check component health
   curl http://localhost:8000/stats
   ```

For additional support, check the project repository issues or documentation.
