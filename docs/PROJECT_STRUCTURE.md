# SELM Project Structure

## Overview
Complete project structure for the Smart Enhanced Language Model (SELM) system.

## Directory Structure

```
SELM/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   └── main.py              # Main API server
│   ├── models/                   # Language model components
│   │   ├── __init__.py
│   │   └── language_model.py    # LLaMA-like model management
│   ├── memory/                   # Vector memory store
│   │   ├── __init__.py
│   │   └── vector_store.py      # ChromaDB integration
│   ├── feedback/                 # Feedback system
│   │   ├── __init__.py
│   │   └── queue.py             # Redis feedback queue
│   └── training/                 # Training components
│       ├── __init__.py
│       ├── lora_trainer.py      # LoRA/QLoRA training
│       └── worker.py            # Training worker
├── scripts/                      # Utility scripts
│   ├── initialize_system.py     # System initialization
│   ├── test_api.py              # API testing client
│   ├── deploy.sh               # Bash deployment script
│   └── deploy.ps1              # PowerShell deployment script
├── docker/                       # Docker configurations
│   ├── Dockerfile              # Main application image
│   └── docker-compose.yml      # Multi-service setup
├── kubernetes/                   # Kubernetes manifests
│   ├── 01-namespace-config.yaml
│   ├── 02-redis.yaml
│   ├── 03-api.yaml
│   ├── 04-training-worker.yaml
│   └── 05-monitoring.yaml
├── docs/                        # Documentation
│   ├── API.md                  # API documentation
│   └── DEPLOYMENT.md           # Deployment guide
├── tests/                       # Test suite
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # Main documentation
```

## Component Descriptions

### Core Components

1. **Language Model (`src/models/language_model.py`)**
   - Manages LLaMA-like model loading and inference
   - Implements LoRA adapter support
   - Handles quantization and GPU optimization
   - Provides chat and text generation interfaces

2. **Vector Memory Store (`src/memory/vector_store.py`)**
   - ChromaDB integration for semantic search
   - Stores corrections, facts, and context
   - Embedding generation and similarity search
   - Memory management and statistics

3. **Feedback Queue (`src/feedback/queue.py`)**
   - Redis-based feedback collection
   - Training request management
   - Queue statistics and monitoring
   - Background task scheduling

4. **LoRA Trainer (`src/training/lora_trainer.py`)**
   - LoRA/QLoRA fine-tuning implementation
   - Dataset preparation from feedback
   - Model adaptation without full retraining
   - Training history and evaluation

5. **FastAPI Server (`src/api/main.py`)**
   - RESTful API endpoints
   - Chat interface with context search
   - Feedback submission and processing
   - System monitoring and statistics

### Configuration

- **Environment Variables**: Configured via `.env` file
- **Settings Management**: Centralized in `src/config.py`
- **Docker Configuration**: Multi-service setup with persistence
- **Kubernetes Manifests**: Production-ready deployment

### Deployment Options

1. **Local Development**
   - Virtual environment setup
   - Direct Python execution
   - Development dependencies

2. **Docker Deployment**
   - Containerized services
   - Volume persistence
   - Service networking
   - Monitoring stack

3. **Kubernetes Deployment**
   - Scalable pod deployment
   - Persistent volume claims
   - Service discovery
   - Auto-scaling configuration

### Key Features

- **Adaptive Learning**: Model learns from user corrections
- **Vector Memory**: Semantic search for relevant context
- **Background Training**: Asynchronous model fine-tuning
- **Monitoring**: Prometheus metrics and health checks
- **Scalability**: Horizontal scaling support
- **Persistence**: Data and model checkpoint storage

### Dependencies

- **ML/AI**: PyTorch, Transformers, PEFT, ChromaDB
- **API**: FastAPI, Uvicorn, Pydantic
- **Queue**: Redis, Celery
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Kubernetes

### Development Workflow

1. **Setup**: Use initialization scripts
2. **Development**: Local environment with hot reload
3. **Testing**: Comprehensive test suite
4. **Deployment**: Automated deployment scripts
5. **Monitoring**: Real-time metrics and logging

This structure provides a complete, production-ready system for adaptive language model hosting with continuous learning capabilities.
