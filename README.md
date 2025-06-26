# Smart Enhanced Language Model (SELM) System

A Python-based system that hosts a LLaMA-like language model with vector memory, feedback loops, and adaptive learning capabilities.

## Features

- 🤖 **Language Model Hosting**: LLaMA-like model using Hugging Face Transformers
- 🧠 **Vector Memory Store**: ChromaDB for storing and retrieving previous corrections and facts
- 🔄 **Feedback Loop**: Redis-based system for queuing user corrections
- 📚 **Adaptive Learning**: LoRA/QLoRA fine-tuning without full retraining
- 🚀 **RESTful API**: FastAPI for high-performance web interface
- 🐳 **Containerized**: Docker for easy deployment
- ☸️ **Scalable**: Kubernetes deployment files for production

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Language      │    │   Vector        │
│   REST API      │◄──►│   Model         │◄──►│   Memory        │
│                 │    │   (LLaMA)       │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Redis         │    │   LoRA/QLoRA    │    │   Feedback      │
│   Queue         │◄──►│   Adapter       │◄──►│   Processor     │
│                 │    │   Training      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker & Docker Compose
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Installation

1. Clone and setup:
```bash
git clone https://github.com/Dwarak18/SELM.git
cd SELM
pip install -r requirements.txt
```

2. Environment setup:
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. Start services:
```bash
docker-compose up -d
```

4. Initialize the system:
```bash
python scripts/initialize_system.py
```

### API Usage

```python
import requests

# Chat with the model
response = requests.post("http://localhost:8000/chat", 
    json={"message": "Hello, how are you?"})

# Provide feedback
requests.post("http://localhost:8000/feedback", 
    json={"original": "Hello", "correction": "Hi there", "context": "greeting"})
```

## Project Structure

```
SELM/
├── src/
│   ├── api/                 # FastAPI application
│   ├── models/              # Language model components
│   ├── memory/              # Vector memory store
│   ├── feedback/            # Feedback processing
│   └── training/            # LoRA/QLoRA training
├── docker/                  # Docker configurations
├── kubernetes/              # K8s deployment files
├── scripts/                 # Setup and utility scripts
├── tests/                   # Test suite
└── docs/                    # Documentation 
```

## Configuration

Key environment variables:

- `MODEL_NAME`: Hugging Face model identifier
- `CHROMA_DB_PATH`: ChromaDB storage path
- `REDIS_URL`: Redis connection string
- `API_HOST`: FastAPI host
- `API_PORT`: FastAPI port
- `GPU_ENABLED`: Enable GPU acceleration

## Deployment

### Local Development
```bash
python -m src.api.main
```

### Docker
```bash
docker-compose up
```

### Kubernetes
```bash
kubectl apply -f kubernetes/
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## License

MIT License - see LICENSE file for details.
