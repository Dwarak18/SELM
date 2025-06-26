# SELM API Documentation

## Overview

The Smart Enhanced Language Model (SELM) provides a RESTful API for interacting with an adaptive language model that learns from user feedback and maintains a vector memory store.

## Base URL

- Local development: `http://localhost:8000`
- Production: Replace with your deployed URL

## Authentication

Currently, the API does not require authentication for most endpoints. In production, implement proper authentication mechanisms.

## Endpoints

### Health and Status

#### GET /health
Check if the API is healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-26T10:30:00Z"
}
```

#### GET /status
Get detailed system status.

**Response:**
```json
{
  "status": "operational",
  "model_loaded": true,
  "vector_store_initialized": true,
  "redis_connected": true,
  "queue_stats": {
    "feedback_queue_size": 5,
    "training_queue_size": 1,
    "total_feedback_items": 25,
    "processed_feedback": 20,
    "unprocessed_feedback": 5
  },
  "model_info": {
    "model_name": "microsoft/DialoGPT-medium",
    "device": "cuda",
    "has_lora": true,
    "trainable_params": 1048576,
    "total_params": 117000000
  }
}
```

### Chat Interface

#### POST /chat
Generate a chat response with optional context search.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "context_search": true
}
```

**Response:**
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "context_used": {
    "corrections_found": 0,
    "facts_found": 2,
    "context": {
      "corrections": [],
      "facts": [
        {
          "document": "Machine learning is a subset of artificial intelligence...",
          "metadata": {
            "category": "ai",
            "confidence": 1.0
          },
          "similarity": 0.95
        }
      ]
    }
  },
  "model_info": {
    "model_name": "microsoft/DialoGPT-medium",
    "device": "cuda"
  }
}
```

### Feedback System

#### POST /feedback
Submit feedback for model improvement.

**Request:**
```json
{
  "original_text": "The capitol of France is Paris",
  "corrected_text": "The capital of France is Paris",
  "context": "Spelling correction",
  "feedback_type": "spelling",
  "user_id": "user123",
  "session_id": "session456",
  "metadata": {
    "source": "web_interface",
    "priority": 1
  }
}
```

**Response:**
```json
{
  "feedback_id": "uuid-here",
  "message": "Feedback submitted successfully"
}
```

### Vector Memory

#### POST /memory/search
Search vector memory for relevant information.

**Request:**
```json
{
  "query": "Python programming",
  "search_type": "both",
  "limit": 5
}
```

**Response:**
```json
{
  "corrections": [
    {
      "document": "Original: print 'hello' Corrected: print('hello')",
      "metadata": {
        "type": "correction",
        "correction_type": "syntax",
        "original_text": "print 'hello'",
        "corrected_text": "print('hello')"
      },
      "similarity": 0.85
    }
  ],
  "facts": [
    {
      "document": "Python is a high-level programming language...",
      "metadata": {
        "type": "fact",
        "category": "programming",
        "confidence": 1.0
      },
      "similarity": 0.90
    }
  ],
  "query": "Python programming"
}
```

#### POST /memory/add_fact
Add a fact to the vector memory.

**Request:**
```json
{
  "fact": "FastAPI is a modern web framework for Python",
  "category": "programming",
  "source": "documentation",
  "confidence": 1.0
}
```

**Response:**
```json
{
  "fact_id": "uuid-here",
  "message": "Fact added successfully"
}
```

### Training

#### GET /training/history
Get training history.

**Response:**
```json
{
  "training_history": [
    {
      "request_id": "training-uuid",
      "adapter_path": "/app/models/lora_checkpoints/adapter_training-uuid",
      "status": "completed",
      "feedback_count": 10,
      "timestamp": "2025-06-26T09:00:00Z",
      "metadata": {
        "training_loss": 0.25,
        "training_steps": 50
      }
    }
  ]
}
```

#### POST /training/trigger
Manually trigger model training.

**Response:**
```json
{
  "message": "Training triggered successfully"
}
```

### Statistics

#### GET /stats
Get comprehensive system statistics.

**Response:**
```json
{
  "queue_stats": {
    "feedback_queue_size": 5,
    "training_queue_size": 1,
    "total_feedback_items": 25,
    "processed_feedback": 20,
    "unprocessed_feedback": 5,
    "training_statuses": {
      "completed": 3,
      "pending": 1,
      "processing": 0,
      "failed": 0
    }
  },
  "memory_stats": {
    "total_entries": 150,
    "corrections": 75,
    "facts": 75,
    "correction_types": {
      "grammar": 30,
      "spelling": 25,
      "style": 20
    },
    "fact_categories": {
      "programming": 40,
      "ai": 20,
      "general": 15
    }
  },
  "model_info": {
    "model_name": "microsoft/DialoGPT-medium",
    "device": "cuda",
    "has_lora": true,
    "trainable_params": 1048576,
    "total_params": 117000000,
    "trainable_percent": 0.9
  },
  "timestamp": "2025-06-26T10:30:00Z"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200`: Success
- `400`: Bad Request - Invalid input parameters
- `404`: Not Found - Resource not found
- `500`: Internal Server Error - Server error
- `503`: Service Unavailable - Service not ready

Error responses include a detail message:

```json
{
  "detail": "Language model not loaded"
}
```

## Rate Limiting

Currently, no rate limiting is implemented. Consider adding rate limiting for production use.

## Examples

### Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Chat example
response = requests.post(f"{BASE_URL}/chat", json={
    "messages": [
        {"role": "user", "content": "Explain machine learning"}
    ],
    "max_tokens": 150
})
print(response.json()["response"])

# Feedback example
requests.post(f"{BASE_URL}/feedback", json={
    "original_text": "ML is a AI subset",
    "corrected_text": "ML is an AI subset",
    "feedback_type": "grammar"
})
```

### JavaScript Client Example

```javascript
// Chat example
fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        messages: [
            {role: 'user', content: 'What is Python?'}
        ],
        max_tokens: 100
    })
})
.then(response => response.json())
.then(data => console.log(data.response));
```

### cURL Examples

```bash
# Health check
curl -X GET http://localhost:8000/health

# Chat request
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "original_text": "Hello world",
    "corrected_text": "Hello, world!",
    "feedback_type": "punctuation"
  }'
```

## WebSocket Support

Currently, the API does not support WebSocket connections. All interactions are through REST endpoints.

## Monitoring

The API exposes metrics at `/metrics` (if Prometheus integration is enabled) for monitoring purposes.
