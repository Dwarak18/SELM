"""Configuration management for SELM system."""

import os
from typing import Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Language model configuration."""
    
    name: str = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
    cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models")
    max_length: int = int(os.getenv("MAX_LENGTH", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    device: str = os.getenv("DEVICE", "auto")
    precision: str = os.getenv("PRECISION", "fp16")
    gpu_enabled: bool = os.getenv("GPU_ENABLED", "true").lower() == "true"


@dataclass
class DatabaseConfig:
    """Database and vector store configuration."""
    
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", "./data/chroma")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "memory_store")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@dataclass
class RedisConfig:
    """Redis configuration."""
    
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    feedback_queue: str = os.getenv("REDIS_FEEDBACK_QUEUE", "feedback_queue")
    training_queue: str = os.getenv("REDIS_TRAINING_QUEUE", "training_queue")


@dataclass
class APIConfig:
    """API server configuration."""
    
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("API_DEBUG", "false").lower() == "true"
    workers: int = int(os.getenv("API_WORKERS", "4"))
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


@dataclass
class TrainingConfig:
    """Training and fine-tuning configuration."""
    
    lora_rank: int = int(os.getenv("LORA_RANK", "16"))
    lora_alpha: int = int(os.getenv("LORA_ALPHA", "32"))
    lora_dropout: float = float(os.getenv("LORA_DROPOUT", "0.1"))
    batch_size: int = int(os.getenv("TRAINING_BATCH_SIZE", "4"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "5e-4"))
    max_steps: int = int(os.getenv("MAX_TRAINING_STEPS", "100"))


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv("LOG_FORMAT", "json")


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    metrics_enabled: bool = os.getenv("METRICS_ENABLED", "true").lower() == "true"


@dataclass
class Settings:
    """Main application settings."""
    
    model: ModelConfig
    database: DatabaseConfig
    redis: RedisConfig
    api: APIConfig
    training: TrainingConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    
    def __init__(self):
        self.model = ModelConfig()
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.api = APIConfig()
        self.training = TrainingConfig()
        self.logging = LoggingConfig()
        self.monitoring = MonitoringConfig()


# Global settings instance
settings = Settings()
