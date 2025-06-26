#!/usr/bin/env python3
"""
System initialization script for SELM.
This script sets up the system components and performs initial configuration.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings
from src.models.language_model import language_model
from src.memory.vector_store import vector_memory
from src.feedback.queue import feedback_queue

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def initialize_system():
    """Initialize all system components."""
    try:
        logger.info("Starting SELM system initialization...")
        
        # Create necessary directories
        os.makedirs(settings.model.cache_dir, exist_ok=True)
        os.makedirs(settings.database.chroma_db_path, exist_ok=True)
        os.makedirs("./logs", exist_ok=True)
        
        # Initialize language model
        logger.info("Initializing language model...")
        language_model.load_model()
        language_model.setup_lora()
        logger.info("Language model initialized successfully")
        
        # Initialize vector memory
        logger.info("Initializing vector memory store...")
        vector_memory.initialize()
        logger.info("Vector memory store initialized successfully")
        
        # Connect to Redis
        logger.info("Connecting to Redis...")
        feedback_queue.connect()
        logger.info("Redis connection established")
        
        # Add some initial facts to the vector store
        logger.info("Adding initial facts to vector store...")
        await add_initial_facts()
        
        # Test system components
        logger.info("Testing system components...")
        await test_system_components()
        
        logger.info("SELM system initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise


async def add_initial_facts():
    """Add some initial facts to the vector store."""
    initial_facts = [
        {
            "fact": "Python is a high-level, interpreted programming language known for its simplicity and readability.",
            "category": "programming",
            "source": "general_knowledge"
        },
        {
            "fact": "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
            "category": "ai",
            "source": "general_knowledge"
        },
        {
            "fact": "FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.",
            "category": "programming",
            "source": "general_knowledge"
        },
        {
            "fact": "LoRA (Low-Rank Adaptation) is a technique for efficiently fine-tuning large language models by adding trainable rank decomposition matrices.",
            "category": "ai",
            "source": "general_knowledge"
        },
        {
            "fact": "ChromaDB is an open-source embedding database designed for building AI applications with LLMs.",
            "category": "database",
            "source": "general_knowledge"
        }
    ]
    
    for fact_data in initial_facts:
        try:
            vector_memory.add_fact(**fact_data)
            logger.info(f"Added fact: {fact_data['fact'][:50]}...")
        except Exception as e:
            logger.warning(f"Failed to add fact: {e}")


async def test_system_components():
    """Test that all system components are working correctly."""
    try:
        # Test language model
        logger.info("Testing language model...")
        test_response = language_model.generate_response("Hello, how are you?", max_new_tokens=20)
        logger.info(f"Model test response: {test_response[:50]}...")
        
        # Test vector memory search
        logger.info("Testing vector memory search...")
        search_results = vector_memory.search_facts("Python programming", n_results=2)
        logger.info(f"Found {len(search_results)} relevant facts")
        
        # Test Redis feedback queue
        logger.info("Testing feedback queue...")
        feedback_id = feedback_queue.add_feedback(
            original_text="Hello world",
            corrected_text="Hello, world!",
            context="Grammar correction test",
            feedback_type="grammar"
        )
        logger.info(f"Test feedback added with ID: {feedback_id}")
        
        # Get queue stats
        stats = feedback_queue.get_queue_stats()
        logger.info(f"Queue stats: {stats}")
        
        logger.info("All system components tested successfully!")
        
    except Exception as e:
        logger.error(f"System component test failed: {e}")
        raise


async def cleanup_test_data():
    """Clean up any test data created during initialization."""
    try:
        logger.info("Cleaning up test data...")
        # You can add cleanup logic here if needed
        logger.info("Test data cleanup completed")
    except Exception as e:
        logger.warning(f"Test data cleanup failed: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(initialize_system())
    except KeyboardInterrupt:
        logger.info("Initialization interrupted by user")
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
