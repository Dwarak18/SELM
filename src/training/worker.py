#!/usr/bin/env python3
"""
Training worker script that continuously processes training requests from the queue.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import settings
from src.models.language_model import language_model
from src.memory.vector_store import vector_memory
from src.feedback.queue import feedback_queue
from src.training.lora_trainer import lora_trainer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingWorker:
    """Training worker that processes training requests from the queue."""
    
    def __init__(self):
        self.running = False
        self.check_interval = 30  # seconds
        
    async def initialize(self):
        """Initialize all required components."""
        try:
            logger.info("Initializing training worker...")
            
            # Initialize language model
            if not language_model.model:
                logger.info("Loading language model...")
                language_model.load_model()
                language_model.setup_lora()
            
            # Initialize vector memory
            if not vector_memory.collection:
                logger.info("Initializing vector memory...")
                vector_memory.initialize()
            
            # Connect to Redis
            if not feedback_queue.redis_client:
                logger.info("Connecting to Redis...")
                feedback_queue.connect()
            
            logger.info("Training worker initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize training worker: {e}")
            raise
    
    async def process_training_requests(self):
        """Continuously process training requests from the queue."""
        logger.info("Starting training request processing...")
        
        while self.running:
            try:
                # Check for training requests
                training_request = feedback_queue.get_next_training_request()
                
                if training_request:
                    logger.info(f"Processing training request: {training_request.id}")
                    
                    # Process the training request
                    await asyncio.to_thread(lora_trainer.train_on_feedback, training_request)
                    
                    logger.info(f"Training request {training_request.id} completed successfully")
                else:
                    # No training requests, wait before checking again
                    await asyncio.sleep(self.check_interval)
                    
                # Also check if we should create new training requests
                lora_trainer.create_training_schedule(min_feedback_count=10)
                
            except Exception as e:
                logger.error(f"Error processing training request: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def start(self):
        """Start the training worker."""
        try:
            await self.initialize()
            self.running = True
            await self.process_training_requests()
        except Exception as e:
            logger.error(f"Training worker failed: {e}")
            raise
    
    def stop(self):
        """Stop the training worker."""
        logger.info("Stopping training worker...")
        self.running = False


# Global worker instance
worker = TrainingWorker()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    worker.stop()


async def main():
    """Main entry point for the training worker."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Training worker interrupted by user")
    except Exception as e:
        logger.error(f"Training worker failed: {e}")
        sys.exit(1)
    finally:
        logger.info("Training worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
