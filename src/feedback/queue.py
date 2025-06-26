"""Redis-based feedback queue system for collecting and processing user corrections."""

import redis
import json
import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FeedbackItem:
    """Represents a feedback item in the queue."""
    
    id: str
    original_text: str
    corrected_text: str
    context: str
    feedback_type: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str = ""
    metadata: Optional[Dict[str, Any]] = None
    processed: bool = False
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class TrainingRequest:
    """Represents a training request for model adaptation."""
    
    id: str
    feedback_items: List[str]  # List of feedback item IDs
    training_type: str  # "lora", "full", etc.
    priority: int = 1
    timestamp: str = ""
    status: str = "pending"  # pending, processing, completed, failed
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.id:
            self.id = str(uuid.uuid4())


class FeedbackQueue:
    """
    Manages feedback collection and training request queues using Redis.
    """
    
    def __init__(self):
        self.redis_client = None
        self.feedback_queue_key = settings.redis.feedback_queue
        self.training_queue_key = settings.redis.training_queue
        self.feedback_storage_key = "feedback_items"
        self.training_storage_key = "training_requests"
        
    def connect(self) -> None:
        """Connect to Redis server."""
        try:
            self.redis_client = redis.from_url(
                settings.redis.url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def add_feedback(
        self,
        original_text: str,
        corrected_text: str,
        context: str = "",
        feedback_type: str = "correction",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add feedback to the queue.
        
        Args:
            original_text: The original text that needs correction
            corrected_text: The corrected version
            context: Additional context
            feedback_type: Type of feedback (correction, preference, etc.)
            user_id: User identifier
            session_id: Session identifier
            metadata: Additional metadata
            
        Returns:
            Feedback item ID
        """
        try:
            # Create feedback item
            feedback_item = FeedbackItem(
                id=str(uuid.uuid4()),
                original_text=original_text,
                corrected_text=corrected_text,
                context=context,
                feedback_type=feedback_type,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata
            )
            
            # Store feedback item
            feedback_data = json.dumps(asdict(feedback_item))
            self.redis_client.hset(self.feedback_storage_key, feedback_item.id, feedback_data)
            
            # Add to queue
            queue_entry = {
                "id": feedback_item.id,
                "timestamp": feedback_item.timestamp,
                "priority": metadata.get("priority", 1) if metadata else 1
            }
            self.redis_client.lpush(self.feedback_queue_key, json.dumps(queue_entry))
            
            logger.info(f"Added feedback to queue: {feedback_item.id}")
            return feedback_item.id
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            raise
    
    def get_feedback(self, feedback_id: str) -> Optional[FeedbackItem]:
        """
        Retrieve a specific feedback item.
        
        Args:
            feedback_id: ID of the feedback item
            
        Returns:
            FeedbackItem or None if not found
        """
        try:
            feedback_data = self.redis_client.hget(self.feedback_storage_key, feedback_id)
            if feedback_data:
                data = json.loads(feedback_data)
                return FeedbackItem(**data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get feedback {feedback_id}: {e}")
            return None
    
    def get_pending_feedback(self, limit: int = 10) -> List[FeedbackItem]:
        """
        Get pending feedback items from the queue.
        
        Args:
            limit: Maximum number of items to retrieve
            
        Returns:
            List of FeedbackItem objects
        """
        try:
            feedback_items = []
            
            for _ in range(limit):
                # Get item from queue (blocking with timeout)
                queue_data = self.redis_client.brpop(self.feedback_queue_key, timeout=1)
                if not queue_data:
                    break
                
                # Parse queue entry
                queue_entry = json.loads(queue_data[1])
                feedback_id = queue_entry["id"]
                
                # Get full feedback item
                feedback_item = self.get_feedback(feedback_id)
                if feedback_item and not feedback_item.processed:
                    feedback_items.append(feedback_item)
                    
            return feedback_items
            
        except Exception as e:
            logger.error(f"Failed to get pending feedback: {e}")
            return []
    
    def mark_feedback_processed(self, feedback_id: str) -> bool:
        """
        Mark a feedback item as processed.
        
        Args:
            feedback_id: ID of the feedback item
            
        Returns:
            True if successful, False otherwise
        """
        try:
            feedback_item = self.get_feedback(feedback_id)
            if feedback_item:
                feedback_item.processed = True
                feedback_data = json.dumps(asdict(feedback_item))
                self.redis_client.hset(self.feedback_storage_key, feedback_id, feedback_data)
                logger.info(f"Marked feedback as processed: {feedback_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to mark feedback as processed {feedback_id}: {e}")
            return False
    
    def create_training_request(
        self,
        feedback_ids: List[str],
        training_type: str = "lora",
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a training request from feedback items.
        
        Args:
            feedback_ids: List of feedback item IDs to include in training
            training_type: Type of training (lora, full, etc.)
            priority: Priority level (higher number = higher priority)
            metadata: Additional metadata
            
        Returns:
            Training request ID
        """
        try:
            # Create training request
            training_request = TrainingRequest(
                id=str(uuid.uuid4()),
                feedback_items=feedback_ids,
                training_type=training_type,
                priority=priority,
                metadata=metadata
            )
            
            # Store training request
            training_data = json.dumps(asdict(training_request))
            self.redis_client.hset(self.training_storage_key, training_request.id, training_data)
            
            # Add to training queue (sorted by priority)
            self.redis_client.zadd(
                self.training_queue_key,
                {training_request.id: priority}
            )
            
            logger.info(f"Created training request: {training_request.id}")
            return training_request.id
            
        except Exception as e:
            logger.error(f"Failed to create training request: {e}")
            raise
    
    def get_training_request(self, request_id: str) -> Optional[TrainingRequest]:
        """
        Retrieve a specific training request.
        
        Args:
            request_id: ID of the training request
            
        Returns:
            TrainingRequest or None if not found
        """
        try:
            training_data = self.redis_client.hget(self.training_storage_key, request_id)
            if training_data:
                data = json.loads(training_data)
                return TrainingRequest(**data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get training request {request_id}: {e}")
            return None
    
    def get_next_training_request(self) -> Optional[TrainingRequest]:
        """
        Get the next training request from the priority queue.
        
        Returns:
            TrainingRequest or None if queue is empty
        """
        try:
            # Get highest priority item
            result = self.redis_client.zpopmax(self.training_queue_key)
            if result:
                request_id = result[0][0]
                return self.get_training_request(request_id)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get next training request: {e}")
            return None
    
    def update_training_status(
        self,
        request_id: str,
        status: str,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update the status of a training request.
        
        Args:
            request_id: ID of the training request
            status: New status (pending, processing, completed, failed)
            metadata_updates: Additional metadata updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            training_request = self.get_training_request(request_id)
            if training_request:
                training_request.status = status
                if metadata_updates:
                    if training_request.metadata:
                        training_request.metadata.update(metadata_updates)
                    else:
                        training_request.metadata = metadata_updates
                
                # Store updated request
                training_data = json.dumps(asdict(training_request))
                self.redis_client.hset(self.training_storage_key, request_id, training_data)
                
                logger.info(f"Updated training request status: {request_id} -> {status}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update training status {request_id}: {e}")
            return False
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback and training queues.
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            feedback_queue_size = self.redis_client.llen(self.feedback_queue_key)
            training_queue_size = self.redis_client.zcard(self.training_queue_key)
            
            # Count processed vs unprocessed feedback
            all_feedback = self.redis_client.hgetall(self.feedback_storage_key)
            processed_count = 0
            total_feedback = len(all_feedback)
            
            for feedback_data in all_feedback.values():
                feedback = json.loads(feedback_data)
                if feedback.get("processed", False):
                    processed_count += 1
            
            # Count training request statuses
            all_training = self.redis_client.hgetall(self.training_storage_key)
            training_statuses = {}
            
            for training_data in all_training.values():
                training = json.loads(training_data)
                status = training.get("status", "unknown")
                training_statuses[status] = training_statuses.get(status, 0) + 1
            
            return {
                "feedback_queue_size": feedback_queue_size,
                "training_queue_size": training_queue_size,
                "total_feedback_items": total_feedback,
                "processed_feedback": processed_count,
                "unprocessed_feedback": total_feedback - processed_count,
                "training_statuses": training_statuses
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    def cleanup_old_entries(self, days: int = 30) -> int:
        """
        Clean up old processed feedback items and completed training requests.
        
        Args:
            days: Number of days to keep entries
            
        Returns:
            Number of entries cleaned up
        """
        try:
            cutoff_timestamp = (datetime.now() - timedelta(days=days)).isoformat()
            cleaned_count = 0
            
            # Clean up old feedback
            all_feedback = self.redis_client.hgetall(self.feedback_storage_key)
            for feedback_id, feedback_data in all_feedback.items():
                feedback = json.loads(feedback_data)
                if (feedback.get("processed", False) and 
                    feedback.get("timestamp", "") < cutoff_timestamp):
                    self.redis_client.hdel(self.feedback_storage_key, feedback_id)
                    cleaned_count += 1
            
            # Clean up old training requests
            all_training = self.redis_client.hgetall(self.training_storage_key)
            for request_id, training_data in all_training.items():
                training = json.loads(training_data)
                if (training.get("status") in ["completed", "failed"] and
                    training.get("timestamp", "") < cutoff_timestamp):
                    self.redis_client.hdel(self.training_storage_key, request_id)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old entries: {e}")
            return 0


# Global feedback queue instance
feedback_queue = FeedbackQueue()
