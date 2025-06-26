"""FastAPI application for the SELM system."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import asyncio
from datetime import datetime
import uvicorn

from src.config import settings
from src.models.language_model import language_model
from src.memory.vector_store import vector_memory
from src.feedback.queue import feedback_queue
from src.training.lora_trainer import lora_trainer

logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Smart Enhanced Language Model (SELM)",
    description="An adaptive language model with vector memory and feedback learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_tokens: Optional[int] = Field(100, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Override temperature")
    context_search: bool = Field(True, description="Whether to search for relevant context")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    context_used: Dict[str, Any] = Field({}, description="Context information used")
    model_info: Dict[str, Any] = Field({}, description="Model information")


class FeedbackRequest(BaseModel):
    original_text: str = Field(..., description="Original text that needs correction")
    corrected_text: str = Field(..., description="Corrected version")
    context: str = Field("", description="Additional context")
    feedback_type: str = Field("correction", description="Type of feedback")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class FeedbackResponse(BaseModel):
    feedback_id: str = Field(..., description="Unique feedback identifier")
    message: str = Field(..., description="Confirmation message")


class MemorySearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    search_type: str = Field("both", description="Type of search: corrections, facts, or both")
    limit: int = Field(5, description="Maximum number of results")


class MemorySearchResponse(BaseModel):
    corrections: List[Dict[str, Any]] = Field([], description="Relevant corrections")
    facts: List[Dict[str, Any]] = Field([], description="Relevant facts")
    query: str = Field(..., description="Original query")


class SystemStatus(BaseModel):
    status: str = Field(..., description="System status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    vector_store_initialized: bool = Field(..., description="Whether vector store is ready")
    redis_connected: bool = Field(..., description="Whether Redis is connected")
    queue_stats: Dict[str, Any] = Field({}, description="Queue statistics")
    model_info: Dict[str, Any] = Field({}, description="Model information")


# Dependency to ensure services are initialized
async def get_initialized_services():
    """Ensure all services are properly initialized."""
    if not language_model.model:
        raise HTTPException(status_code=503, detail="Language model not loaded")
    if not vector_memory.collection:
        raise HTTPException(status_code=503, detail="Vector memory not initialized")
    if not feedback_queue.redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup."""
    try:
        logger.info("Initializing SELM system...")
        
        # Initialize language model
        logger.info("Loading language model...")
        language_model.load_model()
        language_model.setup_lora()
        
        # Initialize vector memory
        logger.info("Initializing vector memory...")
        vector_memory.initialize()
        
        # Connect to Redis
        logger.info("Connecting to Redis...")
        feedback_queue.connect()
        
        logger.info("SELM system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize SELM system: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down SELM system...")


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to SELM - Smart Enhanced Language Model",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get detailed system status."""
    try:
        queue_stats = feedback_queue.get_queue_stats()
        model_info = language_model.get_model_info()
        
        return SystemStatus(
            status="operational",
            model_loaded=language_model.model is not None,
            vector_store_initialized=vector_memory.collection is not None,
            redis_connected=feedback_queue.redis_client is not None,
            queue_stats=queue_stats,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    services: None = Depends(get_initialized_services)
):
    """
    Generate a chat response with optional context search.
    """
    try:
        # Extract the latest user message for context search
        user_message = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        # Search for relevant context if enabled
        context_info = {}
        if request.context_search and user_message:
            context = vector_memory.get_context_for_query(user_message)
            context_info = {
                "corrections_found": len(context["corrections"]),
                "facts_found": len(context["facts"]),
                "context": context
            }
        
        # Generate response
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Add context to system message if found
        if context_info.get("context"):
            context_text = ""
            for correction in context_info["context"]["corrections"][:2]:
                context_text += f"Previous correction: {correction['metadata']['original_text']} -> {correction['metadata']['corrected_text']}\n"
            for fact in context_info["context"]["facts"][:2]:
                context_text += f"Relevant fact: {fact['document']}\n"
            
            if context_text:
                system_message = {
                    "role": "system",
                    "content": f"Consider this relevant context:\n{context_text}"
                }
                messages_dict.insert(0, system_message)
        
        # Generate response with optional parameter overrides
        generation_kwargs = {}
        if request.temperature is not None:
            generation_kwargs["temperature"] = request.temperature
        
        response = language_model.chat(messages_dict)
        
        return ChatResponse(
            response=response,
            context_used=context_info,
            model_info=language_model.get_model_info()
        )
        
    except Exception as e:
        logger.error(f"Chat generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    services: None = Depends(get_initialized_services)
):
    """
    Submit feedback for model improvement.
    """
    try:
        # Add feedback to queue
        feedback_id = feedback_queue.add_feedback(
            original_text=request.original_text,
            corrected_text=request.corrected_text,
            context=request.context,
            feedback_type=request.feedback_type,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        # Schedule background training check
        background_tasks.add_task(check_training_schedule)
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            message="Feedback submitted successfully"
        )
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search", response_model=MemorySearchResponse)
async def search_memory(
    request: MemorySearchRequest,
    services: None = Depends(get_initialized_services)
):
    """
    Search vector memory for relevant information.
    """
    try:
        corrections = []
        facts = []
        
        if request.search_type in ["corrections", "both"]:
            corrections = vector_memory.search_corrections(
                request.query,
                n_results=request.limit
            )
        
        if request.search_type in ["facts", "both"]:
            facts = vector_memory.search_facts(
                request.query,
                n_results=request.limit
            )
        
        return MemorySearchResponse(
            corrections=corrections,
            facts=facts,
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add_fact")
async def add_fact(
    fact: str,
    category: str = "general",
    source: str = "",
    confidence: float = 1.0,
    services: None = Depends(get_initialized_services)
):
    """
    Add a fact to the vector memory.
    """
    try:
        fact_id = vector_memory.add_fact(
            fact=fact,
            category=category,
            source=source,
            confidence=confidence
        )
        
        return {"fact_id": fact_id, "message": "Fact added successfully"}
        
    except Exception as e:
        logger.error(f"Failed to add fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training/history")
async def get_training_history(services: None = Depends(get_initialized_services)):
    """
    Get training history.
    """
    try:
        history = lora_trainer.get_training_history()
        return {"training_history": history}
        
    except Exception as e:
        logger.error(f"Failed to get training history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/trigger")
async def trigger_training(
    background_tasks: BackgroundTasks,
    services: None = Depends(get_initialized_services)
):
    """
    Manually trigger model training on accumulated feedback.
    """
    try:
        background_tasks.add_task(process_training_queue)
        return {"message": "Training triggered successfully"}
        
    except Exception as e:
        logger.error(f"Failed to trigger training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics(services: None = Depends(get_initialized_services)):
    """
    Get system statistics.
    """
    try:
        queue_stats = feedback_queue.get_queue_stats()
        collection_stats = vector_memory.get_collection_stats()
        model_info = language_model.get_model_info()
        
        return {
            "queue_stats": queue_stats,
            "memory_stats": collection_stats,
            "model_info": model_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks

async def check_training_schedule():
    """Background task to check if training should be scheduled."""
    try:
        lora_trainer.create_training_schedule(min_feedback_count=10)
    except Exception as e:
        logger.error(f"Training schedule check failed: {e}")


async def process_training_queue():
    """Background task to process training requests."""
    try:
        training_request = feedback_queue.get_next_training_request()
        if training_request:
            logger.info(f"Processing training request: {training_request.id}")
            await asyncio.to_thread(lora_trainer.train_on_feedback, training_request)
    except Exception as e:
        logger.error(f"Training processing failed: {e}")


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        workers=1,  # Use 1 worker for development, scale in production
        log_level="info"
    )
