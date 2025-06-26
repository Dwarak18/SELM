"""Vector memory store using ChromaDB for storing and retrieving corrections and facts."""

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import logging
import os
import uuid
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import json
from src.config import settings

logger = logging.getLogger(__name__)


class VectorMemoryStore:
    """
    Manages vector storage and retrieval of corrections, facts, and context.
    """
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.collection_name = settings.database.chroma_collection_name
        
    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure data directory exists
            os.makedirs(settings.database.chroma_db_path, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=settings.database.chroma_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(settings.database.embedding_model)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except ValueError:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            logger.info("Vector memory store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector memory store: {e}")
            raise
    
    def add_correction(
        self,
        original_text: str,
        corrected_text: str,
        context: str = "",
        correction_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a correction in the vector database.
        
        Args:
            original_text: The original text that was incorrect
            corrected_text: The corrected version
            context: Additional context about the correction
            correction_type: Type of correction (grammar, fact, style, etc.)
            metadata: Additional metadata
            
        Returns:
            Unique ID of the stored correction
        """
        try:
            # Create unique ID
            correction_id = str(uuid.uuid4())
            
            # Prepare document for embedding
            document = f"Original: {original_text}\nCorrected: {corrected_text}\nContext: {context}"
            
            # Generate embedding
            embedding = self.embedding_model.encode(document).tolist()
            
            # Prepare metadata
            correction_metadata = {
                "type": "correction",
                "correction_type": correction_type,
                "original_text": original_text,
                "corrected_text": corrected_text,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Store in ChromaDB
            self.collection.add(
                documents=[document],
                embeddings=[embedding],
                metadatas=[correction_metadata],
                ids=[correction_id]
            )
            
            logger.info(f"Stored correction with ID: {correction_id}")
            return correction_id
            
        except Exception as e:
            logger.error(f"Failed to store correction: {e}")
            raise
    
    def add_fact(
        self,
        fact: str,
        category: str = "general",
        source: str = "",
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a fact in the vector database.
        
        Args:
            fact: The factual information
            category: Category of the fact
            source: Source of the fact
            confidence: Confidence score (0.0 to 1.0)
            metadata: Additional metadata
            
        Returns:
            Unique ID of the stored fact
        """
        try:
            # Create unique ID
            fact_id = str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_model.encode(fact).tolist()
            
            # Prepare metadata
            fact_metadata = {
                "type": "fact",
                "category": category,
                "source": source,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Store in ChromaDB
            self.collection.add(
                documents=[fact],
                embeddings=[embedding],
                metadatas=[fact_metadata],
                ids=[fact_id]
            )
            
            logger.info(f"Stored fact with ID: {fact_id}")
            return fact_id
            
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            raise
    
    def search_corrections(
        self,
        query: str,
        n_results: int = 5,
        correction_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant corrections based on query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            correction_type: Filter by correction type
            
        Returns:
            List of relevant corrections
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare where clause
            where_clause = {"type": "correction"}
            if correction_type:
                where_clause["correction_type"] = correction_type
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            corrections = []
            for i in range(len(results["documents"][0])):
                correction = {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "id": results["ids"][0][i] if "ids" in results else None
                }
                corrections.append(correction)
            
            return corrections
            
        except Exception as e:
            logger.error(f"Failed to search corrections: {e}")
            return []
    
    def search_facts(
        self,
        query: str,
        n_results: int = 5,
        category: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant facts based on query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            category: Filter by fact category
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of relevant facts
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare where clause
            where_clause = {
                "type": "fact",
                "confidence": {"$gte": min_confidence}
            }
            if category:
                where_clause["category"] = category
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            facts = []
            for i in range(len(results["documents"][0])):
                fact = {
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "id": results["ids"][0][i] if "ids" in results else None
                }
                facts.append(fact)
            
            return facts
            
        except Exception as e:
            logger.error(f"Failed to search facts: {e}")
            return []
    
    def get_context_for_query(
        self,
        query: str,
        max_corrections: int = 3,
        max_facts: int = 3
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get relevant context (corrections and facts) for a query.
        
        Args:
            query: Input query
            max_corrections: Maximum number of corrections to retrieve
            max_facts: Maximum number of facts to retrieve
            
        Returns:
            Dictionary with corrections and facts
        """
        try:
            corrections = self.search_corrections(query, n_results=max_corrections)
            facts = self.search_facts(query, n_results=max_facts)
            
            return {
                "corrections": corrections,
                "facts": facts,
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Failed to get context for query: {e}")
            return {"corrections": [], "facts": [], "query": query}
    
    def update_entry(self, entry_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing entry.
        
        Args:
            entry_id: ID of the entry to update
            metadata_updates: Metadata fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current entry
            result = self.collection.get(ids=[entry_id], include=["metadatas"])
            
            if not result["metadatas"]:
                logger.warning(f"Entry not found: {entry_id}")
                return False
            
            # Update metadata
            current_metadata = result["metadatas"][0]
            updated_metadata = {**current_metadata, **metadata_updates}
            updated_metadata["last_updated"] = datetime.now().isoformat()
            
            # Update in ChromaDB
            self.collection.update(
                ids=[entry_id],
                metadatas=[updated_metadata]
            )
            
            logger.info(f"Updated entry: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update entry {entry_id}: {e}")
            return False
    
    def delete_entry(self, entry_id: str) -> bool:
        """
        Delete an entry from the vector store.
        
        Args:
            entry_id: ID of the entry to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.collection.delete(ids=[entry_id])
            logger.info(f"Deleted entry: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entry {entry_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get all entries to calculate stats
            all_entries = self.collection.get(include=["metadatas"])
            
            total_count = len(all_entries["metadatas"])
            correction_count = sum(1 for m in all_entries["metadatas"] if m.get("type") == "correction")
            fact_count = sum(1 for m in all_entries["metadatas"] if m.get("type") == "fact")
            
            # Get correction types
            correction_types = {}
            fact_categories = {}
            
            for metadata in all_entries["metadatas"]:
                if metadata.get("type") == "correction":
                    correction_type = metadata.get("correction_type", "unknown")
                    correction_types[correction_type] = correction_types.get(correction_type, 0) + 1
                elif metadata.get("type") == "fact":
                    category = metadata.get("category", "unknown")
                    fact_categories[category] = fact_categories.get(category, 0) + 1
            
            return {
                "total_entries": total_count,
                "corrections": correction_count,
                "facts": fact_count,
                "correction_types": correction_types,
                "fact_categories": fact_categories,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def reset_collection(self) -> None:
        """Reset the collection (delete all entries)."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("Collection reset successfully")
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            raise


# Global vector memory store instance
vector_memory = VectorMemoryStore()
