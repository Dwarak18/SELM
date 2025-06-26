#!/usr/bin/env python3
"""
Simple test client for the SELM API.
"""

import requests
import json
import time
import argparse
from typing import Dict, Any


class SELMClient:
    """Client for interacting with the SELM API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        response = self.session.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()
    
    def chat(self, messages: list, max_tokens: int = 100) -> Dict[str, Any]:
        """Send a chat request."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "context_search": True
        }
        response = self.session.post(f"{self.base_url}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    
    def submit_feedback(
        self,
        original_text: str,
        corrected_text: str,
        context: str = "",
        feedback_type: str = "correction"
    ) -> Dict[str, Any]:
        """Submit feedback."""
        payload = {
            "original_text": original_text,
            "corrected_text": corrected_text,
            "context": context,
            "feedback_type": feedback_type
        }
        response = self.session.post(f"{self.base_url}/feedback", json=payload)
        response.raise_for_status()
        return response.json()
    
    def search_memory(self, query: str, search_type: str = "both") -> Dict[str, Any]:
        """Search vector memory."""
        payload = {
            "query": query,
            "search_type": search_type,
            "limit": 5
        }
        response = self.session.post(f"{self.base_url}/memory/search", json=payload)
        response.raise_for_status()
        return response.json()
    
    def add_fact(self, fact: str, category: str = "general") -> Dict[str, Any]:
        """Add a fact to memory."""
        payload = {
            "fact": fact,
            "category": category,
            "source": "api_client",
            "confidence": 1.0
        }
        response = self.session.post(f"{self.base_url}/memory/add_fact", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def trigger_training(self) -> Dict[str, Any]:
        """Manually trigger training."""
        response = self.session.post(f"{self.base_url}/training/trigger")
        response.raise_for_status()
        return response.json()


def run_basic_tests(client: SELMClient):
    """Run basic functionality tests."""
    print("🔍 Running basic tests...")
    
    # Health check
    print("1. Health check...")
    health = client.health_check()
    print(f"   ✅ Health: {health['status']}")
    
    # System status
    print("2. System status...")
    status = client.get_status()
    print(f"   ✅ Model loaded: {status['model_loaded']}")
    print(f"   ✅ Vector store: {status['vector_store_initialized']}")
    print(f"   ✅ Redis: {status['redis_connected']}")
    
    # Add a fact
    print("3. Adding a fact...")
    fact_result = client.add_fact(
        "The capital of France is Paris",
        category="geography"
    )
    print(f"   ✅ Fact added: {fact_result['fact_id']}")
    
    # Search memory
    print("4. Searching memory...")
    search_result = client.search_memory("France capital")
    print(f"   ✅ Found {len(search_result['facts'])} facts")
    
    # Chat test
    print("5. Chat test...")
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    chat_result = client.chat(messages)
    print(f"   ✅ Response: {chat_result['response'][:100]}...")
    
    # Submit feedback
    print("6. Submitting feedback...")
    feedback_result = client.submit_feedback(
        original_text="Paris is the capitol of France",
        corrected_text="Paris is the capital of France",
        context="Spelling correction",
        feedback_type="spelling"
    )
    print(f"   ✅ Feedback submitted: {feedback_result['feedback_id']}")
    
    # Get statistics
    print("7. Getting statistics...")
    stats = client.get_stats()
    print(f"   ✅ Queue stats: {stats['queue_stats']}")
    
    print("✅ All basic tests passed!")


def run_interactive_chat(client: SELMClient):
    """Run interactive chat session."""
    print("🤖 Starting interactive chat session...")
    print("Type 'quit' to exit, 'clear' to clear history")
    
    messages = []
    
    while True:
        try:
            user_input = input("\n👤 You: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                messages = []
                print("Chat history cleared.")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            print("🤖 Assistant: ", end="", flush=True)
            response = client.chat(messages, max_tokens=150)
            assistant_response = response["response"]
            print(assistant_response)
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_response})
            
            # Show context info if available
            if response.get("context_used"):
                context = response["context_used"]
                if context.get("corrections_found", 0) > 0 or context.get("facts_found", 0) > 0:
                    print(f"ℹ️  Used context: {context['corrections_found']} corrections, {context['facts_found']} facts")
            
        except KeyboardInterrupt:
            print("\nChat session interrupted.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="SELM API Test Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat")
    
    args = parser.parse_args()
    
    client = SELMClient(args.url)
    
    try:
        if args.test:
            run_basic_tests(client)
        elif args.chat:
            run_interactive_chat(client)
        else:
            print("Please specify --test or --chat")
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to SELM API at {args.url}")
        print("Make sure the API server is running.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
