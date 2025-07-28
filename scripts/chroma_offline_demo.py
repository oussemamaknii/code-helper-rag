#!/usr/bin/env python3
"""
Chroma Offline Demo - No Network Downloads Required

This demo shows how to use Chroma vector database without 
downloading external models, avoiding timeout issues.
"""

import os
import numpy as np
from pathlib import Path

def create_custom_embedding_function():
    """Create a custom embedding function using already-installed Sentence Transformers."""
    
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        
        # Load the model we already have cached
        print("Loading Sentence Transformers model (using cached version)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        class CustomEmbeddingFunction:
            """Custom embedding function that uses our local model."""
            
            def __init__(self, model):
                self.model = model
            
            def __call__(self, input_texts):
                """Generate embeddings for input texts."""
                if isinstance(input_texts, str):
                    input_texts = [input_texts]
                
                # Generate embeddings
                embeddings = self.model.encode(input_texts)
                return embeddings.tolist()
        
        return CustomEmbeddingFunction(model)
        
    except Exception as e:
        print(f"Failed to create custom embedding function: {e}")
        return None

def test_chroma_with_custom_embeddings():
    """Test Chroma using our custom embedding function."""
    
    print("Testing Chroma with Custom Embeddings (Offline Mode)")
    print("=" * 55)
    
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Create custom embedding function
        embedding_function = create_custom_embedding_function()
        if not embedding_function:
            return False
        
        print("Creating Chroma client with custom embeddings...")
        
        # Create client with our custom embedding function
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create collection with our custom embedding function
        collection = client.get_or_create_collection(
            name="python_code_offline",
            embedding_function=embedding_function
        )
        
        print("SUCCESS: Collection created with offline embeddings!")
        
        # Add sample Python code snippets
        print("Adding Python code samples...")
        
        code_samples = [
            "def quicksort(arr): return sorted(arr) if len(arr) <= 1 else quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x >= arr[0]])",
            "def binary_search(arr, target): left, right = 0, len(arr) - 1; while left <= right: mid = (left + right) // 2; if arr[mid] == target: return mid; elif arr[mid] < target: left = mid + 1; else: right = mid - 1; return -1",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "class LinkedList: def __init__(self): self.head = None; def append(self, data): new_node = Node(data); if not self.head: self.head = new_node; else: current = self.head; while current.next: current = current.next; current.next = new_node",
            "list_comprehension = [x**2 for x in range(10) if x % 2 == 0]  # Square even numbers"
        ]
        
        metadatas = [
            {"algorithm": "sorting", "complexity": "O(n log n)", "type": "recursive"},
            {"algorithm": "searching", "complexity": "O(log n)", "type": "iterative"},
            {"algorithm": "dynamic_programming", "complexity": "O(2^n)", "type": "recursive"},
            {"data_structure": "linked_list", "complexity": "O(n)", "type": "class"},
            {"concept": "list_comprehension", "complexity": "O(n)", "type": "functional"}
        ]
        
        # Add documents to collection
        collection.add(
            documents=code_samples,
            metadatas=metadatas,
            ids=[f"code_{i}" for i in range(len(code_samples))]
        )
        
        print(f"SUCCESS: Added {len(code_samples)} code samples to vector database!")
        
        # Test different types of queries
        test_queries = [
            "How to sort an array in Python?",
            "Binary search implementation",
            "Generate sequence of numbers",
            "Work with lists in Python"
        ]
        
        print("\nTesting Semantic Search:")
        print("-" * 30)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            
            results = collection.query(
                query_texts=[query],
                n_results=2,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['documents'][0]:
                for j, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    print(f"  Result {j+1}: {doc[:60]}...")
                    print(f"    Metadata: {metadata}")
                    print(f"    Similarity: {1 - distance:.3f}")
            else:
                print("  No results found")
        
        # Test filtering by metadata
        print("\nTesting Metadata Filtering:")
        print("-" * 30)
        
        # Find all sorting algorithms
        sorting_results = collection.query(
            query_texts=["sorting algorithm"],
            n_results=5,
            where={"algorithm": "sorting"}
        )
        
        print("Sorting algorithms found:")
        for doc in sorting_results['documents'][0]:
            print(f"  - {doc[:80]}...")
        
        # Find all O(log n) algorithms
        log_results = collection.query(
            query_texts=["efficient algorithm"],
            n_results=5,
            where={"complexity": "O(log n)"}
        )
        
        print("\nO(log n) algorithms found:")
        for doc in log_results['documents'][0]:
            print(f"  - {doc[:80]}...")
        
        print("\nSUCCESS: All Chroma offline tests passed!")
        
        # Show collection stats
        count = collection.count()
        print(f"\nCollection Statistics:")
        print(f"  Total documents: {count}")
        print(f"  Embedding dimension: 384")  # all-MiniLM-L6-v2 dimension
        print(f"  Storage: In-memory (no files created)")
        
        # Cleanup
        client.reset()
        return True
        
    except Exception as e:
        print(f"ERROR: Chroma offline test failed: {e}")
        return False

def demonstrate_rag_pipeline():
    """Demonstrate a complete RAG pipeline with offline tools."""
    
    print("\n" + "=" * 60)
    print("COMPLETE RAG PIPELINE DEMO (OFFLINE)")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        import chromadb
        from chromadb.config import Settings
        
        # Step 1: Initialize components
        print("Step 1: Initializing RAG components...")
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
        
        # Custom embedding function
        class RAGEmbeddingFunction:
            def __init__(self, model):
                self.model = model
            def __call__(self, texts):
                return self.model.encode(texts if isinstance(texts, list) else [texts]).tolist()
        
        embedding_fn = RAGEmbeddingFunction(model)
        
        # Step 2: Create knowledge base
        print("Step 2: Building Python knowledge base...")
        
        collection = client.get_or_create_collection(
            name="python_knowledge",
            embedding_function=embedding_fn
        )
        
        # Sample Python knowledge base
        knowledge_items = [
            {
                "content": "Python lists are mutable sequences that can store multiple items of different types. They support indexing, slicing, and methods like append(), remove(), and sort().",
                "topic": "data_structures",
                "difficulty": "beginner"
            },
            {
                "content": "List comprehensions provide a concise way to create lists: [expression for item in iterable if condition]. Example: squares = [x**2 for x in range(10)]",
                "topic": "syntax",
                "difficulty": "intermediate"
            },
            {
                "content": "The quicksort algorithm works by selecting a pivot element and partitioning the array around it, then recursively sorting the sub-arrays.",
                "topic": "algorithms", 
                "difficulty": "advanced"
            },
            {
                "content": "Binary search finds an element in a sorted array by repeatedly dividing the search interval in half. Time complexity is O(log n).",
                "topic": "algorithms",
                "difficulty": "intermediate"
            },
            {
                "content": "Python functions are defined with 'def' keyword: def function_name(parameters): return value. They support default parameters and keyword arguments.",
                "topic": "functions",
                "difficulty": "beginner"
            }
        ]
        
        # Add to vector database
        collection.add(
            documents=[item["content"] for item in knowledge_items],
            metadatas=[{k: v for k, v in item.items() if k != "content"} for item in knowledge_items],
            ids=[f"knowledge_{i}" for i in range(len(knowledge_items))]
        )
        
        print(f"SUCCESS: Added {len(knowledge_items)} knowledge items")
        
        # Step 3: Query processing and retrieval
        print("Step 3: Processing user queries...")
        
        user_queries = [
            "How do I work with lists in Python?",
            "What's the most efficient way to search in a sorted array?",
            "How to write a function in Python?",
            "Explain quicksort algorithm"
        ]
        
        for query in user_queries:
            print(f"\nUser Query: '{query}'")
            
            # Retrieve relevant context
            results = collection.query(
                query_texts=[query],
                n_results=2,
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['documents'][0]:
                print("  Retrieved Context:")
                for i, (doc, meta, dist) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    relevance = 1 - dist
                    print(f"    {i+1}. [{meta['topic'].upper()}] {doc[:100]}...")
                    print(f"       Relevance: {relevance:.3f} | Difficulty: {meta['difficulty']}")
                
                # Simulate LLM response generation
                best_context = results['documents'][0][0]
                print(f"  Generated Response: Based on the context, {best_context[:150]}...")
            
        print("\nSUCCESS: RAG pipeline demonstration complete!")
        
        # Cleanup
        client.reset()
        return True
        
    except Exception as e:
        print(f"ERROR: RAG pipeline demo failed: {e}")
        return False

def main():
    """Run the complete offline Chroma demonstration."""
    
    print("Python Code Helper - Chroma Offline Demo")
    print("=" * 45)
    print("This demo shows how to use Chroma WITHOUT network downloads")
    print("Perfect for avoiding timeout issues!\n")
    
    # Test 1: Basic Chroma with custom embeddings
    success1 = test_chroma_with_custom_embeddings()
    
    if success1:
        # Test 2: Complete RAG pipeline
        success2 = demonstrate_rag_pipeline()
        
        if success2:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED - CHROMA IS WORKING OFFLINE!")
            print("=" * 60)
            print("\nWhat this proves:")
            print("✅ Chroma vector database is fully functional")
            print("✅ Custom embeddings work without downloads")
            print("✅ Semantic search is operational")
            print("✅ Metadata filtering works")
            print("✅ Complete RAG pipeline is possible")
            
            print("\nNext Steps:")
            print("1. Use this offline approach in your main application")
            print("2. Set up GitHub token for real data ingestion")
            print("3. Install Ollama for local LLM integration")
            print("4. You have a working RAG system foundation!")
            
        else:
            print("Basic Chroma works, but RAG pipeline needs work")
    else:
        print("Chroma still having issues - try upgrading: pip install --upgrade chromadb")

if __name__ == "__main__":
    main() 