#!/usr/bin/env python3
"""
FastAPI Web Interface for Python Code Helper RAG System

This creates a web interface for our working RAG system using:
- Sentence Transformers for embeddings
- Numpy-based vector search
- Ollama for LLM generation
- GitHub API for data collection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
import logging
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Our working RAG components
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import subprocess
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleVectorStore:
    """Simple in-memory vector store using numpy (no Chroma needed)."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add_documents(self, docs, metadatas=None, ids=None):
        """Add documents to the store."""
        if metadatas is None:
            metadatas = [{}] * len(docs)
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(self.documents), len(self.documents) + len(docs))]
        
        self.documents.extend(docs)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
    
    def add_embeddings(self, embeddings):
        """Add precomputed embeddings."""
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query_embedding, top_k=5):
        """Search for similar documents."""
        if len(self.embeddings) == 0:
            return []
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'similarity': float(similarities[idx]),
                'id': self.ids[idx]
            })
        
        return results

class RAGSystem:
    """Complete RAG system using working components."""
    
    def __init__(self):
        logger.info("Initializing RAG system...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = SimpleVectorStore()
        self.is_initialized = False
        logger.info("RAG system initialized!")
    
    def load_knowledge_base(self):
        """Load the knowledge base with Python programming content."""
        logger.info("Loading knowledge base...")
        
        # Static knowledge base
        static_knowledge = [
            {
                'content': """
Python Lists Tutorial:
Lists are ordered, mutable collections in Python. You can create them with square brackets:
my_list = [1, 2, 3, 'hello', True]

Common operations:
- append(item): Add item to end
- insert(index, item): Insert item at index  
- remove(item): Remove first occurrence
- pop(): Remove and return last item
- sort(): Sort list in place
- len(list): Get length

Example: numbers = [3, 1, 4, 1, 5]; numbers.sort(); print(numbers) # [1, 1, 3, 4, 5]
""".strip(),
                'metadata': {'topic': 'data_structures', 'difficulty': 'beginner', 'type': 'tutorial'}
            },
            {
                'content': """
Python Functions Tutorial:
Functions are reusable blocks of code. Define them with 'def':

def function_name(parameters):
    \"\"\"Optional docstring\"\"\"
    # function body
    return result

Example:
def calculate_area(length, width):
    \"\"\"Calculate rectangle area\"\"\"
    return length * width

area = calculate_area(5, 3)  # Returns 15
""".strip(),
                'metadata': {'topic': 'functions', 'difficulty': 'beginner', 'type': 'tutorial'}
            },
            {
                'content': """
Quicksort Algorithm in Python:
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)

# Usage: sorted_array = quicksort([3, 6, 8, 10, 1, 2, 1])
# Time complexity: O(n log n) average, O(n²) worst case
""".strip(),
                'metadata': {'topic': 'algorithms', 'difficulty': 'intermediate', 'type': 'implementation'}
            },
            {
                'content': """
Binary Search in Python:
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found

# Usage: index = binary_search([1, 2, 3, 4, 5], 3)  # Returns 2
# Time complexity: O(log n), Space complexity: O(1)
""".strip(),
                'metadata': {'topic': 'algorithms', 'difficulty': 'intermediate', 'type': 'implementation'}
            },
            {
                'content': """
List Comprehensions in Python:
List comprehensions provide a concise way to create lists:

# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(10)]  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition: [expression for item in iterable if condition]
even_squares = [x**2 for x in range(10) if x % 2 == 0]  # [0, 4, 16, 36, 64]

# Nested: [expression for item1 in iterable1 for item2 in iterable2]
matrix = [[i*j for j in range(3)] for i in range(3)]  # [[0,0,0], [0,1,2], [0,2,4]]
""".strip(),
                'metadata': {'topic': 'syntax', 'difficulty': 'intermediate', 'type': 'tutorial'}
            }
        ]
        
        # Try to add GitHub data
        try:
            github_docs = self._collect_github_data()
            static_knowledge.extend(github_docs)
        except Exception as e:
            logger.warning(f"GitHub data collection failed: {e}")
        
        # Add documents to vector store
        self.vector_store.add_documents(
            docs=[doc['content'] for doc in static_knowledge],
            metadatas=[doc['metadata'] for doc in static_knowledge],
            ids=[f"doc_{i}" for i in range(len(static_knowledge))]
        )
        
        # Create embeddings
        texts = [doc['content'] for doc in static_knowledge]
        embeddings = self.model.encode(texts)
        self.vector_store.add_embeddings(embeddings)
        
        self.is_initialized = True
        logger.info(f"Knowledge base loaded with {len(static_knowledge)} documents")
    
    def _collect_github_data(self):
        """Collect data from GitHub API."""
        github_docs = []
        
        url = "https://api.github.com/search/repositories"
        params = {
            'q': 'python algorithms language:python',
            'sort': 'stars',
            'per_page': 5
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            for repo in data['items']:
                doc = {
                    'content': f"""
Repository: {repo['name']}
Description: {repo['description'] or 'No description'}
Language: Python
Stars: {repo['stargazers_count']}
Topics: {', '.join(repo.get('topics', []))}
URL: {repo['html_url']}

This repository contains Python code for: {repo['description'] or repo['name']}
""".strip(),
                    'metadata': {
                        'source': 'github',
                        'repo_name': repo['name'],
                        'stars': repo['stargazers_count'],
                        'language': 'python'
                    }
                }
                github_docs.append(doc)
        
        return github_docs
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        if not self.is_initialized:
            raise HTTPException(status_code=500, detail="RAG system not initialized")
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search for relevant documents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        return results
    
    def generate_with_ollama(self, prompt: str) -> str:
        """Generate code with Ollama."""
        try:
            import ollama
            
            response = ollama.chat(
                model='codellama:7b',
                messages=[
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            return response['message']['content']
            
        except ImportError:
            return "Ollama Python library not available. Install with: pip install ollama"
        except Exception as e:
            return f"Ollama generation failed: {e}"
    
    def chat(self, query: str) -> Dict[str, Any]:
        """Complete chat functionality with retrieval and generation."""
        # Search for relevant context
        search_results = self.search(query, top_k=3)
        
        if not search_results:
            return {
                'answer': 'I could not find relevant information in my knowledge base.',
                'sources': [],
                'confidence': 0.0
            }
        
        # Prepare context for LLM
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(result['document'][:300])  # Limit context length
            sources.append({
                'content': result['document'][:150] + '...',
                'metadata': result['metadata'],
                'similarity': result['similarity']
            })
        
        context = '\n\n'.join(context_parts)
        
        # Generate response with Ollama
        prompt = f"""Based on the following context about Python programming, answer the user's question.

Context:
{context}

Question: {query}

Please provide a helpful and accurate answer based on the context provided:"""
        
        answer = self.generate_with_ollama(prompt)
        
        # Calculate confidence based on top similarity score
        confidence = search_results[0]['similarity'] if search_results else 0.0
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'context_used': len(context_parts)
        }

# Initialize FastAPI app
app = FastAPI(
    title="Python Code Helper RAG",
    description="AI-powered Python programming assistant using free and open source tools",
    version="1.0.0"
)

# Initialize RAG system
rag_system = RAGSystem()

# Request/Response models
class ChatRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    context_used: int

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    logger.info("Starting RAG system initialization...")
    rag_system.load_knowledge_base()
    logger.info("RAG system ready!")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Python Code Helper RAG</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        button {
            padding: 15px 30px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background: #5a6fd8;
        }
        .response {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .sources {
            margin-top: 15px;
            padding: 15px;
            background: #e9ecef;
            border-radius: 8px;
        }
        .source-item {
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }
        .confidence {
            display: inline-block;
            padding: 3px 8px;
            background: #28a745;
            color: white;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }
        .loading {
            text-align: center;
            color: #667eea;
            font-style: italic;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        .examples {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .example-button {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: #e9ecef;
            border: 1px solid #ddd;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        }
        .example-button:hover {
            background: #667eea;
            color: white;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐍 Python Code Helper RAG</h1>
        <p>AI-powered Python programming assistant using 100% free tools</p>
        <p>✅ Sentence Transformers • ✅ GitHub API • ✅ Ollama LLM • ✅ Local Vector Search</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-number" id="doc-count">Loading...</div>
            <div>Documents</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">384</div>
            <div>Vector Dimensions</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">$0</div>
            <div>Monthly Cost</div>
        </div>
        <div class="stat-card">
            <div class="stat-number" id="status">🟢 Ready</div>
            <div>System Status</div>
        </div>
    </div>

    <div class="chat-container">
        <h2>Ask me about Python programming!</h2>
        <div class="input-group">
            <input type="text" id="queryInput" placeholder="e.g., How do I implement quicksort in Python?" />
            <button onclick="askQuestion()">Ask</button>
        </div>
        <div id="response"></div>
    </div>

    <div class="examples">
        <h3>Example Questions</h3>
        <div class="example-button" onclick="setQuery('How do I sort a list in Python?')">List Sorting</div>
        <div class="example-button" onclick="setQuery('What is binary search and how do I implement it?')">Binary Search</div>
        <div class="example-button" onclick="setQuery('Show me list comprehension examples')">List Comprehensions</div>
        <div class="example-button" onclick="setQuery('How to create Python functions?')">Functions</div>
        <div class="example-button" onclick="setQuery('Find Python repositories for algorithms')">GitHub Repos</div>
        <div class="example-button" onclick="setQuery('Generate a sorting algorithm implementation')">Code Generation</div>
    </div>

    <script>
        // Load system stats
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                document.getElementById('doc-count').textContent = data.documents || 'N/A';
                document.getElementById('status').textContent = data.status === 'healthy' ? '🟢 Ready' : '🔴 Error';
            });

        function setQuery(query) {
            document.getElementById('queryInput').value = query;
        }

        function askQuestion() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) return;

            const responseDiv = document.getElementById('response');
            responseDiv.innerHTML = '<div class="loading">🤔 Thinking and searching knowledge base...</div>';

            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                let sourcesHtml = '';
                if (data.sources && data.sources.length > 0) {
                    sourcesHtml = '<div class="sources"><h4>📚 Sources:</h4>';
                    data.sources.forEach(source => {
                        const confidence = (source.similarity * 100).toFixed(1);
                        sourcesHtml += `
                            <div class="source-item">
                                <strong>${source.metadata.topic || 'General'}</strong>
                                <span class="confidence">${confidence}% match</span>
                                <br>
                                <small>${source.content}</small>
                            </div>
                        `;
                    });
                    sourcesHtml += '</div>';
                }

                const confidenceColor = data.confidence > 0.7 ? '#28a745' : data.confidence > 0.5 ? '#ffc107' : '#dc3545';
                const confidenceText = (data.confidence * 100).toFixed(1);

                responseDiv.innerHTML = `
                    <div class="response">
                        <h3>💡 Answer <span style="color: ${confidenceColor}; font-size: 14px;">(${confidenceText}% confidence)</span></h3>
                        <div style="white-space: pre-wrap; line-height: 1.6;">${data.answer}</div>
                        ${sourcesHtml}
                        <small style="color: #666; margin-top: 10px; display: block;">
                            Used ${data.context_used} context sources • Generated with Ollama CodeLlama
                        </small>
                    </div>
                `;
            })
            .catch(error => {
                responseDiv.innerHTML = `<div class="response" style="border-left-color: #dc3545;">
                    <h3>❌ Error</h3>
                    <p>Sorry, there was an error processing your question: ${error.message}</p>
                </div>`;
            });
        }

        // Allow Enter key to submit
        document.getElementById('queryInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askQuestion();
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for conversational interface."""
    try:
        result = rag_system.chat(request.query)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search endpoint for document retrieval."""
    try:
        results = rag_system.search(request.query, top_k=request.top_k)
        return SearchResponse(
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        'status': 'healthy' if rag_system.is_initialized else 'initializing',
        'documents': len(rag_system.vector_store.documents) if rag_system.is_initialized else 0,
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_store': 'numpy-based',
        'llm': 'ollama-codellama:7b'
    }

@app.post("/generate")
async def generate_code(request: dict):
    """Generate code using Ollama."""
    try:
        prompt = request.get('prompt', '')
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        result = rag_system.generate_with_ollama(prompt)
        return {'generated_code': result}
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 