"""
Chroma Alternative - Drop-in Replacement

This provides a Chroma-like interface using our working numpy-based vector store.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
import uuid

class ChromaAlternativeCollection:
    """A collection that mimics Chroma API but uses our working vector store."""
    
    def __init__(self, name: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.name = name
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = []
        self.metadatas = []
        self.ids = []
    
    def add(self, documents: List[str], metadatas: Optional[List[Dict]] = None, ids: Optional[List[str]] = None):
        """Add documents to the collection."""
        if metadatas is None:
            metadatas = [{}] * len(documents)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings
        embeddings = self.model.encode(documents)
        
        # Store everything
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)
        
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def query(self, query_texts: List[str], n_results: int = 10, where: Optional[Dict] = None, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query the collection."""
        if not self.documents:
            return {'documents': [[]], 'metadatas': [[]], 'distances': [[]], 'ids': [[]]}
        
        # Generate query embedding
        query_embedding = self.model.encode(query_texts)[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Convert to distances (Chroma uses distances, lower is better)
        distances = 1 - similarities
        
        # Get top results
        top_indices = np.argsort(distances)[:n_results]
        
        # Filter by metadata if specified
        if where:
            filtered_indices = []
            for idx in top_indices:
                metadata = self.metadatas[idx]
                match = True
                for key, value in where.items():
                    if metadata.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_indices.append(idx)
            top_indices = filtered_indices[:n_results]
        
        # Prepare results
        result_docs = [self.documents[idx] for idx in top_indices]
        result_metadatas = [self.metadatas[idx] for idx in top_indices]
        result_distances = [float(distances[idx]) for idx in top_indices]
        result_ids = [self.ids[idx] for idx in top_indices]
        
        # Format like Chroma (nested lists for multiple queries)
        return {
            'documents': [result_docs],
            'metadatas': [result_metadatas],
            'distances': [result_distances],
            'ids': [result_ids]
        }
    
    def count(self) -> int:
        """Get number of documents in collection."""
        return len(self.documents)
    
    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        indices_to_remove = []
        for i, doc_id in enumerate(self.ids):
            if doc_id in ids:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for idx in sorted(indices_to_remove, reverse=True):
            del self.documents[idx]
            del self.metadatas[idx]
            del self.ids[idx]
            if len(self.embeddings) > 0:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)

class ChromaAlternativeClient:
    """A client that mimics Chroma API."""
    
    def __init__(self):
        self.collections = {}
    
    def create_collection(self, name: str, **kwargs) -> ChromaAlternativeCollection:
        """Create a new collection."""
        if name in self.collections:
            raise ValueError(f"Collection {name} already exists")
        
        collection = ChromaAlternativeCollection(name)
        self.collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> ChromaAlternativeCollection:
        """Get an existing collection."""
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist")
        return self.collections[name]
    
    def get_or_create_collection(self, name: str, **kwargs) -> ChromaAlternativeCollection:
        """Get or create a collection."""
        if name in self.collections:
            return self.collections[name]
        else:
            return self.create_collection(name, **kwargs)
    
    def delete_collection(self, name: str):
        """Delete a collection."""
        if name in self.collections:
            del self.collections[name]
    
    def list_collections(self):
        """List all collections."""
        return [{"name": name} for name in self.collections.keys()]
    
    def reset(self):
        """Reset the client (clear all collections)."""
        self.collections.clear()

# Convenience function
def Client():
    """Create a Chroma alternative client."""
    return ChromaAlternativeClient()

def test_chroma_alternative():
    """Test the Chroma alternative."""
    print("Testing Chroma Alternative...")
    
    try:
        # Create client and collection
        client = Client()
        collection = client.create_collection("test")
        
        # Add documents
        collection.add(
            documents=["Python is a programming language", "FastAPI is a web framework"],
            metadatas=[{"topic": "language"}, {"topic": "web"}],
            ids=["doc1", "doc2"]
        )
        
        # Query
        results = collection.query(
            query_texts=["programming"],
            n_results=2
        )
        
        success = len(results['documents'][0]) > 0
        print(f"Chroma Alternative test: {'✅ PASSED' if success else '❌ FAILED'}")
        
        if success:
            print(f"  Found {len(results['documents'][0])} results")
            print(f"  Top result: {results['documents'][0][0][:50]}...")
            print(f"  Distance: {results['distances'][0][0]:.3f}")
        
        return success
        
    except Exception as e:
        print(f"Chroma Alternative test: ❌ FAILED - {e}")
        return False

if __name__ == "__main__":
    test_chroma_alternative()
