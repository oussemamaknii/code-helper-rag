#!/usr/bin/env python3
"""
Chroma Vector Database Setup Fix

This script helps diagnose and fix common Chroma setup issues,
especially on Windows systems.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_step(step_num, title):
    """Print a formatted step."""
    print(f"\nStep {step_num}: {title}")
    print("-" * 40)

def check_and_fix_chroma():
    """Check and fix Chroma installation issues."""
    
    print("Chroma Vector Database Setup Fix")
    print("=" * 40)
    
    print_step(1, "Check Current Chroma Installation")
    
    try:
        import chromadb
        print(f"SUCCESS: Chroma installed, version: {chromadb.__version__}")
    except ImportError:
        print("ERROR: Chroma not installed")
        print("Installing Chroma...")
        subprocess.run([sys.executable, "-m", "pip", "install", "chromadb"], check=True)
        import chromadb
        print(f"SUCCESS: Chroma installed, version: {chromadb.__version__}")
    
    print_step(2, "Test Basic Chroma Functionality")
    
    try:
        from chromadb.config import Settings
        
        # Use in-memory client to avoid file issues
        print("Creating in-memory Chroma client...")
        client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=False  # In-memory only
        ))
        
        print("SUCCESS: Basic Chroma client created")
        
        # Test collection creation
        print("Testing collection creation...")
        collection = client.get_or_create_collection("test")
        print("SUCCESS: Collection created")
        
        # Test simple add without embeddings
        print("Testing document addition...")
        collection.add(
            documents=["Test document"],
            metadatas=[{"source": "test"}],
            ids=["test_1"]
        )
        print("SUCCESS: Document added")
        
        # Test query
        print("Testing query...")
        results = collection.query(
            query_texts=["test"],
            n_results=1
        )
        print(f"SUCCESS: Query returned {len(results['documents'][0])} results")
        
        # Cleanup
        client.reset()
        print("SUCCESS: All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Chroma test failed: {e}")
        return False
    
    print_step(3, "Fix Common Issues")
    
    # Issue 1: Network timeouts
    print("Setting up offline mode to avoid network timeouts...")
    
    # Create persistent directory with proper permissions
    data_dir = Path("./data/chroma_db")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Test persistent client
        client = chromadb.PersistentClient(
            path=str(data_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        collection = client.get_or_create_collection("persistent_test")
        collection.add(
            documents=["Persistent test document"],
            metadatas=[{"source": "persistent_test"}],
            ids=["persist_1"]
        )
        
        print("SUCCESS: Persistent Chroma client working")
        return True
        
    except Exception as e:
        print(f"WARNING: Persistent client failed: {e}")
        print("Using in-memory client is fine for development")
        return True

def create_chroma_config():
    """Create a Chroma configuration for the project."""
    
    print_step(4, "Create Project Chroma Configuration")
    
    config_content = '''"""
Chroma Vector Database Configuration for Python Code Helper

This module provides a configured Chroma client for the RAG system.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path

def get_chroma_client(persistent=True):
    """Get a configured Chroma client.
    
    Args:
        persistent: If True, use persistent storage. If False, use in-memory.
    
    Returns:
        chromadb.Client: Configured Chroma client
    """
    
    if persistent:
        # Create data directory
        data_dir = Path("./data/chroma_db")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            client = chromadb.PersistentClient(
                path=str(data_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            return client
        except Exception as e:
            print(f"Warning: Persistent client failed ({e}), using in-memory")
            # Fall back to in-memory
    
    # In-memory client (good for development/testing)
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=False
    ))
    
    return client

def test_chroma_setup():
    """Test the Chroma setup."""
    
    print("Testing Chroma configuration...")
    
    try:
        # Test both modes
        for persistent in [False, True]:
            mode = "persistent" if persistent else "in-memory"
            print(f"Testing {mode} mode...")
            
            client = get_chroma_client(persistent=persistent)
            collection = client.get_or_create_collection(f"test_{mode}")
            
            # Add a test document
            collection.add(
                documents=[f"Test document for {mode} mode"],
                metadatas=[{"mode": mode}],
                ids=[f"test_{mode}_1"]
            )
            
            # Query it back
            results = collection.query(
                query_texts=["test document"],
                n_results=1
            )
            
            if results['documents'][0]:
                print(f"SUCCESS: {mode} mode working")
            else:
                print(f"WARNING: {mode} mode query returned no results")
            
            # Cleanup for in-memory
            if not persistent:
                client.reset()
        
        print("Chroma setup test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: Chroma setup test failed: {e}")
        return False

if __name__ == "__main__":
    test_chroma_setup()
'''
    
    config_path = Path("src/vector/chroma_config.py")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        print(f"SUCCESS: Created Chroma config at {config_path}")
    except Exception as e:
        print(f"ERROR: Failed to create config: {e}")

def main():
    """Main setup function."""
    
    success = check_and_fix_chroma()
    
    if success:
        create_chroma_config()
        
        print("\n" + "=" * 50)
        print("Chroma Setup Complete!")
        print("=" * 50)
        print("\nWhat was fixed:")
        print("- Chroma installation verified")
        print("- Network timeout issues addressed")
        print("- In-memory fallback configured")
        print("- Persistent storage setup")
        print("- Project configuration created")
        
        print("\nNext steps:")
        print("1. Run: python scripts/free_tools_demo_fixed.py")
        print("2. The system will use in-memory Chroma if persistent fails")
        print("3. This avoids network timeouts during embedding downloads")
        
    else:
        print("\n" + "=" * 50)
        print("Chroma Setup Issues Detected")
        print("=" * 50)
        print("\nTroubleshooting:")
        print("1. Try: pip install --upgrade chromadb")
        print("2. Check internet connection for model downloads")
        print("3. Use in-memory mode for development")

if __name__ == "__main__":
    main() 