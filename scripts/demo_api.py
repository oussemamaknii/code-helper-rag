#!/usr/bin/env python3
"""
Simple API demonstration script.

This script demonstrates the basic FastAPI functionality with mock responses
to show the API structure and endpoints.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import httpx
    import uvicorn
    from multiprocessing import Process
    import time
    
    # Test imports
    from src.api.app import app
    
    print("🎉 All required dependencies are available!")
    
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Please install with: pip install -r requirements_api.txt")
    sys.exit(1)


class SimpleAPIDemo:
    """Simple API demonstration."""
    
    def __init__(self):
        """Initialize API demo."""
        self.base_url = "http://localhost:8000"
        self.api_key = "test_key_12345"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def demo_basic_endpoints(self):
        """Demonstrate basic API endpoints."""
        print("\n🚀 Python Code Helper API Demo")
        print("=" * 50)
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            
            # Test root endpoint
            print("\n1. 🏠 Root Endpoint")
            print("-" * 30)
            try:
                response = await client.get(f"{self.base_url}/")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ API Name: {data.get('name')}")
                    print(f"✅ Version: {data.get('version')}")
                    print(f"✅ Status: {data.get('status')}")
                    print(f"✅ Available endpoints: {len(data.get('endpoints', {}))}")
                else:
                    print(f"⚠️ Status: {response.status_code}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test health endpoint
            print("\n2. 🏥 Health Check")
            print("-" * 30)
            try:
                response = await client.get(f"{self.base_url}/api/v1/health")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ System Status: {data.get('status')}")
                    print(f"✅ Uptime: {data.get('uptime', 0):.1f}s")
                    
                    components = data.get('components', {})
                    print(f"✅ Components checked: {len(components)}")
                    for component, status in components.items():
                        health_status = status.get('status', 'unknown')
                        print(f"   • {component}: {health_status}")
                else:
                    print(f"⚠️ Status: {response.status_code}")
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test chat endpoint
            print("\n3. 💬 Chat Endpoint")
            print("-" * 30)
            try:
                chat_request = {
                    "message": "How does the quicksort algorithm work?",
                    "context": {
                        "programming_language": "python",
                        "difficulty_level": "intermediate"
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/api/v1/chat",
                    headers=self.headers,
                    json=chat_request
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Response generated successfully")
                    print(f"✅ Response length: {len(data.get('response', ''))} chars")
                    print(f"✅ Query type: {data.get('query_type')}")
                    print(f"✅ Sources found: {len(data.get('sources', []))}")
                    print(f"✅ Confidence score: {data.get('metrics', {}).get('confidence_score', 0):.3f}")
                    print(f"✅ Suggestions: {len(data.get('suggestions', []))}")
                    
                    # Show preview of response
                    response_preview = data.get('response', '')[:150] + "..." if len(data.get('response', '')) > 150 else data.get('response', '')
                    print(f"\n📝 Response preview:")
                    print(f"   {response_preview}")
                    
                else:
                    print(f"⚠️ Status: {response.status_code}")
                    if response.status_code == 401:
                        print("   Authentication may be required")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test search endpoint  
            print("\n4. 🔍 Search Endpoint")
            print("-" * 30)
            try:
                search_request = {
                    "query": "python sorting algorithms",
                    "top_k": 5,
                    "search_type": "hybrid"
                }
                
                response = await client.post(
                    f"{self.base_url}/api/v1/search",
                    headers=self.headers,
                    json=search_request
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Search completed successfully")
                    print(f"✅ Results found: {len(data.get('results', []))}")
                    print(f"✅ Total results: {data.get('total_results', 0)}")
                    print(f"✅ Search time: {data.get('search_time', 0):.3f}s")
                    
                    # Show first result
                    results = data.get('results', [])
                    if results:
                        first_result = results[0]
                        print(f"\n📄 Top result:")
                        print(f"   Score: {first_result.get('relevance_score', 0):.3f}")
                        print(f"   Source: {first_result.get('source_type', 'unknown')}")
                        content_preview = first_result.get('content', '')[:100] + "..." if len(first_result.get('content', '')) > 100 else first_result.get('content', '')
                        print(f"   Content: {content_preview}")
                        
                else:
                    print(f"⚠️ Status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
            
            # Test documentation endpoints
            print("\n5. 📚 Documentation")
            print("-" * 30)
            try:
                # Check OpenAPI schema
                response = await client.get(f"{self.base_url}/openapi.json")
                if response.status_code == 200:
                    schema = response.json()
                    print(f"✅ OpenAPI schema available")
                    print(f"✅ API title: {schema.get('info', {}).get('title')}")
                    print(f"✅ API version: {schema.get('info', {}).get('version')}")
                    print(f"✅ Endpoints defined: {len(schema.get('paths', {}))}")
                    
                    # Check if Swagger UI is available
                    docs_response = await client.get(f"{self.base_url}/docs")
                    if docs_response.status_code == 200:
                        print(f"✅ Swagger UI available at /docs")
                    
                    # Check if ReDoc is available
                    redoc_response = await client.get(f"{self.base_url}/redoc")
                    if redoc_response.status_code == 200:
                        print(f"✅ ReDoc available at /redoc")
                        
                else:
                    print(f"⚠️ OpenAPI schema status: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n🎉 API Demo Completed!")
        print("\n📋 Summary:")
        print("   • FastAPI application structure implemented")
        print("   • REST endpoints for chat, search, and health checks")
        print("   • Authentication system with API keys")
        print("   • Interactive documentation (Swagger UI + ReDoc)")
        print("   • Rate limiting and middleware")
        print("   • Structured error handling")
        print("   • Mock responses demonstrating full functionality")
        
        print("\n🌐 Access the API:")
        print(f"   • API Root: {self.base_url}/")
        print(f"   • Swagger UI: {self.base_url}/docs")
        print(f"   • ReDoc: {self.base_url}/redoc")
        print(f"   • Health Check: {self.base_url}/api/v1/health")


def start_server():
    """Start the FastAPI server."""
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        log_level="warning",  # Reduce log noise
        access_log=False
    )


async def main():
    """Main demo function."""
    print("🔧 Starting FastAPI server...")
    
    # Start server in background
    server_process = Process(target=start_server)
    server_process.start()
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    try:
        # Check if server is ready
        async with httpx.AsyncClient() as client:
            for attempt in range(10):
                try:
                    response = await client.get("http://localhost:8000/health")
                    if response.status_code == 200:
                        break
                except:
                    if attempt < 9:
                        await asyncio.sleep(0.5)
                    else:
                        print("❌ Server failed to start properly")
                        return
        
        print("✅ Server is ready!")
        
        # Run the demo
        demo = SimpleAPIDemo()
        await demo.demo_basic_endpoints()
        
    finally:
        # Clean up
        print("\n🧹 Shutting down server...")
        server_process.terminate()
        server_process.join(timeout=3)
        if server_process.is_alive():
            server_process.kill()
        print("✅ Server stopped")


if __name__ == "__main__":
    asyncio.run(main()) 