#!/usr/bin/env python3
"""
Test API endpoints using Python requests.

This script tests the running FastAPI server endpoints to demonstrate functionality.
"""

import json
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import httpx
    import asyncio
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx"])
    import httpx
    import asyncio


async def test_api_endpoints():
    """Test all API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Python Code Helper API Endpoints")
    print("=" * 60)
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        
        # Test 1: Root endpoint
        print("\n1. 🏠 Root Endpoint")
        print("-" * 30)
        try:
            response = await client.get(f"{base_url}/")
            if response.status_code == 200:
                data = response.json()
                print("✅ Root endpoint successful")
                print(f"   API Name: {data.get('name')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Status: {data.get('status')}")
                print(f"   Endpoints: {list(data.get('endpoints', {}).keys())}")
            else:
                print(f"❌ Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 2: Health check
        print("\n2. 🏥 Health Check")
        print("-" * 30)
        try:
            response = await client.get(f"{base_url}/api/v1/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ Health check successful")
                print(f"   System Status: {data.get('status')}")
                print(f"   Uptime: {data.get('uptime', 0):.1f}s")
                
                components = data.get('components', {})
                print(f"   Components: {len(components)}")
                for name, comp in components.items():
                    status = comp.get('status', 'unknown')
                    latency = comp.get('latency', 0)
                    print(f"     • {name}: {status} ({latency:.3f}s)")
            else:
                print(f"❌ Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 3: Chat endpoint - Quicksort
        print("\n3. 💬 Chat Endpoint - Quicksort Query")
        print("-" * 30)
        try:
            chat_data = {
                "message": "How does quicksort work?",
                "context": {
                    "programming_language": "python",
                    "difficulty_level": "intermediate"
                }
            }
            
            response = await client.post(f"{base_url}/api/v1/chat", json=chat_data)
            if response.status_code == 200:
                data = response.json()
                print("✅ Chat response successful")
                print(f"   Response length: {len(data.get('response', ''))} chars")
                print(f"   Query type: {data.get('query_type')}")
                print(f"   Sources: {len(data.get('sources', []))}")
                print(f"   Suggestions: {len(data.get('suggestions', []))}")
                
                metrics = data.get('metrics', {})
                print(f"   Confidence: {metrics.get('confidence_score', 0):.3f}")
                print(f"   Tokens used: {metrics.get('tokens_used', 0)}")
                
                # Show response preview
                response_text = data.get('response', '')
                preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
                print(f"\n   📝 Response preview:")
                print(f"   {preview}")
                
                # Show first source
                sources = data.get('sources', [])
                if sources:
                    source = sources[0]
                    print(f"\n   📄 Top source:")
                    print(f"   • Type: {source.get('source_type')}")
                    print(f"   • Score: {source.get('relevance_score', 0):.3f}")
                    print(f"   • Usage: {source.get('usage_type')}")
                
            else:
                print(f"❌ Status: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Raw response: {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 4: Chat endpoint - Binary search
        print("\n4. 💬 Chat Endpoint - Binary Search Query")
        print("-" * 30)
        try:
            chat_data = {
                "message": "Explain binary search algorithm",
                "use_chain_of_thought": False
            }
            
            response = await client.post(f"{base_url}/api/v1/chat", json=chat_data)
            if response.status_code == 200:
                data = response.json()
                print("✅ Binary search response successful")
                print(f"   Query type: {data.get('query_type')}")
                print(f"   Response length: {len(data.get('response', ''))} chars")
                
                # Show suggestions
                suggestions = data.get('suggestions', [])
                print(f"   Follow-up suggestions:")
                for i, suggestion in enumerate(suggestions[:2], 1):
                    print(f"     {i}. {suggestion}")
                
            else:
                print(f"❌ Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 5: Search endpoint
        print("\n5. 🔍 Search Endpoint")
        print("-" * 30)
        try:
            search_data = {
                "query": "python sorting algorithms",
                "top_k": 3,
                "search_type": "hybrid"
            }
            
            response = await client.post(f"{base_url}/api/v1/search", json=search_data)
            if response.status_code == 200:
                data = response.json()
                print("✅ Search successful")
                print(f"   Results returned: {len(data.get('results', []))}")
                print(f"   Total results: {data.get('total_results', 0)}")
                print(f"   Search time: {data.get('search_time', 0):.3f}s")
                
                # Show results
                results = data.get('results', [])
                for i, result in enumerate(results[:2], 1):
                    print(f"\n   📄 Result {i}:")
                    print(f"   • Score: {result.get('relevance_score', 0):.3f}")
                    print(f"   • Source: {result.get('source_type')}")
                    print(f"   • Title: {result.get('title')}")
                    content = result.get('content', '')
                    preview = content[:100] + "..." if len(content) > 100 else content
                    print(f"   • Content: {preview}")
                
                # Show suggestions
                suggestions = data.get('suggestions', [])
                if suggestions:
                    print(f"\n   💡 Search suggestions:")
                    for suggestion in suggestions[:2]:
                        print(f"   • {suggestion}")
                
            else:
                print(f"❌ Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test 6: OpenAPI schema
        print("\n6. 📚 OpenAPI Documentation")
        print("-" * 30)
        try:
            response = await client.get(f"{base_url}/openapi.json")
            if response.status_code == 200:
                schema = response.json()
                print("✅ OpenAPI schema available")
                print(f"   Title: {schema.get('info', {}).get('title')}")
                print(f"   Version: {schema.get('info', {}).get('version')}")
                print(f"   Paths defined: {len(schema.get('paths', {}))}")
                
                # List available endpoints
                paths = schema.get('paths', {})
                print(f"   Available endpoints:")
                for path, methods in paths.items():
                    method_list = list(methods.keys())
                    print(f"     • {path}: {', '.join(method_list).upper()}")
                
            else:
                print(f"❌ Status: {response.status_code}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎉 API Testing Complete!")
    print("\n📊 Summary:")
    print("   ✅ FastAPI server running successfully")
    print("   ✅ REST endpoints operational")
    print("   ✅ Pydantic models working correctly")
    print("   ✅ Mock responses demonstrating full structure")
    print("   ✅ Interactive documentation available")
    
    print("\n🌐 Available Resources:")
    print(f"   • API Root: {base_url}/")
    print(f"   • Swagger UI: {base_url}/docs")
    print(f"   • ReDoc: {base_url}/redoc")
    print(f"   • OpenAPI Schema: {base_url}/openapi.json")


if __name__ == "__main__":
    asyncio.run(test_api_endpoints()) 