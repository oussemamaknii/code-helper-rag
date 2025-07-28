#!/usr/bin/env python3
"""
API integration test script.

This script demonstrates the FastAPI endpoints functionality including
chat, streaming, search, and system management endpoints.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

import httpx
import uvicorn
from multiprocessing import Process

from src.api.app import app
from src.utils.logger import get_logger

logger = get_logger(__name__)


class APITester:
    """API testing client for demonstration and validation."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "test_key_12345"):
        """
        Initialize API tester.
        
        Args:
            base_url: Base URL of the API server
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        self.client = httpx.AsyncClient(timeout=30.0)
        self.results = []
    
    async def test_root_endpoint(self) -> Dict[str, Any]:
        """Test the root endpoint."""
        print("\n🏠 Testing Root Endpoint")
        print("=" * 50)
        
        try:
            response = await self.client.get(f"{self.base_url}/")
            
            result = {
                "endpoint": "GET /",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": None,
                "data": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                print(f"✅ Root endpoint: {result['status_code']}")
                print(f"   API Name: {result['data'].get('name')}")
                print(f"   Version: {result['data'].get('version')}")
                print(f"   Status: {result['data'].get('status')}")
            else:
                print(f"❌ Root endpoint failed: {result['status_code']}")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            result = {"endpoint": "GET /", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health check endpoint."""
        print("\n🏥 Testing Health Check")
        print("=" * 50)
        
        try:
            start_time = time.time()
            response = await self.client.get(f"{self.base_url}/api/v1/health")
            response_time = time.time() - start_time
            
            result = {
                "endpoint": "GET /api/v1/health",
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "response_time": response_time,
                "data": response.json() if response.status_code == 200 else response.text
            }
            
            if result["success"]:
                print(f"✅ Health check: {result['status_code']} ({response_time:.3f}s)")
                print(f"   Status: {result['data'].get('status')}")
                print(f"   Uptime: {result['data'].get('uptime', 0):.1f}s")
                
                components = result['data'].get('components', {})
                for component, status in components.items():
                    health_status = status.get('status', 'unknown')
                    status_icon = "✅" if health_status == "healthy" else "⚠️"
                    print(f"   {status_icon} {component}: {health_status}")
            else:
                print(f"❌ Health check failed: {result['status_code']}")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"❌ Health check error: {e}")
            result = {"endpoint": "GET /api/v1/health", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def test_chat_endpoint(self) -> Dict[str, Any]:
        """Test the chat endpoint."""
        print("\n💬 Testing Chat Endpoint")
        print("=" * 50)
        
        try:
            # Test different types of queries
            test_queries = [
                {
                    "message": "How does the quicksort algorithm work?",
                    "context": {
                        "programming_language": "python",
                        "difficulty_level": "intermediate",
                        "include_examples": True
                    },
                    "use_chain_of_thought": False
                },
                {
                    "message": "Write a binary search function in Python",
                    "context": {
                        "programming_language": "python",
                        "include_tests": True,
                        "difficulty_level": "beginner"
                    }
                },
                {
                    "message": "What's the difference between lists and tuples?",
                    "context": {
                        "programming_language": "python",
                        "difficulty_level": "beginner"
                    }
                }
            ]
            
            results = []
            
            for i, query_data in enumerate(test_queries, 1):
                print(f"\n🔍 Test Query {i}: {query_data['message'][:50]}...")
                
                start_time = time.time()
                response = await self.client.post(
                    f"{self.base_url}/api/v1/chat",
                    headers=self.headers,
                    json=query_data
                )
                response_time = time.time() - start_time
                
                result = {
                    "endpoint": f"POST /api/v1/chat (query {i})",
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": response_time,
                    "query": query_data["message"],
                    "data": response.json() if response.status_code == 200 else response.text
                }
                
                if result["success"]:
                    data = result["data"]
                    print(f"   ✅ Chat response: {result['status_code']} ({response_time:.3f}s)")
                    print(f"   Response length: {len(data.get('response', ''))} chars")
                    print(f"   Query type: {data.get('query_type')}")
                    print(f"   Sources: {len(data.get('sources', []))}")
                    print(f"   Confidence: {data.get('metrics', {}).get('confidence_score', 0):.3f}")
                    print(f"   Tokens used: {data.get('metrics', {}).get('tokens_used', 0)}")
                    
                    if data.get('suggestions'):
                        print(f"   Suggestions: {len(data['suggestions'])}")
                        for suggestion in data['suggestions'][:2]:
                            print(f"     • {suggestion}")
                else:
                    print(f"   ❌ Chat failed: {result['status_code']}")
                    if isinstance(result["data"], dict):
                        print(f"   Error: {result['data'].get('error', {}).get('message', 'Unknown error')}")
                
                results.append(result)
                self.results.append(result)
                
                # Small delay between requests
                await asyncio.sleep(0.5)
            
            return {"chat_tests": results, "success": all(r["success"] for r in results)}
            
        except Exception as e:
            print(f"❌ Chat endpoint error: {e}")
            result = {"endpoint": "POST /api/v1/chat", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def test_streaming_endpoint(self) -> Dict[str, Any]:
        """Test the streaming chat endpoint."""
        print("\n🌊 Testing Streaming Chat")
        print("=" * 50)
        
        try:
            query_data = {
                "message": "Explain Python decorators with examples",
                "context": {
                    "programming_language": "python",
                    "difficulty_level": "intermediate"
                }
            }
            
            print(f"🔍 Streaming Query: {query_data['message']}")
            
            start_time = time.time()
            chunks_received = 0
            total_content = ""
            final_metadata = None
            
            async with self.client.stream('POST', 
                                        f"{self.base_url}/api/v1/chat/stream",
                                        headers=self.headers,
                                        json=query_data) as response:
                
                if response.status_code != 200:
                    print(f"❌ Streaming failed: {response.status_code}")
                    return {"endpoint": "POST /api/v1/chat/stream", "success": False, 
                           "status_code": response.status_code}
                
                print("📡 Receiving stream...")
                
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        chunk_data = json.loads(line[6:])  # Remove 'data: ' prefix
                        chunks_received += 1
                        
                        if chunk_data.get('is_complete'):
                            final_metadata = chunk_data
                            print(f"\n✅ Stream completed ({chunks_received} chunks)")
                        else:
                            content = chunk_data.get('content', '')
                            total_content += content
                            
                            # Print content in real-time (with some limits for demo)
                            if chunks_received <= 50:  # Limit output for demo
                                print(content, end='', flush=True)
            
            response_time = time.time() - start_time
            
            result = {
                "endpoint": "POST /api/v1/chat/stream",
                "status_code": 200,
                "success": True,
                "response_time": response_time,
                "chunks_received": chunks_received,
                "content_length": len(total_content),
                "final_metadata": final_metadata
            }
            
            print(f"\n📊 Streaming Results:")
            print(f"   Response time: {response_time:.3f}s")
            print(f"   Chunks received: {chunks_received}")
            print(f"   Content length: {len(total_content)} chars")
            
            if final_metadata:
                print(f"   Sources: {len(final_metadata.get('sources', []))}")
                metrics = final_metadata.get('metrics', {})
                if metrics:
                    print(f"   Confidence: {metrics.get('confidence_score', 0):.3f}")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"❌ Streaming endpoint error: {e}")
            result = {"endpoint": "POST /api/v1/chat/stream", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def test_search_endpoint(self) -> Dict[str, Any]:
        """Test the search endpoint."""
        print("\n🔍 Testing Search Endpoint")
        print("=" * 50)
        
        try:
            # Test different search queries
            search_queries = [
                {
                    "query": "binary search algorithm",
                    "top_k": 5,
                    "search_type": "hybrid"
                },
                {
                    "query": "python list comprehension",
                    "top_k": 3,
                    "search_type": "semantic",
                    "filters": {
                        "programming_language": "python"
                    }
                },
                {
                    "query": "sorting algorithms comparison",
                    "top_k": 7,
                    "search_type": "keyword"
                }
            ]
            
            results = []
            
            for i, search_data in enumerate(search_queries, 1):
                print(f"\n🔎 Search Query {i}: {search_data['query']}")
                
                start_time = time.time()
                response = await self.client.post(
                    f"{self.base_url}/api/v1/search",
                    headers=self.headers,
                    json=search_data
                )
                response_time = time.time() - start_time
                
                result = {
                    "endpoint": f"POST /api/v1/search (query {i})",
                    "status_code": response.status_code,
                    "success": response.status_code == 200,
                    "response_time": response_time,
                    "query": search_data["query"],
                    "data": response.json() if response.status_code == 200 else response.text
                }
                
                if result["success"]:
                    data = result["data"]
                    print(f"   ✅ Search response: {result['status_code']} ({response_time:.3f}s)")
                    print(f"   Results found: {len(data.get('results', []))}")
                    print(f"   Total results: {data.get('total_results', 0)}")
                    print(f"   Search time: {data.get('search_time', 0):.3f}s")
                    
                    # Show first few results
                    for j, result_item in enumerate(data.get('results', [])[:2]):
                        print(f"   📄 Result {j+1}:")
                        print(f"      Score: {result_item.get('relevance_score', 0):.3f}")
                        print(f"      Source: {result_item.get('source_type', 'unknown')}")
                        print(f"      Content: {result_item.get('content', '')[:100]}...")
                else:
                    print(f"   ❌ Search failed: {result['status_code']}")
                
                results.append(result)
                self.results.append(result)
                
                await asyncio.sleep(0.3)
            
            return {"search_tests": results, "success": all(r["success"] for r in results)}
            
        except Exception as e:
            print(f"❌ Search endpoint error: {e}")
            result = {"endpoint": "POST /api/v1/search", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality."""
        print("\n⏱️ Testing Rate Limiting")
        print("=" * 50)
        
        try:
            # Make multiple rapid requests to test rate limiting
            print("🚀 Making rapid requests to test rate limiting...")
            
            rapid_requests = []
            for i in range(5):  # Make 5 rapid requests
                start_time = time.time()
                response = await self.client.get(f"{self.base_url}/api/v1/health")
                response_time = time.time() - start_time
                
                rapid_requests.append({
                    "request": i + 1,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "rate_limit_headers": {
                        "limit": response.headers.get("X-RateLimit-Limit"),
                        "remaining": response.headers.get("X-RateLimit-Remaining"),
                        "reset": response.headers.get("X-RateLimit-Reset")
                    }
                })
                
                print(f"   Request {i+1}: {response.status_code} "
                      f"(remaining: {response.headers.get('X-RateLimit-Remaining', 'N/A')})")
            
            result = {
                "endpoint": "Rate Limiting Test",
                "success": True,
                "rapid_requests": rapid_requests,
                "all_succeeded": all(req["status_code"] == 200 for req in rapid_requests)
            }
            
            print(f"✅ Rate limiting test completed")
            print(f"   All requests succeeded: {result['all_succeeded']}")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"❌ Rate limiting test error: {e}")
            result = {"endpoint": "Rate Limiting Test", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def test_api_documentation(self) -> Dict[str, Any]:
        """Test API documentation endpoints."""
        print("\n📚 Testing API Documentation")
        print("=" * 50)
        
        try:
            # Test OpenAPI schema
            print("📋 Checking OpenAPI schema...")
            openapi_response = await self.client.get(f"{self.base_url}/openapi.json")
            
            # Test Swagger UI
            print("🎨 Checking Swagger UI...")
            docs_response = await self.client.get(f"{self.base_url}/docs")
            
            # Test ReDoc
            print("📖 Checking ReDoc...")
            redoc_response = await self.client.get(f"{self.base_url}/redoc")
            
            result = {
                "endpoint": "API Documentation",
                "openapi_status": openapi_response.status_code,
                "docs_status": docs_response.status_code,
                "redoc_status": redoc_response.status_code,
                "success": all(resp.status_code == 200 for resp in [openapi_response, docs_response, redoc_response])
            }
            
            if result["success"]:
                print("✅ All documentation endpoints accessible")
                print(f"   OpenAPI schema: {result['openapi_status']}")
                print(f"   Swagger UI: {result['docs_status']}")
                print(f"   ReDoc: {result['redoc_status']}")
                
                # Check OpenAPI schema content
                if openapi_response.status_code == 200:
                    schema = openapi_response.json()
                    print(f"   API Title: {schema.get('info', {}).get('title')}")
                    print(f"   API Version: {schema.get('info', {}).get('version')}")
                    print(f"   Endpoints: {len(schema.get('paths', {}))}")
            else:
                print("❌ Some documentation endpoints failed")
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"❌ Documentation test error: {e}")
            result = {"endpoint": "API Documentation", "success": False, "error": str(e)}
            self.results.append(result)
            return result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all API tests."""
        print("🧪 Python Code Helper API Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run tests in sequence
        tests = [
            ("Root Endpoint", self.test_root_endpoint),
            ("Health Check", self.test_health_endpoint),
            ("Chat Endpoint", self.test_chat_endpoint),
            ("Streaming Chat", self.test_streaming_endpoint),
            ("Search Endpoint", self.test_search_endpoint),
            ("Rate Limiting", self.test_rate_limiting),
            ("API Documentation", self.test_api_documentation)
        ]
        
        test_results = {}
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                test_results[test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                test_results[test_name] = {"success": False, "error": str(e)}
        
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n📊 TEST SUMMARY")
        print("=" * 60)
        
        successful_tests = sum(1 for result in test_results.values() 
                             if result.get("success", False))
        total_tests = len(test_results)
        
        print(f"Tests completed: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {(successful_tests / total_tests) * 100:.1f}%")
        print(f"Total time: {total_time:.2f}s")
        
        # Individual test results
        print(f"\n📋 Individual Results:")
        for test_name, result in test_results.items():
            status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
            print(f"   {status} {test_name}")
            
            if not result.get("success", False) and "error" in result:
                print(f"      Error: {result['error']}")
        
        print(f"\n🎉 API testing completed!")
        
        if successful_tests == total_tests:
            print("🌟 All tests passed! The API is working correctly.")
        else:
            print(f"⚠️ {total_tests - successful_tests} test(s) failed. Check the results above.")
        
        await self.client.aclose()
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": (successful_tests / total_tests) * 100,
                "total_time": total_time
            },
            "test_results": test_results,
            "individual_results": self.results
        }


def start_api_server():
    """Start the API server for testing."""
    uvicorn.run(
        "src.api.app:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False  # Reduce noise during testing
    )


async def main():
    """Run the API tests."""
    print("🚀 Starting Python Code Helper API Tests")
    print("=" * 60)
    
    # Start API server in background
    print("🔧 Starting API server...")
    server_process = Process(target=start_api_server)
    server_process.start()
    
    # Wait for server to start up
    await asyncio.sleep(3)
    
    try:
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        
        async with httpx.AsyncClient() as client:
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = await client.get("http://localhost:8000/health")
                    if response.status_code == 200:
                        print("✅ Server is ready!")
                        break
                except:
                    if i == max_retries - 1:
                        print("❌ Server failed to start")
                        return
                    await asyncio.sleep(1)
        
        # Run tests
        tester = APITester()
        await tester.run_all_tests()
        
    finally:
        # Clean up
        print("\n🧹 Cleaning up...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()
        print("✅ Cleanup completed")


if __name__ == "__main__":
    asyncio.run(main()) 