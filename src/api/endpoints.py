"""
API endpoints for the Python Code Helper system.

This module defines REST and streaming endpoints for chat, search,
and system management functionality.
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.models import (
    ChatRequest,
    ChatResponse,
    SearchRequest,
    SearchResponse,
    StreamingChatResponse,
    SystemHealthResponse,
    UserProfile,
    APIKeyInfo,
    APIKeyCreateRequest,
    ResponseMetrics,
    SourceAttribution,
    QueryType,
    ErrorResponse
)
from src.api.dependencies import (
    get_generation_pipeline,
    get_search_engine,
    get_current_user,
    get_rate_limiter
)
from src.api.auth import require_api_key
from src.generation.pipeline import GenerationPipeline
from src.vector.similarity_search import SimilaritySearchEngine
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Create the main API router
router = APIRouter()


# Chat Endpoints

@router.post("/chat", 
             response_model=ChatResponse,
             summary="Send a chat message",
             description="Send a message to the AI assistant and receive a comprehensive response with source attribution.")
async def chat(
    request: ChatRequest,
    pipeline: GenerationPipeline = Depends(get_generation_pipeline),
    current_user: Optional[UserProfile] = Depends(get_current_user),
    rate_limiter: Any = Depends(get_rate_limiter)
):
    """
    Chat endpoint for conversational interactions.
    
    This endpoint processes user messages and returns intelligent responses
    with source attribution, confidence scoring, and follow-up suggestions.
    """
    
    # Apply rate limiting
    await rate_limiter.check_rate_limit(current_user.user_id if current_user else "anonymous")
    
    # Generate conversation ID if not provided
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    
    start_time = datetime.utcnow()
    
    try:
        logger.info(
            "Processing chat request",
            user_id=current_user.user_id if current_user else "anonymous",
            conversation_id=conversation_id,
            message_length=len(request.message),
            use_cot=request.use_chain_of_thought
        )
        
        # Mock generation pipeline for now
        if not pipeline:
            # Simulate response generation
            await asyncio.sleep(0.1)  # Simulate processing time
            
            mock_response = ChatResponse(
                response=f"I understand you're asking about: '{request.message[:50]}...'. This is a mock response demonstrating the API structure. In the full implementation, I would analyze your question, retrieve relevant context from our knowledge base, and provide a comprehensive answer with source attribution.",
                conversation_id=conversation_id,
                query_type=QueryType.QA_RESPONSE,
                sources=[
                    SourceAttribution(
                        chunk_id="mock_chunk_1",
                        content_snippet="Example code snippet from knowledge base...",
                        source_type="github_code",
                        url="https://github.com/example/repo",
                        title="Example Repository",
                        relevance_score=0.85,
                        confidence=0.80,
                        usage_type="reference"
                    )
                ],
                suggestions=[
                    "Can you provide a code example?",
                    "What are the best practices for this?",
                    "How does this compare to other approaches?"
                ],
                metrics=ResponseMetrics(
                    generation_time=(datetime.utcnow() - start_time).total_seconds(),
                    tokens_used=150,
                    prompt_tokens=80,
                    completion_tokens=70,
                    retrieval_time=0.045,
                    chunks_retrieved=5,
                    confidence_score=0.82
                )
            )
            
            return mock_response
        
        # Use real generation pipeline
        context_metadata = {}
        if request.context:
            context_metadata = {
                'programming_language': request.context.programming_language,
                'difficulty_level': request.context.difficulty_level,
                'domain': request.context.domain,
                'include_examples': request.context.include_examples,
                'include_tests': request.context.include_tests
            }
        
        # Generate response
        generated_response = await pipeline.generate(
            query=request.message,
            context_metadata=context_metadata,
            use_chain_of_thought=request.use_chain_of_thought
        )
        
        # Convert to API response format
        api_response = ChatResponse(
            response=generated_response.content,
            conversation_id=conversation_id,
            query_type=QueryType(generated_response.response_type.value),
            sources=[
                SourceAttribution(
                    chunk_id=source.chunk_id,
                    content_snippet=source.content_snippet,
                    source_type=source.source_type,
                    relevance_score=source.relevance_score,
                    confidence=source.confidence,
                    usage_type=source.usage_type
                )
                for source in generated_response.sources
            ],
            reasoning_steps=generated_response.reasoning_steps,
            suggestions=_generate_follow_up_suggestions(request.message, generated_response),
            metrics=ResponseMetrics(
                generation_time=(datetime.utcnow() - start_time).total_seconds(),
                tokens_used=generated_response.generation_metrics.get('total_tokens', 0),
                prompt_tokens=generated_response.generation_metrics.get('prompt_tokens', 0),
                completion_tokens=generated_response.generation_metrics.get('completion_tokens', 0),
                retrieval_time=0.1,  # Would come from actual metrics
                chunks_retrieved=len(generated_response.sources),
                confidence_score=generated_response.confidence_score
            )
        )
        
        logger.info(
            "Chat request completed",
            conversation_id=conversation_id,
            response_length=len(api_response.response),
            sources_count=len(api_response.sources),
            confidence=api_response.metrics.confidence_score
        )
        
        return api_response
        
    except Exception as e:
        logger.error(f"Chat request failed: {e}", conversation_id=conversation_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )


@router.post("/chat/stream",
             summary="Stream chat response",
             description="Send a message and receive a streaming response in real-time.")
async def chat_stream(
    request: ChatRequest,
    pipeline: GenerationPipeline = Depends(get_generation_pipeline),
    current_user: Optional[UserProfile] = Depends(get_current_user),
    rate_limiter: Any = Depends(get_rate_limiter)
):
    """
    Streaming chat endpoint for real-time responses.
    
    Returns a Server-Sent Events (SSE) stream of response chunks as they're generated.
    """
    
    # Apply rate limiting
    await rate_limiter.check_rate_limit(current_user.user_id if current_user else "anonymous")
    
    conversation_id = request.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            logger.info(
                "Starting streaming chat",
                conversation_id=conversation_id,
                user_id=current_user.user_id if current_user else "anonymous"
            )
            
            if not pipeline:
                # Mock streaming response
                mock_response = "I understand you're asking about streaming responses. This is a demonstration of the streaming API endpoint. In the full implementation, I would process your request and stream the response in real-time as it's generated by the AI model."
                
                words = mock_response.split()
                for i, word in enumerate(words):
                    chunk = StreamingChatResponse(
                        content=word + (" " if i < len(words) - 1 else ""),
                        conversation_id=conversation_id,
                        is_complete=False,
                        chunk_index=i
                    )
                    
                    yield f"data: {chunk.json()}\n\n"
                    await asyncio.sleep(0.05)  # Simulate typing delay
                
                # Send final chunk with metadata
                final_chunk = StreamingChatResponse(
                    content="",
                    conversation_id=conversation_id,
                    is_complete=True,
                    chunk_index=len(words),
                    query_type=QueryType.QA_RESPONSE,
                    sources=[
                        SourceAttribution(
                            chunk_id="mock_streaming_chunk",
                            content_snippet="Mock source for streaming demo",
                            source_type="documentation",
                            relevance_score=0.8,
                            confidence=0.75,
                            usage_type="reference"
                        )
                    ],
                    metrics=ResponseMetrics(
                        generation_time=2.5,
                        tokens_used=120,
                        prompt_tokens=50,
                        completion_tokens=70,
                        retrieval_time=0.1,
                        chunks_retrieved=3,
                        confidence_score=0.78
                    )
                )
                
                yield f"data: {final_chunk.json()}\n\n"
                return
            
            # Use real streaming generation
            # This would integrate with the pipeline's streaming capabilities
            context_metadata = {}
            if request.context:
                context_metadata = {
                    'programming_language': request.context.programming_language,
                    'difficulty_level': request.context.difficulty_level,
                    'domain': request.context.domain
                }
            
            # For now, simulate streaming from a generated response
            generated_response = await pipeline.generate(
                query=request.message,
                context_metadata=context_metadata,
                use_chain_of_thought=request.use_chain_of_thought
            )
            
            # Stream the response word by word
            words = generated_response.content.split()
            for i, word in enumerate(words):
                chunk = StreamingChatResponse(
                    content=word + (" " if i < len(words) - 1 else ""),
                    conversation_id=conversation_id,
                    is_complete=False,
                    chunk_index=i
                )
                
                yield f"data: {chunk.json()}\n\n"
                await asyncio.sleep(0.02)  # Small delay for realistic streaming
            
            # Send completion chunk
            final_chunk = StreamingChatResponse(
                content="",
                conversation_id=conversation_id,
                is_complete=True,
                chunk_index=len(words),
                query_type=QueryType(generated_response.response_type.value),
                sources=[
                    SourceAttribution(
                        chunk_id=source.chunk_id,
                        content_snippet=source.content_snippet,
                        source_type=source.source_type,
                        relevance_score=source.relevance_score,
                        confidence=source.confidence,
                        usage_type=source.usage_type
                    )
                    for source in generated_response.sources
                ],
                metrics=ResponseMetrics(
                    generation_time=2.1,
                    tokens_used=generated_response.generation_metrics.get('total_tokens', 0),
                    prompt_tokens=generated_response.generation_metrics.get('prompt_tokens', 0),
                    completion_tokens=generated_response.generation_metrics.get('completion_tokens', 0),
                    retrieval_time=0.15,
                    chunks_retrieved=len(generated_response.sources),
                    confidence_score=generated_response.confidence_score
                )
            )
            
            yield f"data: {final_chunk.json()}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}", conversation_id=conversation_id)
            error_chunk = StreamingChatResponse(
                content=f"Error: {str(e)}",
                conversation_id=conversation_id,
                is_complete=True,
                chunk_index=0
            )
            yield f"data: {error_chunk.json()}\n\n"
    
    return EventSourceResponse(generate_stream())


# Search Endpoints

@router.post("/search",
             response_model=SearchResponse,
             summary="Search knowledge base",
             description="Search the knowledge base for relevant code examples, documentation, and Q&A content.")
async def search(
    request: SearchRequest,
    search_engine: SimilaritySearchEngine = Depends(get_search_engine),
    current_user: Optional[UserProfile] = Depends(get_current_user),
    rate_limiter: Any = Depends(get_rate_limiter)
):
    """
    Search endpoint for knowledge base queries.
    
    Performs hybrid search across code repositories, Stack Overflow Q&A,
    and documentation to find relevant content.
    """
    
    # Apply rate limiting
    await rate_limiter.check_rate_limit(current_user.user_id if current_user else "anonymous")
    
    start_time = datetime.utcnow()
    
    try:
        logger.info(
            "Processing search request",
            user_id=current_user.user_id if current_user else "anonymous",
            query=request.query,
            search_type=request.search_type,
            top_k=request.top_k
        )
        
        if not search_engine:
            # Mock search response
            mock_results = [
                {
                    "chunk_id": f"mock_result_{i}",
                    "content": f"Mock search result {i+1} for query: '{request.query}'. This demonstrates the search functionality structure.",
                    "title": f"Mock Result {i+1}",
                    "source_type": "github_code" if i % 2 == 0 else "stackoverflow_qa",
                    "url": f"https://example.com/source_{i+1}",
                    "metadata": {
                        "programming_language": "python",
                        "stars": 100 + i * 50,
                        "last_updated": "2025-01-26"
                    },
                    "relevance_score": 0.9 - (i * 0.1),
                    "highlights": [request.query[:20], "python", "algorithm"]
                }
                for i in range(min(request.top_k, 5))
            ]
            
            return SearchResponse(
                results=mock_results,
                total_results=25,
                search_time=(datetime.utcnow() - start_time).total_seconds(),
                query_interpretation=f"Interpreted as: {request.search_type} search for '{request.query}'",
                suggestions=[
                    f"{request.query} implementation",
                    f"{request.query} examples",
                    f"{request.query} best practices"
                ]
            )
        
        # Use real search engine
        from src.vector.similarity_search import SearchRequest as VectorSearchRequest, SearchType
        
        # Map API search type to vector search type
        search_type_mapping = {
            "semantic": SearchType.SEMANTIC,
            "keyword": SearchType.KEYWORD,
            "hybrid": SearchType.HYBRID,
            "code": SearchType.CODE_SEARCH
        }
        
        vector_search_request = VectorSearchRequest(
            query=request.query,
            top_k=request.top_k,
            search_type=search_type_mapping.get(request.search_type, SearchType.HYBRID),
            filters=request.filters or {}
        )
        
        search_results = await search_engine.search(vector_search_request)
        
        # Convert to API response format
        api_results = []
        for result in search_results:
            api_results.append({
                "chunk_id": result.id,
                "content": result.content,
                "title": result.metadata.get("title"),
                "source_type": result.source_type,
                "url": result.metadata.get("url"),
                "metadata": result.metadata,
                "relevance_score": result.final_score,
                "highlights": _extract_highlights(request.query, result.content)
            })
        
        response = SearchResponse(
            results=api_results,
            total_results=len(search_results),
            search_time=(datetime.utcnow() - start_time).total_seconds(),
            query_interpretation=f"Processed as {request.search_type} search",
            suggestions=_generate_search_suggestions(request.query)
        )
        
        logger.info(
            "Search request completed",
            results_count=len(response.results),
            search_time=response.search_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Search request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search request failed: {str(e)}"
        )


# System Management Endpoints

@router.get("/health",
            response_model=SystemHealthResponse,
            summary="System health check",
            description="Get comprehensive system health status and metrics.")
async def health_check(
    pipeline: GenerationPipeline = Depends(get_generation_pipeline),
    search_engine: SimilaritySearchEngine = Depends(get_search_engine)
):
    """
    System health check endpoint.
    
    Returns detailed health information for all system components
    including generation pipeline, search engine, and overall metrics.
    """
    
    try:
        components = {}
        
        # Check generation pipeline
        if pipeline:
            pipeline_health = await pipeline.get_health_status()
            components["generation_pipeline"] = pipeline_health
        else:
            components["generation_pipeline"] = {
                "status": "initializing",
                "message": "Pipeline not yet initialized"
            }
        
        # Check search engine
        if search_engine:
            search_health = await search_engine.health_check()
            components["vector_search"] = search_health
        else:
            components["vector_search"] = {
                "status": "initializing",
                "message": "Search engine not yet initialized"
            }
        
        # Overall system status
        all_healthy = all(
            comp.get("status") == "healthy" 
            for comp in components.values()
        )
        
        overall_status = "healthy" if all_healthy else "degraded"
        
        # Get system metrics
        metrics = {}
        if pipeline:
            pipeline_metrics = await pipeline.get_metrics()
            metrics.update(pipeline_metrics.get("metrics", {}))
        
        return SystemHealthResponse(
            status=overall_status,
            uptime=3600.0,  # Would be calculated from actual startup time
            components=components,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealthResponse(
            status="unhealthy",
            uptime=0.0,
            components={"error": {"status": "error", "message": str(e)}},
            metrics={}
        )


# Helper Functions

def _generate_follow_up_suggestions(original_query: str, response: Any) -> List[str]:
    """Generate follow-up question suggestions based on the query and response."""
    suggestions = []
    
    query_lower = original_query.lower()
    
    if "how" in query_lower:
        suggestions.extend([
            "Can you show me a code example?",
            "What are the pros and cons?",
            "Are there any alternatives?"
        ])
    elif "what" in query_lower:
        suggestions.extend([
            "How do I implement this?",
            "When should I use this?",
            "Can you provide more details?"
        ])
    elif "implement" in query_lower or "write" in query_lower:
        suggestions.extend([
            "How can I test this code?",
            "What are the edge cases?",
            "How can I optimize this?"
        ])
    else:
        suggestions.extend([
            "Can you elaborate on this?",
            "Show me an example",
            "What are best practices?"
        ])
    
    return suggestions[:3]


def _generate_search_suggestions(query: str) -> List[str]:
    """Generate search query suggestions."""
    suggestions = [
        f"{query} implementation",
        f"{query} examples",
        f"{query} best practices",
        f"{query} tutorial"
    ]
    return suggestions[:3]


def _extract_highlights(query: str, content: str) -> List[str]:
    """Extract highlighted text snippets from content."""
    query_words = query.lower().split()
    content_words = content.lower().split()
    
    highlights = []
    for word in query_words:
        if word in content_words:
            highlights.append(word)
    
    # Add some common programming terms if found
    programming_terms = ["python", "function", "class", "algorithm", "code", "example"]
    for term in programming_terms:
        if term in content.lower() and term not in highlights:
            highlights.append(term)
    
    return highlights[:5] 