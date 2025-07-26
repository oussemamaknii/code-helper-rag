"""
Response generation with source attribution and reasoning.

This module provides contextual response generation with source grounding,
chain-of-thought reasoning, and different response types for various use cases.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re

from src.generation.llm_providers import BaseLLMProvider, GenerationRequest, GenerationResponse
from src.generation.prompt_engineering import PromptTemplate, PromptContext, PromptBuilder
from src.generation.context_processor import ContextProcessor
from src.vector.similarity_search import RetrievalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseType(Enum):
    """Types of responses that can be generated."""
    DIRECT_ANSWER = "direct_answer"
    CODE_EXPLANATION = "code_explanation"
    CODE_GENERATION = "code_generation"
    DEBUGGING_HELP = "debugging_help"
    TUTORIAL = "tutorial"
    COMPARISON = "comparison"
    BEST_PRACTICES = "best_practices"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class SourceAttribution:
    """Attribution information for sources used in response."""
    chunk_id: str
    content_snippet: str
    source_type: str
    relevance_score: float
    confidence: float = 0.0
    usage_type: str = "reference"  # reference, example, inspiration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'chunk_id': self.chunk_id,
            'content_snippet': self.content_snippet,
            'source_type': self.source_type,
            'relevance_score': self.relevance_score,
            'confidence': self.confidence,
            'usage_type': self.usage_type
        }


@dataclass
class GeneratedResponse:
    """Complete generated response with metadata and attributions."""
    content: str
    response_type: ResponseType  
    sources: List[SourceAttribution] = field(default_factory=list)
    confidence_score: float = 0.0
    reasoning_steps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_sources(self) -> bool:
        """Check if response has source attributions."""
        return len(self.sources) > 0
    
    @property
    def primary_sources(self) -> List[SourceAttribution]:
        """Get primary sources (highest relevance)."""
        return sorted(self.sources, key=lambda x: x.relevance_score, reverse=True)[:3]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'content': self.content,
            'response_type': self.response_type.value,
            'sources': [source.to_dict() for source in self.sources],
            'confidence_score': self.confidence_score,
            'reasoning_steps': self.reasoning_steps,
            'metadata': self.metadata,
            'generation_metrics': self.generation_metrics,
            'created_at': self.created_at.isoformat()
        }


class ContextualResponseGenerator:
    """Generates contextual responses with source attribution."""
    
    def __init__(self,
                 llm_provider: BaseLLMProvider,
                 prompt_builder: PromptBuilder,
                 context_processor: ContextProcessor):
        """
        Initialize contextual response generator.
        
        Args:
            llm_provider: LLM provider for text generation
            prompt_builder: Prompt builder for creating prompts
            context_processor: Context processor for optimizing retrieved content
        """
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder
        self.context_processor = context_processor
        
        self.logger = get_logger(__name__, component="response_generator")
    
    async def generate_response(self,
                              query: str,
                              retrieved_chunks: List[RetrievalResult],
                              template: PromptTemplate,
                              context_metadata: Optional[Dict[str, Any]] = None) -> GeneratedResponse:
        """
        Generate a contextual response with source attribution.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            template: Prompt template to use
            context_metadata: Additional context metadata
            
        Returns:
            GeneratedResponse: Complete response with sources and metadata
        """
        self.logger.info(
            "Generating contextual response",
            query_length=len(query),
            chunks_count=len(retrieved_chunks),
            template=template.name
        )
        
        try:
            # Step 1: Process context for optimal LLM input
            processed_context = self.context_processor.process_context(
                retrieved_chunks, query
            )
            
            # Step 2: Build prompt context
            prompt_context = PromptContext(
                query=query,
                retrieved_chunks=retrieved_chunks,
                additional_context=context_metadata or {}
            )
            
            # Extract context information
            if context_metadata:
                prompt_context.programming_language = context_metadata.get('programming_language')
                prompt_context.difficulty_level = context_metadata.get('difficulty_level')  
                prompt_context.domain = context_metadata.get('domain')
                prompt_context.user_intent = context_metadata.get('user_intent')
            
            # Step 3: Build messages for LLM
            messages = self.prompt_builder.build_prompt(template, prompt_context)
            
            # Step 4: Generate response
            generation_request = GenerationRequest(messages=messages)
            llm_response = await self.llm_provider.generate(generation_request)
            
            # Step 5: Extract source attributions
            sources = self._extract_source_attributions(
                llm_response.text, retrieved_chunks
            )
            
            # Step 6: Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                llm_response, retrieved_chunks, sources
            )
            
            # Step 7: Create final response
            response = GeneratedResponse(
                content=llm_response.text,
                response_type=self._detect_response_type(template.prompt_type, llm_response.text),
                sources=sources,
                confidence_score=confidence_score,
                metadata={
                    'query': query,
                    'template_used': template.name,
                    'context_chunks': len(retrieved_chunks),
                    'processed_context_summary': processed_context.get('summary', {}),
                    **(context_metadata or {})
                },
                generation_metrics={
                    'prompt_tokens': llm_response.prompt_tokens,
                    'completion_tokens': llm_response.completion_tokens,
                    'total_tokens': llm_response.total_tokens,
                    'model': llm_response.model,
                    'provider': llm_response.provider
                }
            )
            
            self.logger.info(
                "Response generated successfully",
                response_length=len(response.content),
                sources_count=len(response.sources),
                confidence=response.confidence_score,
                total_tokens=llm_response.total_tokens
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            
            # Return error response
            error_response = GeneratedResponse(
                content=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                response_type=ResponseType.DIRECT_ANSWER,
                confidence_score=0.0,
                metadata={'error': str(e)}
            )
            
            return error_response
    
    def _extract_source_attributions(self,
                                   response_text: str,
                                   retrieved_chunks: List[RetrievalResult]) -> List[SourceAttribution]:
        """Extract source attributions from generated response."""
        sources = []
        
        # Simple approach: check for content overlap
        response_words = set(response_text.lower().split())
        
        for chunk in retrieved_chunks[:5]:  # Check top 5 chunks
            chunk_words = set(chunk.content.lower().split())
            
            # Calculate overlap
            overlap = len(response_words.intersection(chunk_words))
            overlap_ratio = overlap / len(response_words) if response_words else 0
            
            # If significant overlap, consider it a source
            if overlap_ratio > 0.1 or overlap > 10:
                # Extract snippet that appears in both
                snippet = self._extract_common_snippet(response_text, chunk.content)
                
                attribution = SourceAttribution(
                    chunk_id=chunk.id,
                    content_snippet=snippet,
                    source_type=chunk.source_type,
                    relevance_score=chunk.final_score,
                    confidence=min(overlap_ratio * 2, 1.0),  # Scale to 0-1
                    usage_type=self._determine_usage_type(chunk, response_text)
                )
                sources.append(attribution)
        
        # Sort by confidence and relevance
        sources.sort(key=lambda x: (x.confidence, x.relevance_score), reverse=True)
        
        return sources[:5]  # Return top 5 sources
    
    def _extract_common_snippet(self, response_text: str, chunk_content: str) -> str:
        """Extract common snippet between response and chunk."""
        # Simple approach: find longest common substring
        response_lines = response_text.split('\n')
        chunk_lines = chunk_content.split('\n')
        
        best_snippet = ""
        max_length = 0
        
        for resp_line in response_lines:
            for chunk_line in chunk_lines:
                # Check if lines are similar (allowing for some variation)
                if len(resp_line.strip()) > 20 and len(chunk_line.strip()) > 20:
                    resp_words = resp_line.lower().split()
                    chunk_words = chunk_line.lower().split()
                    
                    common_words = set(resp_words).intersection(set(chunk_words))
                    if len(common_words) > max_length:
                        max_length = len(common_words)
                        best_snippet = chunk_line.strip()[:200]  # Limit snippet length
        
        return best_snippet if best_snippet else chunk_content[:150] + "..."
    
    def _determine_usage_type(self, chunk: RetrievalResult, response_text: str) -> str:
        """Determine how the chunk was used in the response."""
        response_lower = response_text.lower()
        
        # Check for code patterns
        if chunk.chunk_type in ['function', 'class'] and '```' in response_text:
            return "example"
        
        # Check for explanatory usage
        elif any(phrase in response_lower for phrase in ['as shown', 'for example', 'similar to']):
            return "reference"
        
        # Check for inspirational usage
        elif chunk.final_score > 0.8:
            return "inspiration"
        
        else:
            return "reference"
    
    def _calculate_confidence_score(self,
                                  llm_response: GenerationResponse,
                                  retrieved_chunks: List[RetrievalResult],
                                  sources: List[SourceAttribution]) -> float:
        """Calculate confidence score for the generated response."""
        # Base confidence from having sources
        base_confidence = 0.3 if sources else 0.1
        
        # Boost for high-relevance sources
        source_boost = 0.0
        if sources:
            avg_source_relevance = sum(s.relevance_score for s in sources) / len(sources)
            source_boost = min(avg_source_relevance * 0.4, 0.4)
        
        # Boost for response length and structure
        response_quality_boost = 0.0
        if len(llm_response.text) > 100:
            response_quality_boost += 0.1
        if '```' in llm_response.text:  # Has code examples
            response_quality_boost += 0.1
        if len(llm_response.text.split('\n')) > 5:  # Well structured
            response_quality_boost += 0.1
        
        # Penalty for errors or uncertainty indicators
        uncertainty_penalty = 0.0
        uncertainty_phrases = ['not sure', 'might be', 'possibly', 'i think', 'maybe']
        if any(phrase in llm_response.text.lower() for phrase in uncertainty_phrases):
            uncertainty_penalty = 0.2
        
        # Final confidence score
        confidence = min(
            base_confidence + source_boost + response_quality_boost - uncertainty_penalty,
            1.0
        )
        
        return max(confidence, 0.0)  # Ensure non-negative
    
    def _detect_response_type(self, prompt_type, response_text: str) -> ResponseType:
        """Detect the actual response type from prompt type and content."""
        response_lower = response_text.lower()
        
        # Check for code generation
        if '```' in response_text and any(keyword in response_lower for keyword in ['def ', 'class ', 'function']):
            return ResponseType.CODE_GENERATION
        
        # Check for debugging help
        elif any(keyword in response_lower for keyword in ['error', 'fix', 'debug', 'issue', 'problem']):
            return ResponseType.DEBUGGING_HELP
        
        # Check for explanations
        elif any(keyword in response_lower for keyword in ['works by', 'explanation', 'step by step']):
            return ResponseType.CODE_EXPLANATION
        
        # Check for tutorials
        elif any(keyword in response_lower for keyword in ['first', 'then', 'next', 'finally']) and len(response_text) > 500:
            return ResponseType.TUTORIAL
        
        # Check for comparisons
        elif any(keyword in response_lower for keyword in ['difference', 'vs', 'compared', 'better']):
            return ResponseType.COMPARISON
        
        # Check for best practices
        elif any(keyword in response_lower for keyword in ['best practice', 'recommended', 'should', 'avoid']):
            return ResponseType.BEST_PRACTICES
        
        # Default to direct answer
        else:
            return ResponseType.DIRECT_ANSWER


class ChainOfThoughtGenerator:
    """Generates responses with explicit chain-of-thought reasoning."""
    
    def __init__(self, base_generator: ContextualResponseGenerator):
        """
        Initialize chain-of-thought generator.
        
        Args:
            base_generator: Base contextual response generator
        """
        self.base_generator = base_generator
        self.logger = get_logger(__name__, component="cot_generator")
    
    async def generate_with_reasoning(self,
                                    query: str,
                                    retrieved_chunks: List[RetrievalResult],
                                    template: PromptTemplate,
                                    context_metadata: Optional[Dict[str, Any]] = None) -> GeneratedResponse:
        """
        Generate response with explicit chain-of-thought reasoning.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            template: Prompt template to use
            context_metadata: Additional context metadata
            
        Returns:
            GeneratedResponse: Response with reasoning steps
        """
        self.logger.info(
            "Generating response with chain-of-thought reasoning",
            query=query[:100]
        )
        
        # Step 1: Generate reasoning prompt
        reasoning_template = self._create_reasoning_template(template)
        
        # Step 2: Generate response with reasoning
        response = await self.base_generator.generate_response(
            query, retrieved_chunks, reasoning_template, context_metadata
        )
        
        # Step 3: Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(response.content)
        response.reasoning_steps = reasoning_steps
        
        # Step 4: Clean up response content (remove reasoning markers)
        response.content = self._clean_response_content(response.content)
        
        self.logger.info(
            "Chain-of-thought response generated",
            reasoning_steps=len(reasoning_steps)
        )
        
        return response
    
    def _create_reasoning_template(self, base_template: PromptTemplate) -> PromptTemplate:
        """Create a reasoning-enhanced version of the template."""
        reasoning_prompt = base_template.template + """

REASONING APPROACH:
Before providing your final answer, think through the problem step by step:

1. **Understanding**: What is the user really asking for?
2. **Analysis**: What information from the context is most relevant?
3. **Solution**: How can I best address their need?
4. **Verification**: Does my answer make sense and use the provided context appropriately?

Structure your response as:
REASONING:
[Your step-by-step reasoning here]

ANSWER:
[Your final answer here]"""
        
        # Create new template with reasoning
        reasoning_template = PromptTemplate(
            name=f"{base_template.name} (Chain-of-Thought)",
            prompt_type=base_template.prompt_type,
            template=reasoning_prompt,
            description=f"{base_template.description} - Enhanced with chain-of-thought reasoning",
            few_shot_examples=base_template.few_shot_examples,
            required_context=base_template.required_context,
            metadata={**base_template.metadata, 'reasoning_enhanced': True}
        )
        
        return reasoning_template
    
    def _extract_reasoning_steps(self, response_content: str) -> List[str]:
        """Extract reasoning steps from response content."""
        reasoning_steps = []
        
        # Look for REASONING section
        reasoning_match = re.search(r'REASONING:\s*(.*?)\s*ANSWER:', response_content, re.DOTALL | re.IGNORECASE)
        
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            
            # Split into steps (look for numbered items or bullet points)
            steps = re.split(r'\n\d+\.|\n-|\n\*', reasoning_text)
            reasoning_steps = [step.strip() for step in steps if step.strip()]
        
        else:
            # Fallback: look for step indicators in the text
            lines = response_content.split('\n')
            for line in lines:
                if any(indicator in line.lower() for indicator in ['step ', 'first', 'then', 'next', 'finally']):
                    reasoning_steps.append(line.strip())
        
        return reasoning_steps
    
    def _clean_response_content(self, response_content: str) -> str:
        """Clean response content by removing reasoning markers."""
        # Remove REASONING section and keep only ANSWER
        answer_match = re.search(r'ANSWER:\s*(.*)', response_content, re.DOTALL | re.IGNORECASE)
        
        if answer_match:
            return answer_match.group(1).strip()
        else:
            # If no clear ANSWER section, return original content
            return response_content 