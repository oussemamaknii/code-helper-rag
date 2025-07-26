"""
Advanced prompt engineering for RAG responses.

This module provides template-based prompt engineering with few-shot examples,
context integration, and specialized prompts for different query types.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from jinja2 import Template, Environment, BaseLoader
import json

from src.vector.similarity_search import RetrievalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of prompts for different use cases."""
    CODE_EXPLANATION = "code_explanation"
    CODE_GENERATION = "code_generation"
    DEBUGGING_HELP = "debugging_help"
    QA_RESPONSE = "qa_response"
    CONCEPT_EXPLANATION = "concept_explanation"
    BEST_PRACTICES = "best_practices"
    CODE_REVIEW = "code_review"
    TUTORIAL = "tutorial"
    COMPARISON = "comparison"
    TROUBLESHOOTING = "troubleshooting"


@dataclass
class FewShotExample:
    """Few-shot example for prompt engineering."""
    input: str
    output: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'input': self.input,
            'output': self.output,
            'context': self.context,
            'metadata': self.metadata
        }


@dataclass
class PromptContext:
    """Context information for prompt building."""
    query: str
    retrieved_chunks: List[RetrievalResult]
    user_intent: Optional[str] = None
    programming_language: Optional[str] = None
    difficulty_level: Optional[str] = None
    domain: Optional[str] = None
    previous_conversation: List[Dict[str, str]] = field(default_factory=list)
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptTemplate:
    """Template for generating prompts."""
    name: str
    prompt_type: PromptType
    template: str
    description: str
    few_shot_examples: List[FewShotExample] = field(default_factory=list)
    required_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def render(self, context: Dict[str, Any]) -> str:
        """
        Render the template with provided context.
        
        Args:
            context: Context variables for template rendering
            
        Returns:
            str: Rendered prompt
        """
        try:
            template = Template(self.template)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render template {self.name}: {e}")
            return self.template  # Return raw template as fallback


class PromptBuilder:
    """Builder for creating context-aware prompts."""
    
    def __init__(self):
        """Initialize prompt builder."""
        self.logger = get_logger(__name__, component="prompt_builder")
    
    def build_prompt(self, 
                    template: PromptTemplate,
                    context: PromptContext) -> List[Dict[str, str]]:
        """
        Build a complete prompt with system message, examples, and user query.
        
        Args:
            template: Prompt template to use
            context: Context information for prompt building
            
        Returns:
            List[Dict[str, str]]: Messages for LLM API
        """
        messages = []
        
        # Build context variables for template rendering
        template_context = self._build_template_context(context)
        
        # Render system prompt
        system_prompt = template.render(template_context)
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add few-shot examples if available
        if template.few_shot_examples:
            examples_to_use = self._select_examples(template.few_shot_examples, context)
            for example in examples_to_use:
                messages.append({
                    "role": "user",
                    "content": example.input
                })
                messages.append({
                    "role": "assistant", 
                    "content": example.output
                })
        
        # Add conversation history if available
        if context.previous_conversation:
            messages.extend(context.previous_conversation[-6:])  # Last 3 exchanges
        
        # Add current user query
        messages.append({
            "role": "user",
            "content": context.query
        })
        
        self.logger.debug(
            "Prompt built successfully",
            template=template.name,
            messages_count=len(messages),
            has_examples=len(template.few_shot_examples) > 0
        )
        
        return messages
    
    def _build_template_context(self, context: PromptContext) -> Dict[str, Any]:
        """Build context variables for template rendering."""
        # Process retrieved chunks
        code_chunks = []
        qa_chunks = []
        
        for chunk in context.retrieved_chunks[:5]:  # Top 5 most relevant
            chunk_info = {
                'content': chunk.content,
                'score': chunk.final_score,
                'source': chunk.source_type,
                'type': chunk.chunk_type
            }
            
            if chunk.chunk_type in ['function', 'class', 'module']:
                code_chunks.append(chunk_info)
            else:
                qa_chunks.append(chunk_info)
        
        # Build comprehensive context
        template_context = {
            'query': context.query,
            'code_chunks': code_chunks,
            'qa_chunks': qa_chunks,
            'programming_language': context.programming_language or 'Python',
            'difficulty_level': context.difficulty_level or 'intermediate',
            'domain': context.domain or 'general',
            'user_intent': context.user_intent or 'help',
            'total_chunks': len(context.retrieved_chunks),
            'has_code_examples': len(code_chunks) > 0,
            'has_qa_examples': len(qa_chunks) > 0,
            **context.additional_context
        }
        
        return template_context
    
    def _select_examples(self, 
                        examples: List[FewShotExample], 
                        context: PromptContext) -> List[FewShotExample]:
        """Select most relevant few-shot examples."""
        if len(examples) <= 2:
            return examples
        
        # Simple relevance scoring based on context similarity
        scored_examples = []
        query_words = set(context.query.lower().split())
        
        for example in examples:
            # Score based on word overlap with query
            example_words = set(example.input.lower().split())
            overlap_score = len(query_words.intersection(example_words)) / len(query_words)
            
            # Boost score if programming language matches
            if (context.programming_language and 
                context.programming_language.lower() in example.input.lower()):
                overlap_score += 0.3
            
            scored_examples.append((overlap_score, example))
        
        # Sort by score and take top 2
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        return [example for _, example in scored_examples[:2]]


class PromptTemplateLibrary:
    """Library of pre-built prompt templates."""
    
    def __init__(self):
        """Initialize template library."""
        self.templates: Dict[PromptType, PromptTemplate] = {}
        self._initialize_templates()
        self.logger = get_logger(__name__, component="template_library")
    
    def get_template(self, prompt_type: PromptType) -> Optional[PromptTemplate]:
        """Get template by type."""
        return self.templates.get(prompt_type)
    
    def add_template(self, template: PromptTemplate) -> None:
        """Add custom template to library."""
        self.templates[template.prompt_type] = template
        self.logger.info(f"Added template: {template.name}")
    
    def _initialize_templates(self) -> None:
        """Initialize built-in templates."""
        
        # Code Explanation Template
        code_explanation_template = PromptTemplate(
            name="Code Explanation",
            prompt_type=PromptType.CODE_EXPLANATION,
            description="Explains how code works with context from retrieved examples",
            template="""You are an expert Python developer helping users understand code. 

Your task is to explain code clearly and comprehensively, using relevant examples from the codebase.

CONTEXT INFORMATION:
{% if code_chunks %}
Relevant code examples:
{% for chunk in code_chunks %}
- {{ chunk.type|title }}: {{ chunk.content[:200] }}... (Score: {{ "%.2f"|format(chunk.score) }})
{% endfor %}
{% endif %}

{% if qa_chunks %}
Related Q&A:
{% for chunk in qa_chunks %}
- {{ chunk.content[:150] }}... (Score: {{ "%.2f"|format(chunk.score) }})
{% endfor %}
{% endif %}

GUIDELINES:
1. Provide clear, step-by-step explanations
2. Use the retrieved examples to support your explanation
3. Explain not just WHAT the code does, but WHY and HOW
4. Include relevant concepts and best practices
5. Tailor explanation to {{ difficulty_level }} level
6. Focus on {{ programming_language }} specifics when relevant

Always cite your sources when referencing the provided examples.""",
            few_shot_examples=[
                FewShotExample(
                    input="Can you explain how this quicksort function works?",
                    output="""This quicksort implementation uses the divide-and-conquer strategy. Here's how it works:

1. **Base Case**: If the array has 1 or fewer elements, it's already sorted, so return it
2. **Pivot Selection**: Choose the middle element as the pivot
3. **Partitioning**: Split the array into three parts:
   - Elements less than pivot (left)
   - Elements equal to pivot (middle) 
   - Elements greater than pivot (right)
4. **Recursive Sorting**: Recursively apply quicksort to left and right partitions
5. **Combine**: Concatenate sorted left + middle + sorted right

The key insight is that after partitioning, the pivot is in its final sorted position, and we only need to sort the sub-arrays."""
                ),
                FewShotExample(
                    input="What does this list comprehension do: [x**2 for x in range(10) if x % 2 == 0]?",
                    output="""This list comprehension creates a new list containing the squares of even numbers from 0 to 9.

Breaking it down:
- `range(10)` generates numbers 0, 1, 2, ..., 9
- `if x % 2 == 0` filters for even numbers (0, 2, 4, 6, 8)
- `x**2` squares each even number
- Result: [0, 4, 16, 36, 64]

This is equivalent to:
```python
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x**2)
```

List comprehensions are more concise and often faster than traditional loops for creating lists."""
                )
            ]
        )
        
        # Code Generation Template
        code_generation_template = PromptTemplate(
            name="Code Generation",
            prompt_type=PromptType.CODE_GENERATION,
            description="Generates code based on requirements with examples",
            template="""You are an expert {{ programming_language }} developer. Generate clean, efficient, and well-documented code based on the user's requirements.

CONTEXT INFORMATION:
{% if code_chunks %}
Relevant code patterns:
{% for chunk in code_chunks %}
- {{ chunk.content[:200] }}...
{% endfor %}
{% endif %}

REQUIREMENTS:
- Write clean, readable code following {{ programming_language }} best practices
- Include docstrings and comments for clarity
- Handle edge cases and errors appropriately
- Use efficient algorithms and data structures
- Target {{ difficulty_level }} level implementation

GUIDELINES:
1. Start with a clear function/class signature
2. Add comprehensive docstring with parameters, returns, and examples
3. Implement the logic step by step
4. Include error handling where appropriate
5. Add usage examples if helpful
6. Follow PEP 8 style guidelines for Python

Use the provided code patterns as inspiration for structure and style.""",
            few_shot_examples=[
                FewShotExample(
                    input="Create a function to find the longest common subsequence of two strings.",
                    output="""```python
def longest_common_subsequence(str1: str, str2: str) -> str:
    '''
    Find the longest common subsequence of two strings using dynamic programming.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        str: The longest common subsequence
        
    Example:
        >>> longest_common_subsequence("ABCDGH", "AEDFHR")
        'ADH'
    '''
    m, n = len(str1), len(str2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return ''.join(reversed(lcs))
```

This implementation uses dynamic programming with O(m*n) time complexity and space complexity."""
                )
            ]
        )
        
        # Q&A Response Template
        qa_response_template = PromptTemplate(
            name="Q&A Response",
            prompt_type=PromptType.QA_RESPONSE,
            description="Answers questions using retrieved Q&A context",
            template="""You are a helpful {{ programming_language }} expert answering programming questions.

CONTEXT FROM COMMUNITY Q&A:
{% if qa_chunks %}
{% for chunk in qa_chunks %}
Relevant Q&A {{ loop.index }}:
{{ chunk.content[:300] }}...
---
{% endfor %}
{% endif %}

{% if code_chunks %}
Relevant Code Examples:
{% for chunk in code_chunks %}
{{ chunk.content[:200] }}...
---
{% endfor %}
{% endif %}

GUIDELINES:
1. Provide accurate, helpful answers based on the context
2. Include practical code examples when relevant
3. Explain the reasoning behind your recommendations
4. Mention common pitfalls or alternatives
5. Adapt complexity to {{ difficulty_level }} level
6. Reference the provided Q&A context when applicable

Answer the user's question comprehensively while being concise.""",
            few_shot_examples=[
                FewShotExample(
                    input="What's the difference between == and is in Python?",
                    output="""The difference between `==` and `is` in Python:

**`==` (Equality operator):**
- Compares the **values** of objects
- Calls the `__eq__()` method
- Returns `True` if objects have the same value

**`is` (Identity operator):**
- Compares **object identity** (memory location)
- Returns `True` if both variables point to the same object in memory

**Examples:**
```python
# Values are equal, but different objects
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)  # True (same values)
print(a is b)  # False (different objects)

# Same object
c = a
print(a is c)  # True (same object)

# Special case: small integers and strings are cached
x = 5
y = 5
print(x is y)  # True (Python caches small integers)
```

**When to use:**
- Use `==` for value comparison (most common)
- Use `is` for identity checks, especially with `None`: `if x is None:`"""
                )
            ]
        )
        
        # Debugging Help Template
        debugging_template = PromptTemplate(
            name="Debugging Help",
            prompt_type=PromptType.DEBUGGING_HELP,
            description="Helps debug code issues with context",
            template="""You are an expert debugger helping to identify and fix code issues.

CONTEXT INFORMATION:
{% if code_chunks %}
Similar code patterns:
{% for chunk in code_chunks %}
{{ chunk.content[:250] }}...
---
{% endfor %}
{% endif %}

DEBUGGING APPROACH:
1. Analyze the error message and code structure
2. Identify the root cause of the issue
3. Provide a clear explanation of what's wrong
4. Suggest specific fixes with code examples
5. Explain how to prevent similar issues
6. Recommend debugging techniques

Focus on {{ programming_language }} specific debugging practices and common pitfalls."""
        )
        
        # Store templates
        self.templates[PromptType.CODE_EXPLANATION] = code_explanation_template
        self.templates[PromptType.CODE_GENERATION] = code_generation_template
        self.templates[PromptType.QA_RESPONSE] = qa_response_template
        self.templates[PromptType.DEBUGGING_HELP] = debugging_template
        
        self.logger.info(f"Initialized {len(self.templates)} prompt templates")
    
    def detect_prompt_type(self, query: str) -> PromptType:
        """
        Automatically detect the most appropriate prompt type for a query.
        
        Args:
            query: User query to analyze
            
        Returns:
            PromptType: Most appropriate prompt type
        """
        query_lower = query.lower()
        
        # Code generation indicators
        if any(keyword in query_lower for keyword in [
            'write', 'create', 'implement', 'build', 'generate', 'make a function',
            'code for', 'algorithm for', 'script to'
        ]):
            return PromptType.CODE_GENERATION
        
        # Debugging indicators
        elif any(keyword in query_lower for keyword in [
            'error', 'bug', 'debug', 'fix', 'broken', 'not working', 'traceback',
            'exception', 'fails', 'wrong output'
        ]):
            return PromptType.DEBUGGING_HELP
        
        # Code explanation indicators
        elif any(keyword in query_lower for keyword in [
            'explain', 'how does', 'what does', 'understand', 'works', 'meaning',
            'breakdown', 'walk through'
        ]):
            return PromptType.CODE_EXPLANATION
        
        # Best practices indicators
        elif any(keyword in query_lower for keyword in [
            'best practice', 'recommended', 'should i', 'better way',
            'optimize', 'improve', 'convention'
        ]):
            return PromptType.BEST_PRACTICES
        
        # Comparison indicators
        elif any(keyword in query_lower for keyword in [
            'difference', 'vs', 'versus', 'compare', 'better', 'choose between'
        ]):
            return PromptType.COMPARISON
        
        # Default to Q&A response
        else:
            return PromptType.QA_RESPONSE 