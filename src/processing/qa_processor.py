"""
Q&A processor for Stack Overflow content.

This module implements a specialized processor for Stack Overflow Q&A pairs,
with intelligent chunking and metadata extraction.
"""

from typing import AsyncGenerator, List, Dict, Any
import re

from src.processing.base_processor import BaseProcessor, ProcessedChunk
from src.processing.chunking_strategies import (
    ChunkingStrategyFactory, ChunkingConfig, ChunkingStrategy
)
from src.ingestion.base_collector import CollectedItem
from src.utils.logger import get_logger
from src.utils.text_utils import extract_code_blocks, remove_html_tags, clean_text
from src.config.settings import settings

logger = get_logger(__name__)


class QAProcessor(BaseProcessor):
    """
    Specialized processor for Stack Overflow Q&A content.
    
    Features:
    - Intelligent Q&A pair chunking
    - Code block extraction and analysis
    - Tag-based categorization
    - Quality scoring
    - Answer relevance assessment
    """
    
    def __init__(self, **kwargs):
        """Initialize Q&A processor."""
        super().__init__(name="QAProcessor", **kwargs)
        
        # Configure chunking strategy for Q&A
        self.qa_config = ChunkingConfig(
            strategy=ChunkingStrategy.QA_PAIR,
            max_chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap // 2,  # Less overlap for Q&A
            min_chunk_size=50,  # Allow shorter chunks for Q&A
            preserve_structure=True,
            include_context=True
        )
    
    async def get_supported_types(self) -> List[str]:
        """Get supported source types."""
        return ["stackoverflow_qa"]
    
    async def process_item(self, item: CollectedItem) -> AsyncGenerator[ProcessedChunk, None]:
        """
        Process a Stack Overflow Q&A item into chunks.
        
        Args:
            item: Collected Stack Overflow Q&A item
            
        Yields:
            ProcessedChunk: Q&A chunks with enhanced metadata
        """
        self.logger.debug(
            "Processing Stack Overflow Q&A item",
            item_id=item.id,
            question_id=item.metadata.get('question_id'),
            answer_id=item.metadata.get('answer_id'),
            content_length=len(item.content)
        )
        
        # Extract and analyze code blocks
        code_blocks = extract_code_blocks(item.content)
        
        # Enhanced metadata with Q&A analysis
        enhanced_metadata = {
            **item.metadata,
            'source_item_id': item.id,
            'content_analysis': await self._analyze_qa_content(item.content),
            'code_blocks': [
                {
                    'language': block.language,
                    'line_count': len(block.content.splitlines()),
                    'char_count': len(block.content)
                }
                for block in code_blocks
            ],
            'quality_indicators': self._assess_quality(item),
            'topic_categories': self._categorize_content(item.metadata.get('tags', [])),
            'answer_characteristics': self._analyze_answer_characteristics(item.content, item.metadata)
        }
        
        # Create chunking strategy
        strategy = ChunkingStrategyFactory.create_strategy(self.qa_config)
        
        # Generate chunks
        chunks = strategy.chunk_content(item.content, enhanced_metadata)
        
        for chunk in chunks:
            # Additional processing for Q&A chunks
            await self._enrich_qa_chunk(chunk, code_blocks)
            yield chunk
    
    async def _analyze_qa_content(self, content: str) -> Dict[str, Any]:
        """Analyze Q&A content structure and characteristics."""
        
        lines = content.splitlines()
        analysis = {
            'total_lines': len(lines),
            'question_lines': 0,
            'answer_lines': 0,
            'code_lines': 0,
            'has_code_examples': False,
            'has_error_messages': False,
            'has_links': False,
            'readability_score': 0.0
        }
        
        current_section = None
        in_code_block = False
        
        for line in lines:
            line_clean = line.strip()
            
            # Detect sections
            if line.startswith("Question:"):
                current_section = "question"
                continue
            elif line.startswith("Answer:"):
                current_section = "answer"
                continue
            
            # Count lines by section
            if current_section == "question":
                analysis['question_lines'] += 1
            elif current_section == "answer":
                analysis['answer_lines'] += 1
            
            # Detect code blocks
            if line_clean.startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    analysis['has_code_examples'] = True
                continue
            
            if in_code_block or line.startswith('    '):
                analysis['code_lines'] += 1
            
            # Detect other patterns
            if re.search(r'error|exception|traceback|failed', line_clean, re.IGNORECASE):
                analysis['has_error_messages'] = True
            
            if re.search(r'https?://|www\.', line_clean):
                analysis['has_links'] = True
        
        # Calculate readability score (simple heuristic)
        if analysis['total_lines'] > 0:
            code_ratio = analysis['code_lines'] / analysis['total_lines']
            has_structure = analysis['question_lines'] > 0 and analysis['answer_lines'] > 0
            has_examples = analysis['has_code_examples']
            
            score = 50.0  # Base score
            if has_structure:
                score += 20.0
            if has_examples:
                score += 15.0
            if 0.1 <= code_ratio <= 0.5:  # Good balance of code and explanation
                score += 15.0
            
            analysis['readability_score'] = min(100.0, score)
        
        return analysis
    
    def _assess_quality(self, item: CollectedItem) -> Dict[str, Any]:
        """Assess quality indicators for Q&A content."""
        
        quality = {
            'score_based': {
                'question_score': item.metadata.get('question_score', 0),
                'answer_score': item.metadata.get('answer_score', 0),
                'total_score': item.metadata.get('question_score', 0) + item.metadata.get('answer_score', 0),
                'is_accepted': item.metadata.get('is_accepted', False)
            },
            'content_based': {
                'has_code': item.metadata.get('has_code', False),
                'content_length': len(item.content),
                'question_length': item.metadata.get('question_length', 0),
                'answer_length': item.metadata.get('answer_length', 0)
            },
            'engagement': {
                'view_count': item.metadata.get('view_count', 0),
                'python_related': item.metadata.get('python_related', False)
            }
        }
        
        # Calculate overall quality score
        score = 0.0
        
        # Score-based factors (40% weight)
        if quality['score_based']['is_accepted']:
            score += 20.0
        if quality['score_based']['total_score'] > 10:
            score += 15.0
        elif quality['score_based']['total_score'] > 5:
            score += 10.0
        elif quality['score_based']['total_score'] > 0:
            score += 5.0
        
        # Content-based factors (40% weight)
        if quality['content_based']['has_code']:
            score += 15.0
        if quality['content_based']['answer_length'] > 200:
            score += 15.0
        elif quality['content_based']['answer_length'] > 100:
            score += 10.0
        
        # Engagement factors (20% weight)
        if quality['engagement']['view_count'] > 1000:
            score += 10.0
        elif quality['engagement']['view_count'] > 100:
            score += 5.0
        
        if quality['engagement']['python_related']:
            score += 10.0
        
        quality['overall_score'] = min(100.0, score)
        
        return quality
    
    def _categorize_content(self, tags: List[str]) -> Dict[str, List[str]]:
        """Categorize content based on tags."""
        
        categories = {
            'core_python': [],
            'web_frameworks': [],
            'data_science': [],
            'databases': [],
            'testing': [],
            'async_programming': [],
            'gui': [],
            'tools': [],
            'other': []
        }
        
        # Tag categorization mapping
        tag_mapping = {
            'core_python': ['python', 'python-3.x', 'python-2.7', 'list', 'dictionary', 'string', 'loops', 'functions', 'classes', 'oop'],
            'web_frameworks': ['django', 'flask', 'fastapi', 'tornado', 'pyramid', 'bottle', 'web', 'rest', 'api'],
            'data_science': ['pandas', 'numpy', 'matplotlib', 'scipy', 'scikit-learn', 'jupyter', 'data-analysis', 'machine-learning'],
            'databases': ['sql', 'sqlite', 'postgresql', 'mysql', 'mongodb', 'orm', 'sqlalchemy', 'database'],
            'testing': ['testing', 'unittest', 'pytest', 'mock', 'debugging'],
            'async_programming': ['asyncio', 'async', 'await', 'concurrent', 'threading', 'multiprocessing'],
            'gui': ['tkinter', 'qt', 'kivy', 'gui', 'desktop'],
            'tools': ['pip', 'virtualenv', 'conda', 'git', 'deployment', 'packaging']
        }
        
        for tag in tags:
            tag_lower = tag.lower()
            categorized = False
            
            for category, category_tags in tag_mapping.items():
                if tag_lower in category_tags:
                    categories[category].append(tag)
                    categorized = True
                    break
            
            if not categorized:
                categories['other'].append(tag)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _analyze_answer_characteristics(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of the answer portion."""
        
        characteristics = {
            'answer_type': 'unknown',
            'explanation_quality': 'unknown',
            'code_to_text_ratio': 0.0,
            'has_step_by_step': False,
            'has_alternatives': False,
            'has_warnings': False,
            'references_documentation': False
        }
        
        # Extract answer section
        lines = content.splitlines()
        answer_lines = []
        in_answer = False
        
        for line in lines:
            if line.startswith("Answer:"):
                in_answer = True
                continue
            if in_answer:
                answer_lines.append(line)
        
        answer_text = '\n'.join(answer_lines)
        
        if not answer_text.strip():
            return characteristics
        
        # Analyze answer type
        code_blocks = extract_code_blocks(answer_text)
        text_length = len(remove_html_tags(answer_text))
        code_length = sum(len(block.content) for block in code_blocks)
        
        if code_length > 0:
            characteristics['code_to_text_ratio'] = code_length / (text_length + code_length)
        
        if len(code_blocks) > 0 and text_length > 100:
            characteristics['answer_type'] = 'code_with_explanation'
        elif len(code_blocks) > 0:
            characteristics['answer_type'] = 'code_only'
        elif text_length > 200:
            characteristics['answer_type'] = 'explanation_only'
        else:
            characteristics['answer_type'] = 'brief'
        
        # Check for quality indicators
        answer_lower = answer_text.lower()
        
        # Step-by-step indicators
        step_indicators = ['step 1', 'first', 'then', 'next', 'finally', '1.', '2.', '3.']
        characteristics['has_step_by_step'] = any(indicator in answer_lower for indicator in step_indicators)
        
        # Alternative solutions
        alt_indicators = ['alternatively', 'another way', 'you could also', 'option']
        characteristics['has_alternatives'] = any(indicator in answer_lower for indicator in alt_indicators)
        
        # Warnings and caveats
        warning_indicators = ['warning', 'caution', 'note that', 'be careful', 'important']
        characteristics['has_warnings'] = any(indicator in answer_lower for indicator in warning_indicators)
        
        # Documentation references
        doc_indicators = ['docs', 'documentation', 'official', 'pep', 'according to']
        characteristics['references_documentation'] = any(indicator in answer_lower for indicator in doc_indicators)
        
        # Explanation quality heuristic
        if characteristics['has_step_by_step'] and characteristics['code_to_text_ratio'] > 0.2:
            characteristics['explanation_quality'] = 'excellent'
        elif len(code_blocks) > 0 and text_length > 100:
            characteristics['explanation_quality'] = 'good'
        elif text_length > 50:
            characteristics['explanation_quality'] = 'basic'
        else:
            characteristics['explanation_quality'] = 'minimal'
        
        return characteristics
    
    async def _enrich_qa_chunk(self, chunk: ProcessedChunk, code_blocks: List) -> None:
        """Enrich Q&A chunk with additional metadata."""
        
        # Add chunk-specific analysis
        if chunk.chunk_type == "question":
            chunk.metadata.update({
                'question_characteristics': self._analyze_question_characteristics(chunk.content),
                'urgency_indicators': self._detect_urgency_indicators(chunk.content)
            })
        
        elif chunk.chunk_type == "answer":
            chunk.metadata.update({
                'answer_completeness': self._assess_answer_completeness(chunk.content),
                'solution_confidence': self._assess_solution_confidence(chunk.content)
            })
        
        # Add code block analysis for chunks containing code
        chunk_code_blocks = [block for block in code_blocks 
                           if block.content in chunk.content]
        
        if chunk_code_blocks:
            chunk.metadata['code_analysis'] = {
                'languages': list(set(block.language for block in chunk_code_blocks)),
                'total_code_lines': sum(len(block.content.splitlines()) for block in chunk_code_blocks),
                'code_snippets': len(chunk_code_blocks)
            }
    
    def _analyze_question_characteristics(self, content: str) -> Dict[str, Any]:
        """Analyze characteristics of question content."""
        
        characteristics = {
            'question_type': 'unknown',
            'has_error': False,
            'has_code_attempt': False,
            'specificity_level': 'unknown'
        }
        
        content_lower = content.lower()
        
        # Question type detection
        if re.search(r'how (to|do|can)', content_lower):
            characteristics['question_type'] = 'how_to'
        elif re.search(r'why (does|is|do)', content_lower):
            characteristics['question_type'] = 'why'
        elif re.search(r'what (is|does|should)', content_lower):
            characteristics['question_type'] = 'what'
        elif re.search(r'(error|exception|problem|issue)', content_lower):
            characteristics['question_type'] = 'troubleshooting'
        elif content.endswith('?'):
            characteristics['question_type'] = 'general_question'
        
        # Error detection
        error_patterns = ['error', 'exception', 'traceback', 'failed', 'not working']
        characteristics['has_error'] = any(pattern in content_lower for pattern in error_patterns)
        
        # Code attempt detection
        characteristics['has_code_attempt'] = '```' in content or content.count('    ') > 2
        
        # Specificity assessment
        if characteristics['has_code_attempt'] and len(content) > 200:
            characteristics['specificity_level'] = 'high'
        elif len(content) > 100:
            characteristics['specificity_level'] = 'medium'
        else:
            characteristics['specificity_level'] = 'low'
        
        return characteristics
    
    def _detect_urgency_indicators(self, content: str) -> List[str]:
        """Detect urgency indicators in question content."""
        
        urgency_patterns = [
            ('urgent', 'urgent'),
            ('asap', 'asap'),
            ('quickly', 'time_sensitive'),
            ('deadline', 'deadline'),
            ('stuck', 'blocked'),
            ('help', 'help_needed'),
            ('please', 'polite_request')
        ]
        
        content_lower = content.lower()
        indicators = []
        
        for pattern, indicator in urgency_patterns:
            if pattern in content_lower:
                indicators.append(indicator)
        
        return indicators
    
    def _assess_answer_completeness(self, content: str) -> str:
        """Assess how complete an answer appears to be."""
        
        content_length = len(content)
        has_code = '```' in content or content.count('    ') > 2
        has_explanation = len(remove_html_tags(content)) > 100
        
        if has_code and has_explanation and content_length > 300:
            return 'comprehensive'
        elif has_code and has_explanation:
            return 'good'
        elif has_code or has_explanation:
            return 'partial'
        else:
            return 'minimal'
    
    def _assess_solution_confidence(self, content: str) -> str:
        """Assess confidence level of the solution."""
        
        content_lower = content.lower()
        
        # High confidence indicators
        high_confidence = ['this will', 'this works', 'guaranteed', 'definitely', 'always']
        
        # Low confidence indicators
        low_confidence = ['might work', 'try this', 'maybe', 'possibly', 'should work']
        
        # Uncertainty indicators
        uncertainty = ['not sure', 'think', 'believe', 'probably']
        
        if any(indicator in content_lower for indicator in high_confidence):
            return 'high'
        elif any(indicator in content_lower for indicator in uncertainty):
            return 'uncertain'
        elif any(indicator in content_lower for indicator in low_confidence):
            return 'tentative'
        else:
            return 'neutral'
    
    async def _perform_health_check(self) -> None:
        """Perform health check for Q&A processor."""
        
        # Test Q&A processing with sample content
        test_content = """Title: How to print in Python?

Question:
I want to print a message in Python. How do I do this?

Answer:
You can use the print() function:

```python
print("Hello, World!")
```

This will output the message to the console."""
        
        test_metadata = {
            'question_id': 123,
            'answer_id': 456,
            'tags': ['python', 'print'],
            'question_score': 5,
            'answer_score': 10,
            'is_accepted': True,
            'source_item_id': 'test',
            'source_type': 'stackoverflow_qa'
        }
        
        try:
            # Test content analysis
            analysis = await self._analyze_qa_content(test_content)
            if not analysis:
                raise Exception("Q&A content analysis not working")
            
            # Test chunking strategy
            strategy = ChunkingStrategyFactory.create_strategy(self.qa_config)
            chunks = strategy.chunk_content(test_content, test_metadata)
            
            if not chunks:
                raise Exception("Q&A chunking strategy not working")
            
            self.logger.debug("Q&A processor health check passed")
            
        except Exception as e:
            self.logger.error(f"Q&A processor health check failed: {e}")
            raise 