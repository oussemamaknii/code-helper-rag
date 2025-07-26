"""
Text processing utilities and helpers.

This module provides text processing functions including code extraction,
text cleaning, similarity calculations, and other text-related operations
used throughout the application.
"""

import re
import hashlib
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import unicodedata
from difflib import SequenceMatcher

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CodeBlock:
    """Represents an extracted code block."""
    content: str
    language: str
    start_line: int
    end_line: int
    is_inline: bool = False


def clean_text(text: str, preserve_code: bool = True) -> str:
    """
    Clean and normalize text while optionally preserving code blocks.
    
    Args:
        text: Input text to clean
        preserve_code: Whether to preserve code block formatting
    
    Returns:
        Cleaned text
    
    Examples:
        >>> text = "  This is some text with\\nextra   spaces  "
        >>> clean_text(text)
        'This is some text with extra spaces'
    """
    if not text:
        return ""
    
    # Store code blocks if preserving them
    code_blocks = []
    if preserve_code:
        code_blocks = extract_code_blocks(text)
        # Replace code blocks with placeholders
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(block.content, placeholder)
    
    # Unicode normalization
    text = unicodedata.normalize('NFKD', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Restore code blocks if they were preserved
    if preserve_code:
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            text = text.replace(placeholder, block.content)
    
    return text


def extract_code_blocks(text: str) -> List[CodeBlock]:
    """
    Extract code blocks from text (markdown, HTML, or other formats).
    
    Args:
        text: Input text containing code blocks
    
    Returns:
        List of extracted code blocks
    
    Examples:
        >>> text = '''
        ... Here's some Python code:
        ... ```python
        ... def hello():
        ...     print("Hello, world!")
        ... ```
        ... '''
        >>> blocks = extract_code_blocks(text)
        >>> len(blocks)
        1
    """
    code_blocks = []
    lines = text.split('\n')
    
    # Pattern for fenced code blocks (```language)
    fenced_pattern = re.compile(r'^```(\w+)?')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for fenced code block
        match = fenced_pattern.match(line)
        if match:
            language = match.group(1) if match.group(1) else 'text'
            start_line = i
            i += 1
            
            # Find the closing fence
            code_content = []
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_content.append(lines[i])
                i += 1
            
            if i < len(lines):  # Found closing fence
                code_blocks.append(CodeBlock(
                    content='\n'.join(code_content),
                    language=language,
                    start_line=start_line,
                    end_line=i,
                    is_inline=False
                ))
        
        # Check for indented code blocks (4+ spaces)
        elif line.startswith('    ') or line.startswith('\t'):
            start_line = i
            code_content = []
            
            while i < len(lines) and (lines[i].startswith('    ') or lines[i].startswith('\t') or lines[i].strip() == ''):
                code_content.append(lines[i][4:] if lines[i].startswith('    ') else lines[i][1:])
                i += 1
                continue
            
            # Remove trailing empty lines
            while code_content and code_content[-1].strip() == '':
                code_content.pop()
            
            if code_content:
                code_blocks.append(CodeBlock(
                    content='\n'.join(code_content),
                    language='text',  # Can't determine language from indented blocks
                    start_line=start_line,
                    end_line=i - 1,
                    is_inline=False
                ))
            continue
        
        i += 1
    
    # Extract inline code (backticks)
    inline_pattern = re.compile(r'`([^`\n]+)`')
    for match in inline_pattern.finditer(text):
        code_blocks.append(CodeBlock(
            content=match.group(1),
            language='text',
            start_line=0,  # Line numbers not meaningful for inline code
            end_line=0,
            is_inline=True
        ))
    
    return code_blocks


def extract_python_imports(code: str) -> List[str]:
    """
    Extract import statements from Python code.
    
    Args:
        code: Python code string
    
    Returns:
        List of import statements
    
    Examples:
        >>> code = '''
        ... import os
        ... from typing import List, Dict
        ... import numpy as np
        ... '''
        >>> imports = extract_python_imports(code)
        >>> 'import os' in imports
        True
    """
    imports = []
    
    # Pattern for import statements
    import_patterns = [
        r'^import\s+[\w\., ]+',
        r'^from\s+[\w\.]+\s+import\s+[\w\., \*\(\)\\n]+',
    ]
    
    for line in code.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        for pattern in import_patterns:
            if re.match(pattern, line):
                imports.append(line)
                break
    
    return imports


def extract_function_signatures(code: str) -> List[str]:
    """
    Extract function signatures from Python code.
    
    Args:
        code: Python code string
    
    Returns:
        List of function signatures
    
    Examples:
        >>> code = '''
        ... def hello_world():
        ...     print("Hello!")
        ... 
        ... async def fetch_data(url: str) -> dict:
        ...     return {}
        ... '''
        >>> signatures = extract_function_signatures(code)
        >>> len(signatures)
        2
    """
    signatures = []
    
    # Pattern for function definitions
    func_pattern = re.compile(r'^(async\s+)?def\s+\w+\([^)]*\)(\s*->\s*[^:]+)?:', re.MULTILINE)
    
    for match in func_pattern.finditer(code):
        signature = match.group(0).rstrip(':')
        signatures.append(signature)
    
    return signatures


def calculate_similarity(text1: str, text2: str, method: str = 'sequence') -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity calculation method ('sequence', 'jaccard', 'cosine')
    
    Returns:
        Similarity score between 0.0 and 1.0
    
    Examples:
        >>> similarity = calculate_similarity("hello world", "hello python")
        >>> 0.0 <= similarity <= 1.0
        True
    """
    if not text1 or not text2:
        return 0.0
    
    if method == 'sequence':
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    elif method == 'jaccard':
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if union else 0.0
    
    elif method == 'cosine':
        # Simple cosine similarity based on word frequency
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Create word frequency vectors
        all_words = set(words1 + words2)
        vector1 = [words1.count(word) for word in all_words]
        vector2 = [words2.count(word) for word in all_words]
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        magnitude1 = sum(a * a for a in vector1) ** 0.5
        magnitude2 = sum(b * b for b in vector2) ** 0.5
        
        if magnitude1 * magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def generate_text_hash(text: str, algorithm: str = 'md5') -> str:
    """
    Generate hash for text content.
    
    Args:
        text: Input text
        algorithm: Hash algorithm ('md5', 'sha256', 'sha1')
    
    Returns:
        Hexadecimal hash string
    
    Examples:
        >>> text_hash = generate_text_hash("Hello, world!")
        >>> len(text_hash)
        32
    """
    if algorithm == 'md5':
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    elif algorithm == 'sha256':
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    elif algorithm == 'sha1':
        return hashlib.sha1(text.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unknown hash algorithm: {algorithm}")


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with suffix.
    
    Args:
        text: Input text
        max_length: Maximum length including suffix
        suffix: Suffix to add when truncating
    
    Returns:
        Truncated text
    
    Examples:
        >>> truncated = truncate_text("This is a long text", 10)
        >>> truncated
        'This is...'
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
    
    Returns:
        List of extracted URLs
    
    Examples:
        >>> text = "Visit https://example.com or http://test.org"
        >>> urls = extract_urls(text)
        >>> len(urls)
        2
    """
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags from text.
    
    Args:
        text: Input text with HTML tags
    
    Returns:
        Text with HTML tags removed
    
    Examples:
        >>> html_text = "<p>Hello <b>world</b>!</p>"
        >>> clean = remove_html_tags(html_text)
        >>> clean
        'Hello world!'
    """
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities
    html_entities = {
        '&lt;': '<',
        '&gt;': '>',
        '&amp;': '&',
        '&quot;': '"',
        '&#39;': "'",
        '&nbsp;': ' ',
    }
    
    for entity, char in html_entities.items():
        clean = clean.replace(entity, char)
    
    return clean


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    
    Examples:
        >>> text = "Hello world. How are you? Fine, thanks!"
        >>> sentences = split_into_sentences(text)
        >>> len(sentences)
        3
    """
    # Simple sentence splitting (can be improved with NLP libraries)
    sentence_endings = re.compile(r'[.!?]+\s+')
    sentences = sentence_endings.split(text.strip())
    
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count for text (useful for LLM token estimation).
    
    Args:
        text: Input text
    
    Returns:
        Approximate number of tokens
    
    Note:
        This is a rough approximation. For accurate counts, use tiktoken.
    
    Examples:
        >>> text = "Hello world, this is a test."
        >>> token_count = count_tokens_approximate(text)
        >>> token_count > 0
        True
    """
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    Args:
        text: Input text
    
    Returns:
        Text with normalized whitespace
    
    Examples:
        >>> text = "Hello    world\\n\\n\\nThis   is   a   test."
        >>> normalized = normalize_whitespace(text)
        >>> normalized
        'Hello world This is a test.'
    """
    # Replace multiple whitespace characters with single space
    normalized = re.sub(r'\s+', ' ', text)
    return normalized.strip()


class TextSummarizer:
    """
    Simple extractive text summarization utility.
    
    Provides basic text summarization by extracting the most relevant sentences
    based on word frequency and sentence position.
    """
    
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences
    
    def summarize(self, text: str) -> str:
        """
        Summarize text by extracting key sentences.
        
        Args:
            text: Input text to summarize
        
        Returns:
            Summarized text
        
        Examples:
            >>> summarizer = TextSummarizer(max_sentences=2)
            >>> long_text = "..." # Long text
            >>> summary = summarizer.summarize(long_text)
        """
        sentences = split_into_sentences(text)
        
        if len(sentences) <= self.max_sentences:
            return text
        
        # Calculate word frequencies
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word = re.sub(r'[^\w]', '', word)  # Remove punctuation
            if word:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences based on word frequencies
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            words_in_sentence = sentence.lower().split()
            
            for word in words_in_sentence:
                word = re.sub(r'[^\w]', '', word)
                if word in word_freq:
                    score += word_freq[word]
            
            # Normalize by sentence length
            if len(words_in_sentence) > 0:
                score = score / len(words_in_sentence)
            
            # Slight preference for earlier sentences
            position_weight = 1.0 - (i * 0.1 / len(sentences))
            score *= position_weight
            
            sentence_scores.append((score, i, sentence))
        
        # Select top sentences
        sentence_scores.sort(reverse=True)
        selected_sentences = sentence_scores[:self.max_sentences]
        
        # Sort by original order
        selected_sentences.sort(key=lambda x: x[1])
        
        return ' '.join([sentence for _, _, sentence in selected_sentences]) 