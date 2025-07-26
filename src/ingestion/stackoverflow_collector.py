"""
Stack Overflow data collector for Q&A content.

This module implements an async Stack Overflow collector that retrieves
Python-related questions and answers with proper rate limiting and filtering.
"""

import asyncio
import gzip
import json
import hashlib
from typing import AsyncGenerator, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlencode

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from src.ingestion.base_collector import BaseCollector, CollectedItem
from src.utils.logger import get_logger
from src.utils.async_utils import AsyncRetry, async_timeout
from src.utils.text_utils import remove_html_tags, clean_text
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class StackOverflowQAItem(CollectedItem):
    """Collected Q&A item from Stack Overflow."""
    
    def __init__(self,
                 question_id: int,
                 question_title: str,
                 question_body: str,
                 answer_body: str,
                 answer_id: int,
                 tags: List[str],
                 question_score: int,
                 answer_score: int,
                 is_accepted: bool,
                 created_date: str,
                 view_count: int = 0):
        """Initialize Stack Overflow Q&A item."""
        
        # Generate unique ID from question and answer IDs
        item_id = f"so_{question_id}_{answer_id}"
        
        # Clean HTML from content
        clean_question = remove_html_tags(question_body)
        clean_answer = remove_html_tags(answer_body)
        
        # Combine question and answer for content
        content = f"Title: {question_title}\n\nQuestion:\n{clean_question}\n\nAnswer:\n{clean_answer}"
        
        metadata = {
            "question_id": question_id,
            "answer_id": answer_id,
            "question_title": question_title,
            "tags": tags,
            "question_score": question_score,
            "answer_score": answer_score,
            "is_accepted": is_accepted,
            "created_date": created_date,
            "view_count": view_count,
            "url": f"https://stackoverflow.com/questions/{question_id}",
            "answer_url": f"https://stackoverflow.com/a/{answer_id}",
            "has_code": "```" in answer_body or "<code>" in answer_body,
            "question_length": len(clean_question),
            "answer_length": len(clean_answer),
            "total_score": question_score + answer_score,
            "python_related": any(tag in ["python", "python-3.x", "python-2.7"] for tag in tags)
        }
        
        super().__init__(
            id=item_id,
            content=content,
            metadata=metadata,
            source_type="stackoverflow_qa"
        )


class StackOverflowCollector(BaseCollector):
    """
    Async Stack Overflow collector for Python Q&A content.
    
    Features:
    - API rate limiting (300 requests per day for unauthenticated)
    - Question filtering by score, tags, date
    - Answer filtering by score, acceptance status
    - HTML content cleaning
    - Comprehensive error handling
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 min_question_score: int = None,
                 min_answer_score: int = None,
                 tags: List[str] = None,
                 include_accepted_only: bool = False,
                 **kwargs):
        """
        Initialize Stack Overflow collector.
        
        Args:
            api_key: Stack Overflow API key (optional, increases rate limit)
            min_question_score: Minimum question score filter
            min_answer_score: Minimum answer score filter  
            tags: List of tags to filter by
            include_accepted_only: Whether to include only accepted answers
            **kwargs: Additional arguments for BaseCollector
        """
        super().__init__(
            name="StackOverflowCollector",
            rate_limit=300 if not api_key else 10000,  # Requests per day
            **kwargs
        )
        
        self.api_key = api_key
        self.base_url = "https://api.stackexchange.com/2.3"
        self.min_question_score = min_question_score or settings.so_min_score
        self.min_answer_score = min_answer_score or 0
        self.tags = tags or settings.so_tags
        self.include_accepted_only = include_accepted_only
        
        # Track processed questions to avoid duplicates
        self.processed_questions: Set[int] = set()
        
        # HTTP session for connection pooling
        self.session: Optional[ClientSession] = None
        
        self.logger.info(
            "StackOverflowCollector initialized",
            min_question_score=self.min_question_score,
            min_answer_score=self.min_answer_score,
            tags=self.tags,
            include_accepted_only=self.include_accepted_only,
            has_api_key=bool(self.api_key)
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = ClientSession(
            timeout=ClientTimeout(total=30),
            headers={"User-Agent": "PythonCodeHelper/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_total_count(self, **kwargs) -> Optional[int]:
        """Get approximate total count of questions to process."""
        try:
            max_questions = kwargs.get('max_questions', 1000)
            
            # Use search API to get count
            params = self._build_search_params(page=1, pagesize=1)
            
            async with self.session.get(f"{self.base_url}/search", params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                total_available = data.get('total', 0)
                
                return min(total_available, max_questions)
                
        except Exception as e:
            self.logger.warning(f"Could not get total count: {e}")
            return None
    
    async def collect_items(self,
                           max_questions: int = 1000,
                           max_answers_per_question: int = 5) -> AsyncGenerator[CollectedItem, None]:
        """
        Collect Python Q&A from Stack Overflow.
        
        Args:
            max_questions: Maximum number of questions to process
            max_answers_per_question: Maximum answers to collect per question
            
        Yields:
            StackOverflowQAItem: Collected Q&A pairs
        """
        if not self.session:
            async with self:
                async for item in self._collect_with_session(max_questions, max_answers_per_question):
                    yield item
        else:
            async for item in self._collect_with_session(max_questions, max_answers_per_question):
                yield item
    
    async def _collect_with_session(self,
                                   max_questions: int,
                                   max_answers_per_question: int) -> AsyncGenerator[StackOverflowQAItem, None]:
        """Internal collection method with active session."""
        
        self.logger.info(
            "Starting Stack Overflow collection",
            max_questions=max_questions,
            max_answers_per_question=max_answers_per_question,
            tags=self.tags
        )
        
        collected_questions = 0
        page = 1
        
        while collected_questions < max_questions:
            try:
                # Get questions page
                questions = await self._fetch_questions_page(page)
                
                if not questions:
                    self.logger.info("No more questions available")
                    break
                
                # Process each question
                for question in questions:
                    if collected_questions >= max_questions:
                        break
                    
                    # Skip if already processed
                    if question['question_id'] in self.processed_questions:
                        continue
                    
                    # Filter question
                    if not self._should_process_question(question):
                        continue
                    
                    # Get answers for this question
                    async for qa_item in self._process_question_answers(question, max_answers_per_question):
                        yield qa_item
                    
                    self.processed_questions.add(question['question_id'])
                    collected_questions += 1
                    
                    # Small delay to be respectful
                    await asyncio.sleep(0.1)
                
                page += 1
                
                # Rate limiting delay between pages
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error processing page {page}: {e}")
                break
    
    async def _fetch_questions_page(self, page: int) -> List[Dict]:
        """Fetch a page of questions from Stack Overflow API."""
        try:
            params = self._build_search_params(page=page, pagesize=100)
            
            async with async_timeout(
                self.session.get(f"{self.base_url}/search", params=params),
                timeout=30.0
            ) as response:
                
                if response.status == 429:  # Rate limited
                    self.logger.warning("Rate limited, waiting...")
                    await asyncio.sleep(60)
                    return []
                
                if response.status != 200:
                    self.logger.error(f"API error: {response.status}")
                    return []
                
                # Handle gzipped response
                content = await response.read()
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(content)
                
                data = json.loads(content)
                return data.get('items', [])
                
        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching page {page}")
            return []
        except Exception as e:
            self.logger.error(f"Error fetching questions page {page}: {e}")
            return []
    
    async def _process_question_answers(self,
                                       question: Dict,
                                       max_answers: int) -> AsyncGenerator[StackOverflowQAItem, None]:
        """Process answers for a question."""
        question_id = question['question_id']
        
        try:
            # Get answers for this question
            params = self._build_answers_params()
            
            async with async_timeout(
                self.session.get(f"{self.base_url}/questions/{question_id}/answers", params=params),
                timeout=30.0
            ) as response:
                
                if response.status != 200:
                    self.logger.debug(f"Could not fetch answers for question {question_id}")
                    return
                
                # Handle gzipped response  
                content = await response.read()
                if response.headers.get('content-encoding') == 'gzip':
                    content = gzip.decompress(content)
                
                data = json.loads(content)
                answers = data.get('items', [])
                
                # Process answers
                answer_count = 0
                for answer in answers:
                    if answer_count >= max_answers:
                        break
                    
                    if self._should_process_answer(answer):
                        qa_item = self._create_qa_item(question, answer)
                        if qa_item:
                            yield qa_item
                            answer_count += 1
                
        except Exception as e:
            self.logger.warning(f"Error processing answers for question {question_id}: {e}")
    
    def _create_qa_item(self, question: Dict, answer: Dict) -> Optional[StackOverflowQAItem]:
        """Create Q&A item from question and answer data."""
        try:
            return StackOverflowQAItem(
                question_id=question['question_id'],
                question_title=question.get('title', ''),
                question_body=question.get('body', ''),
                answer_body=answer.get('body', ''),
                answer_id=answer['answer_id'],
                tags=question.get('tags', []),
                question_score=question.get('score', 0),
                answer_score=answer.get('score', 0),
                is_accepted=answer.get('is_accepted', False),
                created_date=datetime.fromtimestamp(
                    answer.get('creation_date', 0), 
                    tz=timezone.utc
                ).isoformat(),
                view_count=question.get('view_count', 0)
            )
        except Exception as e:
            self.logger.warning(f"Error creating Q&A item: {e}")
            return None
    
    def _should_process_question(self, question: Dict) -> bool:
        """Check if question should be processed."""
        try:
            # Check score
            if question.get('score', 0) < self.min_question_score:
                return False
            
            # Check if it has answers
            if question.get('answer_count', 0) == 0:
                return False
            
            # Check tags
            question_tags = question.get('tags', [])
            if not any(tag in self.tags for tag in question_tags):
                return False
            
            # Check if question has body content
            if not question.get('body', '').strip():
                return False
            
            return True
            
        except Exception:
            return False
    
    def _should_process_answer(self, answer: Dict) -> bool:
        """Check if answer should be processed."""
        try:
            # Check score
            if answer.get('score', 0) < self.min_answer_score:
                return False
            
            # Check if accepted only mode is enabled
            if self.include_accepted_only and not answer.get('is_accepted', False):
                return False
            
            # Check if answer has body content
            if not answer.get('body', '').strip():
                return False
            
            # Skip very short answers
            body = remove_html_tags(answer.get('body', ''))
            if len(body.strip()) < 50:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _build_search_params(self, page: int = 1, pagesize: int = 100) -> Dict:
        """Build parameters for questions search."""
        params = {
            'order': 'desc',
            'sort': 'votes',
            'site': 'stackoverflow',
            'page': page,
            'pagesize': pagesize,
            'filter': 'withbody',  # Include question body
            'tagged': ';'.join(self.tags),
            'min': self.min_question_score
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        return params
    
    def _build_answers_params(self) -> Dict:
        """Build parameters for answers request."""
        params = {
            'order': 'desc',
            'sort': 'votes',
            'site': 'stackoverflow',
            'filter': 'withbody',  # Include answer body
            'pagesize': 100
        }
        
        if self.api_key:
            params['key'] = self.api_key
        
        return params
    
    async def _perform_health_check(self) -> None:
        """Perform Stack Overflow API health check."""
        try:
            if not self.session:
                self.session = ClientSession(timeout=ClientTimeout(total=10))
            
            # Test API access
            params = {'site': 'stackoverflow', 'pagesize': 1}
            if self.api_key:
                params['key'] = self.api_key
            
            async with self.session.get(f"{self.base_url}/questions", params=params) as response:
                if response.status != 200:
                    raise Exception(f"API returned status {response.status}")
                
                # Check rate limits
                quota_remaining = response.headers.get('x-quota-remaining')
                if quota_remaining and int(quota_remaining) < 10:
                    raise Exception(f"Low API quota remaining: {quota_remaining}")
            
            self.logger.debug("Stack Overflow API health check passed")
            
        except Exception as e:
            self.logger.error(f"Stack Overflow API health check failed: {e}")
            raise
    
    async def get_question_details(self, question_id: int) -> Optional[Dict]:
        """Get detailed information about a specific question."""
        try:
            if not self.session:
                self.session = ClientSession()
            
            params = {
                'site': 'stackoverflow',
                'filter': 'withbody'
            }
            if self.api_key:
                params['key'] = self.api_key
            
            async with self.session.get(f"{self.base_url}/questions/{question_id}", params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                questions = data.get('items', [])
                
                return questions[0] if questions else None
                
        except Exception as e:
            self.logger.error(f"Error getting question details for {question_id}: {e}")
            return None 