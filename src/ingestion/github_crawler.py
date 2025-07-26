"""
GitHub repository crawler for collecting Python code.

This module implements an async GitHub crawler that can collect Python code
from popular repositories with comprehensive error handling and rate limiting.
"""

import asyncio
import base64
from typing import AsyncGenerator, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import hashlib

from github import Github, Repository, ContentFile, GithubException
from github.PaginatedList import PaginatedList

from src.ingestion.base_collector import BaseCollector, CollectedItem
from src.utils.logger import get_logger
from src.utils.async_utils import run_async
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class GitHubCodeItem(CollectedItem):
    """Collected item from GitHub repository."""
    
    def __init__(self, 
                 file_path: str,
                 content: str, 
                 repository_name: str,
                 repository_url: str,
                 file_size: int,
                 file_sha: str,
                 last_modified: str,
                 language: str = "python"):
        """Initialize GitHub code item."""
        
        # Generate unique ID from repo and file path
        item_id = hashlib.md5(f"{repository_name}:{file_path}".encode()).hexdigest()
        
        metadata = {
            "file_path": file_path,
            "repository_name": repository_name,
            "repository_url": repository_url,
            "file_size": file_size,
            "file_sha": file_sha,
            "last_modified": last_modified,
            "language": language,
            "lines_of_code": len(content.split('\n')),
            "contains_classes": 'class ' in content,
            "contains_functions": 'def ' in content,
            "contains_imports": any(line.strip().startswith(('import ', 'from ')) 
                                   for line in content.split('\n'))
        }
        
        super().__init__(
            id=item_id,
            content=content,
            metadata=metadata,
            source_type="github_code"
        )


class GitHubCrawler(BaseCollector):
    """
    Async GitHub repository crawler for Python code collection.
    
    Features:
    - Rate limiting to respect API limits
    - Repository filtering by stars, language, size
    - File filtering by size, type, content
    - Comprehensive error handling
    - Progress tracking and metrics
    """
    
    def __init__(self, 
                 github_token: str,
                 max_file_size: int = None,
                 min_stars: int = None,
                 exclude_forks: bool = None,
                 languages: List[str] = None,
                 **kwargs):
        """
        Initialize GitHub crawler.
        
        Args:
            github_token: GitHub personal access token
            max_file_size: Maximum file size in bytes
            min_stars: Minimum repository stars
            exclude_forks: Whether to exclude forked repositories
            languages: List of programming languages to collect
            **kwargs: Additional arguments for BaseCollector
        """
        super().__init__(
            name="GitHubCrawler",
            rate_limit=settings.github_rate_limit if hasattr(settings, 'github_rate_limit') else 5000,
            **kwargs
        )
        
        self.github = Github(github_token)
        self.max_file_size = max_file_size or settings.github_max_file_size
        self.min_stars = min_stars or settings.github_min_stars
        self.exclude_forks = exclude_forks if exclude_forks is not None else settings.github_exclude_forks
        self.languages = languages or settings.github_languages
        
        # Track processed repositories to avoid duplicates
        self.processed_repos: Set[str] = set()
        
        self.logger.info(
            "GitHubCrawler initialized",
            max_file_size=self.max_file_size,
            min_stars=self.min_stars,
            exclude_forks=self.exclude_forks,
            languages=self.languages
        )
    
    async def get_total_count(self, **kwargs) -> Optional[int]:
        """Get approximate total count of repositories to process."""
        try:
            query = kwargs.get('query', self._build_search_query())
            max_repos = kwargs.get('max_repos', 100)
            
            # Use GitHub search to get count
            repositories = await run_async(
                self.github.search_repositories,
                query=query,
                sort="stars",
                order="desc"
            )
            
            # Return minimum of available repos and max_repos
            return min(repositories.totalCount, max_repos)
            
        except Exception as e:
            self.logger.warning(f"Could not get total count: {e}")
            return None
    
    async def collect_items(self, 
                           query: Optional[str] = None,
                           max_repos: int = 100,
                           max_files_per_repo: int = 50) -> AsyncGenerator[CollectedItem, None]:
        """
        Collect Python code files from GitHub repositories.
        
        Args:
            query: Custom search query (uses default if None)
            max_repos: Maximum number of repositories to process
            max_files_per_repo: Maximum files to collect per repository
            
        Yields:
            GitHubCodeItem: Collected Python code files
        """
        search_query = query or self._build_search_query()
        
        self.logger.info(
            "Starting GitHub collection",
            query=search_query,
            max_repos=max_repos,
            max_files_per_repo=max_files_per_repo
        )
        
        try:
            # Search for repositories
            repositories = await run_async(
                self.github.search_repositories,
                query=search_query,
                sort="stars",
                order="desc"
            )
            
            repo_count = 0
            async for repo_item in self._process_repositories(repositories, max_repos):
                if repo_item:
                    yield repo_item
                    
                # Update progress
                if hasattr(repo_item, 'metadata') and repo_item.metadata.get('repository_name'):
                    current_repo = repo_item.metadata['repository_name']
                    if current_repo not in self.processed_repos:
                        self.processed_repos.add(current_repo)
                        repo_count += 1
                        
                        self.logger.debug(
                            "Repository processed",
                            repository=current_repo,
                            total_processed=repo_count
                        )
                
        except GithubException as e:
            self.logger.error(f"GitHub API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during collection: {e}")
            raise
    
    async def _process_repositories(self, 
                                   repositories: PaginatedList,
                                   max_repos: int) -> AsyncGenerator[GitHubCodeItem, None]:
        """Process repositories and yield code items."""
        processed_count = 0
        
        for repo in repositories:
            if processed_count >= max_repos:
                break
            
            try:
                # Filter repository
                if not await self._should_process_repository(repo):
                    continue
                
                self.logger.debug(f"Processing repository: {repo.full_name}")
                
                # Process Python files in repository
                async for code_item in self._process_repository_files(repo):
                    yield code_item
                
                processed_count += 1
                
            except GithubException as e:
                if e.status == 403:  # Rate limit
                    self.logger.warning("Rate limit hit, waiting...")
                    await asyncio.sleep(60)  # Wait 1 minute
                    continue
                else:
                    self.logger.error(f"GitHub error processing {repo.full_name}: {e}")
                    continue
            except Exception as e:
                self.logger.error(f"Error processing repository {repo.full_name}: {e}")
                continue
    
    async def _process_repository_files(self, repo: Repository) -> AsyncGenerator[GitHubCodeItem, None]:
        """Process Python files in a repository."""
        try:
            # Get repository contents
            contents = await run_async(repo.get_contents, "")
            file_count = 0
            max_files = 50  # Limit files per repo
            
            while contents and file_count < max_files:
                file_content = contents.pop(0)
                
                try:
                    if file_content.type == "dir":
                        # Add directory contents to queue
                        dir_contents = await run_async(repo.get_contents, file_content.path)
                        contents.extend(dir_contents)
                        
                    elif await self._should_process_file(file_content):
                        # Process Python file
                        code_item = await self._extract_code_item(file_content, repo)
                        if code_item:
                            yield code_item
                            file_count += 1
                            
                except GithubException as e:
                    if e.status == 403:  # Rate limit
                        await asyncio.sleep(60)
                    else:
                        self.logger.warning(f"Error processing file {file_content.path}: {e}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Error processing file {file_content.path}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error accessing repository contents for {repo.full_name}: {e}")
    
    async def _extract_code_item(self, 
                               file_content: ContentFile, 
                               repo: Repository) -> Optional[GitHubCodeItem]:
        """Extract code content from file."""
        try:
            # Check file size
            if file_content.size > self.max_file_size:
                self.logger.debug(f"Skipping large file: {file_content.path} ({file_content.size} bytes)")
                return None
            
            # Get file content
            content_bytes = await run_async(lambda: file_content.decoded_content)
            content = content_bytes.decode('utf-8', errors='ignore')
            
            # Basic content validation
            if not content.strip():
                return None
            
            # Create code item
            return GitHubCodeItem(
                file_path=file_content.path,
                content=content,
                repository_name=repo.full_name,
                repository_url=repo.html_url,
                file_size=file_content.size,
                file_sha=file_content.sha,
                last_modified=repo.updated_at.isoformat() if repo.updated_at else "",
                language="python"
            )
            
        except UnicodeDecodeError:
            self.logger.debug(f"Could not decode file: {file_content.path}")
            return None
        except Exception as e:
            self.logger.warning(f"Error extracting content from {file_content.path}: {e}")
            return None
    
    async def _should_process_repository(self, repo: Repository) -> bool:
        """Check if repository should be processed."""
        try:
            # Check if already processed
            if repo.full_name in self.processed_repos:
                return False
            
            # Check stars
            if repo.stargazers_count < self.min_stars:
                return False
            
            # Check if fork (if exclusion enabled)
            if self.exclude_forks and repo.fork:
                return False
            
            # Check language
            if repo.language and repo.language.lower() not in [lang.lower() for lang in self.languages]:
                return False
            
            # Check if repository is accessible
            if repo.private:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking repository {repo.full_name}: {e}")
            return False
    
    async def _should_process_file(self, file_content: ContentFile) -> bool:
        """Check if file should be processed."""
        try:
            # Check if it's a Python file
            if not file_content.name.endswith('.py'):
                return False
            
            # Skip certain directories/files
            skip_patterns = [
                '__pycache__',
                '.git',
                'test_',
                'tests/',
                'test/',
                '.pytest',
                'conftest.py',
                'setup.py',
                '__init__.py'  # Often empty or minimal
            ]
            
            file_path_lower = file_content.path.lower()
            if any(pattern in file_path_lower for pattern in skip_patterns):
                return False
            
            # Check file size
            if file_content.size > self.max_file_size:
                return False
            
            # Skip very small files (likely empty or minimal)
            if file_content.size < 100:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking file {file_content.path}: {e}")
            return False
    
    def _build_search_query(self) -> str:
        """Build GitHub search query."""
        query_parts = []
        
        # Add language filters
        for language in self.languages:
            query_parts.append(f"language:{language}")
        
        # Add stars filter
        query_parts.append(f"stars:>={self.min_stars}")
        
        # Exclude forks if configured
        if self.exclude_forks:
            query_parts.append("fork:false")
        
        # Add other filters for quality
        query_parts.extend([
            "size:>100",  # Repository has some content
            "pushed:>2020-01-01"  # Recently updated
        ])
        
        return " ".join(query_parts)
    
    async def _perform_health_check(self) -> None:
        """Perform GitHub API health check."""
        try:
            # Check API rate limit
            rate_limit = await run_async(self.github.get_rate_limit)
            
            if rate_limit.core.remaining < 100:
                raise Exception(f"Low API rate limit remaining: {rate_limit.core.remaining}")
            
            # Test a simple API call
            await run_async(self.github.get_user)
            
            self.logger.debug(
                "GitHub API health check passed",
                remaining_requests=rate_limit.core.remaining,
                reset_time=rate_limit.core.reset
            )
            
        except Exception as e:
            self.logger.error(f"GitHub API health check failed: {e}")
            raise
    
    async def get_repository_info(self, repo_name: str) -> Optional[Dict]:
        """Get detailed information about a specific repository."""
        try:
            repo = await run_async(self.github.get_repo, repo_name)
            
            return {
                "name": repo.full_name,
                "description": repo.description,
                "stars": repo.stargazers_count,
                "forks": repo.forks_count,
                "language": repo.language,
                "size": repo.size,
                "created_at": repo.created_at.isoformat() if repo.created_at else None,
                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                "topics": repo.get_topics(),
                "license": repo.license.name if repo.license else None,
                "url": repo.html_url
            }
            
        except Exception as e:
            self.logger.error(f"Error getting repository info for {repo_name}: {e}")
            return None 