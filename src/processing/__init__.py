"""
Data processing module.

This module handles code parsing, semantic chunking, metadata extraction,
and other data preprocessing operations for the RAG system.
"""

from .base_processor import (
    BaseProcessor,
    ProcessedChunk,
    ProcessingStatus,
    ProcessingMetrics,
    DefaultChunkValidator
)
from .python_analyzer import (
    PythonASTAnalyzer,
    CodeElement,
    CodeElementType,
    CodeAnalysis,
    DependencyInfo
)
from .chunking_strategies import (
    ChunkingStrategy,
    ChunkingConfig,
    BaseChunkingStrategy,
    FixedSizeChunkingStrategy,
    FunctionLevelChunkingStrategy,
    QAPairChunkingStrategy,
    ChunkingStrategyFactory
)
from .code_processor import CodeProcessor
from .qa_processor import QAProcessor
from .pipeline import DataProcessingPipeline, ProcessingPipelineConfig

__all__ = [
    # Base processing
    "BaseProcessor",
    "ProcessedChunk",
    "ProcessingStatus",
    "ProcessingMetrics",
    "DefaultChunkValidator",
    
    # Python analysis
    "PythonASTAnalyzer",
    "CodeElement",
    "CodeElementType", 
    "CodeAnalysis",
    "DependencyInfo",
    
    # Chunking strategies
    "ChunkingStrategy",
    "ChunkingConfig",
    "BaseChunkingStrategy",
    "FixedSizeChunkingStrategy",
    "FunctionLevelChunkingStrategy",
    "QAPairChunkingStrategy",
    "ChunkingStrategyFactory",
    
    # Specialized processors
    "CodeProcessor",
    "QAProcessor",
    
    # Pipeline orchestration
    "DataProcessingPipeline",
    "ProcessingPipelineConfig"
] 