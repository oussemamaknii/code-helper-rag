"""
Evaluation and monitoring system for the Python Code Helper RAG system.

This module provides comprehensive evaluation metrics, performance monitoring,
A/B testing, and quality assessment tools for continuous improvement.
"""

from .ragas_evaluator import (
    RAGASEvaluator,
    RAGASMetrics,
    EvaluationResult,
    EvaluationDataset,
    EvaluationConfig
)
from .performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    MetricCollector,
    AlertManager,
    ThresholdConfig
)
from .ab_testing import (
    ABTestManager,
    ABTestConfig,
    TestVariant,
    TestResult,
    StatisticalSignificance
)
from .analytics import (
    AnalyticsPipeline,
    UsageAnalytics,
    UserBehaviorAnalyzer,
    QueryPatternAnalyzer,
    PerformanceAnalyzer
)
from .feedback import (
    FeedbackCollector,
    FeedbackAnalyzer,
    UserFeedback,
    FeedbackMetrics
)
from .quality_metrics import (
    QualityMetricsCollector,
    ResponseQualityAssessor,
    SourceRelevanceEvaluator,
    CoherenceAnalyzer
)
from .evaluation_pipeline import (
    EvaluationPipeline,
    EvaluationPipelineConfig,
    EvaluationRunner,
    BenchmarkSuite
)

__all__ = [
    # RAGAS Evaluation
    "RAGASEvaluator",
    "RAGASMetrics", 
    "EvaluationResult",
    "EvaluationDataset",
    "EvaluationConfig",
    
    # Performance Monitoring
    "PerformanceMonitor",
    "PerformanceMetrics",
    "MetricCollector",
    "AlertManager", 
    "ThresholdConfig",
    
    # A/B Testing
    "ABTestManager",
    "ABTestConfig",
    "TestVariant",
    "TestResult",
    "StatisticalSignificance",
    
    # Analytics
    "AnalyticsPipeline",
    "UsageAnalytics",
    "UserBehaviorAnalyzer",
    "QueryPatternAnalyzer",
    "PerformanceAnalyzer",
    
    # Feedback
    "FeedbackCollector",
    "FeedbackAnalyzer", 
    "UserFeedback",
    "FeedbackMetrics",
    
    # Quality Metrics
    "QualityMetricsCollector",
    "ResponseQualityAssessor",
    "SourceRelevanceEvaluator",
    "CoherenceAnalyzer",
    
    # Evaluation Pipeline
    "EvaluationPipeline",
    "EvaluationPipelineConfig", 
    "EvaluationRunner",
    "BenchmarkSuite"
] 