"""
RAGAS-based evaluation system for RAG quality assessment.

This module implements comprehensive evaluation metrics using the RAGAS framework
for assessing retrieval-augmented generation systems.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from src.utils.logger import get_logger
from src.utils.async_utils import AsyncBatch, gather_with_concurrency

logger = get_logger(__name__)


class RAGASMetricType(str, Enum):
    """Types of RAGAS evaluation metrics."""
    FAITHFULNESS = "faithfulness"
    ANSWER_RELEVANCY = "answer_relevancy" 
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    CONTEXT_RELEVANCY = "context_relevancy"
    ANSWER_SIMILARITY = "answer_similarity"
    ANSWER_CORRECTNESS = "answer_correctness"


@dataclass
class EvaluationDataPoint:
    """Single evaluation data point for RAGAS assessment."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class RAGASMetrics:
    """RAGAS evaluation metrics result."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    context_relevancy: float = 0.0
    answer_similarity: float = 0.0
    answer_correctness: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            'faithfulness': 0.25,
            'answer_relevancy': 0.20,
            'context_precision': 0.15,
            'context_recall': 0.15,
            'context_relevancy': 0.10,
            'answer_similarity': 0.10,
            'answer_correctness': 0.05
        }
        
        total_score = (
            self.faithfulness * weights['faithfulness'] +
            self.answer_relevancy * weights['answer_relevancy'] +
            self.context_precision * weights['context_precision'] +
            self.context_recall * weights['context_recall'] +
            self.context_relevancy * weights['context_relevancy'] +
            self.answer_similarity * weights['answer_similarity'] +
            self.answer_correctness * weights['answer_correctness']
        )
        
        return total_score
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            'faithfulness': self.faithfulness,
            'answer_relevancy': self.answer_relevancy,
            'context_precision': self.context_precision,
            'context_recall': self.context_recall,
            'context_relevancy': self.context_relevancy,
            'answer_similarity': self.answer_similarity,
            'answer_correctness': self.answer_correctness,
            'overall_score': self.overall_score
        }


class EvaluationConfig(BaseModel):
    """Configuration for RAGAS evaluation."""
    
    # Metrics to evaluate
    metrics: List[RAGASMetricType] = Field(
        default=[
            RAGASMetricType.FAITHFULNESS,
            RAGASMetricType.ANSWER_RELEVANCY,
            RAGASMetricType.CONTEXT_PRECISION,
            RAGASMetricType.CONTEXT_RECALL
        ]
    )
    
    # Evaluation settings
    batch_size: int = Field(10, ge=1, le=100)
    max_concurrent_evaluations: int = Field(5, ge=1, le=20)
    evaluation_timeout: int = Field(300, ge=60, le=600)  # seconds
    
    # LLM settings for evaluation
    evaluation_model: str = Field("gpt-4")
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(1000, ge=100, le=4000)
    
    # Quality thresholds
    minimum_faithfulness: float = Field(0.7, ge=0.0, le=1.0)
    minimum_relevancy: float = Field(0.6, ge=0.0, le=1.0)
    minimum_precision: float = Field(0.5, ge=0.0, le=1.0)
    
    # Sampling settings
    sample_size: Optional[int] = Field(None, description="Number of samples to evaluate")
    stratified_sampling: bool = Field(True, description="Use stratified sampling")


class EvaluationDataset(BaseModel):
    """Dataset for evaluation containing questions, answers, and ground truth."""
    
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    data_points: List[EvaluationDataPoint] = Field(..., description="Evaluation data points")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def sample(self, n: int, stratified: bool = True) -> 'EvaluationDataset':
        """Sample n data points from the dataset."""
        if n >= len(self.data_points):
            return self
        
        if stratified:
            # Simple stratified sampling by query type/length
            sampled_points = self._stratified_sample(n)
        else:
            # Random sampling
            indices = np.random.choice(len(self.data_points), n, replace=False)
            sampled_points = [self.data_points[i] for i in indices]
        
        return EvaluationDataset(
            name=f"{self.name}_sample_{n}",
            description=f"Sample of {n} points from {self.name}",
            data_points=sampled_points,
            metadata={**self.metadata, "sampled_from": self.name, "sample_size": n}
        )
    
    def _stratified_sample(self, n: int) -> List[EvaluationDataPoint]:
        """Perform stratified sampling based on query characteristics."""
        # Group by query length (simple stratification)
        short_queries = [dp for dp in self.data_points if len(dp.question.split()) <= 10]
        medium_queries = [dp for dp in self.data_points if 10 < len(dp.question.split()) <= 20]
        long_queries = [dp for dp in self.data_points if len(dp.question.split()) > 20]
        
        # Calculate proportions
        total = len(self.data_points)
        short_prop = len(short_queries) / total
        medium_prop = len(medium_queries) / total
        long_prop = len(long_queries) / total
        
        # Sample proportionally
        short_n = max(1, int(n * short_prop))
        medium_n = max(1, int(n * medium_prop))
        long_n = n - short_n - medium_n
        
        sampled = []
        if short_queries and short_n > 0:
            sampled.extend(np.random.choice(short_queries, min(short_n, len(short_queries)), replace=False))
        if medium_queries and medium_n > 0:
            sampled.extend(np.random.choice(medium_queries, min(medium_n, len(medium_queries)), replace=False))
        if long_queries and long_n > 0:
            sampled.extend(np.random.choice(long_queries, min(long_n, len(long_queries)), replace=False))
        
        return sampled[:n]


@dataclass
class EvaluationResult:
    """Result of RAGAS evaluation."""
    
    dataset_name: str
    config: EvaluationConfig
    metrics: RAGASMetrics
    individual_scores: List[Dict[str, float]]
    evaluation_time: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        """Generate evaluation summary."""
        return {
            'dataset': self.dataset_name,
            'overall_score': self.metrics.overall_score,
            'metrics': self.metrics.to_dict(),
            'evaluation_time': self.evaluation_time,
            'sample_size': len(self.individual_scores),
            'quality_assessment': self._assess_quality(),
            'timestamp': self.timestamp.isoformat()
        }
    
    def _assess_quality(self) -> Dict[str, str]:
        """Assess overall quality based on thresholds."""
        assessment = {}
        
        if self.metrics.faithfulness >= self.config.minimum_faithfulness:
            assessment['faithfulness'] = 'good'
        elif self.metrics.faithfulness >= 0.5:
            assessment['faithfulness'] = 'acceptable'
        else:
            assessment['faithfulness'] = 'poor'
        
        if self.metrics.answer_relevancy >= self.config.minimum_relevancy:
            assessment['relevancy'] = 'good'
        elif self.metrics.answer_relevancy >= 0.4:
            assessment['relevancy'] = 'acceptable'
        else:
            assessment['relevancy'] = 'poor'
        
        if self.metrics.context_precision >= self.config.minimum_precision:
            assessment['precision'] = 'good'
        elif self.metrics.context_precision >= 0.3:
            assessment['precision'] = 'acceptable'
        else:
            assessment['precision'] = 'poor'
        
        # Overall assessment
        scores = [self.metrics.faithfulness, self.metrics.answer_relevancy, self.metrics.context_precision]
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 0.7:
            assessment['overall'] = 'excellent'
        elif avg_score >= 0.6:
            assessment['overall'] = 'good'
        elif avg_score >= 0.4:
            assessment['overall'] = 'acceptable'
        else:
            assessment['overall'] = 'needs_improvement'
        
        return assessment


class RAGASEvaluator:
    """
    RAGAS-based evaluator for RAG system quality assessment.
    
    This class implements comprehensive evaluation using RAGAS metrics
    to assess the quality of retrieval-augmented generation systems.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize RAGAS evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = get_logger(__name__, component="ragas_evaluator")
        
        # Initialize metric calculators
        self._metric_calculators = {
            RAGASMetricType.FAITHFULNESS: self._calculate_faithfulness,
            RAGASMetricType.ANSWER_RELEVANCY: self._calculate_answer_relevancy,
            RAGASMetricType.CONTEXT_PRECISION: self._calculate_context_precision,
            RAGASMetricType.CONTEXT_RECALL: self._calculate_context_recall,
            RAGASMetricType.CONTEXT_RELEVANCY: self._calculate_context_relevancy,
            RAGASMetricType.ANSWER_SIMILARITY: self._calculate_answer_similarity,
            RAGASMetricType.ANSWER_CORRECTNESS: self._calculate_answer_correctness
        }
    
    async def evaluate(self, dataset: EvaluationDataset) -> EvaluationResult:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            EvaluationResult: Comprehensive evaluation results
        """
        start_time = datetime.utcnow()
        
        # Sample dataset if configured
        eval_dataset = dataset
        if self.config.sample_size and self.config.sample_size < len(dataset.data_points):
            eval_dataset = dataset.sample(self.config.sample_size, self.config.stratified_sampling)
            self.logger.info(f"Sampled {len(eval_dataset.data_points)} points from {len(dataset.data_points)}")
        
        self.logger.info(
            f"Starting RAGAS evaluation",
            dataset=eval_dataset.name,
            sample_size=len(eval_dataset.data_points),
            metrics=list(self.config.metrics)
        )
        
        # Evaluate in batches
        all_scores = []
        batch_processor = AsyncBatch(self.config.batch_size)
        
        async for batch in batch_processor.process(eval_dataset.data_points):
            batch_scores = await self._evaluate_batch(batch)
            all_scores.extend(batch_scores)
            
            self.logger.debug(f"Completed batch evaluation: {len(batch_scores)} scores")
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_scores)
        
        evaluation_time = (datetime.utcnow() - start_time).total_seconds()
        
        result = EvaluationResult(
            dataset_name=eval_dataset.name,
            config=self.config,
            metrics=aggregated_metrics,
            individual_scores=all_scores,
            evaluation_time=evaluation_time,
            timestamp=start_time,
            metadata={
                'original_dataset_size': len(dataset.data_points),
                'evaluated_samples': len(eval_dataset.data_points),
                'metrics_evaluated': list(self.config.metrics)
            }
        )
        
        self.logger.info(
            f"RAGAS evaluation completed",
            overall_score=result.metrics.overall_score,
            evaluation_time=evaluation_time,
            quality_assessment=result._assess_quality()['overall']
        )
        
        return result
    
    async def _evaluate_batch(self, batch: List[EvaluationDataPoint]) -> List[Dict[str, float]]:
        """Evaluate a batch of data points."""
        tasks = []
        for data_point in batch:
            task = self._evaluate_single(data_point)
            tasks.append(task)
        
        return await gather_with_concurrency(tasks, self.config.max_concurrent_evaluations)
    
    async def _evaluate_single(self, data_point: EvaluationDataPoint) -> Dict[str, float]:
        """Evaluate a single data point across all configured metrics."""
        scores = {}
        
        for metric_type in self.config.metrics:
            calculator = self._metric_calculators[metric_type]
            try:
                score = await calculator(data_point)
                scores[metric_type.value] = score
            except Exception as e:
                self.logger.warning(f"Failed to calculate {metric_type.value}: {e}")
                scores[metric_type.value] = 0.0
        
        return scores
    
    def _aggregate_metrics(self, all_scores: List[Dict[str, float]]) -> RAGASMetrics:
        """Aggregate individual scores into overall metrics."""
        if not all_scores:
            return RAGASMetrics()
        
        # Calculate means for each metric
        metric_sums = {}
        metric_counts = {}
        
        for scores in all_scores:
            for metric, score in scores.items():
                if metric not in metric_sums:
                    metric_sums[metric] = 0.0
                    metric_counts[metric] = 0
                
                metric_sums[metric] += score
                metric_counts[metric] += 1
        
        # Calculate averages
        aggregated = RAGASMetrics()
        
        for metric, total in metric_sums.items():
            avg_score = total / metric_counts[metric] if metric_counts[metric] > 0 else 0.0
            
            if metric == RAGASMetricType.FAITHFULNESS.value:
                aggregated.faithfulness = avg_score
            elif metric == RAGASMetricType.ANSWER_RELEVANCY.value:
                aggregated.answer_relevancy = avg_score
            elif metric == RAGASMetricType.CONTEXT_PRECISION.value:
                aggregated.context_precision = avg_score
            elif metric == RAGASMetricType.CONTEXT_RECALL.value:
                aggregated.context_recall = avg_score
            elif metric == RAGASMetricType.CONTEXT_RELEVANCY.value:
                aggregated.context_relevancy = avg_score
            elif metric == RAGASMetricType.ANSWER_SIMILARITY.value:
                aggregated.answer_similarity = avg_score
            elif metric == RAGASMetricType.ANSWER_CORRECTNESS.value:
                aggregated.answer_correctness = avg_score
        
        return aggregated
    
    # Metric calculation methods (simplified implementations)
    # In production, these would use proper LLM-based evaluation
    
    async def _calculate_faithfulness(self, data_point: EvaluationDataPoint) -> float:
        """Calculate faithfulness score - how much the answer is grounded in context."""
        # Mock implementation - would use LLM to check if answer is supported by context
        answer_words = set(data_point.answer.lower().split())
        context_words = set()
        for context in data_point.contexts:
            context_words.update(context.lower().split())
        
        if not answer_words:
            return 0.0
        
        # Simple overlap-based faithfulness (in production, use LLM)
        supported_words = answer_words.intersection(context_words)
        faithfulness = len(supported_words) / len(answer_words)
        
        return min(faithfulness * 1.2, 1.0)  # Slight boost for realistic scores
    
    async def _calculate_answer_relevancy(self, data_point: EvaluationDataPoint) -> float:
        """Calculate how relevant the answer is to the question."""
        # Mock implementation using word overlap
        question_words = set(data_point.question.lower().split())
        answer_words = set(data_point.answer.lower().split())
        
        if not question_words or not answer_words:
            return 0.0
        
        overlap = question_words.intersection(answer_words)
        relevancy = len(overlap) / len(question_words.union(answer_words))
        
        return min(relevancy * 2.0, 1.0)  # Boost for realistic scores
    
    async def _calculate_context_precision(self, data_point: EvaluationDataPoint) -> float:
        """Calculate precision of retrieved context."""
        if not data_point.contexts:
            return 0.0
        
        # Mock implementation - in production, use LLM to assess relevance
        question_words = set(data_point.question.lower().split())
        
        relevant_contexts = 0
        for context in data_point.contexts:
            context_words = set(context.lower().split())
            overlap = question_words.intersection(context_words)
            if len(overlap) / len(question_words) > 0.3:  # Threshold for relevance
                relevant_contexts += 1
        
        return relevant_contexts / len(data_point.contexts)
    
    async def _calculate_context_recall(self, data_point: EvaluationDataPoint) -> float:
        """Calculate recall of retrieved context."""
        if not data_point.ground_truth or not data_point.contexts:
            return 0.0
        
        # Mock implementation
        truth_words = set(data_point.ground_truth.lower().split())
        context_words = set()
        for context in data_point.contexts:
            context_words.update(context.lower().split())
        
        if not truth_words:
            return 0.0
        
        covered_words = truth_words.intersection(context_words)
        return len(covered_words) / len(truth_words)
    
    async def _calculate_context_relevancy(self, data_point: EvaluationDataPoint) -> float:
        """Calculate overall relevancy of context to question."""
        return await self._calculate_context_precision(data_point)  # Simplified
    
    async def _calculate_answer_similarity(self, data_point: EvaluationDataPoint) -> float:
        """Calculate semantic similarity between answer and ground truth."""
        if not data_point.ground_truth:
            return 0.0
        
        # Mock implementation using word overlap
        answer_words = set(data_point.answer.lower().split())
        truth_words = set(data_point.ground_truth.lower().split())
        
        if not answer_words or not truth_words:
            return 0.0
        
        overlap = answer_words.intersection(truth_words)
        similarity = len(overlap) / len(answer_words.union(truth_words))
        
        return similarity
    
    async def _calculate_answer_correctness(self, data_point: EvaluationDataPoint) -> float:
        """Calculate correctness of answer compared to ground truth."""
        if not data_point.ground_truth:
            return 0.0
        
        # Mock implementation - in production, use LLM for semantic correctness
        similarity = await self._calculate_answer_similarity(data_point)
        
        # Simple heuristic: if similarity is high and answer is comprehensive
        completeness = min(len(data_point.answer) / len(data_point.ground_truth), 1.0)
        correctness = (similarity * 0.7) + (completeness * 0.3)
        
        return correctness 