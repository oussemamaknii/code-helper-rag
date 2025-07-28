#!/usr/bin/env python3
"""
Evaluation system demonstration script.

This script demonstrates the comprehensive evaluation and monitoring
capabilities of the Python Code Helper RAG system.
"""

import asyncio
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.ragas_evaluator import (
    RAGASEvaluator, EvaluationConfig, EvaluationDataset, EvaluationDataPoint, RAGASMetricType
)
from src.evaluation.performance_monitor import (
    PerformanceMonitor, ThresholdConfig
)
from src.evaluation.ab_testing import (
    ABTestManager, ABTestConfig, TestVariant, TestResult, VariantType
)


class EvaluationDemo:
    """Comprehensive evaluation system demonstration."""
    
    def __init__(self):
        """Initialize evaluation demo."""
        self.ragas_evaluator = None
        self.performance_monitor = None
        self.ab_test_manager = ABTestManager()
        
        print("🧪 Python Code Helper Evaluation System Demo")
        print("=" * 60)
    
    async def run_full_demo(self):
        """Run comprehensive evaluation demonstration."""
        
        # Demo 1: RAGAS Evaluation
        print("\n📊 1. RAGAS Quality Evaluation")
        print("-" * 40)
        await self.demo_ragas_evaluation()
        
        # Demo 2: Performance Monitoring
        print("\n⚡ 2. Performance Monitoring")
        print("-" * 40)
        await self.demo_performance_monitoring()
        
        # Demo 3: A/B Testing
        print("\n🔬 3. A/B Testing Infrastructure")
        print("-" * 40)
        await self.demo_ab_testing()
        
        # Demo 4: Integration Example
        print("\n🔄 4. Integrated Evaluation Workflow")
        print("-" * 40)
        await self.demo_integrated_workflow()
        
        print("\n🎉 Evaluation System Demo Complete!")
        
    async def demo_ragas_evaluation(self):
        """Demonstrate RAGAS evaluation system."""
        print("Setting up RAGAS evaluation...")
        
        # Create evaluation configuration
        config = EvaluationConfig(
            metrics=[
                RAGASMetricType.FAITHFULNESS,
                RAGASMetricType.ANSWER_RELEVANCY,
                RAGASMetricType.CONTEXT_PRECISION,
                RAGASMetricType.CONTEXT_RECALL
            ],
            batch_size=5,
            max_concurrent_evaluations=3,
            minimum_faithfulness=0.7,
            minimum_relevancy=0.6,
            sample_size=20
        )
        
        # Create evaluator
        self.ragas_evaluator = RAGASEvaluator(config)
        
        # Create sample evaluation dataset
        dataset = self._create_sample_dataset()
        
        print(f"✅ Created evaluation dataset with {len(dataset.data_points)} samples")
        print(f"✅ Configured RAGAS metrics: {[m.value for m in config.metrics]}")
        
        # Run evaluation
        print("\n🔍 Running RAGAS evaluation...")
        start_time = time.time()
        
        try:
            results = await self.ragas_evaluator.evaluate(dataset)
            evaluation_time = time.time() - start_time
            
            print(f"✅ Evaluation completed in {evaluation_time:.2f}s")
            
            # Display results
            print(f"\n📈 RAGAS Evaluation Results:")
            print(f"   Overall Score: {results.metrics.overall_score:.3f}")
            print(f"   Faithfulness: {results.metrics.faithfulness:.3f}")
            print(f"   Answer Relevancy: {results.metrics.answer_relevancy:.3f}")
            print(f"   Context Precision: {results.metrics.context_precision:.3f}")
            print(f"   Context Recall: {results.metrics.context_recall:.3f}")
            
            # Quality assessment
            quality = results._assess_quality()
            print(f"\n🎯 Quality Assessment:")
            print(f"   Overall: {quality['overall']}")
            print(f"   Faithfulness: {quality['faithfulness']}")
            print(f"   Relevancy: {quality['relevancy']}")
            print(f"   Precision: {quality['precision']}")
            
            # Sample size analysis
            print(f"\n📊 Analysis Summary:")
            print(f"   Samples evaluated: {len(results.individual_scores)}")
            print(f"   Evaluation time per sample: {evaluation_time/len(results.individual_scores):.3f}s")
            
            # Show some individual scores
            print(f"\n🔍 Sample Individual Scores:")
            for i, score in enumerate(results.individual_scores[:3]):
                print(f"   Sample {i+1}: {score}")
                
        except Exception as e:
            print(f"❌ RAGAS evaluation failed: {e}")
            
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring system."""
        print("Setting up performance monitoring...")
        
        # Create threshold configuration
        threshold_config = ThresholdConfig(
            max_response_time=3.0,
            max_search_time=1.0,
            max_generation_time=5.0,
            max_error_rate=10.0,  # Higher for demo
            max_cpu_usage=90.0,   # Higher for demo
            max_memory_usage=90.0,
            alert_window_minutes=1,  # Shorter for demo
            alert_cooldown_minutes=2
        )
        
        # Create performance monitor
        self.performance_monitor = PerformanceMonitor(threshold_config)
        
        print("✅ Performance monitor configured")
        print(f"   Response time threshold: {threshold_config.max_response_time}s")
        print(f"   Error rate threshold: {threshold_config.max_error_rate}%")
        print(f"   CPU usage threshold: {threshold_config.max_cpu_usage}%")
        
        # Start monitoring
        print("\n📡 Starting performance monitoring...")
        await self.performance_monitor.start_monitoring(interval_seconds=5)
        
        # Simulate some API requests with metrics
        print("\n🚀 Simulating API requests...")
        await self._simulate_api_requests()
        
        # Get performance summary
        await asyncio.sleep(2)  # Let monitoring collect some data
        summary = self.performance_monitor.get_performance_summary(hours=1)
        
        print(f"\n📊 Performance Summary:")
        print(f"   Monitoring active: {summary['monitoring_active']}")
        print(f"   System uptime: {summary['uptime_seconds']:.1f}s")
        
        perf = summary.get('performance', {})
        if perf.get('response_time'):
            rt_stats = perf['response_time']
            print(f"   Response time - Mean: {rt_stats.get('mean', 0):.3f}s, P95: {rt_stats.get('p95', 0):.3f}s")
        
        sys_resources = summary.get('system_resources', {})
        if sys_resources.get('cpu_usage'):
            cpu_stats = sys_resources['cpu_usage']
            print(f"   CPU usage - Mean: {cpu_stats.get('mean', 0):.1f}%, Max: {cpu_stats.get('max', 0):.1f}%")
        
        endpoint_stats = summary.get('endpoint_stats', {})
        print(f"   Total requests: {sum(endpoint_stats.get('total_requests', {}).values())}")
        print(f"   Total errors: {sum(endpoint_stats.get('total_errors', {}).values())}")
        
        # Stop monitoring
        await self.performance_monitor.stop_monitoring()
        print("✅ Performance monitoring stopped")
        
    async def demo_ab_testing(self):
        """Demonstrate A/B testing infrastructure."""
        print("Setting up A/B testing...")
        
        # Create test variants
        variants = [
            TestVariant(
                id="control",
                name="Current Model",
                description="Existing GPT-4 model",
                config={"model": "gpt-4", "temperature": 0.7},
                traffic_percentage=50.0,
                is_control=True
            ),
            TestVariant(
                id="treatment",
                name="Enhanced Model",
                description="GPT-4 with optimized prompts",
                config={"model": "gpt-4", "temperature": 0.5, "enhanced_prompts": True},
                traffic_percentage=50.0,
                is_control=False
            )
        ]
        
        # Create A/B test configuration
        test_config = ABTestConfig(
            test_id="model_optimization_test_1",
            name="Model Response Quality Test",
            description="Testing enhanced prompts vs current implementation",
            variant_type=VariantType.PROMPT_OPTIMIZATION,
            variants=variants,
            primary_metric="user_satisfaction_score",
            secondary_metrics=["response_time", "accuracy_score"],
            minimum_sample_size=20,  # Lower for demo
            significance_level=0.05,
            maximum_duration_days=7
        )
        
        # Create test
        await self.ab_test_manager.create_test(test_config)
        print(f"✅ A/B test created: {test_config.name}")
        print(f"   Test ID: {test_config.test_id}")
        print(f"   Variants: {len(test_config.variants)}")
        print(f"   Primary metric: {test_config.primary_metric}")
        
        # Simulate user assignments and results
        print("\n👥 Simulating user assignments and results...")
        await self._simulate_ab_test_traffic(test_config.test_id)
        
        # Analyze results
        print("\n📈 Analyzing A/B test results...")
        analysis = await self.ab_test_manager.analyze_test(test_config.test_id)
        
        print(f"✅ Analysis completed")
        print(f"   Total samples: {analysis['total_samples']}")
        
        # Show variant analysis
        for variant_id, stats in analysis.get('variant_analysis', {}).items():
            print(f"   Variant '{variant_id}':")
            print(f"     Sample size: {stats['sample_size']}")
            print(f"     Mean score: {stats['mean']:.3f}")
            print(f"     Std deviation: {stats['std']:.3f}")
        
        # Show statistical tests
        for variant_id, test_result in analysis.get('statistical_tests', {}).items():
            significance = test_result.get('significance', {})
            print(f"   Statistical test (Control vs {variant_id}):")
            print(f"     Improvement: {test_result['improvement']:.1f}%")
            print(f"     P-value: {significance.get('p_value', 0):.4f}")
            print(f"     Significant: {significance.get('is_significant', False)}")
            print(f"     Effect size: {significance.get('effect_size', 0):.3f}")
        
        # Show active tests
        active_tests = self.ab_test_manager.get_active_tests()
        print(f"\n📋 Active tests: {len(active_tests)}")
        for test in active_tests:
            print(f"   • {test['name']} ({test['results_count']} results)")
            
    async def demo_integrated_workflow(self):
        """Demonstrate integrated evaluation workflow."""
        print("Running integrated evaluation workflow...")
        
        # Simulate a complete evaluation cycle
        workflow_steps = [
            "🔍 Collecting user queries and responses",
            "📊 Running RAGAS quality assessment", 
            "⚡ Monitoring system performance",
            "🔬 Analyzing A/B test results",
            "📈 Generating evaluation report",
            "🎯 Identifying optimization opportunities"
        ]
        
        for i, step in enumerate(workflow_steps, 1):
            print(f"   {i}. {step}")
            await asyncio.sleep(0.5)  # Simulate processing time
        
        # Generate mock evaluation report
        report = {
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "system_health": "healthy",
            "quality_metrics": {
                "overall_ragas_score": 0.782,
                "faithfulness": 0.845,
                "answer_relevancy": 0.756,
                "context_precision": 0.678,
                "improvement_vs_baseline": "+12.3%"
            },
            "performance_metrics": {
                "avg_response_time": 1.234,
                "p95_response_time": 2.456,
                "error_rate": 0.8,
                "system_utilization": {
                    "cpu": 45.2,
                    "memory": 67.8
                }
            },
            "ab_test_results": {
                "active_tests": 1,
                "significant_improvements": 1,
                "recommended_winner": "Enhanced Model (treatment)"
            },
            "recommendations": [
                "Deploy enhanced prompt optimization from A/B test",
                "Investigate context precision improvements",
                "Monitor memory usage trends",
                "Increase evaluation dataset size"
            ]
        }
        
        print("\n📋 Integrated Evaluation Report:")
        print(f"   Timestamp: {report['evaluation_timestamp']}")
        print(f"   System Health: ✅ {report['system_health']}")
        
        quality = report['quality_metrics']
        print(f"   Quality Score: {quality['overall_ragas_score']:.3f} ({quality['improvement_vs_baseline']})")
        
        performance = report['performance_metrics']
        print(f"   Avg Response Time: {performance['avg_response_time']:.3f}s")
        print(f"   Error Rate: {performance['error_rate']:.1f}%")
        
        ab_results = report['ab_test_results']
        print(f"   A/B Tests: {ab_results['active_tests']} active, {ab_results['significant_improvements']} significant")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✅ Integrated evaluation workflow completed successfully!")
        
    def _create_sample_dataset(self) -> EvaluationDataset:
        """Create sample evaluation dataset."""
        sample_data = [
            {
                "question": "How does the quicksort algorithm work?",
                "answer": "Quicksort is a divide-and-conquer algorithm that works by selecting a pivot element and partitioning the array around it. Elements smaller than the pivot go to the left, larger elements go to the right. The process is then recursively applied to the sub-arrays.",
                "contexts": [
                    "Quicksort is a sorting algorithm that uses divide-and-conquer strategy",
                    "The algorithm chooses a pivot element and partitions the array",
                    "Time complexity is O(n log n) on average, O(n²) worst case"
                ],
                "ground_truth": "Quicksort is an efficient divide-and-conquer sorting algorithm that partitions arrays around pivot elements."
            },
            {
                "question": "What is the difference between lists and tuples in Python?",
                "answer": "Lists are mutable sequences in Python, meaning you can modify them after creation. Tuples are immutable sequences - once created, you cannot change their contents. Lists use square brackets [], tuples use parentheses ().",
                "contexts": [
                    "Python lists are mutable data structures created with square brackets",
                    "Tuples are immutable sequences created with parentheses", 
                    "Lists support item assignment, tuples do not"
                ],
                "ground_truth": "Lists are mutable and use [], tuples are immutable and use ()."
            },
            {
                "question": "How do you implement binary search in Python?",
                "answer": "Binary search works on sorted arrays by repeatedly dividing the search space in half. Compare the target with the middle element: if equal, found; if target is smaller, search left half; if larger, search right half.",
                "contexts": [
                    "Binary search requires a sorted array as input",
                    "The algorithm compares target with middle element",
                    "Search space is halved in each iteration"
                ],
                "ground_truth": "Binary search divides sorted arrays in half repeatedly until the target is found."
            }
        ]
        
        # Create more samples by varying the existing ones
        data_points = []
        for base_data in sample_data:
            for i in range(8):  # Create 8 variations of each
                variation = EvaluationDataPoint(
                    question=base_data["question"],
                    answer=base_data["answer"],
                    contexts=base_data["contexts"],
                    ground_truth=base_data["ground_truth"],
                    metadata={"variation": i, "category": "programming_concepts"}
                )
                data_points.append(variation)
        
        return EvaluationDataset(
            name="programming_concepts_eval",
            description="Evaluation dataset for programming concept explanations",
            data_points=data_points,
            metadata={"created_for": "demo", "domain": "programming"}
        )
    
    async def _simulate_api_requests(self):
        """Simulate API requests for performance monitoring."""
        endpoints = ["/api/v1/chat", "/api/v1/search", "/api/v1/health"]
        
        for i in range(20):
            endpoint = random.choice(endpoints)
            
            # Simulate response time (with some variation)
            base_time = {"chat": 1.5, "search": 0.5, "health": 0.1}
            response_time = base_time.get(endpoint.split('/')[-1], 1.0) + random.uniform(-0.2, 0.8)
            
            # Simulate status codes (mostly successful)
            status_code = 200 if random.random() > 0.1 else random.choice([400, 429, 500])
            
            # Record metrics
            self.performance_monitor.record_request(
                endpoint=endpoint,
                response_time=response_time,
                status_code=status_code,
                user_id=f"user_{i % 5}"
            )
            
            # Simulate some generation metrics for chat endpoint
            if endpoint == "/api/v1/chat":
                self.performance_monitor.record_generation_metrics(
                    query_type="code_explanation",
                    generation_time=response_time * 0.8,
                    tokens_used=random.randint(100, 500),
                    sources_retrieved=random.randint(3, 8),
                    confidence_score=random.uniform(0.6, 0.95)
                )
            
            await asyncio.sleep(0.1)  # Small delay between requests
    
    async def _simulate_ab_test_traffic(self, test_id: str):
        """Simulate A/B test traffic and results."""
        user_ids = [f"user_{i}" for i in range(50)]
        
        for user_id in user_ids:
            # Assign user to variant
            variant_id = await self.ab_test_manager.assign_user_to_test(test_id, user_id)
            
            if variant_id:
                # Simulate different performance based on variant
                if variant_id == "control":
                    satisfaction_score = random.gauss(7.2, 1.5)  # Control group
                    response_time = random.gauss(1.8, 0.4)
                    accuracy_score = random.gauss(0.75, 0.15)
                else:  # treatment
                    satisfaction_score = random.gauss(7.8, 1.4)  # Slightly better
                    response_time = random.gauss(1.6, 0.3)  # Slightly faster
                    accuracy_score = random.gauss(0.82, 0.12)  # More accurate
                
                # Clamp values to reasonable ranges
                satisfaction_score = max(1.0, min(10.0, satisfaction_score))
                response_time = max(0.1, response_time)
                accuracy_score = max(0.0, min(1.0, accuracy_score))
                
                # Record test result
                result = TestResult(
                    variant_id=variant_id,
                    user_id=user_id,
                    session_id=f"session_{random.randint(1000, 9999)}",
                    metric_values={
                        "user_satisfaction_score": satisfaction_score,
                        "response_time": response_time,
                        "accuracy_score": accuracy_score
                    },
                    metadata={"query_type": "code_explanation"}
                )
                
                await self.ab_test_manager.record_result(test_id, result)


async def main():
    """Run the evaluation system demonstration."""
    demo = EvaluationDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main()) 