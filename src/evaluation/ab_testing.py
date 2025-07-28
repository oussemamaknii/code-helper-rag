"""
A/B testing infrastructure for RAG system optimization.

This module provides comprehensive A/B testing capabilities for evaluating
different models, prompts, retrieval strategies, and system configurations.
"""

import asyncio
import hashlib
import random
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict

import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TestStatus(str, Enum):
    """A/B test status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantType(str, Enum):
    """Types of test variants."""
    MODEL_COMPARISON = "model_comparison"
    PROMPT_OPTIMIZATION = "prompt_optimization"
    RETRIEVAL_STRATEGY = "retrieval_strategy"
    RANKING_ALGORITHM = "ranking_algorithm"
    CONTEXT_LENGTH = "context_length"
    TEMPERATURE_SETTING = "temperature_setting"
    CHUNK_SIZE = "chunk_size"


class SignificanceTest(str, Enum):
    """Statistical significance tests."""
    T_TEST = "t_test"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    BOOTSTRAP = "bootstrap"


@dataclass
class TestVariant:
    """Configuration for an A/B test variant."""
    id: str
    name: str
    description: str
    config: Dict[str, Any]
    traffic_percentage: float = 50.0
    is_control: bool = False
    
    def __post_init__(self):
        """Validate variant configuration."""
        if not 0 < self.traffic_percentage <= 100:
            raise ValueError("Traffic percentage must be between 0 and 100")


class ABTestConfig(BaseModel):
    """Configuration for A/B test."""
    
    # Test identification
    test_id: str = Field(..., description="Unique test identifier")
    name: str = Field(..., description="Human-readable test name")
    description: str = Field(..., description="Test description and objectives")
    variant_type: VariantType = Field(..., description="Type of variant being tested")
    
    # Test variants
    variants: List[TestVariant] = Field(..., min_items=2, description="Test variants")
    
    # Test parameters
    significance_level: float = Field(0.05, ge=0.01, le=0.1, description="Statistical significance level")
    minimum_sample_size: int = Field(100, ge=10, le=10000, description="Minimum samples per variant")
    maximum_duration_days: int = Field(30, ge=1, le=365, description="Maximum test duration")
    
    # Success metrics
    primary_metric: str = Field(..., description="Primary success metric")
    secondary_metrics: List[str] = Field(default_factory=list, description="Secondary metrics")
    
    # Traffic allocation
    traffic_allocation: Dict[str, float] = Field(default_factory=dict, description="Traffic allocation per variant")
    
    # Targeting
    user_targeting: Optional[Dict[str, Any]] = Field(None, description="User targeting criteria")
    query_targeting: Optional[Dict[str, Any]] = Field(None, description="Query targeting criteria")
    
    def model_post_init(self, __context: Any) -> None:
        """Validate configuration after initialization."""
        # Validate traffic allocation
        if self.traffic_allocation:
            total_traffic = sum(self.traffic_allocation.values())
            if abs(total_traffic - 100.0) > 0.01:
                raise ValueError(f"Traffic allocation must sum to 100%, got {total_traffic}%")
        else:
            # Auto-allocate traffic equally
            traffic_per_variant = 100.0 / len(self.variants)
            self.traffic_allocation = {v.id: traffic_per_variant for v in self.variants}


@dataclass
class TestResult:
    """Result from A/B test variant."""
    variant_id: str
    user_id: str
    session_id: str
    metric_values: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StatisticalSignificance:
    """Statistical significance analysis result."""
    test_statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    is_significant: bool
    effect_size: Optional[float] = None
    test_type: SignificanceTest = SignificanceTest.T_TEST
    
    @property
    def significance_level(self) -> str:
        """Get significance level description."""
        if self.p_value < 0.001:
            return "highly_significant"
        elif self.p_value < 0.01:
            return "very_significant"
        elif self.p_value < 0.05:
            return "significant"
        else:
            return "not_significant"


class UserAssignment:
    """
    User assignment for A/B testing with consistent allocation.
    
    Features:
    - Consistent user assignment across sessions
    - Configurable traffic allocation
    - User targeting support
    - Assignment logging and tracking
    """
    
    def __init__(self):
        """Initialize user assignment system."""
        self.logger = get_logger(__name__, component="user_assignment")
        
        # Store user assignments for consistency
        self._user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)
    
    def assign_user(self, test_config: ABTestConfig, user_id: str, 
                   user_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Assign user to test variant.
        
        Args:
            test_config: A/B test configuration
            user_id: User identifier
            user_metadata: Optional user metadata for targeting
            
        Returns:
            Variant ID if user is assigned, None if not eligible
        """
        # Check if user already assigned to this test
        if user_id in self._user_assignments and test_config.test_id in self._user_assignments[user_id]:
            return self._user_assignments[user_id][test_config.test_id]
        
        # Check user targeting criteria
        if not self._check_user_targeting(test_config, user_metadata):
            return None
        
        # Generate consistent hash for user + test
        hash_input = f"{user_id}:{test_config.test_id}"
        user_hash = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        hash_percentage = (user_hash % 10000) / 100.0  # 0-99.99%
        
        # Determine variant based on traffic allocation
        cumulative_percentage = 0.0
        for variant in test_config.variants:
            variant_traffic = test_config.traffic_allocation.get(variant.id, 0)
            cumulative_percentage += variant_traffic
            
            if hash_percentage < cumulative_percentage:
                # Assign user to this variant
                self._user_assignments[user_id][test_config.test_id] = variant.id
                
                self.logger.debug(
                    f"User assigned to variant",
                    user_id=user_id,
                    test_id=test_config.test_id,
                    variant_id=variant.id,
                    hash_percentage=hash_percentage
                )
                
                return variant.id
        
        # No variant assigned (shouldn't happen if traffic allocation is correct)
        return None
    
    def _check_user_targeting(self, test_config: ABTestConfig, 
                            user_metadata: Optional[Dict[str, Any]]) -> bool:
        """Check if user meets targeting criteria."""
        if not test_config.user_targeting:
            return True
        
        if not user_metadata:
            return False
        
        targeting = test_config.user_targeting
        
        # Check each targeting criterion
        for key, expected_value in targeting.items():
            user_value = user_metadata.get(key)
            
            if isinstance(expected_value, list):
                if user_value not in expected_value:
                    return False
            elif isinstance(expected_value, dict):
                # Range-based targeting
                if 'min' in expected_value and user_value < expected_value['min']:
                    return False
                if 'max' in expected_value and user_value > expected_value['max']:
                    return False
            else:
                if user_value != expected_value:
                    return False
        
        return True


class StatisticalAnalyzer:
    """
    Statistical analysis for A/B test results.
    
    Features:
    - Multiple statistical tests
    - Effect size calculation
    - Confidence intervals
    - Sample size estimation
    - Power analysis
    """
    
    def __init__(self):
        """Initialize statistical analyzer."""
        self.logger = get_logger(__name__, component="statistical_analyzer")
    
    def analyze_test_results(self, control_data: List[float], 
                           treatment_data: List[float],
                           test_type: SignificanceTest = SignificanceTest.T_TEST,
                           significance_level: float = 0.05) -> StatisticalSignificance:
        """
        Analyze A/B test results for statistical significance.
        
        Args:
            control_data: Control group measurements
            treatment_data: Treatment group measurements
            test_type: Type of statistical test to perform
            significance_level: Significance level (alpha)
            
        Returns:
            Statistical significance analysis
        """
        if not control_data or not treatment_data:
            raise ValueError("Both control and treatment data must be provided")
        
        control_array = np.array(control_data)
        treatment_array = np.array(treatment_data)
        
        if test_type == SignificanceTest.T_TEST:
            return self._t_test_analysis(control_array, treatment_array, significance_level)
        elif test_type == SignificanceTest.MANN_WHITNEY:
            return self._mann_whitney_analysis(control_array, treatment_array, significance_level)
        elif test_type == SignificanceTest.BOOTSTRAP:
            return self._bootstrap_analysis(control_array, treatment_array, significance_level)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def _t_test_analysis(self, control: np.ndarray, treatment: np.ndarray,
                        significance_level: float) -> StatisticalSignificance:
        """Perform t-test analysis."""
        # Welch's t-test (assumes unequal variances)
        t_statistic, p_value = stats.ttest_ind(treatment, control, equal_var=False)
        
        # Calculate confidence interval for difference in means
        n1, n2 = len(control), len(treatment)
        mean1, mean2 = np.mean(control), np.mean(treatment)
        var1, var2 = np.var(control, ddof=1), np.var(treatment, ddof=1)
        
        # Standard error of difference
        se_diff = np.sqrt(var1/n1 + var2/n2)
        
        # Degrees of freedom (Welch's formula)
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Critical value
        t_critical = stats.t.ppf(1 - significance_level/2, df)
        
        # Confidence interval
        diff = mean2 - mean1
        margin_error = t_critical * se_diff
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        return StatisticalSignificance(
            test_statistic=t_statistic,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=p_value < significance_level,
            effect_size=effect_size,
            test_type=SignificanceTest.T_TEST
        )
    
    def _mann_whitney_analysis(self, control: np.ndarray, treatment: np.ndarray,
                              significance_level: float) -> StatisticalSignificance:
        """Perform Mann-Whitney U test analysis."""
        statistic, p_value = stats.mannwhitneyu(
            treatment, control, alternative='two-sided'
        )
        
        # Effect size (rank biserial correlation)
        n1, n2 = len(control), len(treatment)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return StatisticalSignificance(
            test_statistic=statistic,
            p_value=p_value,
            confidence_interval=(0, 0),  # Complex to calculate for Mann-Whitney
            is_significant=p_value < significance_level,
            effect_size=effect_size,
            test_type=SignificanceTest.MANN_WHITNEY
        )
    
    def _bootstrap_analysis(self, control: np.ndarray, treatment: np.ndarray,
                           significance_level: float, n_bootstrap: int = 10000) -> StatisticalSignificance:
        """Perform bootstrap analysis."""
        # Original difference in means
        original_diff = np.mean(treatment) - np.mean(control)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_control = np.random.choice(control, size=len(control), replace=True)
            boot_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
            
            # Calculate difference
            boot_diff = np.mean(boot_treatment) - np.mean(boot_control)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Confidence interval
        alpha = significance_level
        ci_lower = np.percentile(bootstrap_diffs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        
        # P-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= 0),
            np.mean(bootstrap_diffs <= 0)
        )
        
        return StatisticalSignificance(
            test_statistic=original_diff,
            p_value=p_value,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=not (ci_lower <= 0 <= ci_upper),
            effect_size=original_diff / np.std(np.concatenate([control, treatment])),
            test_type=SignificanceTest.BOOTSTRAP
        )
    
    def calculate_required_sample_size(self, effect_size: float, 
                                     power: float = 0.8,
                                     significance_level: float = 0.05) -> int:
        """Calculate required sample size for detecting effect size."""
        # Using formula for two-sample t-test
        z_alpha = stats.norm.ppf(1 - significance_level/2)
        z_beta = stats.norm.ppf(power)
        
        # Required sample size per group
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))


class ABTestManager:
    """
    A/B test manager for coordinating experiments.
    
    Features:
    - Test lifecycle management
    - User assignment and tracking
    - Results collection and analysis
    - Statistical significance testing
    - Automatic test stopping criteria
    """
    
    def __init__(self):
        """Initialize A/B test manager."""
        self.logger = get_logger(__name__, component="ab_test_manager")
        
        # Test management
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._test_results: Dict[str, List[TestResult]] = defaultdict(list)
        
        # Components
        self.user_assignment = UserAssignment()
        self.statistical_analyzer = StatisticalAnalyzer()
    
    async def create_test(self, config: ABTestConfig) -> None:
        """Create and start a new A/B test."""
        # Validate configuration
        await self._validate_test_config(config)
        
        # Store test configuration
        self._active_tests[config.test_id] = config
        
        self.logger.info(
            f"A/B test created",
            test_id=config.test_id,
            name=config.name,
            variants=len(config.variants),
            primary_metric=config.primary_metric
        )
    
    async def assign_user_to_test(self, test_id: str, user_id: str,
                                user_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Assign user to test variant."""
        if test_id not in self._active_tests:
            return None
        
        test_config = self._active_tests[test_id]
        variant_id = self.user_assignment.assign_user(test_config, user_id, user_metadata)
        
        return variant_id
    
    async def record_result(self, test_id: str, result: TestResult) -> None:
        """Record test result for analysis."""
        if test_id not in self._active_tests:
            self.logger.warning(f"Recording result for unknown test: {test_id}")
            return
        
        self._test_results[test_id].append(result)
        
        # Check if test should be stopped
        await self._check_stopping_criteria(test_id)
    
    async def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if test_id not in self._active_tests:
            raise ValueError(f"Test not found: {test_id}")
        
        test_config = self._active_tests[test_id]
        results = self._test_results[test_id]
        
        if not results:
            return {"status": "no_data", "message": "No results recorded yet"}
        
        # Group results by variant
        variant_results = defaultdict(list)
        for result in results:
            variant_results[result.variant_id].append(result)
        
        # Analyze each metric
        analysis = {
            "test_id": test_id,
            "test_name": test_config.name,
            "total_samples": len(results),
            "variant_analysis": {},
            "statistical_tests": {}
        }
        
        # Find control variant
        control_variant = None
        for variant in test_config.variants:
            if variant.is_control:
                control_variant = variant.id
                break
        
        if not control_variant:
            control_variant = test_config.variants[0].id  # Use first as control
        
        # Analyze primary metric
        primary_metric = test_config.primary_metric
        control_values = [
            result.metric_values.get(primary_metric, 0)
            for result in variant_results[control_variant]
        ]
        
        for variant_id, variant_results_list in variant_results.items():
            if variant_id == control_variant:
                continue
            
            treatment_values = [
                result.metric_values.get(primary_metric, 0)
                for result in variant_results_list
            ]
            
            if control_values and treatment_values:
                significance = self.statistical_analyzer.analyze_test_results(
                    control_values, treatment_values
                )
                
                analysis["statistical_tests"][variant_id] = {
                    "control_mean": np.mean(control_values),
                    "treatment_mean": np.mean(treatment_values),
                    "improvement": (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values) * 100,
                    "significance": significance.__dict__
                }
        
        # Summary statistics per variant
        for variant_id, variant_results_list in variant_results.items():
            values = [result.metric_values.get(primary_metric, 0) for result in variant_results_list]
            
            analysis["variant_analysis"][variant_id] = {
                "sample_size": len(values),
                "mean": np.mean(values) if values else 0,
                "std": np.std(values) if values else 0,
                "min": np.min(values) if values else 0,
                "max": np.max(values) if values else 0
            }
        
        return analysis
    
    async def stop_test(self, test_id: str, reason: str = "manual") -> None:
        """Stop an active A/B test."""
        if test_id in self._active_tests:
            del self._active_tests[test_id]
            
            self.logger.info(
                f"A/B test stopped",
                test_id=test_id,
                reason=reason
            )
    
    async def _validate_test_config(self, config: ABTestConfig) -> None:
        """Validate A/B test configuration."""
        # Check for duplicate test ID
        if config.test_id in self._active_tests:
            raise ValueError(f"Test with ID {config.test_id} already exists")
        
        # Validate variant configurations
        total_traffic = sum(config.traffic_allocation.values())
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_traffic}%")
        
        # Check for control variant
        control_count = sum(1 for v in config.variants if v.is_control)
        if control_count != 1:
            raise ValueError("Exactly one variant must be marked as control")
    
    async def _check_stopping_criteria(self, test_id: str) -> None:
        """Check if test should be stopped based on criteria."""
        test_config = self._active_tests[test_id]
        results = self._test_results[test_id]
        
        # Check minimum sample size
        variant_counts = defaultdict(int)
        for result in results:
            variant_counts[result.variant_id] += 1
        
        min_samples_reached = all(
            count >= test_config.minimum_sample_size
            for count in variant_counts.values()
        )
        
        if min_samples_reached:
            # Check for statistical significance
            analysis = await self.analyze_test(test_id)
            
            for variant_id, test_result in analysis.get("statistical_tests", {}).items():
                significance = test_result.get("significance", {})
                if significance.get("is_significant", False):
                    self.logger.info(
                        f"Test reached statistical significance",
                        test_id=test_id,
                        variant_id=variant_id,
                        p_value=significance.get("p_value")
                    )
                    # Optionally auto-stop test here
                    # await self.stop_test(test_id, "statistical_significance")
                    break
    
    def get_active_tests(self) -> List[Dict[str, Any]]:
        """Get list of active tests."""
        return [
            {
                "test_id": test_config.test_id,
                "name": test_config.name,
                "variant_type": test_config.variant_type,
                "variants": len(test_config.variants),
                "primary_metric": test_config.primary_metric,
                "results_count": len(self._test_results[test_config.test_id])
            }
            for test_config in self._active_tests.values()
        ] 