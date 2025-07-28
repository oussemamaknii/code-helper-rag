"""
Performance monitoring system for RAG pipeline.

This module provides comprehensive performance monitoring, metrics collection,
alerting, and real-time analysis of system performance.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum

import psutil
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of performance metrics."""
    LATENCY = "latency"
    THROUGHPUT = "throughput" 
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_USAGE = "network_usage"
    QUEUE_SIZE = "queue_size"
    ACTIVE_CONNECTIONS = "active_connections"
    CACHE_HIT_RATE = "cache_hit_rate"


class AlertLevel(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_processes: int
    load_average: Optional[float] = None


class ThresholdConfig(BaseModel):
    """Configuration for performance thresholds and alerting."""
    
    # Latency thresholds (seconds)
    max_response_time: float = Field(5.0, ge=0.1, le=60.0)
    max_search_time: float = Field(2.0, ge=0.1, le=30.0)
    max_generation_time: float = Field(10.0, ge=0.1, le=120.0)
    
    # Throughput thresholds
    min_requests_per_second: float = Field(0.1, ge=0.01, le=1000.0)
    max_queue_size: int = Field(100, ge=1, le=10000)
    
    # Error rate thresholds (percentage)
    max_error_rate: float = Field(5.0, ge=0.0, le=100.0)
    max_timeout_rate: float = Field(2.0, ge=0.0, le=100.0)
    
    # System resource thresholds
    max_cpu_usage: float = Field(80.0, ge=10.0, le=100.0)
    max_memory_usage: float = Field(85.0, ge=10.0, le=100.0)
    max_disk_usage: float = Field(90.0, ge=10.0, le=100.0)
    
    # Cache performance
    min_cache_hit_rate: float = Field(60.0, ge=0.0, le=100.0)
    
    # Alert settings
    alert_window_minutes: int = Field(5, ge=1, le=60)
    alert_cooldown_minutes: int = Field(15, ge=1, le=120)


class PerformanceAlert(NamedTuple):
    """Performance alert notification."""
    level: AlertLevel
    metric: str
    current_value: float
    threshold_value: float
    message: str
    timestamp: datetime
    metadata: Dict[str, Any]


class MetricCollector:
    """
    Collector for performance metrics with time-series storage.
    
    Features:
    - Time-series metric storage with configurable retention
    - Efficient aggregation and querying
    - Memory-efficient circular buffers
    - Real-time metric calculation
    """
    
    def __init__(self, max_points: int = 10000, retention_hours: int = 24):
        """
        Initialize metric collector.
        
        Args:
            max_points: Maximum points to store per metric
            retention_hours: Hours to retain metrics
        """
        self.max_points = max_points
        self.retention_hours = retention_hours
        
        # Time-series storage: metric_name -> deque of (timestamp, value, labels)
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._last_cleanup = time.time()
        
        self.logger = get_logger(__name__, component="metric_collector")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None,
                     timestamp: Optional[datetime] = None) -> None:
        """Record a performance metric."""
        timestamp = timestamp or datetime.utcnow()
        labels = labels or {}
        
        self._metrics[name].append((timestamp, value, labels))
        
        # Periodic cleanup of old metrics
        current_time = time.time()
        if current_time - self._last_cleanup > 3600:  # Cleanup every hour
            self._cleanup_old_metrics()
            self._last_cleanup = current_time
    
    def get_metrics(self, name: str, hours: int = 1) -> List[PerformanceMetric]:
        """Get metrics for the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        metrics = []
        for timestamp, value, labels in self._metrics[name]:
            if timestamp >= cutoff_time:
                metrics.append(PerformanceMetric(
                    name=name,
                    value=value,
                    unit=self._get_metric_unit(name),
                    timestamp=timestamp,
                    labels=labels
                ))
        
        return metrics
    
    def calculate_statistics(self, name: str, hours: int = 1) -> Dict[str, float]:
        """Calculate statistics for a metric over time window."""
        metrics = self.get_metrics(name, hours)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'p50': self._percentile(values, 50),
            'p95': self._percentile(values, 95),
            'p99': self._percentile(values, 99)
        }
    
    def get_current_rate(self, name: str, minutes: int = 5) -> float:
        """Calculate current rate (events per second) for a metric."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        count = 0
        for timestamp, _, _ in self._metrics[name]:
            if timestamp >= cutoff_time:
                count += 1
        
        return count / (minutes * 60) if count > 0 else 0.0
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up metrics older than retention period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        
        for name, metric_deque in self._metrics.items():
            # Remove old entries from the left
            while metric_deque and metric_deque[0][0] < cutoff_time:
                metric_deque.popleft()
    
    def _get_metric_unit(self, name: str) -> str:
        """Get unit for metric name."""
        unit_mapping = {
            'latency': 'seconds',
            'response_time': 'seconds',
            'throughput': 'requests/sec',
            'error_rate': 'percent',
            'cpu_usage': 'percent',
            'memory_usage': 'percent',
            'disk_usage': 'percent',
            'queue_size': 'count',
            'cache_hit_rate': 'percent'
        }
        
        for key, unit in unit_mapping.items():
            if key in name.lower():
                return unit
        
        return 'count'
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = max(0, min(index, len(sorted_values) - 1))
        
        return sorted_values[index]


class AlertManager:
    """
    Alert manager for performance monitoring.
    
    Features:
    - Threshold-based alerting
    - Alert cooldown to prevent spam
    - Multiple alert channels
    - Alert aggregation and deduplication
    """
    
    def __init__(self, config: ThresholdConfig):
        """
        Initialize alert manager.
        
        Args:
            config: Threshold configuration
        """
        self.config = config
        self.logger = get_logger(__name__, component="alert_manager")
        
        # Alert state tracking
        self._last_alerts: Dict[str, datetime] = {}
        self._alert_counts: Dict[str, int] = defaultdict(int)
        self._alert_handlers: List[Callable] = []
    
    def add_alert_handler(self, handler: Callable[[PerformanceAlert], None]) -> None:
        """Add an alert handler function."""
        self._alert_handlers.append(handler)
    
    async def check_thresholds(self, metric_collector: MetricCollector) -> List[PerformanceAlert]:
        """Check all thresholds and generate alerts."""
        alerts = []
        
        # Check latency thresholds
        response_time_stats = metric_collector.calculate_statistics('response_time', hours=1)
        if response_time_stats and response_time_stats['p95'] > self.config.max_response_time:
            alert = self._create_alert(
                level=AlertLevel.WARNING,
                metric='response_time_p95',
                current_value=response_time_stats['p95'],
                threshold_value=self.config.max_response_time,
                message=f"High response time: P95 is {response_time_stats['p95']:.2f}s"
            )
            alerts.append(alert)
        
        # Check error rate
        error_rate = metric_collector.get_current_rate('errors', minutes=self.config.alert_window_minutes)
        total_requests = metric_collector.get_current_rate('requests', minutes=self.config.alert_window_minutes)
        
        if total_requests > 0:
            error_percentage = (error_rate / total_requests) * 100
            if error_percentage > self.config.max_error_rate:
                alert = self._create_alert(
                    level=AlertLevel.ERROR,
                    metric='error_rate',
                    current_value=error_percentage,
                    threshold_value=self.config.max_error_rate,
                    message=f"High error rate: {error_percentage:.1f}%"
                )
                alerts.append(alert)
        
        # Check system metrics
        system_alerts = await self._check_system_thresholds()
        alerts.extend(system_alerts)
        
        # Filter alerts by cooldown period
        filtered_alerts = self._filter_alerts_by_cooldown(alerts)
        
        # Send alerts
        for alert in filtered_alerts:
            await self._send_alert(alert)
        
        return filtered_alerts
    
    async def _check_system_thresholds(self) -> List[PerformanceAlert]:
        """Check system resource thresholds."""
        alerts = []
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.max_cpu_usage:
                alert = self._create_alert(
                    level=AlertLevel.WARNING,
                    metric='cpu_usage',
                    current_value=cpu_percent,
                    threshold_value=self.config.max_cpu_usage,
                    message=f"High CPU usage: {cpu_percent:.1f}%"
                )
                alerts.append(alert)
            
            # Memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.config.max_memory_usage:
                alert = self._create_alert(
                    level=AlertLevel.WARNING,
                    metric='memory_usage',
                    current_value=memory.percent,
                    threshold_value=self.config.max_memory_usage,
                    message=f"High memory usage: {memory.percent:.1f}%"
                )
                alerts.append(alert)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > self.config.max_disk_usage:
                alert = self._create_alert(
                    level=AlertLevel.ERROR,
                    metric='disk_usage',
                    current_value=disk_percent,
                    threshold_value=self.config.max_disk_usage,
                    message=f"High disk usage: {disk_percent:.1f}%"
                )
                alerts.append(alert)
        
        except Exception as e:
            self.logger.warning(f"Failed to check system metrics: {e}")
        
        return alerts
    
    def _create_alert(self, level: AlertLevel, metric: str, current_value: float,
                     threshold_value: float, message: str, 
                     metadata: Optional[Dict[str, Any]] = None) -> PerformanceAlert:
        """Create a performance alert."""
        return PerformanceAlert(
            level=level,
            metric=metric,
            current_value=current_value,
            threshold_value=threshold_value,
            message=message,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
    
    def _filter_alerts_by_cooldown(self, alerts: List[PerformanceAlert]) -> List[PerformanceAlert]:
        """Filter alerts based on cooldown period."""
        filtered = []
        current_time = datetime.utcnow()
        cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
        
        for alert in alerts:
            alert_key = f"{alert.metric}_{alert.level}"
            
            if alert_key not in self._last_alerts:
                # First time seeing this alert
                filtered.append(alert)
                self._last_alerts[alert_key] = current_time
            else:
                # Check if cooldown period has passed
                if current_time - self._last_alerts[alert_key] > cooldown:
                    filtered.append(alert)
                    self._last_alerts[alert_key] = current_time
        
        return filtered
    
    async def _send_alert(self, alert: PerformanceAlert) -> None:
        """Send alert to all registered handlers."""
        self._alert_counts[alert.metric] += 1
        
        self.logger.warning(
            f"Performance alert: {alert.message}",
            level=alert.level,
            metric=alert.metric,
            current_value=alert.current_value,
            threshold=alert.threshold_value
        )
        
        # Send to registered handlers
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    
    Features:
    - Real-time metric collection
    - System resource monitoring
    - Application-specific metrics
    - Threshold-based alerting
    - Performance trend analysis
    """
    
    def __init__(self, config: ThresholdConfig, metric_collector: Optional[MetricCollector] = None):
        """
        Initialize performance monitor.
        
        Args:
            config: Threshold configuration
            metric_collector: Optional custom metric collector
        """
        self.config = config
        self.metric_collector = metric_collector or MetricCollector()
        self.alert_manager = AlertManager(config)
        
        self.logger = get_logger(__name__, component="performance_monitor")
        
        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._start_time = datetime.utcnow()
        
        # Performance tracking
        self._request_counts = defaultdict(int)
        self._error_counts = defaultdict(int)
        
    async def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start continuous performance monitoring."""
        if self._monitoring_active:
            self.logger.warning("Performance monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        self.logger.info(f"Performance monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self) -> None:
        """Stop continuous performance monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Check thresholds and generate alerts
                alerts = await self.alert_manager.check_thresholds(self.metric_collector)
                
                if alerts:
                    self.logger.info(f"Generated {len(alerts)} performance alerts")
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent()
            self.metric_collector.record_metric('cpu_usage', cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metric_collector.record_metric('memory_usage', memory.percent)
            self.metric_collector.record_metric('memory_used_gb', memory.used / (1024**3))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.metric_collector.record_metric('disk_usage', disk_percent)
            self.metric_collector.record_metric('disk_free_gb', disk.free / (1024**3))
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metric_collector.record_metric('network_bytes_sent', network.bytes_sent)
            self.metric_collector.record_metric('network_bytes_recv', network.bytes_recv)
            
            # Process metrics
            self.metric_collector.record_metric('active_processes', len(psutil.pids()))
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def record_request(self, endpoint: str, response_time: float, status_code: int,
                      user_id: Optional[str] = None) -> None:
        """Record API request metrics."""
        labels = {'endpoint': endpoint, 'status_code': str(status_code)}
        if user_id:
            labels['user_id'] = user_id
        
        # Record metrics
        self.metric_collector.record_metric('response_time', response_time, labels)
        self.metric_collector.record_metric('requests', 1, labels)
        
        # Track request counts
        self._request_counts[endpoint] += 1
        
        # Track errors
        if status_code >= 400:
            self.metric_collector.record_metric('errors', 1, labels)
            self._error_counts[endpoint] += 1
    
    def record_generation_metrics(self, query_type: str, generation_time: float,
                                tokens_used: int, sources_retrieved: int,
                                confidence_score: float) -> None:
        """Record LLM generation metrics."""
        labels = {'query_type': query_type}
        
        self.metric_collector.record_metric('generation_time', generation_time, labels)
        self.metric_collector.record_metric('tokens_used', tokens_used, labels)
        self.metric_collector.record_metric('sources_retrieved', sources_retrieved, labels)
        self.metric_collector.record_metric('confidence_score', confidence_score, labels)
    
    def record_search_metrics(self, search_type: str, search_time: float,
                            results_count: int, query_length: int) -> None:
        """Record search performance metrics."""
        labels = {'search_type': search_type}
        
        self.metric_collector.record_metric('search_time', search_time, labels)
        self.metric_collector.record_metric('search_results', results_count, labels)
        self.metric_collector.record_metric('query_length', query_length, labels)
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        # Request statistics
        response_time_stats = self.metric_collector.calculate_statistics('response_time', hours)
        request_rate = self.metric_collector.get_current_rate('requests', minutes=60)
        error_rate = self.metric_collector.get_current_rate('errors', minutes=60)
        
        # System statistics  
        cpu_stats = self.metric_collector.calculate_statistics('cpu_usage', hours)
        memory_stats = self.metric_collector.calculate_statistics('memory_usage', hours)
        
        # Generation statistics
        generation_stats = self.metric_collector.calculate_statistics('generation_time', hours)
        search_stats = self.metric_collector.calculate_statistics('search_time', hours)
        
        return {
            'uptime_seconds': uptime,
            'monitoring_active': self._monitoring_active,
            'performance': {
                'requests_per_hour': request_rate * 3600,
                'error_rate_per_hour': error_rate * 3600,
                'response_time': response_time_stats,
                'generation_time': generation_stats,
                'search_time': search_stats
            },
            'system_resources': {
                'cpu_usage': cpu_stats,
                'memory_usage': memory_stats
            },
            'endpoint_stats': {
                'total_requests': dict(self._request_counts),
                'total_errors': dict(self._error_counts)
            }
        } 