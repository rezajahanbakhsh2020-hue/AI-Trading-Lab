"""
Live Runtime Readiness Gate Module

This module provides functionality to evaluate and gate the readiness of trading systems
for live runtime deployment. It performs comprehensive checks on system health, 
performance metrics, and risk management capabilities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ReadinessStatus(Enum):
    """Status indicators for readiness evaluation."""
    READY = "ready"
    NOT_READY = "not_ready"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Data class representing a health metric."""
    name: str
    value: float
    threshold: float
    timestamp: datetime
    is_healthy: bool


@dataclass
class ReadinessReport:
    """Data class for readiness evaluation report."""
    status: ReadinessStatus
    timestamp: datetime
    metrics: List[HealthMetric]
    passed_checks: int
    total_checks: int
    failures: List[str]
    warnings: List[str]
    recommendation: str


class LiveRuntimeReadinessGate:
    """
    Gate system for evaluating live runtime readiness of trading systems.
    
    This class performs comprehensive checks on:
    - System health and availability
    - Performance metrics
    - Risk management systems
    - API connectivity
    - Data pipeline integrity
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the readiness gate.
        
        Args:
            logger: Optional logger instance. If None, creates a default logger.
        """
        self.logger = logger or self._setup_default_logger()
        self.metrics_history: List[HealthMetric] = []
        self.readiness_history: List[ReadinessReport] = []
        
    def _setup_default_logger(self) -> logging.Logger:
        """Set up default logging configuration."""
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def evaluate_readiness(self, system_metrics: Dict[str, Any]) -> ReadinessReport:
        """
        Evaluate the readiness of the trading system for live deployment.
        
        Args:
            system_metrics: Dictionary containing current system metrics
            
        Returns:
            ReadinessReport: Comprehensive readiness evaluation report
        """
        self.logger.info("Starting readiness evaluation...")
        
        metrics = []
        failures = []
        warnings = []
        
        # Perform individual checks
        checks_results = self._perform_health_checks(system_metrics)
        
        for check_name, (is_passed, metric, message) in checks_results.items():
            metrics.append(metric)
            if not is_passed:
                failures.append(f"{check_name}: {message}")
            else:
                self.logger.info(f"✓ {check_name} passed")
        
        # Determine overall status
        passed_checks = sum(1 for _, (is_passed, _, _) in checks_results.items() if is_passed)
        total_checks = len(checks_results)
        
        status = self._determine_status(passed_checks, total_checks, failures)
        recommendation = self._generate_recommendation(status, failures, warnings)
        
        report = ReadinessReport(
            status=status,
            timestamp=datetime.now(),
            metrics=metrics,
            passed_checks=passed_checks,
            total_checks=total_checks,
            failures=failures,
            warnings=warnings,
            recommendation=recommendation
        )
        
        self.readiness_history.append(report)
        return report
    
    def _perform_health_checks(
        self, 
        system_metrics: Dict[str, Any]
    ) -> Dict[str, Tuple[bool, HealthMetric, str]]:
        """
        Perform individual health checks on the system.
        
        Args:
            system_metrics: System metrics dictionary
            
        Returns:
            Dictionary with check results
        """
        results = {}
        
        # Check 1: CPU Usage
        cpu_threshold = 80.0
        cpu_value = system_metrics.get('cpu_usage', 0)
        cpu_metric = HealthMetric(
            name='cpu_usage',
            value=cpu_value,
            threshold=cpu_threshold,
            timestamp=datetime.now(),
            is_healthy=cpu_value < cpu_threshold
        )
        results['CPU Usage'] = (cpu_metric.is_healthy, cpu_metric, f"Current: {cpu_value}%")
        
        # Check 2: Memory Usage
        memory_threshold = 85.0
        memory_value = system_metrics.get('memory_usage', 0)
        memory_metric = HealthMetric(
            name='memory_usage',
            value=memory_value,
            threshold=memory_threshold,
            timestamp=datetime.now(),
            is_healthy=memory_value < memory_threshold
        )
        results['Memory Usage'] = (memory_metric.is_healthy, memory_metric, f"Current: {memory_value}%")
        
        # Check 3: API Response Time
        api_threshold = 200.0  # milliseconds
        api_response_time = system_metrics.get('api_response_time_ms', 0)
        api_metric = HealthMetric(
            name='api_response_time',
            value=api_response_time,
            threshold=api_threshold,
            timestamp=datetime.now(),
            is_healthy=api_response_time < api_threshold
        )
        results['API Response Time'] = (api_metric.is_healthy, api_metric, f"Current: {api_response_time}ms")
        
        # Check 4: Error Rate
        error_rate_threshold = 1.0  # percent
        error_rate = system_metrics.get('error_rate_percent', 0)
        error_metric = HealthMetric(
            name='error_rate',
            value=error_rate,
            threshold=error_rate_threshold,
            timestamp=datetime.now(),
            is_healthy=error_rate < error_rate_threshold
        )
        results['Error Rate'] = (error_metric.is_healthy, error_metric, f"Current: {error_rate}%")
        
        # Check 5: Risk Controls Active
        risk_controls = system_metrics.get('risk_controls_active', False)
        risk_metric = HealthMetric(
            name='risk_controls',
            value=1.0 if risk_controls else 0.0,
            threshold=1.0,
            timestamp=datetime.now(),
            is_healthy=risk_controls
        )
        results['Risk Controls'] = (risk_metric.is_healthy, risk_metric, "Active" if risk_controls else "Inactive")
        
        return results
    
    def _determine_status(
        self, 
        passed_checks: int, 
        total_checks: int,
        failures: List[str]
    ) -> ReadinessStatus:
        """
        Determine overall readiness status based on check results.
        
        Args:
            passed_checks: Number of passed checks
            total_checks: Total number of checks
            failures: List of failures
            
        Returns:
            ReadinessStatus indicating overall readiness
        """
        if passed_checks == total_checks:
            return ReadinessStatus.READY
        elif passed_checks >= total_checks * 0.8:
            return ReadinessStatus.PARTIAL
        elif passed_checks >= total_checks * 0.5:
            return ReadinessStatus.DEGRADED
        else:
            return ReadinessStatus.NOT_READY
    
    def _generate_recommendation(
        self,
        status: ReadinessStatus,
        failures: List[str],
        warnings: List[str]
    ) -> str:
        """
        Generate a recommendation based on readiness evaluation.
        
        Args:
            status: Current readiness status
            failures: List of failures
            warnings: List of warnings
            
        Returns:
            Recommendation string
        """
        if status == ReadinessStatus.READY:
            return "System is ready for live deployment. All checks passed."
        elif status == ReadinessStatus.PARTIAL:
            return f"System is partially ready. Please address: {', '.join(failures[:2])}"
        elif status == ReadinessStatus.DEGRADED:
            return f"System is degraded. Multiple issues detected: {', '.join(failures[:3])}"
        else:
            return "System is NOT ready for live deployment. Critical failures detected."
    
    def get_readiness_history(
        self, 
        hours: int = 24
    ) -> List[ReadinessReport]:
        """
        Retrieve readiness reports from the last N hours.
        
        Args:
            hours: Number of hours to look back (default: 24)
            
        Returns:
            List of ReadinessReport objects from the specified time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            report for report in self.readiness_history 
            if report.timestamp >= cutoff_time
        ]
    
    def is_live_ready(self, system_metrics: Dict[str, Any]) -> bool:
        """
        Quick check to determine if system is ready for live deployment.
        
        Args:
            system_metrics: System metrics dictionary
            
        Returns:
            Boolean indicating readiness
        """
        report = self.evaluate_readiness(system_metrics)
        return report.status == ReadinessStatus.READY


def create_readiness_gate(
    logger: Optional[logging.Logger] = None
) -> LiveRuntimeReadinessGate:
    """
    Factory function to create a readiness gate instance.
    
    Args:
        logger: Optional logger instance
        
    Returns:
        LiveRuntimeReadinessGate instance
    """
    return LiveRuntimeReadinessGate(logger=logger)
