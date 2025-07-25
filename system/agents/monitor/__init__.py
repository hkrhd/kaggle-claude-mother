"""
Monitor Agent システム

全エージェント活動・実験進捗・リソース使用状況・パフォーマンス指標の
リアルタイム監視とアラート生成を行う統合監視システム。
"""

from .monitor_agent import (
    MonitorAgent,
    MonitoringLevel,
    AlertSeverity,
    SystemHealth,
    PerformanceMetrics,
    CompetitionProgress,
    SystemAlert,
    MonitoringReport
)

__all__ = [
    "MonitorAgent",
    "MonitoringLevel",
    "AlertSeverity", 
    "SystemHealth",
    "PerformanceMetrics",
    "CompetitionProgress",
    "SystemAlert",
    "MonitoringReport"
]

__version__ = "1.0.0"
__author__ = "Claude Mother System"
__description__ = "Real-time system monitoring and alerting for Kaggle competition automation"