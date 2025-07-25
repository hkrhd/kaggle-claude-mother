"""
Orchestrator システム

全エージェント統合オーケストレーション・自動競技実行・
システム管理機能を提供するメイン統合システム。
"""

from .master_orchestrator import (
    MasterOrchestrator,
    SystemPhase,
    OrchestrationMode,
    CompetitionContext,
    OrchestrationResult
)

__all__ = [
    "MasterOrchestrator",
    "SystemPhase",
    "OrchestrationMode",
    "CompetitionContext", 
    "OrchestrationResult"
]

__version__ = "1.0.0"
__author__ = "Claude Mother System"
__description__ = "Master orchestration system for complete Kaggle competition automation"