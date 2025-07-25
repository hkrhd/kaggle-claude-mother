"""
Retrospective Agent システム

競技終了後の全活動振り返り・成功失敗分析・改善点特定・
知識蓄積による継続学習システム。
"""

from .retrospective_agent import (
    RetrospectiveAgent,
    RetrospectiveDepth,
    LearningCategory,
    ImprovementPriority,
    CompetitionSummary,
    TechniqueAnalysis,
    LearningInsight,
    RetrospectiveReport
)

__all__ = [
    "RetrospectiveAgent",
    "RetrospectiveDepth",
    "LearningCategory", 
    "ImprovementPriority",
    "CompetitionSummary",
    "TechniqueAnalysis",
    "LearningInsight",
    "RetrospectiveReport"
]

__version__ = "1.0.0"
__author__ = "Claude Mother System"
__description__ = "Post-competition learning and improvement system for Kaggle automation"