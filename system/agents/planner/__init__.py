"""
戦略プランニングエージェント

メダル確率算出・戦略的コンペ選択・撤退判断を担当する核心エージェント。
GitHub Issue連携による他エージェントとの安全な協調動作を実現。
"""

__version__ = "0.1.0"

from .planner_agent import PlannerAgent
from .models.competition import CompetitionInfo, CompetitionStatus, AnalysisResult
from .models.probability import MedalProbability, ProbabilityFactors
from .calculators.medal_probability import MedalProbabilityCalculator
from .strategies.selection_strategy import CompetitionSelectionStrategy
from .strategies.withdrawal_strategy import WithdrawalStrategy

__all__ = [
    "PlannerAgent",
    "CompetitionInfo",
    "CompetitionStatus", 
    "AnalysisResult",
    "MedalProbability",
    "ProbabilityFactors",
    "MedalProbabilityCalculator",
    "CompetitionSelectionStrategy",
    "WithdrawalStrategy"
]