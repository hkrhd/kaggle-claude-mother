"""
動的コンペ管理システム

週2回の動的最適化による最大3コンペ同時進行管理システム。
メダル確率算出・ポートフォリオ最適化・撤退判断・自動スケジューリングを統合。
"""

__version__ = "0.1.0"

# メダル確率算出
from .medal_probability_calculators.medal_probability_calculator import (
    MedalProbabilityCalculator,
    MedalProbabilityResult,
    CompetitionData,
    CompetitionType,
    PrizeType,
    ExpertiseProfile
)

# ポートフォリオ最適化
from .portfolio_optimizers.competition_portfolio_optimizer import (
    CompetitionPortfolioOptimizer,
    PortfolioOptimizationResult,
    CompetitionPortfolioItem,
    PortfolioStrategy,
    ResourceConstraints
)

# 撤退・入れ替え判断
from .decision_engines.withdrawal_decision_maker import (
    WithdrawalDecisionMaker,
    WithdrawalAnalysis,
    CompetitionStatus,
    CompetitionPhase,
    WithdrawalReason,
    ReplacementOpportunity
)

# 動的スケジューラー
from .schedulers.dynamic_scheduler import (
    DynamicScheduler,
    ScheduledTask,
    ScheduleType,
    ExecutionResult,
    ExecutionStatus
)

# 統合管理システム
from .dynamic_competition_manager import (
    DynamicCompetitionManager,
    ActiveCompetition
)

__all__ = [
    # メダル確率算出
    "MedalProbabilityCalculator",
    "MedalProbabilityResult", 
    "CompetitionData",
    "CompetitionType",
    "PrizeType",
    "ExpertiseProfile",
    
    # ポートフォリオ最適化
    "CompetitionPortfolioOptimizer",
    "PortfolioOptimizationResult",
    "CompetitionPortfolioItem", 
    "PortfolioStrategy",
    "ResourceConstraints",
    
    # 撤退・入れ替え判断
    "WithdrawalDecisionMaker",
    "WithdrawalAnalysis",
    "CompetitionStatus",
    "CompetitionPhase", 
    "WithdrawalReason",
    "ReplacementOpportunity",
    
    # 動的スケジューラー
    "DynamicScheduler",
    "ScheduledTask",
    "ScheduleType",
    "ExecutionResult",
    "ExecutionStatus",
    
    # 統合管理システム
    "DynamicCompetitionManager",
    "ActiveCompetition"
]