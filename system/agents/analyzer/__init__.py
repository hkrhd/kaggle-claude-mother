"""
深層分析エージェント

グランドマスター級技術調査・最新手法研究・実装可能性判定を担当する
技術特化エージェント。Kaggle優勝解法とarXiv最新論文の分析による
実装戦略策定システム。
"""

from .analyzer_agent import AnalyzerAgent
from .knowledge_base.grandmaster_patterns import GrandmasterPatterns
from .analyzers.technical_feasibility import TechnicalFeasibilityAnalyzer
from .collectors.kaggle_solutions import KaggleSolutionCollector
from .collectors.arxiv_papers import ArxivPaperCollector

__all__ = [
    "AnalyzerAgent",
    "GrandmasterPatterns", 
    "TechnicalFeasibilityAnalyzer",
    "KaggleSolutionCollector",
    "ArxivPaperCollector"
]

# バージョン情報
__version__ = "1.0.0"
__author__ = "Kaggle Claude Mother System"