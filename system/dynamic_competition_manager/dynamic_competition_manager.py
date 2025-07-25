"""
統合動的コンペ管理システム

メダル確率算出・ポートフォリオ最適化・撤退判断・自動スケジューリングを統合した
動的コンペ管理システムのメインクラス。
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .medal_probability_calculators.medal_probability_calculator import (
    MedalProbabilityCalculator, CompetitionData, CompetitionType, PrizeType
)
from .portfolio_optimizers.competition_portfolio_optimizer import (
    CompetitionPortfolioOptimizer, PortfolioStrategy
)
from .decision_engines.withdrawal_decision_maker import WithdrawalDecisionMaker
from .schedulers.dynamic_scheduler import DynamicScheduler


@dataclass
class ActiveCompetition:
    """アクティブな競技情報"""
    id: str
    name: str
    type: str
    deadline: datetime
    participants: int
    prize_amount: float
    medal_probability: float
    start_date: datetime
    current_rank: Optional[int] = None
    submissions_made: int = 0
    estimated_work_hours: float = 0.0


class DynamicCompetitionManager:
    """統合動的コンペ管理システム"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        self.github_token = github_token
        self.repo_name = repo_name
        
        # コンポーネント初期化
        self.medal_calculator = MedalProbabilityCalculator()
        self.portfolio_optimizer = CompetitionPortfolioOptimizer()
        self.withdrawal_decision_maker = WithdrawalDecisionMaker()
        self.scheduler = DynamicScheduler()
        
        # 状態管理
        self.active_competitions: List[ActiveCompetition] = []
        self.last_scan_time: Optional[datetime] = None
        
        self.logger.info("DynamicCompetitionManager初期化完了")
    
    async def scan_new_competitions(self) -> List[Dict[str, Any]]:
        """新競技スキャン（週2回実行）"""
        
        self.logger.info("新競技スキャン開始")
        
        try:
            # 模擬的な新競技データ（実際の実装では Kaggle API等から取得）
            mock_competitions = [
                {
                    "id": "house-prices-advanced",
                    "name": "House Prices - Advanced Regression Techniques",
                    "type": "tabular",
                    "deadline": (datetime.utcnow() + timedelta(days=45)).isoformat(),
                    "participants": 890,
                    "prize_amount": 0,  # Knowledge competition
                    "url": "https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques"
                },
                {
                    "id": "nlp-getting-started", 
                    "name": "Natural Language Processing with Disaster Tweets",
                    "type": "nlp",
                    "deadline": (datetime.utcnow() + timedelta(days=60)).isoformat(),
                    "participants": 1250,
                    "prize_amount": 0,
                    "url": "https://www.kaggle.com/competitions/nlp-getting-started"
                }
            ]
            
            # メダル確率算出
            evaluated_competitions = []
            for comp in mock_competitions:
                medal_prob = await self._calculate_medal_probability(comp)
                comp["medal_probability"] = medal_prob
                
                # 閾値フィルタ（確率30%以上）
                if medal_prob >= 0.3:
                    evaluated_competitions.append(comp)
            
            self.last_scan_time = datetime.utcnow()
            self.logger.info(f"新競技スキャン完了: {len(evaluated_competitions)}件が候補")
            
            return evaluated_competitions
            
        except Exception as e:
            self.logger.error(f"新競技スキャンエラー: {e}")
            return []
    
    async def _calculate_medal_probability(self, competition: Dict[str, Any]) -> float:
        """競技のメダル確率算出"""
        
        try:
            # CompetitionDataオブジェクト作成（タイタニック向け実データ構造）
            comp_data = CompetitionData(
                competition_id=competition["id"],
                title=competition["name"],
                competition_type=CompetitionType.TABULAR if competition["type"] == "tabular" else CompetitionType.NLP,
                participant_count=competition["participants"],
                total_prize=competition["prize_amount"],
                prize_type=PrizeType.KNOWLEDGE if competition["prize_amount"] == 0 else PrizeType.MONETARY,
                days_remaining=(datetime.fromisoformat(competition["deadline"]).replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days,
                data_characteristics={
                    "dataset_size": "medium",
                    "feature_count": 10 if competition["type"] == "tabular" else 1000,
                    "data_quality": "high"
                },
                skill_requirements=["feature_engineering", "ensemble_methods"] if competition["type"] == "tabular" else ["text_processing", "transformers"],
                leaderboard_competition=min(1.0, competition["participants"] / 1000)  # 参加者数ベースの競争度
            )
            
            # 確率算出実行
            result = await self.medal_calculator.calculate_medal_probability(comp_data)
            return result.overall_medal_probability
            
        except Exception as e:
            self.logger.error(f"メダル確率算出エラー: {e}")
            return 0.0
    
    async def get_active_competitions(self) -> List[Dict[str, Any]]:
        """アクティブ競技一覧取得"""
        
        return [
            {
                "id": comp.id,
                "name": comp.name,
                "type": comp.type,
                "deadline": comp.deadline.isoformat(),
                "medal_probability": comp.medal_probability,
                "current_rank": comp.current_rank,
                "submissions_made": comp.submissions_made
            }
            for comp in self.active_competitions
        ]
    
    async def analyze_portfolio(self) -> Dict[str, Any]:
        """競技ポートフォリオ分析"""
        
        try:
            portfolio_items = []
            
            for comp in self.active_competitions:
                portfolio_items.append({
                    "competition_id": comp.id,
                    "medal_probability": comp.medal_probability,
                    "expected_work_hours": comp.estimated_work_hours,
                    "deadline": comp.deadline,
                    "current_investment": comp.submissions_made * 2.0  # 模擬的な投資時間
                })
            
            # ポートフォリオ最適化実行
            optimization_result = await self.portfolio_optimizer.optimize_portfolio(
                available_competitions=portfolio_items,
                strategy=PortfolioStrategy.BALANCED
            )
            
            return {
                "active_competitions": len(self.active_competitions),
                "total_expected_probability": sum(comp.medal_probability for comp in self.active_competitions),
                "recommendations": optimization_result.recommended_actions if hasattr(optimization_result, 'recommended_actions') else []
            }
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ分析エラー: {e}")
            return {
                "active_competitions": len(self.active_competitions),
                "recommendations": []
            }
    
    async def update_competition_status(self, competition_id: str, status: str) -> bool:
        """競技状態更新"""
        
        try:
            for comp in self.active_competitions:
                if comp.id == competition_id:
                    self.logger.info(f"競技状態更新: {competition_id} -> {status}")
                    return True
            
            self.logger.warning(f"競技が見つかりません: {competition_id}")
            return False
            
        except Exception as e:
            self.logger.error(f"競技状態更新エラー: {e}")
            return False
    
    async def should_withdraw_from_competition(self, competition_id: str) -> Dict[str, Any]:
        """撤退判断分析"""
        
        try:
            target_comp = None
            for comp in self.active_competitions:
                if comp.id == competition_id:
                    target_comp = comp
                    break
            
            if not target_comp:
                return {"should_withdraw": False, "reason": "Competition not found"}
            
            # 撤退判断実行
            analysis = await self.withdrawal_decision_maker.analyze_withdrawal_decision({
                "competition_id": competition_id,
                "current_rank": target_comp.current_rank or 500,
                "days_remaining": (target_comp.deadline - datetime.utcnow()).days,
                "medal_probability": target_comp.medal_probability,
                "investment_hours": target_comp.submissions_made * 2.0
            })
            
            return {
                "should_withdraw": analysis.should_withdraw if hasattr(analysis, 'should_withdraw') else False,
                "confidence": analysis.confidence if hasattr(analysis, 'confidence') else 0.5,
                "reasons": analysis.reasons if hasattr(analysis, 'reasons') else []
            }
            
        except Exception as e:
            self.logger.error(f"撤退判断エラー: {e}")
            return {"should_withdraw": False, "reason": f"Error: {e}"}
    
    async def add_competition(self, competition_data: Dict[str, Any]) -> bool:
        """新競技追加"""
        
        try:
            new_comp = ActiveCompetition(
                id=competition_data["id"],
                name=competition_data["name"],
                type=competition_data["type"],
                deadline=datetime.fromisoformat(competition_data["deadline"]),
                participants=competition_data.get("participants", 1000),
                prize_amount=competition_data.get("prize_amount", 0.0),
                medal_probability=competition_data.get("medal_probability", 0.5),
                start_date=datetime.utcnow()
            )
            
            self.active_competitions.append(new_comp)
            self.logger.info(f"新競技追加: {new_comp.name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"競技追加エラー: {e}")
            return False
    
    async def remove_competition(self, competition_id: str) -> bool:
        """競技削除（撤退）"""
        
        try:
            for i, comp in enumerate(self.active_competitions):
                if comp.id == competition_id:
                    removed_comp = self.active_competitions.pop(i)
                    self.logger.info(f"競技削除: {removed_comp.name}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"競技削除エラー: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        
        return {
            "active_competitions_count": len(self.active_competitions),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "total_medal_probability": sum(comp.medal_probability for comp in self.active_competitions),
            "system_health": "healthy"
        }