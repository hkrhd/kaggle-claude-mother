"""
統合動的コンペ管理システム

メダル確率算出・ポートフォリオ最適化・撤退判断・自動スケジューリングを統合した
動的コンペ管理システムのメインクラス。LLMベース競技選択エージェント統合。
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

# LLMベース競技選択エージェント
from ..agents.competition_selector.competition_selector_agent import (
    CompetitionSelectorAgent, SelectionStrategy
)


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
    """統合動的コンペ管理システム（LLMベース選択統合）"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        self.github_token = github_token
        self.repo_name = repo_name
        
        # コンポーネント初期化
        self.medal_calculator = MedalProbabilityCalculator()
        self.portfolio_optimizer = CompetitionPortfolioOptimizer()
        self.withdrawal_decision_maker = WithdrawalDecisionMaker()
        self.scheduler = DynamicScheduler()
        
        # LLMベース競技選択エージェント
        self.competition_selector = CompetitionSelectorAgent(
            github_token=github_token,
            repo_name=repo_name
        )
        
        # 状態管理
        self.active_competitions: List[ActiveCompetition] = []
        self.last_scan_time: Optional[datetime] = None
        self.last_llm_selection_time: Optional[datetime] = None
        
        self.logger.info("DynamicCompetitionManager初期化完了（LLM選択統合）")
    
    async def scan_new_competitions(self) -> List[Dict[str, Any]]:
        """新競技スキャン・LLM選択統合実行（週2回実行）"""
        
        self.logger.info("新競技スキャン・LLM選択開始")
        
        try:
            # 模擬的な新競技データ（実際の実装では Kaggle API等から取得）
            mock_competitions = [
                {
                    "id": "tabular-playground-series-apr-2024",
                    "name": "Tabular Playground Series - Apr 2024",
                    "type": "tabular",
                    "deadline": (datetime.utcnow() + timedelta(days=45)).isoformat(),
                    "participants": 2150,
                    "prize_amount": 25000,  # Featured competition with prize
                    "competition_category": "featured",
                    "awards_medals": True,
                    "url": "https://www.kaggle.com/competitions/tabular-playground-series-apr-2024"
                },
                {
                    "id": "plant-pathology-2024-fgvc11",
                    "name": "Plant Pathology 2024 - FGVC11",
                    "type": "computer_vision", 
                    "deadline": (datetime.utcnow() + timedelta(days=60)).isoformat(),
                    "participants": 1800,
                    "prize_amount": 15000,  # Research competition with prize
                    "competition_category": "research",
                    "awards_medals": True,
                    "url": "https://www.kaggle.com/competitions/plant-pathology-2024-fgvc11"
                },
                {
                    "id": "llm-classification-finetuning",
                    "name": "LLM Classification Finetuning",
                    "type": "nlp",
                    "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(), 
                    "participants": 3500,
                    "prize_amount": 100000,  # High-value Featured competition
                    "competition_category": "featured",
                    "awards_medals": True,
                    "url": "https://www.kaggle.com/competitions/llm-classification-finetuning"
                },
                # Knowledge competitions (should be filtered out) - 削除済み
            ]
            
            # **段階1**: アワードポイント・メダル獲得可能コンペフィルタ（従来ロジック）
            eligible_competitions = []
            for comp in mock_competitions:
                if self._is_medal_eligible_competition(comp):
                    # 基本メダル確率算出
                    medal_prob = await self._calculate_medal_probability(comp)
                    comp["medal_probability"] = medal_prob
                    
                    # 最低閾値フィルタ（確率15%以上）
                    if medal_prob >= 0.15:
                        eligible_competitions.append(comp)
                        self.logger.info(f"🔍 一次フィルタ通過: {comp['name']} (確率: {medal_prob:.1%})")
                    else:
                        self.logger.info(f"📊 除外 (確率不足): {comp['name']} (確率: {medal_prob:.1%})")
                else:
                    self.logger.info(f"❌ 除外 (アワード・メダル対象外): {comp['name']}")
            
            # **段階2**: LLMベース戦略的選択実行
            selected_competitions = []
            if eligible_competitions:
                # LLM選択頻度制御（12時間間隔）
                current_time = datetime.utcnow()
                should_perform_llm_selection = (
                    self.last_llm_selection_time is None or 
                    (current_time - self.last_llm_selection_time).total_seconds() > 12 * 3600
                )
                
                if should_perform_llm_selection:
                    self.logger.info(f"🧠 LLM戦略的選択実行: {len(eligible_competitions)}競技評価")
                    
                    # 戦略的選択実行
                    selection_decision = await self.competition_selector.evaluate_available_competitions(
                        available_competitions=eligible_competitions,
                        strategy=SelectionStrategy.BALANCED  # デフォルト戦略
                    )
                    
                    # 選択結果を従来形式に変換
                    for comp_profile in selection_decision.selected_competitions:
                        # 元の競技データに LLM 評価結果をマージ
                        for orig_comp in eligible_competitions:
                            if orig_comp["id"] == comp_profile.id:
                                orig_comp["llm_selected"] = True
                                orig_comp["llm_recommendation"] = comp_profile.llm_recommendation
                                orig_comp["llm_score"] = comp_profile.llm_score
                                orig_comp["strategic_value"] = comp_profile.strategic_value
                                selected_competitions.append(orig_comp)
                                break
                    
                    self.last_llm_selection_time = current_time
                    self.logger.info(f"🎯 LLM選択完了: {len(selected_competitions)}競技選択")
                
                else:
                    # LLM選択間隔未達成時は既存ポートフォリオを維持
                    self.logger.info("🕐 LLM選択間隔未達成 - 既存ポートフォリオ維持")
                    selected_competitions = eligible_competitions  # フォールバック
            
            else:
                self.logger.info("🔍 一次フィルタ通過競技なし")
            
            self.last_scan_time = datetime.utcnow()
            self.logger.info(
                f"新競技スキャン・LLM選択完了: {len(selected_competitions)}競技選択 "
                f"(/{len(mock_competitions)}件スキャン, {len(eligible_competitions)}件一次通過)"
            )
            
            return selected_competitions
            
        except Exception as e:
            self.logger.error(f"新競技スキャン・LLM選択エラー: {e}")
            return []
    
    def _is_medal_eligible_competition(self, competition: Dict[str, Any]) -> bool:
        """アワードポイント・メダル獲得可能コンペ判定"""
        
        # 1. 明示的にメダル対象外の場合
        if competition.get("awards_medals", True) is False:
            return False
        
        # 2. カテゴリによる除外
        excluded_categories = {
            "getting-started",    # Getting Started competitions
            "playground",         # Playground competitions (一部例外あり)
            "inclass",           # InClass competitions
            "knowledge"          # Knowledge competitions
        }
        
        category = competition.get("competition_category", "").lower()
        if category in excluded_categories:
            # Playgroundでも賞金付きは例外的に許可
            if category == "playground" and competition.get("prize_amount", 0) > 0:
                return True
            return False
        
        # 3. 賞金額による判定
        prize_amount = competition.get("prize_amount", 0)
        
        # Featured競技: 通常賞金付きでメダル対象
        if category == "featured":
            return prize_amount > 0
        
        # Research競技: 賞金付きのみメダル対象  
        if category == "research":
            return prize_amount > 0
        
        # その他のカテゴリ: 基本的に賞金があればメダル対象
        return prize_amount > 0
    
    async def _calculate_medal_probability(self, competition: Dict[str, Any]) -> float:
        """競技のメダル確率算出"""
        
        try:
            # CompetitionTypeマッピング
            type_mapping = {
                "tabular": CompetitionType.TABULAR,
                "computer_vision": CompetitionType.COMPUTER_VISION,
                "nlp": CompetitionType.NLP,
                "time_series": CompetitionType.TIME_SERIES,
                "audio": CompetitionType.AUDIO,
                "graph": CompetitionType.GRAPH
            }
            
            comp_type = type_mapping.get(competition["type"], CompetitionType.TABULAR)
            
            # PrizeType判定: 賞金がある場合のみMONETARY、それ以外は対象外
            prize_amount = competition["prize_amount"]
            if prize_amount > 0:
                prize_type = PrizeType.MONETARY
            else:
                # 賞金なしコンペは本来フィルタで除外されているはずだが、安全のため
                self.logger.warning(f"賞金なしコンペが算出対象に: {competition['name']}")
                prize_type = PrizeType.KNOWLEDGE
            
            # CompetitionDataオブジェクト作成
            comp_data = CompetitionData(
                competition_id=competition["id"],
                title=competition["name"],
                competition_type=comp_type,
                participant_count=competition["participants"],
                total_prize=prize_amount,
                prize_type=prize_type,
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
    
    async def get_llm_selection_status(self) -> Dict[str, Any]:
        """LLM選択システム状態取得"""
        
        try:
            # 競技選択エージェント状態取得
            selector_status = await self.competition_selector.get_current_portfolio_status()
            
            return {
                "llm_selection_agent_id": self.competition_selector.agent_id,
                "last_llm_selection_time": self.last_llm_selection_time.isoformat() if self.last_llm_selection_time else None,
                "selection_decisions_made": len(self.competition_selector.selection_history),
                "current_portfolio_size": selector_status.get("active_competitions_count", 0),
                "portfolio_medal_probability": selector_status.get("total_medal_probability", 0.0),
                "gpu_budget_utilization": selector_status.get("gpu_budget_utilization", 0.0),
                "last_selection_strategy": selector_status.get("last_selection_strategy"),
                "selection_performance": self.competition_selector.get_selection_performance_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"LLM選択状態取得エラー: {e}")
            return {
                "error": str(e),
                "llm_selection_enabled": False
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得（LLM選択統合版）"""
        
        return {
            "active_competitions_count": len(self.active_competitions),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "last_llm_selection_time": self.last_llm_selection_time.isoformat() if self.last_llm_selection_time else None,
            "total_medal_probability": sum(comp.medal_probability for comp in self.active_competitions),
            "llm_selection_enabled": True,
            "system_health": "healthy"
        }