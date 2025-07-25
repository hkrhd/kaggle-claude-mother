"""
撤退戦略システム

リアルタイム分析による戦略的撤退判断・リソース最適化。
plan_planner.md の撤退判断アルゴリズム準拠の損切りルール実装。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    import numpy as np
except ImportError:
    # テスト環境用モック実装
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            return (sum((x - mean) ** 2 for x in data) / len(data)) ** 0.5
        @staticmethod
        def exp(x):
            return math.exp(x)
        @staticmethod
        def log(x):
            return math.log(x)
        @staticmethod
        def argmax(data):
            return data.index(max(data)) if data else 0
        @staticmethod
        def min(data):
            return min(data) if data else 0
        @staticmethod
        def max(data):
            return max(data) if data else 0
    np = MockNumpy()

from ..models.competition import CompetitionInfo, CompetitionStatus, CompetitionPhase
from ..models.probability import MedalProbability
from ..calculators.medal_probability import MedalProbabilityCalculator


class WithdrawalReason(Enum):
    """撤退理由"""
    LOW_PROBABILITY = "low_probability"         # 確率低下
    POOR_RANKING = "poor_ranking"              # 順位不振  
    TIME_CONSTRAINT = "time_constraint"         # 時間制約
    BETTER_OPPORTUNITY = "better_opportunity"   # より良い機会
    RESOURCE_SHORTAGE = "resource_shortage"     # リソース不足
    STRATEGIC_PIVOT = "strategic_pivot"         # 戦略転換
    FORCE_MAJEURE = "force_majeure"            # 不可抗力


class WithdrawalUrgency(Enum):
    """撤退緊急度"""
    IMMEDIATE = "immediate"        # 即座撤退
    URGENT = "urgent"             # 緊急撤退
    MODERATE = "moderate"         # 中程度
    LOW = "low"                   # 低優先度
    MONITOR = "monitor"           # 監視継続


@dataclass
class WithdrawalAnalysis:
    """撤退分析結果"""
    competition_info: CompetitionInfo
    current_status: Optional[CompetitionStatus]
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 撤退判定
    should_withdraw: bool = False
    withdrawal_urgency: WithdrawalUrgency = WithdrawalUrgency.MONITOR
    primary_reason: Optional[WithdrawalReason] = None
    secondary_reasons: List[WithdrawalReason] = field(default_factory=list)
    
    # スコア分析
    withdrawal_score: float = 0.0     # 0-1: 高いほど撤退推奨
    continuation_score: float = 0.0   # 0-1: 高いほど継続推奨
    opportunity_cost: float = 0.0     # 機会コスト
    
    # 要因分析
    ranking_factor: float = 0.0       # 順位要因
    probability_factor: float = 0.0   # 確率要因
    time_factor: float = 0.0          # 時間要因
    resource_factor: float = 0.0      # リソース要因
    opportunity_factor: float = 0.0   # 機会要因
    
    # 予測・推奨
    predicted_final_rank: Optional[int] = None
    minimum_viable_rank: Optional[int] = None
    recommended_action: str = ""
    action_timeline: str = ""
    
    # 代替案
    alternative_competitions: List[str] = field(default_factory=list)
    reallocation_suggestions: List[str] = field(default_factory=list)
    
    # メタデータ
    confidence_level: float = 0.0
    analysis_version: str = "1.0"


@dataclass
class WithdrawalConditions:
    """撤退条件設定"""
    
    # 確率閾値
    min_medal_probability: float = 0.15      # 最低メダル確率
    probability_decline_rate: float = 0.5    # 確率低下率閾値
    
    # 順位閾値
    max_rank_percentile: float = 0.7         # 最大順位パーセンタイル
    stagnation_days: int = 14                # 順位停滞日数
    
    # 時間制約
    min_remaining_days: int = 7              # 最少残り日数
    max_time_investment: float = 30.0        # 最大時間投資（時間）
    
    # 機会コスト
    opportunity_cost_threshold: float = 0.3   # 機会コスト閾値
    better_option_probability_gap: float = 0.2  # より良い選択肢との確率差
    
    # リソース制約
    resource_utilization_max: float = 0.9    # 最大リソース使用率
    concurrent_competition_limit: int = 3     # 同時参加上限


class WithdrawalStrategy:
    """撤退戦略システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.probability_calculator = MedalProbabilityCalculator()
        
        # デフォルト撤退条件
        self.default_conditions = WithdrawalConditions()
        
        # 段階別重み設定
        self.phase_weights = {
            CompetitionPhase.UPCOMING: {
                "probability": 0.4, "ranking": 0.1, "time": 0.2, 
                "resource": 0.15, "opportunity": 0.15
            },
            CompetitionPhase.ACTIVE: {
                "probability": 0.3, "ranking": 0.3, "time": 0.15,
                "resource": 0.1, "opportunity": 0.15
            },
            CompetitionPhase.SUBMISSION_ONLY: {
                "probability": 0.25, "ranking": 0.45, "time": 0.2,
                "resource": 0.05, "opportunity": 0.05
            },
            CompetitionPhase.COMPLETED: {
                "probability": 0.0, "ranking": 0.0, "time": 0.0,
                "resource": 0.0, "opportunity": 0.0
            }
        }
    
    async def analyze_withdrawal_decision(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[CompetitionStatus] = None,
        available_alternatives: Optional[List[CompetitionInfo]] = None,
        custom_conditions: Optional[WithdrawalConditions] = None
    ) -> WithdrawalAnalysis:
        """撤退判断分析実行"""
        
        self.logger.info(f"撤退判断分析開始: {competition_info.title}")
        
        try:
            conditions = custom_conditions or self.default_conditions
            
            # 現在のメダル確率算出
            current_probability = await self.probability_calculator.calculate_medal_probability(
                competition_info
            )
            
            # 各要因スコア計算
            factors = await self._calculate_withdrawal_factors(
                competition_info, current_status, current_probability, conditions
            )
            
            # 段階別重み付け統合
            phase = self._determine_competition_phase(competition_info, current_status)
            weighted_score = await self._calculate_weighted_withdrawal_score(factors, phase)
            
            # 撤退判定
            withdrawal_decision = await self._make_withdrawal_decision(
                weighted_score, factors, conditions
            )
            
            # 機会コスト分析
            opportunity_analysis = await self._analyze_opportunity_cost(
                competition_info, current_probability, available_alternatives or []
            )
            
            # 代替案分析
            alternatives = await self._analyze_alternatives(
                competition_info, available_alternatives or []
            )
            
            # 最終推奨アクション
            recommended_action = await self._generate_withdrawal_recommendation(
                withdrawal_decision, factors, opportunity_analysis
            )
            
            # 分析結果構築
            analysis = WithdrawalAnalysis(
                competition_info=competition_info,
                current_status=current_status,
                should_withdraw=withdrawal_decision["should_withdraw"],
                withdrawal_urgency=withdrawal_decision["urgency"],
                primary_reason=withdrawal_decision["primary_reason"],
                secondary_reasons=withdrawal_decision["secondary_reasons"],
                withdrawal_score=weighted_score["withdrawal_score"],
                continuation_score=1.0 - weighted_score["withdrawal_score"],
                opportunity_cost=opportunity_analysis["cost"],
                ranking_factor=factors["ranking_factor"],
                probability_factor=factors["probability_factor"],
                time_factor=factors["time_factor"],
                resource_factor=factors["resource_factor"],
                opportunity_factor=factors["opportunity_factor"],
                predicted_final_rank=await self._predict_final_rank(
                    competition_info, current_status, current_probability
                ),
                minimum_viable_rank=await self._calculate_minimum_viable_rank(competition_info),
                recommended_action=recommended_action["action"],
                action_timeline=recommended_action["timeline"],
                alternative_competitions=alternatives["competitions"],
                reallocation_suggestions=alternatives["suggestions"],
                confidence_level=await self._calculate_confidence_level(factors),
            )
            
            self.logger.info(f"撤退判断分析完了: {analysis.should_withdraw} ({competition_info.title})")
            return analysis
            
        except Exception as e:
            self.logger.error(f"撤退判断分析失敗: {e}")
            raise
    
    async def monitor_withdrawal_triggers(
        self,
        active_competitions: List[Tuple[CompetitionInfo, CompetitionStatus]],
        conditions: Optional[WithdrawalConditions] = None
    ) -> Dict[str, WithdrawalAnalysis]:
        """撤退トリガー監視"""
        
        self.logger.info(f"撤退トリガー監視開始: {len(active_competitions)}件")
        
        withdrawal_analyses = {}
        conditions = conditions or self.default_conditions
        
        for comp_info, status in active_competitions:
            try:
                analysis = await self.analyze_withdrawal_decision(
                    comp_info, status, None, conditions
                )
                
                # 緊急撤退の場合のみ記録
                if analysis.withdrawal_urgency in [WithdrawalUrgency.IMMEDIATE, WithdrawalUrgency.URGENT]:
                    withdrawal_analyses[comp_info.competition_id] = analysis
                    
            except Exception as e:
                self.logger.warning(f"撤退監視失敗 ({comp_info.competition_id}): {e}")
        
        self.logger.info(f"撤退トリガー監視完了: {len(withdrawal_analyses)}件の緊急案件")
        return withdrawal_analyses
    
    async def _calculate_withdrawal_factors(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[CompetitionStatus],
        current_probability: MedalProbability,
        conditions: WithdrawalConditions
    ) -> Dict[str, float]:
        """撤退要因計算"""
        
        factors = {}
        
        # 確率要因 (低いほど撤退スコア高)
        prob = current_probability.overall_probability
        if prob < conditions.min_medal_probability:
            factors["probability_factor"] = 1.0  # 確実に撤退推奨
        elif prob < 0.3:
            factors["probability_factor"] = 0.8
        elif prob < 0.5:
            factors["probability_factor"] = 0.5
        else:
            factors["probability_factor"] = 0.2  # 高確率は継続
        
        # 順位要因
        factors["ranking_factor"] = await self._calculate_ranking_factor(
            current_status, conditions
        )
        
        # 時間要因
        factors["time_factor"] = await self._calculate_time_factor(
            competition_info, current_status, conditions
        )
        
        # リソース要因
        factors["resource_factor"] = await self._calculate_resource_factor(
            competition_info, current_status, conditions
        )
        
        # 機会要因
        factors["opportunity_factor"] = await self._calculate_opportunity_factor(
            competition_info, current_probability
        )
        
        return factors
    
    async def _calculate_ranking_factor(
        self,
        current_status: Optional[CompetitionStatus],
        conditions: WithdrawalConditions
    ) -> float:
        """順位要因計算"""
        
        if not current_status or current_status.current_rank is None:
            return 0.4  # 情報不足時のデフォルト
        
        # 順位パーセンタイル計算
        if current_status.current_rank > 0 and current_status.competition_info.participant_count > 0:
            percentile = current_status.current_rank / current_status.competition_info.participant_count
        else:
            percentile = 0.5  # デフォルト
        
        # 順位による撤退スコア
        if percentile > conditions.max_rank_percentile:
            ranking_score = 0.9  # 下位は撤退推奨
        elif percentile > 0.5:
            ranking_score = 0.6  # 中位
        elif percentile > 0.2:
            ranking_score = 0.3  # 上位
        else:
            ranking_score = 0.1  # 最上位は継続
        
        # 停滞期間による調整
        if (current_status.last_improvement_date and 
            (datetime.utcnow() - current_status.last_improvement_date).days > conditions.stagnation_days):
            ranking_score += 0.2  # 停滞ペナルティ
        
        return min(1.0, ranking_score)
    
    async def _calculate_time_factor(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[CompetitionStatus],
        conditions: WithdrawalConditions
    ) -> float:
        """時間要因計算"""
        
        days_remaining = competition_info.days_remaining
        
        # 残り時間不足
        if days_remaining < conditions.min_remaining_days:
            time_score = 0.9  # 時間切れ近し
        elif days_remaining < 14:
            time_score = 0.6  # 時間不足
        elif days_remaining < 30:
            time_score = 0.3  # やや時間不足
        else:
            time_score = 0.1  # 十分な時間
        
        # 投資時間による調整
        if current_status and current_status.time_invested_hours > conditions.max_time_investment:
            time_score += 0.3  # 過剰投資ペナルティ
        
        return min(1.0, time_score)
    
    async def _calculate_resource_factor(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[CompetitionStatus],
        conditions: WithdrawalConditions
    ) -> float:
        """リソース要因計算"""
        
        # 基本リソース負荷
        base_load = 0.3  # デフォルト値
        
        # コンペ種別による負荷
        type_loads = {
            "tabular": 0.2,
            "computer_vision": 0.6,
            "nlp": 0.5,
            "multi_modal": 0.8,
            "reinforcement_learning": 0.9
        }
        
        comp_type = competition_info.competition_type.value
        resource_load = type_loads.get(comp_type, base_load)
        
        # データサイズによる調整
        if competition_info.data_size_gb > 5.0:
            resource_load += 0.2
        elif competition_info.data_size_gb > 10.0:
            resource_load += 0.4
        
        # 現在の投資状況
        if current_status:
            # 成果に対する投資効率
            if current_status.time_invested_hours > 20 and current_status.experiments_completed < 5:
                resource_load += 0.3  # 低効率投資
        
        return min(1.0, resource_load)
    
    async def _calculate_opportunity_factor(
        self,
        competition_info: CompetitionInfo,
        current_probability: MedalProbability
    ) -> float:
        """機会要因計算"""
        
        # 現在のコンペの期待価値
        current_value = current_probability.expected_medal_value
        
        # 機会コストの推定（簡易実装）
        # 実際の実装では利用可能な代替コンペとの比較が必要
        
        if current_value < 2.0:  # 低い期待価値
            return 0.8  # 機会コスト高
        elif current_value < 4.0:
            return 0.5  # 中程度の機会コスト
        else:
            return 0.2  # 機会コスト低
    
    async def _calculate_weighted_withdrawal_score(
        self,
        factors: Dict[str, float],
        phase: CompetitionPhase
    ) -> Dict[str, float]:
        """重み付き撤退スコア計算"""
        
        weights = self.phase_weights[phase]
        
        withdrawal_score = (
            factors["probability_factor"] * weights["probability"] +
            factors["ranking_factor"] * weights["ranking"] +
            factors["time_factor"] * weights["time"] +
            factors["resource_factor"] * weights["resource"] +
            factors["opportunity_factor"] * weights["opportunity"]
        )
        
        return {
            "withdrawal_score": withdrawal_score,
            "weighted_factors": {
                key: factors[key] * weights[key.replace("_factor", "")]
                for key in factors.keys()
            }
        }
    
    async def _make_withdrawal_decision(
        self,
        weighted_score: Dict[str, float],
        factors: Dict[str, float],
        conditions: WithdrawalConditions
    ) -> Dict[str, Any]:
        """撤退決定"""
        
        withdrawal_score = weighted_score["withdrawal_score"]
        
        # 基本撤退判定
        should_withdraw = withdrawal_score > 0.6
        
        # 緊急度判定
        if withdrawal_score > 0.9:
            urgency = WithdrawalUrgency.IMMEDIATE
        elif withdrawal_score > 0.8:
            urgency = WithdrawalUrgency.URGENT
        elif withdrawal_score > 0.6:
            urgency = WithdrawalUrgency.MODERATE
        elif withdrawal_score > 0.4:
            urgency = WithdrawalUrgency.LOW
        else:
            urgency = WithdrawalUrgency.MONITOR
        
        # 主要理由特定
        primary_reason = None
        secondary_reasons = []
        
        factor_reasons = [
            (factors["probability_factor"], WithdrawalReason.LOW_PROBABILITY),
            (factors["ranking_factor"], WithdrawalReason.POOR_RANKING),
            (factors["time_factor"], WithdrawalReason.TIME_CONSTRAINT),
            (factors["resource_factor"], WithdrawalReason.RESOURCE_SHORTAGE),
            (factors["opportunity_factor"], WithdrawalReason.BETTER_OPPORTUNITY)
        ]
        
        # 最高スコアの要因を主要理由に
        factor_reasons.sort(key=lambda x: x[0], reverse=True)
        primary_reason = factor_reasons[0][1]
        
        # 閾値を超える副次的理由
        for score, reason in factor_reasons[1:]:
            if score > 0.6:
                secondary_reasons.append(reason)
        
        return {
            "should_withdraw": should_withdraw,
            "urgency": urgency,
            "primary_reason": primary_reason,
            "secondary_reasons": secondary_reasons
        }
    
    async def _analyze_opportunity_cost(
        self,
        competition_info: CompetitionInfo,
        current_probability: MedalProbability,
        available_alternatives: List[CompetitionInfo]
    ) -> Dict[str, Any]:
        """機会コスト分析"""
        
        current_value = current_probability.expected_medal_value
        
        if not available_alternatives:
            return {"cost": 0.0, "better_options": []}
        
        # 代替案の期待価値計算
        better_options = []
        max_alternative_value = 0.0
        
        for alt_comp in available_alternatives:
            # 簡易期待価値推定（実際は詳細計算が必要）
            alt_prob = await self.probability_calculator.calculate_medal_probability(alt_comp)
            alt_value = alt_prob.expected_medal_value
            
            if alt_value > current_value * 1.2:  # 20%以上良い
                better_options.append({
                    "competition": alt_comp.title,
                    "expected_value": alt_value,
                    "improvement": (alt_value - current_value) / current_value
                })
            
            max_alternative_value = max(max_alternative_value, alt_value)
        
        # 機会コスト計算
        if current_value > 0:
            opportunity_cost = max(0.0, (max_alternative_value - current_value) / current_value)
        else:
            opportunity_cost = 1.0 if max_alternative_value > 0 else 0.0
        
        return {
            "cost": min(1.0, opportunity_cost),
            "better_options": better_options,
            "max_alternative_value": max_alternative_value
        }
    
    async def _analyze_alternatives(
        self,
        competition_info: CompetitionInfo,
        available_alternatives: List[CompetitionInfo]
    ) -> Dict[str, List[str]]:
        """代替案分析"""
        
        competitions = []
        suggestions = []
        
        if not available_alternatives:
            suggestions.append("現在、代替コンペが利用できません")
            return {"competitions": competitions, "suggestions": suggestions}
        
        # 同種コンペの代替案
        same_type_alternatives = [
            comp for comp in available_alternatives
            if comp.competition_type == competition_info.competition_type
        ]
        
        if same_type_alternatives:
            competitions.extend([comp.title for comp in same_type_alternatives[:3]])
            suggestions.append("同種コンペでスキル活用継続")
        
        # 高確率代替案
        high_prob_alternatives = [
            comp for comp in available_alternatives
            if comp.participant_count < competition_info.participant_count
        ]
        
        if high_prob_alternatives:
            competitions.extend([comp.title for comp in high_prob_alternatives[:2]])
            suggestions.append("低競争コンペで勝率向上")
        
        # リソース効率的代替案
        efficient_alternatives = [
            comp for comp in available_alternatives
            if comp.competition_type.value in ["tabular", "getting_started"]
        ]
        
        if efficient_alternatives:
            suggestions.append("軽量コンペでリソース効率化")
        
        return {"competitions": list(set(competitions)), "suggestions": suggestions}
    
    async def _generate_withdrawal_recommendation(
        self,
        withdrawal_decision: Dict[str, Any],
        factors: Dict[str, float],
        opportunity_analysis: Dict[str, Any]
    ) -> Dict[str, str]:
        """撤退推奨生成"""
        
        if withdrawal_decision["should_withdraw"]:
            urgency = withdrawal_decision["urgency"]
            
            if urgency == WithdrawalUrgency.IMMEDIATE:
                action = "即座に撤退し、リソースを代替機会に集中"
                timeline = "24時間以内"
            elif urgency == WithdrawalUrgency.URGENT:
                action = "早急な撤退を検討、代替案の準備"
                timeline = "3日以内"
            else:
                action = "段階的撤退または条件付き継続"
                timeline = "1週間以内に再評価"
        else:
            if factors["ranking_factor"] > 0.5:
                action = "継続だが順位改善に集中"
            elif factors["time_factor"] > 0.5:
                action = "継続だが効率的な時間利用"
            else:
                action = "継続推奨、現在の戦略維持"
            
            timeline = "次回定期評価まで継続"
        
        return {"action": action, "timeline": timeline}
    
    def _determine_competition_phase(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[CompetitionStatus]
    ) -> CompetitionPhase:
        """コンペ段階判定"""
        
        if current_status:
            return current_status.phase
        
        # 残り日数による推定
        days_remaining = competition_info.days_remaining
        
        if days_remaining <= 0:
            return CompetitionPhase.COMPLETED
        elif days_remaining <= 7:
            return CompetitionPhase.SUBMISSION_ONLY
        else:
            return CompetitionPhase.ACTIVE
    
    async def _predict_final_rank(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[CompetitionStatus],
        current_probability: MedalProbability
    ) -> Optional[int]:
        """最終順位予測"""
        
        if not current_status or current_status.current_rank is None:
            # 確率から推定
            expected_percentile = 1.0 - current_probability.overall_probability
            return int(competition_info.participant_count * expected_percentile)
        
        # 現在順位からの改善/悪化予測
        current_rank = current_status.current_rank
        
        # トレンド分析
        if current_status.score_history:
            # 改善トレンドがあれば順位向上を予測
            recent_scores = current_status.score_history[-3:]
            if len(recent_scores) >= 2:
                trend = recent_scores[-1]["score"] - recent_scores[0]["score"]
                if trend > 0:
                    rank_improvement = max(1, int(current_rank * 0.1))
                    return max(1, current_rank - rank_improvement)
        
        return current_rank  # 現状維持予測
    
    async def _calculate_minimum_viable_rank(self, competition_info: CompetitionInfo) -> int:
        """最低限必要順位計算"""
        
        participant_count = competition_info.participant_count
        
        if participant_count <= 0:
            return 1
        
        # メダル圏の推定（上位3%）
        medal_cutoff = max(1, int(participant_count * 0.03))
        
        return medal_cutoff
    
    async def _calculate_confidence_level(self, factors: Dict[str, float]) -> float:
        """信頼度計算"""
        
        # 要因スコアの分散から信頼度を計算
        factor_values = list(factors.values())
        
        if len(factor_values) <= 1:
            return 0.5
        
        variance = np.var(factor_values)
        
        # 分散が小さい（要因が一致）ほど信頼度高
        confidence = max(0.1, min(1.0, 1.0 - variance))
        
        return confidence