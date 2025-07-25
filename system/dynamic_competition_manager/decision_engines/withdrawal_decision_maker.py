"""
撤退・入れ替え意思決定システム

リアルタイム分析による戦略的撤退判断・新機会への切り替え。
損切りルールと機会コスト最適化による効率的リソース配分。
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
import pandas as pd

from ..medal_probability_calculators.medal_probability_calculator import (
    CompetitionData, MedalProbabilityResult, MedalProbabilityCalculator
)
from ..portfolio_optimizers.competition_portfolio_optimizer import (
    CompetitionPortfolioItem, CompetitionPortfolioOptimizer
)


class WithdrawalReason(Enum):
    """撤退理由"""
    LOW_PROBABILITY = "low_probability"         # 確率低下
    BETTER_OPPORTUNITY = "better_opportunity"   # より良い機会
    RESOURCE_CONSTRAINT = "resource_constraint" # リソース制約
    TIME_CONSTRAINT = "time_constraint"         # 時間制約
    STRATEGIC_FOCUS = "strategic_focus"         # 戦略的集中
    FORCE_MAJEURE = "force_majeure"            # 不可抗力


class CompetitionPhase(Enum):
    """コンペティション進行段階"""
    EARLY = "early"           # 開始初期
    MID_EARLY = "mid_early"   # 前半
    MID_LATE = "mid_late"     # 後半
    FINAL = "final"           # 終了間際


@dataclass
class CompetitionStatus:
    """コンペ進行状況"""
    competition_data: CompetitionData
    current_ranking: Optional[int] = None
    current_percentile: Optional[float] = None
    best_score: Optional[float] = None
    recent_score_trend: str = "stable"  # improving, stable, declining
    time_invested_hours: float = 0.0
    experiments_completed: int = 0
    last_improvement_days_ago: int = 0
    submission_count: int = 0
    phase: CompetitionPhase = CompetitionPhase.EARLY


@dataclass 
class WithdrawalAnalysis:
    """撤退分析結果"""
    should_withdraw: bool
    withdrawal_urgency: float  # 0-1スケール
    primary_reason: WithdrawalReason
    secondary_reasons: List[WithdrawalReason]
    withdrawal_score: float
    continuation_score: float
    opportunity_cost: float
    risk_assessment: Dict[str, float]
    recommended_action: str
    alternative_options: List[str]
    analysis_metadata: Dict[str, Any]


@dataclass
class ReplacementOpportunity:
    """代替機会"""
    new_competition: CompetitionData
    medal_probability: MedalProbabilityResult
    opportunity_score: float
    transition_cost: float
    net_benefit: float
    replacement_rationale: str


class WithdrawalDecisionMaker:
    """撤退・入れ替え意思決定システム"""
    
    def __init__(self):
        self.medal_calculator = MedalProbabilityCalculator()
        self.portfolio_optimizer = CompetitionPortfolioOptimizer()
        self.logger = logging.getLogger(__name__)
        
        # 撤退判断閾値
        self.withdrawal_thresholds = {
            "medal_probability_min": 0.15,     # 最低メダル確率
            "ranking_percentile_max": 0.8,     # 最大順位パーセンタイル
            "stagnation_days_max": 14,         # 最大停滞日数
            "opportunity_cost_min": 0.3,       # 最小機会コスト
            "time_investment_max": 30.0        # 最大時間投資（時間）
        }
        
        # 段階別重み係数
        self.phase_weights = {
            CompetitionPhase.EARLY: {
                "probability": 0.4, "ranking": 0.2, "trend": 0.2, "opportunity": 0.2
            },
            CompetitionPhase.MID_EARLY: {
                "probability": 0.3, "ranking": 0.3, "trend": 0.2, "opportunity": 0.2
            },
            CompetitionPhase.MID_LATE: {
                "probability": 0.25, "ranking": 0.4, "trend": 0.25, "opportunity": 0.1
            },
            CompetitionPhase.FINAL: {
                "probability": 0.2, "ranking": 0.5, "trend": 0.2, "opportunity": 0.1
            }
        }
    
    async def analyze_withdrawal_decision(
        self,
        competition_status: CompetitionStatus,
        available_alternatives: Optional[List[CompetitionData]] = None,
        current_portfolio: Optional[List[CompetitionStatus]] = None
    ) -> WithdrawalAnalysis:
        """撤退判断分析実行"""
        
        try:
            self.logger.info(f"撤退分析開始: {competition_status.competition_data.title}")
            
            # 現在のメダル確率再算出
            current_probability = await self.medal_calculator.calculate_medal_probability(
                competition_status.competition_data
            )
            
            # 各要因の評価
            probability_score = self.evaluate_probability_factor(
                current_probability, competition_status
            )
            
            ranking_score = self.evaluate_ranking_factor(competition_status)
            trend_score = self.evaluate_trend_factor(competition_status)
            opportunity_score = await self.evaluate_opportunity_factor(
                competition_status, available_alternatives
            )
            
            # 段階別重み付き総合スコア
            phase_weights = self.phase_weights[competition_status.phase]
            withdrawal_score = (
                probability_score * phase_weights["probability"] +
                ranking_score * phase_weights["ranking"] +
                trend_score * phase_weights["trend"] +
                opportunity_score * phase_weights["opportunity"]
            )
            
            # 継続スコア（逆）
            continuation_score = 1.0 - withdrawal_score
            
            # 撤退判断
            should_withdraw = withdrawal_score > 0.6  # 60%閾値
            
            # 撤退理由特定
            primary_reason, secondary_reasons = self.identify_withdrawal_reasons(
                probability_score, ranking_score, trend_score, opportunity_score
            )
            
            # 機会コスト算出
            opportunity_cost = await self.calculate_opportunity_cost(
                competition_status, available_alternatives
            )
            
            # リスク評価
            risk_assessment = self.assess_withdrawal_risks(
                competition_status, current_probability
            )
            
            # 推奨アクション
            recommended_action = self.generate_recommendation(
                should_withdraw, withdrawal_score, competition_status
            )
            
            # 代替選択肢
            alternative_options = self.generate_alternative_options(
                competition_status, should_withdraw
            )
            
            analysis = WithdrawalAnalysis(
                should_withdraw=should_withdraw,
                withdrawal_urgency=min(1.0, withdrawal_score * 1.2),
                primary_reason=primary_reason,
                secondary_reasons=secondary_reasons,
                withdrawal_score=withdrawal_score,
                continuation_score=continuation_score,
                opportunity_cost=opportunity_cost,
                risk_assessment=risk_assessment,
                recommended_action=recommended_action,
                alternative_options=alternative_options,
                analysis_metadata={
                    "analysis_time": datetime.utcnow().isoformat(),
                    "competition_phase": competition_status.phase.value,
                    "factor_scores": {
                        "probability": probability_score,
                        "ranking": ranking_score,
                        "trend": trend_score,
                        "opportunity": opportunity_score
                    }
                }
            )
            
            self.logger.info(f"撤退分析完了: {'撤退推奨' if should_withdraw else '継続推奨'}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"撤退分析失敗: {e}")
            raise
    
    def evaluate_probability_factor(
        self,
        current_probability: MedalProbabilityResult,
        status: CompetitionStatus
    ) -> float:
        """確率要因評価"""
        
        overall_prob = current_probability.overall_probability
        
        # 基本確率評価（低いほど撤退スコア高）
        if overall_prob < self.withdrawal_thresholds["medal_probability_min"]:
            prob_score = 1.0  # 確実に撤退推奨
        elif overall_prob < 0.3:
            prob_score = 0.8
        elif overall_prob < 0.5:
            prob_score = 0.5
        elif overall_prob < 0.7:
            prob_score = 0.3
        else:
            prob_score = 0.1  # 高確率は継続
        
        # 信頼区間による調整
        ci_lower, ci_upper = current_probability.confidence_interval
        uncertainty = ci_upper - ci_lower
        
        # 不確実性が高い場合は撤退リスク増加
        uncertainty_penalty = min(0.2, uncertainty * 0.5)
        prob_score += uncertainty_penalty
        
        return min(1.0, prob_score)
    
    def evaluate_ranking_factor(self, status: CompetitionStatus) -> float:
        """順位要因評価"""
        
        if status.current_percentile is None:
            return 0.4  # 情報不足時のデフォルト
        
        percentile = status.current_percentile
        
        # 順位が悪いほど撤退スコア高
        if percentile > self.withdrawal_thresholds["ranking_percentile_max"]:
            ranking_score = 0.9  # 下位20%は撤退推奨
        elif percentile > 0.6:
            ranking_score = 0.7  # 下位40%
        elif percentile > 0.4:
            ranking_score = 0.4  # 中位
        elif percentile > 0.2:
            ranking_score = 0.2  # 上位40%
        else:
            ranking_score = 0.1  # 上位20%は継続
        
        # 段階別調整
        if status.phase == CompetitionPhase.FINAL:
            # 終盤では順位がより重要
            ranking_score *= 1.3
        elif status.phase == CompetitionPhase.EARLY:
            # 初期は順位の重要度を下げる
            ranking_score *= 0.7
        
        return min(1.0, ranking_score)
    
    def evaluate_trend_factor(self, status: CompetitionStatus) -> float:
        """トレンド要因評価"""
        
        # 最近の改善からの経過日数
        stagnation_days = status.last_improvement_days_ago
        
        # スコア改善トレンド
        trend_scores = {
            "improving": 0.1,  # 改善中は継続
            "stable": 0.4,     # 安定
            "declining": 0.8   # 悪化は撤退検討
        }
        
        base_trend_score = trend_scores.get(status.recent_score_trend, 0.5)
        
        # 停滞期間による調整
        if stagnation_days > self.withdrawal_thresholds["stagnation_days_max"]:
            stagnation_penalty = 0.4  # 長期停滞ペナルティ
        elif stagnation_days > 7:
            stagnation_penalty = 0.2
        else:
            stagnation_penalty = 0.0
        
        # 実験数による調整
        if status.experiments_completed < 5:
            # 実験不足は継続の余地あり
            experiment_adjustment = -0.2
        elif status.experiments_completed > 20:
            # 十分な実験済みは撤退検討
            experiment_adjustment = 0.1
        else:
            experiment_adjustment = 0.0
        
        trend_score = base_trend_score + stagnation_penalty + experiment_adjustment
        
        return max(0.0, min(1.0, trend_score))
    
    async def evaluate_opportunity_factor(
        self,
        status: CompetitionStatus,
        available_alternatives: Optional[List[CompetitionData]]
    ) -> float:
        """機会要因評価"""
        
        if not available_alternatives:
            return 0.3  # 代替なしは継続寄り
        
        # 現在のコンペの機会コスト算出
        current_expected_value = await self.calculate_expected_value(
            status.competition_data
        )
        
        # 代替機会の評価
        best_alternative_value = 0.0
        for alt_comp in available_alternatives:
            alt_value = await self.calculate_expected_value(alt_comp)
            best_alternative_value = max(best_alternative_value, alt_value)
        
        # 機会コスト比率
        if current_expected_value > 0:
            opportunity_ratio = (best_alternative_value - current_expected_value) / current_expected_value
        else:
            opportunity_ratio = 1.0 if best_alternative_value > 0.1 else 0.0
        
        # 機会コストが高いほど撤退スコア高
        if opportunity_ratio > 0.5:  # 50%以上の機会コスト
            opportunity_score = 0.9
        elif opportunity_ratio > 0.3:
            opportunity_score = 0.7
        elif opportunity_ratio > 0.1:
            opportunity_score = 0.4
        elif opportunity_ratio > -0.1:
            opportunity_score = 0.2
        else:
            opportunity_score = 0.1  # 現在が最良
        
        return opportunity_score
    
    async def calculate_expected_value(self, competition: CompetitionData) -> float:
        """期待値算出"""
        
        medal_result = await self.medal_calculator.calculate_medal_probability(competition)
        
        # メダル価値重み付け
        medal_values = {
            "gold": 10.0,
            "silver": 6.0,
            "bronze": 3.0
        }
        
        expected_value = (
            medal_result.gold_probability * medal_values["gold"] +
            medal_result.silver_probability * medal_values["silver"] +
            medal_result.bronze_probability * medal_values["bronze"]
        )
        
        return expected_value
    
    def identify_withdrawal_reasons(
        self,
        prob_score: float,
        rank_score: float,
        trend_score: float,
        opp_score: float
    ) -> Tuple[WithdrawalReason, List[WithdrawalReason]]:
        """撤退理由特定"""
        
        # 各要因のスコアと理由のマッピング
        factor_reasons = [
            (prob_score, WithdrawalReason.LOW_PROBABILITY),
            (opp_score, WithdrawalReason.BETTER_OPPORTUNITY),
            (rank_score, WithdrawalReason.STRATEGIC_FOCUS),
            (trend_score, WithdrawalReason.TIME_CONSTRAINT)
        ]
        
        # スコア順でソート
        factor_reasons.sort(key=lambda x: x[0], reverse=True)
        
        primary_reason = factor_reasons[0][1]
        
        # 閾値を超える副次的理由
        secondary_reasons = [
            reason for score, reason in factor_reasons[1:]
            if score > 0.6
        ]
        
        return primary_reason, secondary_reasons
    
    async def calculate_opportunity_cost(
        self,
        status: CompetitionStatus,
        alternatives: Optional[List[CompetitionData]]
    ) -> float:
        """機会コスト算出"""
        
        if not alternatives:
            return 0.0
        
        # 現在のコンペの残り期待価値
        current_remaining_value = await self.calculate_remaining_expected_value(status)
        
        # 最良代替案の期待価値
        best_alternative_value = 0.0
        for alt in alternatives:
            alt_value = await self.calculate_expected_value(alt)
            best_alternative_value = max(best_alternative_value, alt_value)
        
        # 正規化された機会コスト
        if current_remaining_value > 0:
            opportunity_cost = (best_alternative_value - current_remaining_value) / max(best_alternative_value, current_remaining_value)
        else:
            opportunity_cost = 1.0 if best_alternative_value > 0 else 0.0
        
        return max(0.0, min(1.0, opportunity_cost))
    
    async def calculate_remaining_expected_value(self, status: CompetitionStatus) -> float:
        """残り期待価値算出"""
        
        # 基本期待価値
        base_expected_value = await self.calculate_expected_value(status.competition_data)
        
        # 既投資時間による調整（サンクコスト考慮せず）
        # 残り時間割合
        total_competition_days = 90  # 仮定：平均コンペ期間
        elapsed_days = total_competition_days - status.competition_data.days_remaining
        remaining_time_ratio = status.competition_data.days_remaining / total_competition_days
        
        # 進捗段階による価値減衰
        if status.phase == CompetitionPhase.FINAL:
            phase_multiplier = 0.3  # 終盤は改善余地少
        elif status.phase == CompetitionPhase.MID_LATE:
            phase_multiplier = 0.6
        elif status.phase == CompetitionPhase.MID_EARLY:
            phase_multiplier = 0.8
        else:
            phase_multiplier = 1.0  # 初期は満額
        
        # 現在順位による調整
        if status.current_percentile:
            if status.current_percentile < 0.1:
                position_multiplier = 1.2  # 上位はボーナス
            elif status.current_percentile < 0.3:
                position_multiplier = 1.0
            elif status.current_percentile < 0.7:
                position_multiplier = 0.7
            else:
                position_multiplier = 0.4  # 下位は大幅減額
        else:
            position_multiplier = 0.8
        
        remaining_value = base_expected_value * remaining_time_ratio * phase_multiplier * position_multiplier
        
        return max(0.0, remaining_value)
    
    def assess_withdrawal_risks(
        self,
        status: CompetitionStatus,
        probability_result: MedalProbabilityResult
    ) -> Dict[str, float]:
        """撤退リスク評価"""
        
        risks = {}
        
        # 早期撤退リスク（逆転可能性）
        if status.phase in [CompetitionPhase.EARLY, CompetitionPhase.MID_EARLY]:
            risks["early_withdrawal"] = 0.3 if status.current_percentile and status.current_percentile < 0.5 else 0.1
        else:
            risks["early_withdrawal"] = 0.1
        
        # 投資回収リスク
        time_invested_ratio = status.time_invested_hours / 40.0  # 週40時間基準
        risks["sunk_cost"] = min(0.5, time_invested_ratio * 0.3)
        
        # 機会逸失リスク
        if probability_result.overall_probability > 0.4:
            risks["missed_opportunity"] = 0.4
        elif probability_result.overall_probability > 0.2:
            risks["missed_opportunity"] = 0.2
        else:
            risks["missed_opportunity"] = 0.1
        
        # 代替なしリスク
        risks["no_alternatives"] = 0.3  # デフォルト値（実際は代替数による）
        
        return risks
    
    def generate_recommendation(
        self,
        should_withdraw: bool,
        withdrawal_score: float,
        status: CompetitionStatus
    ) -> str:
        """推奨アクション生成"""
        
        if should_withdraw:
            if withdrawal_score > 0.8:
                urgency = "即座に"
            elif withdrawal_score > 0.7:
                urgency = "早急に"
            else:
                urgency = "慎重に検討して"
            
            base_rec = f"{urgency}撤退を推奨"
            
            # 段階別補足
            if status.phase == CompetitionPhase.EARLY:
                supplement = "（早期判断により機会コスト削減）"
            elif status.phase == CompetitionPhase.FINAL:
                supplement = "（終盤での戦略的撤退）"
            else:
                supplement = ""
            
            return f"{base_rec}{supplement}"
        
        else:
            if withdrawal_score < 0.3:
                confidence = "強く"
            elif withdrawal_score < 0.5:
                confidence = ""
            else:
                confidence = "条件付きで"
            
            base_rec = f"{confidence}継続を推奨"
            
            # 改善点の提案
            if status.last_improvement_days_ago > 7:
                supplement = "（新しいアプローチの実験を推奨）"
            elif status.experiments_completed < 10:
                supplement = "（追加実験による改善を期待）"
            else:
                supplement = ""
            
            return f"{base_rec}{supplement}"
    
    def generate_alternative_options(
        self,
        status: CompetitionStatus,
        withdrawal_recommended: bool
    ) -> List[str]:
        """代替選択肢生成"""
        
        options = []
        
        if withdrawal_recommended:
            options.extend([
                "即座撤退：リソースを新機会に集中",
                "段階的撤退：最小限の継続で様子見",
                "条件付き継続：短期改善目標設定"
            ])
        else:
            options.extend([
                "戦略見直し：新しい手法・アプローチの導入",
                "リソース追加：時間投資の増加",
                "目標調整：順位目標の現実的見直し"
            ])
            
        # 段階別追加オプション
        if status.phase == CompetitionPhase.EARLY:
            options.append("データ理解深化：EDAの再実行")
        elif status.phase == CompetitionPhase.MID_EARLY:
            options.append("モデル多様化：新しいアルゴリズム試行")
        elif status.phase == CompetitionPhase.MID_LATE:
            options.append("アンサンブル最適化：既存モデルの組み合わせ")
        else:  # FINAL
            options.append("最終調整：ハイパーパラメータ微調整")
        
        return options[:5]  # 最大5つの選択肢
    
    async def find_replacement_opportunities(
        self,
        withdrawn_competition: CompetitionStatus,
        available_competitions: List[CompetitionData]
    ) -> List[ReplacementOpportunity]:
        """代替機会発見"""
        
        if not available_competitions:
            return []
        
        opportunities = []
        
        # 撤退したコンペの特性
        withdrawn_value = await self.calculate_expected_value(
            withdrawn_competition.competition_data
        )
        
        for comp in available_competitions:
            # 代替機会の評価
            medal_result = await self.medal_calculator.calculate_medal_probability(comp)
            opportunity_score = self.calculate_replacement_opportunity_score(
                comp, medal_result, withdrawn_competition
            )
            
            # 移行コスト
            transition_cost = self.calculate_transition_cost(
                withdrawn_competition, comp
            )
            
            # 純便益
            new_value = await self.calculate_expected_value(comp)
            net_benefit = new_value - withdrawn_value - transition_cost
            
            # 代替理由
            rationale = self.generate_replacement_rationale(
                comp, medal_result, withdrawn_competition
            )
            
            opportunity = ReplacementOpportunity(
                new_competition=comp,
                medal_probability=medal_result,
                opportunity_score=opportunity_score,
                transition_cost=transition_cost,
                net_benefit=net_benefit,
                replacement_rationale=rationale
            )
            
            opportunities.append(opportunity)
        
        # 純便益順でソート
        opportunities.sort(key=lambda x: x.net_benefit, reverse=True)
        
        return opportunities[:5]  # 上位5つの機会
    
    def calculate_replacement_opportunity_score(
        self,
        new_comp: CompetitionData,
        medal_result: MedalProbabilityResult,
        old_status: CompetitionStatus
    ) -> float:
        """代替機会スコア算出"""
        
        # 基本メダル確率
        base_score = medal_result.overall_probability * 100
        
        # 専門性マッチボーナス
        domain_bonus = medal_result.factor_breakdown.get("domain_matching", 0.5) * 20
        
        # 時間的優位性
        if new_comp.days_remaining > old_status.competition_data.days_remaining:
            time_bonus = 15  # より多くの時間
        elif new_comp.days_remaining > 30:
            time_bonus = 10
        else:
            time_bonus = 0
        
        # 競争レベル比較
        if new_comp.participant_count < old_status.competition_data.participant_count:
            competition_bonus = 10  # より低競争
        else:
            competition_bonus = 0
        
        opportunity_score = base_score + domain_bonus + time_bonus + competition_bonus
        
        return opportunity_score
    
    def calculate_transition_cost(
        self,
        old_status: CompetitionStatus,
        new_comp: CompetitionData
    ) -> float:
        """移行コスト算出"""
        
        base_cost = 5.0  # 基本移行コスト
        
        # 分野変更コスト
        if old_status.competition_data.competition_type != new_comp.competition_type:
            domain_change_cost = 10.0
        else:
            domain_change_cost = 2.0
        
        # 学習曲線コスト
        complexity_cost = {
            "tabular": 3.0,
            "computer_vision": 8.0,
            "nlp": 7.0,
            "time_series": 5.0,
            "audio": 9.0,
            "graph": 6.0,
            "multi_modal": 12.0,
            "reinforcement_learning": 15.0
        }.get(new_comp.competition_type.value, 7.0)
        
        # 投資済み時間のサンクコスト（心理的コスト）
        sunk_cost_penalty = min(5.0, old_status.time_invested_hours * 0.1)
        
        total_cost = base_cost + domain_change_cost + complexity_cost + sunk_cost_penalty
        
        return total_cost
    
    def generate_replacement_rationale(
        self,
        new_comp: CompetitionData,
        medal_result: MedalProbabilityResult,
        old_status: CompetitionStatus
    ) -> str:
        """代替理由生成"""
        
        reasons = []
        
        # 確率改善
        old_prob = 0.3  # 撤退判断された場合の推定確率
        if medal_result.overall_probability > old_prob:
            improvement = (medal_result.overall_probability - old_prob) * 100
            reasons.append(f"メダル確率{improvement:.0f}%改善")
        
        # 専門性マッチ
        domain_match = medal_result.factor_breakdown.get("domain_matching", 0.5)
        if domain_match > 0.7:
            reasons.append("高い専門性マッチング")
        
        # 競争レベル
        if new_comp.participant_count < 1000:
            reasons.append("低競争環境")
        
        # 時間的優位
        if new_comp.days_remaining > 45:
            reasons.append("十分な開発時間")
        
        # 賞金・動機
        if new_comp.total_prize > 50000:
            reasons.append("高い賞金による動機")
        elif new_comp.total_prize < 10000:
            reasons.append("低賞金による低競争")
        
        if reasons:
            return "、".join(reasons[:3])  # 最大3つの理由
        else:
            return "総合的な期待値改善"
    
    def get_withdrawal_statistics(self) -> Dict[str, Any]:
        """撤退統計取得"""
        
        return {
            "withdrawal_thresholds": self.withdrawal_thresholds.copy(),
            "phase_weights": {
                phase.value: weights for phase, weights in self.phase_weights.items()
            },
            "last_updated": datetime.utcnow().isoformat()
        }