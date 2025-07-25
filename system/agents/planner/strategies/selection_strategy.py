"""
コンペ選択戦略システム

メダル確率・リソース効率・機会コストを総合評価した最適コンペ選択。
動的コンペ管理システムとの連携による戦略的ポートフォリオ構築。
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
        def sum(data):
            return sum(data)
    np = MockNumpy()

from ..models.competition import CompetitionInfo, CompetitionType, AnalysisResult
from ..models.probability import MedalProbability, ProbabilityTier
from ..calculators.medal_probability import MedalProbabilityCalculator


class SelectionStrategy(Enum):
    """選択戦略種別"""
    MAX_PROBABILITY = "max_probability"      # 最高確率優先
    BALANCED = "balanced"                    # バランス重視
    RISK_DIVERSIFIED = "risk_diversified"   # リスク分散
    HIGH_REWARD = "high_reward"             # 高報酬優先
    QUICK_WIN = "quick_win"                 # 短期勝利
    SKILL_MATCH = "skill_match"             # スキルマッチ優先


class SelectionCriteria(Enum):
    """選択基準"""
    MEDAL_PROBABILITY = "medal_probability"
    EXPECTED_VALUE = "expected_value"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    SKILL_ALIGNMENT = "skill_alignment"
    TIME_EFFICIENCY = "time_efficiency"
    RISK_LEVEL = "risk_level"


@dataclass
class SelectionScore:
    """選択スコア"""
    competition_info: CompetitionInfo
    medal_probability: MedalProbability
    
    # 基本スコア
    overall_score: float = 0.0
    probability_score: float = 0.0
    value_score: float = 0.0
    efficiency_score: float = 0.0
    
    # 詳細評価
    risk_score: float = 0.0
    time_score: float = 0.0
    skill_score: float = 0.0
    strategic_score: float = 0.0
    
    # 選択根拠
    selection_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    opportunity_factors: List[str] = field(default_factory=list)
    
    # メタデータ
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy_used: str = ""


@dataclass
class PortfolioRecommendation:
    """ポートフォリオ推奨"""
    recommended_competitions: List[SelectionScore]
    portfolio_score: float
    expected_medal_count: float
    risk_level: str
    resource_utilization: float
    
    # 戦略説明
    strategy_rationale: str
    portfolio_balance: Dict[str, float]
    risk_mitigation: List[str]
    
    # 代替案
    alternative_portfolios: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


class CompetitionSelectionStrategy:
    """コンペ選択戦略システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.probability_calculator = MedalProbabilityCalculator()
        
        # 戦略設定
        self.max_portfolio_size = 3     # 最大同時参加数
        self.min_medal_probability = 0.1  # 最低メダル確率
        self.risk_tolerance = 0.6       # リスク許容度
        
        # 重み設定（戦略別）
        self.strategy_weights = {
            SelectionStrategy.MAX_PROBABILITY: {
                "probability": 0.5, "value": 0.2, "efficiency": 0.15, 
                "risk": 0.05, "time": 0.05, "skill": 0.05
            },
            SelectionStrategy.BALANCED: {
                "probability": 0.25, "value": 0.2, "efficiency": 0.2,
                "risk": 0.15, "time": 0.1, "skill": 0.1
            },
            SelectionStrategy.RISK_DIVERSIFIED: {
                "probability": 0.2, "value": 0.15, "efficiency": 0.15,
                "risk": 0.3, "time": 0.1, "skill": 0.1
            },
            SelectionStrategy.HIGH_REWARD: {
                "probability": 0.3, "value": 0.4, "efficiency": 0.1,
                "risk": 0.1, "time": 0.05, "skill": 0.05
            },
            SelectionStrategy.QUICK_WIN: {
                "probability": 0.35, "value": 0.15, "efficiency": 0.15,
                "risk": 0.1, "time": 0.2, "skill": 0.05
            },
            SelectionStrategy.SKILL_MATCH: {
                "probability": 0.2, "value": 0.15, "efficiency": 0.15,
                "risk": 0.1, "time": 0.1, "skill": 0.3
            }
        }
    
    async def analyze_competition_for_selection(
        self,
        competition_info: CompetitionInfo,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        current_portfolio: Optional[List[CompetitionInfo]] = None
    ) -> AnalysisResult:
        """コンペ選択分析"""
        
        start_time = datetime.utcnow()
        self.logger.info(f"コンペ選択分析開始: {competition_info.title}")
        
        try:
            # メダル確率算出
            medal_probability = await self.probability_calculator.calculate_medal_probability(
                competition_info
            )
            
            # 選択スコア計算
            selection_score = await self._calculate_selection_score(
                competition_info, medal_probability, strategy
            )
            
            # ポートフォリオ適合性評価
            portfolio_fit = await self._evaluate_portfolio_fit(
                competition_info, current_portfolio or []
            )
            
            # 戦略的評価
            strategic_evaluation = await self._evaluate_strategic_value(
                competition_info, medal_probability, strategy
            )
            
            # 推奨アクション決定
            recommended_action = await self._determine_recommended_action(
                selection_score, portfolio_fit, strategic_evaluation
            )
            
            # 分析結果構築
            analysis_result = AnalysisResult(
                competition_info=competition_info,
                analysis_timestamp=start_time,
                medal_probability=medal_probability.overall_probability,
                gold_probability=medal_probability.gold_probability,
                silver_probability=medal_probability.silver_probability,
                bronze_probability=medal_probability.bronze_probability,
                confidence_interval=medal_probability.confidence_interval,
                probability_factors=medal_probability.factor_breakdown,
                risk_factors=medal_probability.risk_factors,
                opportunity_factors=await self._identify_opportunity_factors(competition_info),
                strategic_score=selection_score.overall_score,
                resource_efficiency=selection_score.efficiency_score,
                time_opportunity_cost=await self._calculate_time_opportunity_cost(competition_info),
                skill_match_score=selection_score.skill_score,
                recommended_action=recommended_action,
                action_confidence=selection_score.overall_score,
                action_reasoning=selection_score.selection_reasons,
                participation_strategy=await self._generate_participation_strategy(
                    competition_info, medal_probability
                ) if recommended_action == "participate" else None,
                resource_allocation=await self._calculate_resource_allocation(competition_info),
                milestone_targets=await self._generate_milestone_targets(competition_info),
                withdrawal_triggers=await self._define_withdrawal_triggers(competition_info),
                withdrawal_thresholds=await self._define_withdrawal_thresholds(competition_info),
                alternative_competitions=await self._find_alternative_competitions(competition_info),
                next_analysis_scheduled=datetime.utcnow() + timedelta(days=7),
                analysis_duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
            
            self.logger.info(f"コンペ選択分析完了: {recommended_action} ({competition_info.title})")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"コンペ選択分析失敗: {e}")
            raise
    
    async def recommend_optimal_portfolio(
        self,
        available_competitions: List[CompetitionInfo],
        strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        current_portfolio: Optional[List[CompetitionInfo]] = None
    ) -> PortfolioRecommendation:
        """最適ポートフォリオ推奨"""
        
        self.logger.info(f"最適ポートフォリオ分析開始: {len(available_competitions)}件の候補")
        
        try:
            # 各コンペの選択スコア計算
            competition_scores = []
            for comp in available_competitions:
                medal_prob = await self.probability_calculator.calculate_medal_probability(comp)
                score = await self._calculate_selection_score(comp, medal_prob, strategy)
                competition_scores.append(score)
            
            # スコア順でソート
            competition_scores.sort(key=lambda x: x.overall_score, reverse=True)
            
            # 最適ポートフォリオ構築
            optimal_portfolio = await self._construct_optimal_portfolio(
                competition_scores, strategy, current_portfolio or []
            )
            
            # ポートフォリオ評価
            portfolio_evaluation = await self._evaluate_portfolio(optimal_portfolio, strategy)
            
            # 推奨結果作成
            recommendation = PortfolioRecommendation(
                recommended_competitions=optimal_portfolio,
                portfolio_score=portfolio_evaluation["overall_score"],
                expected_medal_count=portfolio_evaluation["expected_medals"],
                risk_level=portfolio_evaluation["risk_level"],
                resource_utilization=portfolio_evaluation["resource_utilization"],
                strategy_rationale=await self._generate_strategy_rationale(strategy, optimal_portfolio),
                portfolio_balance=portfolio_evaluation["balance_metrics"],
                risk_mitigation=portfolio_evaluation["risk_mitigation"],
                alternative_portfolios=await self._generate_alternative_portfolios(
                    competition_scores, strategy
                ),
                optimization_suggestions=await self._generate_optimization_suggestions(
                    optimal_portfolio, available_competitions
                )
            )
            
            self.logger.info(f"最適ポートフォリオ推奨完了: {len(optimal_portfolio)}件選択")
            return recommendation
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ推奨失敗: {e}")
            raise
    
    async def _calculate_selection_score(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability,
        strategy: SelectionStrategy
    ) -> SelectionScore:
        """選択スコア計算"""
        
        # 基本スコア計算
        probability_score = medal_probability.overall_probability
        value_score = await self._calculate_value_score(competition_info, medal_probability)
        efficiency_score = await self._calculate_efficiency_score(competition_info)
        risk_score = 1.0 - sum(medal_probability.risk_factors.values()) / len(medal_probability.risk_factors) if medal_probability.risk_factors else 0.8
        time_score = await self._calculate_time_score(competition_info)
        skill_score = medal_probability.factor_breakdown.get("専門性マッチング", 0.5)
        
        # 戦略別重み適用
        weights = self.strategy_weights[strategy]
        overall_score = (
            probability_score * weights["probability"] +
            value_score * weights["value"] +
            efficiency_score * weights["efficiency"] +
            risk_score * weights["risk"] +
            time_score * weights["time"] +
            skill_score * weights["skill"]
        )
        
        # 選択根拠生成
        selection_reasons = await self._generate_selection_reasons(
            competition_info, medal_probability, strategy
        )
        
        return SelectionScore(
            competition_info=competition_info,
            medal_probability=medal_probability,
            overall_score=overall_score,
            probability_score=probability_score,
            value_score=value_score,
            efficiency_score=efficiency_score,
            risk_score=risk_score,
            time_score=time_score,
            skill_score=skill_score,
            selection_reasons=selection_reasons,
            strategy_used=strategy.value
        )
    
    async def _calculate_value_score(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability
    ) -> float:
        """価値スコア計算"""
        
        # 期待メダル価値
        expected_medal_value = medal_probability.expected_medal_value
        
        # 賞金による価値調整
        if competition_info.total_prize > 0:
            prize_value = min(1.0, competition_info.total_prize / 100000.0)  # 10万ドル基準で正規化
        else:
            prize_value = 0.5  # 非金銭賞のデフォルト価値
        
        # 参加者数による希少価値
        scarcity_value = 1.0
        if competition_info.participant_count > 0:
            # 参加者が少ないほど希少価値高い
            scarcity_value = max(0.3, min(1.0, 1000.0 / competition_info.participant_count))
        
        # 重み付き統合
        value_score = 0.5 * expected_medal_value + 0.3 * prize_value + 0.2 * scarcity_value
        
        return min(1.0, value_score)
    
    async def _calculate_efficiency_score(self, competition_info: CompetitionInfo) -> float:
        """効率性スコア計算"""
        
        # 推定必要時間
        estimated_hours = await self._estimate_time_investment(competition_info)
        
        # 時間効率性（少ない時間で高い成果を期待）
        if estimated_hours > 0:
            time_efficiency = max(0.1, min(1.0, 20.0 / estimated_hours))  # 20時間基準
        else:
            time_efficiency = 0.5
        
        # データサイズ効率性
        data_efficiency = 1.0
        if competition_info.data_size_gb > 0:
            # データが小さいほど効率的
            data_efficiency = max(0.3, min(1.0, 2.0 / competition_info.data_size_gb))  # 2GB基準
        
        # 複雑性効率性
        complexity_factor = 1.0
        if competition_info.competition_type in [
            CompetitionType.MULTI_MODAL, CompetitionType.REINFORCEMENT_LEARNING
        ]:
            complexity_factor = 0.6  # 複雑な分野は効率性低下
        elif competition_info.competition_type in [
            CompetitionType.TABULAR, CompetitionType.GETTING_STARTED
        ]:
            complexity_factor = 1.2  # シンプルな分野は効率性向上
        
        efficiency_score = 0.5 * time_efficiency + 0.3 * data_efficiency + 0.2 * complexity_factor
        
        return min(1.0, efficiency_score)
    
    async def _calculate_time_score(self, competition_info: CompetitionInfo) -> float:
        """時間スコア計算"""
        
        days_remaining = competition_info.days_remaining
        
        if days_remaining <= 0:
            return 0.0  # 終了済み
        
        # 残り時間による効果曲線
        if days_remaining >= 60:
            return 1.0    # 十分な時間
        elif days_remaining >= 30:
            return 0.9    # 適度な時間
        elif days_remaining >= 14:
            return 0.7    # やや短い
        elif days_remaining >= 7:
            return 0.5    # 短い
        else:
            return 0.3    # 非常に短い
    
    async def _estimate_time_investment(self, competition_info: CompetitionInfo) -> float:
        """時間投資見積もり"""
        
        # コンペ種別による基本時間
        base_hours = {
            CompetitionType.TABULAR: 15,
            CompetitionType.TIME_SERIES: 20,
            CompetitionType.COMPUTER_VISION: 25,
            CompetitionType.NLP: 30,
            CompetitionType.AUDIO: 35,
            CompetitionType.GRAPH: 25,
            CompetitionType.MULTI_MODAL: 40,
            CompetitionType.REINFORCEMENT_LEARNING: 50,
            CompetitionType.CODE_COMPETITION: 10,
            CompetitionType.GETTING_STARTED: 8
        }
        
        base_time = base_hours.get(competition_info.competition_type, 20)
        
        # 参加者数による調整（競争が激しいほど時間増加）
        if competition_info.participant_count > 2000:
            competition_factor = 1.5
        elif competition_info.participant_count > 1000:
            competition_factor = 1.3
        elif competition_info.participant_count > 500:
            competition_factor = 1.1
        else:
            competition_factor = 1.0
        
        # 賞金による調整（高賞金は高品質要求）
        if competition_info.total_prize > 50000:
            prize_factor = 1.4
        elif competition_info.total_prize > 25000:
            prize_factor = 1.2
        else:
            prize_factor = 1.0
        
        estimated_hours = base_time * competition_factor * prize_factor
        
        return max(5.0, min(80.0, estimated_hours))
    
    async def _evaluate_portfolio_fit(
        self,
        competition_info: CompetitionInfo,
        current_portfolio: List[CompetitionInfo]
    ) -> Dict[str, Any]:
        """ポートフォリオ適合性評価"""
        
        fit_analysis = {
            "can_add": len(current_portfolio) < self.max_portfolio_size,
            "diversity_benefit": 0.0,
            "resource_conflict": 0.0,
            "timeline_overlap": 0.0,
            "skill_synergy": 0.0
        }
        
        if not current_portfolio:
            fit_analysis["diversity_benefit"] = 1.0
            return fit_analysis
        
        # 多様性分析
        current_types = set(comp.competition_type for comp in current_portfolio)
        if competition_info.competition_type not in current_types:
            fit_analysis["diversity_benefit"] = 0.8  # 新しい種別による多様性
        else:
            fit_analysis["diversity_benefit"] = 0.3  # 既存種別の重複
        
        # リソース競合分析
        total_estimated_time = sum(
            await self._estimate_time_investment(comp) for comp in current_portfolio
        )
        new_time = await self._estimate_time_investment(competition_info)
        
        if total_estimated_time + new_time > 60:  # 週60時間制限
            fit_analysis["resource_conflict"] = 0.8  # 高い競合
        elif total_estimated_time + new_time > 40:
            fit_analysis["resource_conflict"] = 0.5  # 中程度の競合
        else:
            fit_analysis["resource_conflict"] = 0.2  # 低い競合
        
        # タイムライン重複分析
        overlapping_competitions = 0
        for comp in current_portfolio:
            if abs(comp.days_remaining - competition_info.days_remaining) < 14:
                overlapping_competitions += 1
        
        fit_analysis["timeline_overlap"] = overlapping_competitions / len(current_portfolio)
        
        return fit_analysis
    
    async def _evaluate_strategic_value(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability,
        strategy: SelectionStrategy
    ) -> Dict[str, float]:
        """戦略的価値評価"""
        
        strategic_value = {}
        
        # 戦略適合度
        if strategy == SelectionStrategy.MAX_PROBABILITY:
            strategic_value["strategy_alignment"] = medal_probability.overall_probability
        elif strategy == SelectionStrategy.HIGH_REWARD:
            strategic_value["strategy_alignment"] = min(1.0, competition_info.total_prize / 100000.0)
        elif strategy == SelectionStrategy.QUICK_WIN:
            strategic_value["strategy_alignment"] = 1.0 - (competition_info.days_remaining / 90.0)
        else:
            strategic_value["strategy_alignment"] = 0.7  # デフォルト
        
        # 学習価値
        learning_types = {
            CompetitionType.TABULAR: 0.6,
            CompetitionType.COMPUTER_VISION: 0.8,
            CompetitionType.NLP: 0.9,
            CompetitionType.MULTI_MODAL: 1.0
        }
        strategic_value["learning_value"] = learning_types.get(
            competition_info.competition_type, 0.7
        )
        
        # 実績構築価値
        if medal_probability.overall_probability > 0.6:
            strategic_value["portfolio_building"] = 0.9  # 高確率は実績に良い
        elif medal_probability.overall_probability > 0.3:
            strategic_value["portfolio_building"] = 0.7
        else:
            strategic_value["portfolio_building"] = 0.4
        
        return strategic_value
    
    async def _determine_recommended_action(
        self,
        selection_score: SelectionScore,
        portfolio_fit: Dict[str, Any],
        strategic_evaluation: Dict[str, float]
    ) -> str:
        """推奨アクション決定"""
        
        # 基本的な参加条件チェック
        if selection_score.medal_probability.overall_probability < self.min_medal_probability:
            return "skip"  # 確率が低すぎる
        
        if not portfolio_fit["can_add"]:
            return "monitor"  # ポートフォリオが満杯
        
        if portfolio_fit["resource_conflict"] > 0.7:
            return "analyze_further"  # リソース競合が深刻
        
        # 総合スコアによる判定
        if selection_score.overall_score > 0.8:
            return "participate"  # 高スコア：即座に参加
        elif selection_score.overall_score > 0.6:
            return "participate"  # 中高スコア：参加推奨
        elif selection_score.overall_score > 0.4:
            return "monitor"     # 中スコア：監視継続
        else:
            return "skip"        # 低スコア：スキップ
    
    async def _construct_optimal_portfolio(
        self,
        competition_scores: List[SelectionScore],
        strategy: SelectionStrategy,
        current_portfolio: List[CompetitionInfo]
    ) -> List[SelectionScore]:
        """最適ポートフォリオ構築"""
        
        optimal_portfolio = []
        current_portfolio_ids = set(comp.competition_id for comp in current_portfolio)
        
        # 既存ポートフォリオの継続評価
        existing_to_keep = []
        for score in competition_scores:
            if score.competition_info.competition_id in current_portfolio_ids:
                if score.overall_score > 0.5:  # 継続価値がある
                    existing_to_keep.append(score)
        
        optimal_portfolio.extend(existing_to_keep)
        
        # 新規追加候補の選択
        remaining_slots = self.max_portfolio_size - len(optimal_portfolio)
        candidates = [
            score for score in competition_scores
            if score.competition_info.competition_id not in current_portfolio_ids
            and score.overall_score > 0.4
        ]
        
        # 多様性を考慮した選択
        selected_types = set(
            score.competition_info.competition_type for score in optimal_portfolio
        )
        
        for _ in range(remaining_slots):
            if not candidates:
                break
            
            best_candidate = None
            best_adjusted_score = 0.0
            
            for candidate in candidates:
                adjusted_score = candidate.overall_score
                
                # 多様性ボーナス
                if candidate.competition_info.competition_type not in selected_types:
                    adjusted_score *= 1.2
                
                # リスク分散ボーナス
                if strategy == SelectionStrategy.RISK_DIVERSIFIED:
                    current_risk = np.mean([s.risk_score for s in optimal_portfolio]) if optimal_portfolio else 0.5
                    if abs(candidate.risk_score - current_risk) > 0.3:
                        adjusted_score *= 1.15
                
                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_candidate = candidate
            
            if best_candidate:
                optimal_portfolio.append(best_candidate)
                selected_types.add(best_candidate.competition_info.competition_type)
                candidates.remove(best_candidate)
        
        return optimal_portfolio
    
    async def _evaluate_portfolio(
        self,
        portfolio: List[SelectionScore],
        strategy: SelectionStrategy
    ) -> Dict[str, Any]:
        """ポートフォリオ評価"""
        
        if not portfolio:
            return {
                "overall_score": 0.0,
                "expected_medals": 0.0,
                "risk_level": "high",
                "resource_utilization": 0.0,
                "balance_metrics": {},
                "risk_mitigation": []
            }
        
        # 基本統計
        overall_scores = [score.overall_score for score in portfolio]
        medal_probabilities = [score.medal_probability.overall_probability for score in portfolio]
        
        # 期待メダル数計算
        expected_medals = sum(medal_probabilities)
        
        # ポートフォリオ全体スコア
        portfolio_score = np.mean(overall_scores)
        
        # リスク評価
        risk_scores = [score.risk_score for score in portfolio]
        avg_risk = np.mean(risk_scores)
        risk_diversity = np.std(risk_scores) if len(risk_scores) > 1 else 0.0
        
        if avg_risk > 0.8 and risk_diversity < 0.1:
            risk_level = "low"
        elif avg_risk > 0.6:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        # リソース利用率
        total_estimated_time = sum(
            await self._estimate_time_investment(score.competition_info)
            for score in portfolio
        )
        resource_utilization = min(1.0, total_estimated_time / 60.0)  # 週60時間基準
        
        # バランス指標
        competition_types = [score.competition_info.competition_type for score in portfolio]
        type_diversity = len(set(competition_types)) / len(competition_types)
        
        balance_metrics = {
            "type_diversity": type_diversity,
            "score_variance": np.var(overall_scores),
            "risk_variance": np.var(risk_scores),
            "time_distribution": resource_utilization
        }
        
        # リスク軽減策
        risk_mitigation = []
        if avg_risk < 0.6:
            risk_mitigation.append("高リスクコンペ含有 - 継続監視必要")
        if type_diversity < 0.7:
            risk_mitigation.append("種別集中 - 分散化推奨")
        if resource_utilization > 0.9:
            risk_mitigation.append("リソース過負荷 - 優先度調整必要")
        
        return {
            "overall_score": portfolio_score,
            "expected_medals": expected_medals,
            "risk_level": risk_level,
            "resource_utilization": resource_utilization,
            "balance_metrics": balance_metrics,
            "risk_mitigation": risk_mitigation
        }
    
    # その他のヘルパーメソッドは省略（実装は続く）
    async def _generate_selection_reasons(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability,
        strategy: SelectionStrategy
    ) -> List[str]:
        """選択理由生成"""
        
        reasons = []
        
        # 確率による理由
        if medal_probability.overall_probability > 0.7:
            reasons.append(f"高いメダル確率 ({medal_probability.overall_probability:.1%})")
        elif medal_probability.overall_probability > 0.5:
            reasons.append(f"適度なメダル確率 ({medal_probability.overall_probability:.1%})")
        
        # 専門性による理由
        domain_match = medal_probability.factor_breakdown.get("専門性マッチング", 0.0)
        if domain_match > 0.8:
            reasons.append("高い専門性マッチング")
        
        # 時間による理由
        if competition_info.days_remaining > 60:
            reasons.append("十分な開発時間")
        elif competition_info.days_remaining < 14:
            reasons.append("緊急性による優先度向上")
        
        # 賞金による理由
        if competition_info.total_prize > 50000:
            reasons.append("高額賞金による価値")
        
        # 戦略固有の理由
        if strategy == SelectionStrategy.QUICK_WIN and competition_info.days_remaining < 30:
            reasons.append("短期勝利戦略に適合")
        
        return reasons
    
    async def _identify_opportunity_factors(self, competition_info: CompetitionInfo) -> Dict[str, float]:
        """機会要因特定"""
        
        return {
            "market_timing": 0.7,  # 市場タイミング
            "skill_leverage": 0.8,  # スキル活用度
            "network_effect": 0.6,  # ネットワーク効果
            "learning_opportunity": 0.9  # 学習機会
        }
    
    async def _calculate_time_opportunity_cost(self, competition_info: CompetitionInfo) -> float:
        """時間機会コスト計算"""
        
        estimated_time = await self._estimate_time_investment(competition_info)
        # 他の活動との機会コスト比較（簡易実装）
        return min(1.0, estimated_time / 40.0)  # 週40時間基準
    
    async def _define_withdrawal_triggers(self, competition_info: CompetitionInfo) -> List[str]:
        """撤退トリガー定義"""
        
        return [
            "順位が下位30%に低下",
            f"残り{competition_info.days_remaining // 3}日で改善なし",
            "より高確率コンペが出現",
            "リソース制約による優先度変更"
        ]
    
    async def _define_withdrawal_thresholds(self, competition_info: CompetitionInfo) -> Dict[str, float]:
        """撤退閾値定義"""
        
        return {
            "rank_percentile_threshold": 0.7,  # 下位70%で撤退検討
            "probability_drop_threshold": 0.5,  # 確率50%低下で撤退検討
            "time_investment_limit": 30.0,     # 30時間投資上限
            "opportunity_cost_threshold": 0.3   # 機会コスト30%で撤退検討
        }
    
    async def _calculate_resource_allocation(self, competition_info: CompetitionInfo) -> Dict[str, float]:
        """リソース配分計算"""
        
        estimated_time = await self._estimate_time_investment(competition_info)
        
        return {
            "time_hours_weekly": min(20.0, estimated_time / 4),  # 週当たり時間
            "compute_hours_weekly": min(10.0, estimated_time / 8),  # 計算リソース
            "priority_weight": 0.7 if competition_info.total_prize > 50000 else 0.5
        }
    
    async def _generate_milestone_targets(self, competition_info: CompetitionInfo) -> List[Dict[str, Any]]:
        """マイルストーン目標生成"""
        
        days_remaining = competition_info.days_remaining
        targets = []
        
        # 段階的目標設定
        if days_remaining > 30:
            targets.extend([
                {"milestone": "データ理解・EDA完了", "target_date": 7, "priority": "high"},
                {"milestone": "ベースライン構築", "target_date": 14, "priority": "high"},
                {"milestone": "特徴量エンジニアリング", "target_date": 21, "priority": "medium"}
            ])
        elif days_remaining > 14:
            targets.extend([
                {"milestone": "データ理解・ベースライン", "target_date": 5, "priority": "high"},
                {"milestone": "改善モデル構築", "target_date": 10, "priority": "high"}
            ])
        else:
            targets.append({"milestone": "最終提出最適化", "target_date": 3, "priority": "critical"})
        
        return targets
    
    async def _find_alternative_competitions(self, competition_info: CompetitionInfo) -> List[str]:
        """代替コンペ発見"""
        
        # 簡易実装：同種別の類似コンペを推奨
        alternatives = []
        
        comp_type = competition_info.competition_type.value
        
        if comp_type == "tabular":
            alternatives = ["Tabular Playground Series", "House Prices", "Titanic"]
        elif comp_type == "computer_vision":
            alternatives = ["CIFAR-10", "Fashion-MNIST", "Plant Pathology"]
        elif comp_type == "nlp":
            alternatives = ["Sentiment Analysis", "Question Answering", "Text Classification"]
        
        return alternatives[:3]  # 最大3つ
    
    async def _generate_participation_strategy(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability
    ) -> str:
        """参加戦略生成"""
        
        if medal_probability.overall_probability > 0.7:
            return "aggressive_optimization"  # 積極的最適化
        elif medal_probability.overall_probability > 0.5:
            return "balanced_approach"       # バランス型
        elif medal_probability.overall_probability > 0.3:
            return "learning_focused"        # 学習重視
        else:
            return "minimal_viable"          # 最小実行可能
    
    async def _generate_strategy_rationale(
        self,
        strategy: SelectionStrategy,
        portfolio: List[SelectionScore]
    ) -> str:
        """戦略根拠生成"""
        
        if strategy == SelectionStrategy.MAX_PROBABILITY:
            return f"最高確率追求戦略：期待メダル数 {sum(s.medal_probability.overall_probability for s in portfolio):.1f}個"
        elif strategy == SelectionStrategy.BALANCED:
            return f"バランス戦略：リスク分散と期待値最適化のバランス"
        elif strategy == SelectionStrategy.RISK_DIVERSIFIED:
            return f"リスク分散戦略：複数分野での安定的メダル獲得"
        else:
            return f"{strategy.value}戦略による最適化"
    
    async def _generate_alternative_portfolios(
        self,
        competition_scores: List[SelectionScore],
        strategy: SelectionStrategy
    ) -> List[Dict[str, Any]]:
        """代替ポートフォリオ生成"""
        
        alternatives = []
        
        # 高リスク・高リターン案
        high_risk_portfolio = sorted(
            competition_scores[:5], 
            key=lambda x: x.medal_probability.overall_probability, 
            reverse=True
        )[:3]
        
        alternatives.append({
            "name": "高リスク・高リターン",
            "competitions": [s.competition_info.title for s in high_risk_portfolio],
            "expected_medals": sum(s.medal_probability.overall_probability for s in high_risk_portfolio)
        })
        
        # 安定重視案
        stable_portfolio = sorted(
            competition_scores[:5],
            key=lambda x: x.risk_score,
            reverse=True
        )[:3]
        
        alternatives.append({
            "name": "安定重視",
            "competitions": [s.competition_info.title for s in stable_portfolio],
            "risk_level": "low"
        })
        
        return alternatives
    
    async def _generate_optimization_suggestions(
        self,
        portfolio: List[SelectionScore],
        available_competitions: List[CompetitionInfo]
    ) -> List[str]:
        """最適化提案生成"""
        
        suggestions = []
        
        if len(portfolio) < self.max_portfolio_size:
            suggestions.append(f"ポートフォリオに{self.max_portfolio_size - len(portfolio)}件追加可能")
        
        if portfolio:
            avg_probability = sum(s.medal_probability.overall_probability for s in portfolio) / len(portfolio)
            if avg_probability < 0.4:
                suggestions.append("より高確率コンペへの入れ替えを検討")
        
        type_diversity = len(set(s.competition_info.competition_type for s in portfolio))
        if type_diversity < 2:
            suggestions.append("コンペ種別の多様化でリスク分散")
        
        return suggestions