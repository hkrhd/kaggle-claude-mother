"""
コンペティションポートフォリオ最適化システム

最大3コンペ同時進行での最適ポートフォリオ選択・リスク分散。
メダル獲得確率最大化とリソース効率の両立を実現。
"""

import asyncio
import itertools
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
from scipy.optimize import minimize
import pandas as pd

from ..medal_probability_calculators.medal_probability_calculator import (
    CompetitionData, MedalProbabilityResult, MedalProbabilityCalculator
)


class PortfolioStrategy(Enum):
    """ポートフォリオ戦略"""
    MAX_PROBABILITY = "max_probability"      # 確率最大化
    RISK_DIVERSIFIED = "risk_diversified"    # リスク分散
    BALANCED = "balanced"                    # バランス型
    AGGRESSIVE = "aggressive"                # 積極型
    CONSERVATIVE = "conservative"            # 保守型


class ResourceType(Enum):
    """リソース種別"""
    TIME = "time"
    GPU_HOURS = "gpu_hours"
    MENTAL_CAPACITY = "mental_capacity"
    FINANCIAL = "financial"


@dataclass
class ResourceConstraints:
    """リソース制約"""
    weekly_time_hours: float = 40.0
    gpu_hours_per_week: float = 20.0
    mental_capacity_points: float = 100.0
    financial_budget: float = 0.0  # 無料前提
    max_concurrent_competitions: int = 3


@dataclass
class CompetitionPortfolioItem:
    """ポートフォリオアイテム"""
    competition_data: CompetitionData
    medal_probability_result: MedalProbabilityResult
    estimated_resource_requirements: Dict[ResourceType, float]
    priority_score: float
    risk_score: float
    opportunity_score: float
    selection_reason: str


@dataclass
class PortfolioOptimizationResult:
    """ポートフォリオ最適化結果"""
    selected_competitions: List[CompetitionPortfolioItem]
    rejected_competitions: List[CompetitionPortfolioItem]
    portfolio_metrics: Dict[str, float]
    resource_utilization: Dict[ResourceType, float]
    expected_medal_count: float
    portfolio_risk_score: float
    diversification_score: float
    optimization_strategy: PortfolioStrategy
    selection_rationale: str
    alternative_portfolios: List[Dict[str, Any]]


class CompetitionPortfolioOptimizer:
    """コンペティションポートフォリオ最適化システム"""
    
    def __init__(self, resource_constraints: Optional[ResourceConstraints] = None):
        self.resource_constraints = resource_constraints or ResourceConstraints()
        self.medal_calculator = MedalProbabilityCalculator()
        self.logger = logging.getLogger(__name__)
        
        # 最適化パラメータ
        self.optimization_weights = {
            "medal_probability": 0.40,
            "resource_efficiency": 0.25,
            "risk_diversification": 0.20,
            "timing_advantage": 0.10,
            "domain_coverage": 0.05
        }
    
    async def optimize_portfolio(
        self,
        available_competitions: List[CompetitionData],
        strategy: PortfolioStrategy = PortfolioStrategy.BALANCED,
        current_portfolio: Optional[List[CompetitionData]] = None
    ) -> PortfolioOptimizationResult:
        """ポートフォリオ最適化実行"""
        
        try:
            self.logger.info(f"ポートフォリオ最適化開始: {len(available_competitions)}コンペ候補")
            
            # 各コンペのメダル確率・リソース要件算出
            portfolio_items = await self.evaluate_competitions(available_competitions)
            
            # 戦略別最適化実行
            if strategy == PortfolioStrategy.MAX_PROBABILITY:
                result = await self.optimize_max_probability(portfolio_items)
            elif strategy == PortfolioStrategy.RISK_DIVERSIFIED:
                result = await self.optimize_risk_diversified(portfolio_items)
            elif strategy == PortfolioStrategy.AGGRESSIVE:
                result = await self.optimize_aggressive(portfolio_items)
            elif strategy == PortfolioStrategy.CONSERVATIVE:
                result = await self.optimize_conservative(portfolio_items)
            else:  # BALANCED
                result = await self.optimize_balanced(portfolio_items)
            
            # 現在のポートフォリオとの比較・切り替え判断
            if current_portfolio:
                result = await self.evaluate_portfolio_transition(
                    current_result=result,
                    current_portfolio=current_portfolio
                )
            
            self.logger.info(f"ポートフォリオ最適化完了: {len(result.selected_competitions)}コンペ選択")
            return result
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ最適化失敗: {e}")
            raise
    
    async def evaluate_competitions(
        self, 
        competitions: List[CompetitionData]
    ) -> List[CompetitionPortfolioItem]:
        """各コンペの評価・ポートフォリオアイテム作成"""
        
        portfolio_items = []
        
        for competition in competitions:
            # メダル確率算出
            medal_result = await self.medal_calculator.calculate_medal_probability(competition)
            
            # リソース要件推定
            resource_requirements = self.estimate_resource_requirements(
                competition, medal_result
            )
            
            # 優先度・リスク・機会スコア算出
            priority_score = self.calculate_priority_score(competition, medal_result)
            risk_score = self.calculate_risk_score(competition, medal_result)
            opportunity_score = self.calculate_opportunity_score(competition, medal_result)
            
            portfolio_item = CompetitionPortfolioItem(
                competition_data=competition,
                medal_probability_result=medal_result,
                estimated_resource_requirements=resource_requirements,
                priority_score=priority_score,
                risk_score=risk_score,
                opportunity_score=opportunity_score,
                selection_reason=""
            )
            
            portfolio_items.append(portfolio_item)
        
        # 優先度順ソート
        portfolio_items.sort(key=lambda x: x.priority_score, reverse=True)
        
        return portfolio_items
    
    def estimate_resource_requirements(
        self,
        competition: CompetitionData,
        medal_result: MedalProbabilityResult
    ) -> Dict[ResourceType, float]:
        """リソース要件推定"""
        
        # 基本時間要件（競技タイプ別）
        base_time_requirements = {
            "tabular": 12.0,         # 週12時間
            "computer_vision": 18.0, # 週18時間（GPU集約）
            "nlp": 15.0,            # 週15時間
            "time_series": 14.0,    # 週14時間
            "audio": 16.0,          # 週16時間
            "graph": 10.0,          # 週10時間
            "multi_modal": 20.0,    # 週20時間（複雑）
            "reinforcement_learning": 22.0  # 週22時間（実験集約）
        }
        
        comp_type = competition.competition_type.value
        base_time = base_time_requirements.get(comp_type, 15.0)
        
        # 参加者数による調整（高競争 = より多くの実験が必要）
        competition_multiplier = 1.0
        if competition.participant_count > 3000:
            competition_multiplier = 1.4
        elif competition.participant_count > 1500:
            competition_multiplier = 1.2
        elif competition.participant_count < 500:
            competition_multiplier = 0.8
        
        # 残り時間による調整
        time_pressure_multiplier = 1.0
        if competition.days_remaining < 14:
            time_pressure_multiplier = 1.6  # 短期集中
        elif competition.days_remaining < 30:
            time_pressure_multiplier = 1.3
        elif competition.days_remaining > 90:
            time_pressure_multiplier = 0.9  # 長期分散
        
        # 最終時間要件
        time_requirement = base_time * competition_multiplier * time_pressure_multiplier
        
        # GPU時間要件
        gpu_intensive_types = {"computer_vision", "nlp", "audio", "multi_modal", "reinforcement_learning"}
        if comp_type in gpu_intensive_types:
            gpu_requirement = time_requirement * 0.7  # 時間の70%がGPU使用
        else:
            gpu_requirement = time_requirement * 0.2  # 時間の20%がGPU使用
        
        # メンタルキャパシティ要件
        mental_capacity = self.estimate_mental_capacity_requirement(
            competition, medal_result
        )
        
        return {
            ResourceType.TIME: time_requirement,
            ResourceType.GPU_HOURS: gpu_requirement,
            ResourceType.MENTAL_CAPACITY: mental_capacity,
            ResourceType.FINANCIAL: 0.0  # 無料前提
        }
    
    def estimate_mental_capacity_requirement(
        self,
        competition: CompetitionData,
        medal_result: MedalProbabilityResult
    ) -> float:
        """メンタルキャパシティ要件推定"""
        
        base_capacity = 30.0  # 基本値
        
        # 競技複雑度による調整
        complexity_factors = {
            "tabular": 1.0,
            "computer_vision": 1.3,
            "nlp": 1.4,
            "time_series": 1.2,
            "audio": 1.5,
            "graph": 1.1,
            "multi_modal": 1.8,
            "reinforcement_learning": 2.0
        }
        
        complexity_multiplier = complexity_factors.get(
            competition.competition_type.value, 1.2
        )
        
        # ドメインマッチングによる調整（得意分野は負荷軽減）
        domain_factor = medal_result.factor_breakdown.get("domain_matching", 0.5)
        domain_adjustment = 2.0 - domain_factor  # 0.5-1.5の範囲
        
        # 競争激化度による調整
        if competition.participant_count > 3000:
            competition_stress = 1.5  # 高ストレス
        elif competition.participant_count > 1500:
            competition_stress = 1.2
        else:
            competition_stress = 1.0
        
        mental_capacity = base_capacity * complexity_multiplier * domain_adjustment * competition_stress
        
        return min(mental_capacity, 80.0)  # 最大80ポイント
    
    def calculate_priority_score(
        self,
        competition: CompetitionData,
        medal_result: MedalProbabilityResult
    ) -> float:
        """優先度スコア算出"""
        
        # メダル確率ベーススコア
        base_score = medal_result.overall_probability * 100
        
        # 金メダル重み付け（「金メダル1個 > 銀メダル2個」原則）
        gold_bonus = medal_result.gold_probability * 50
        silver_bonus = medal_result.silver_probability * 20
        bronze_bonus = medal_result.bronze_probability * 10
        
        medal_weighted_score = base_score + gold_bonus + silver_bonus + bronze_bonus
        
        # 時間的緊急度
        urgency_bonus = 0
        if competition.days_remaining <= 7:
            urgency_bonus = 20  # 緊急度高
        elif competition.days_remaining <= 14:
            urgency_bonus = 10
        elif competition.days_remaining >= 90:
            urgency_bonus = -5  # 時間余裕によるペナルティ
        
        # 専門性マッチボーナス
        domain_match = medal_result.factor_breakdown.get("domain_matching", 0.5)
        domain_bonus = (domain_match - 0.5) * 30  # -15〜+15の範囲
        
        priority_score = medal_weighted_score + urgency_bonus + domain_bonus
        
        return max(0, priority_score)
    
    def calculate_risk_score(
        self,
        competition: CompetitionData,
        medal_result: MedalProbabilityResult
    ) -> float:
        """リスクスコア算出"""
        
        risk_factors = []
        
        # 参加者数リスク
        if competition.participant_count > 4000:
            risk_factors.append(40)  # 高リスク
        elif competition.participant_count > 2000:
            risk_factors.append(20)  # 中リスク
        else:
            risk_factors.append(5)   # 低リスク
        
        # 賞金リスク（高賞金 = 強豪集中）
        if competition.total_prize > 100000:
            risk_factors.append(35)
        elif competition.total_prize > 50000:
            risk_factors.append(20)
        else:
            risk_factors.append(10)
        
        # 専門性不一致リスク
        domain_match = medal_result.factor_breakdown.get("domain_matching", 0.5)
        if domain_match < 0.4:
            risk_factors.append(30)  # 専門外リスク
        elif domain_match < 0.6:
            risk_factors.append(15)
        else:
            risk_factors.append(5)
        
        # 時間制約リスク
        if competition.days_remaining < 7:
            risk_factors.append(25)  # 時間不足リスク
        elif competition.days_remaining < 14:
            risk_factors.append(10)
        
        # 信頼区間幅（不確実性）
        ci_lower, ci_upper = medal_result.confidence_interval
        uncertainty = ci_upper - ci_lower
        uncertainty_risk = uncertainty * 50  # 0-50の範囲
        risk_factors.append(uncertainty_risk)
        
        return sum(risk_factors)
    
    def calculate_opportunity_score(
        self,
        competition: CompetitionData,
        medal_result: MedalProbabilityResult
    ) -> float:
        """機会スコア算出"""
        
        opportunity_factors = []
        
        # 低競争機会
        if competition.participant_count < 800:
            opportunity_factors.append(30)  # 高機会
        elif competition.participant_count < 1500:
            opportunity_factors.append(15)  # 中機会
        
        # 専門性マッチ機会
        domain_match = medal_result.factor_breakdown.get("domain_matching", 0.5)
        if domain_match > 0.8:
            opportunity_factors.append(25)  # 高い専門性マッチ
        elif domain_match > 0.6:
            opportunity_factors.append(15)
        
        # 早期参入機会
        if competition.days_remaining > 60:
            opportunity_factors.append(20)  # 十分な準備時間
        elif competition.days_remaining > 30:
            opportunity_factors.append(10)
        
        # 低賞金機会（参加者質低下）
        if competition.total_prize < 10000:
            opportunity_factors.append(15)
        
        # 過去実績機会
        historical_factor = medal_result.factor_breakdown.get("historical_performance", 0.5)
        if historical_factor > 0.7:
            opportunity_factors.append(20)  # 過去の成功パターン
        
        return sum(opportunity_factors)
    
    async def optimize_max_probability(
        self, 
        portfolio_items: List[CompetitionPortfolioItem]
    ) -> PortfolioOptimizationResult:
        """確率最大化戦略"""
        
        # 上位3コンペを選択（リソース制約チェック付き）
        selected = []
        total_resources = {resource_type: 0.0 for resource_type in ResourceType}
        
        for item in portfolio_items:
            if len(selected) >= self.resource_constraints.max_concurrent_competitions:
                break
            
            # リソース制約チェック
            can_add = True
            temp_resources = total_resources.copy()
            
            for resource_type, requirement in item.estimated_resource_requirements.items():
                temp_resources[resource_type] += requirement
                
                # 制約チェック
                if resource_type == ResourceType.TIME and temp_resources[resource_type] > self.resource_constraints.weekly_time_hours:
                    can_add = False
                    break
                elif resource_type == ResourceType.GPU_HOURS and temp_resources[resource_type] > self.resource_constraints.gpu_hours_per_week:
                    can_add = False
                    break
                elif resource_type == ResourceType.MENTAL_CAPACITY and temp_resources[resource_type] > self.resource_constraints.mental_capacity_points:
                    can_add = False
                    break
            
            if can_add:
                selected.append(item)
                total_resources = temp_resources
                item.selection_reason = f"確率最大化: {item.medal_probability_result.overall_probability:.3f}"
        
        rejected = [item for item in portfolio_items if item not in selected]
        
        return self.create_optimization_result(
            selected, rejected, PortfolioStrategy.MAX_PROBABILITY,
            "メダル確率最大化を優先した選択", total_resources
        )
    
    async def optimize_risk_diversified(
        self, 
        portfolio_items: List[CompetitionPortfolioItem]
    ) -> PortfolioOptimizationResult:
        """リスク分散戦略"""
        
        # 競技タイプ・リスクレベル・時期の分散を考慮
        selected = []
        used_types = set()
        risk_levels = []
        
        # リスクレベル別グループ化
        low_risk = [item for item in portfolio_items if item.risk_score < 50]
        medium_risk = [item for item in portfolio_items if 50 <= item.risk_score < 100]
        high_risk = [item for item in portfolio_items if item.risk_score >= 100]
        
        # 各リスクレベルから1つずつ選択を試行
        candidate_groups = [low_risk, medium_risk, high_risk]
        
        for group in candidate_groups:
            if len(selected) >= self.resource_constraints.max_concurrent_competitions:
                break
            
            # グループ内で最高確率を選択
            group.sort(key=lambda x: x.medal_probability_result.overall_probability, reverse=True)
            
            for item in group:
                # 競技タイプ重複チェック
                comp_type = item.competition_data.competition_type
                if comp_type in used_types:
                    continue
                
                # リソース制約チェック
                if self.can_add_to_portfolio(selected, item):
                    selected.append(item)
                    used_types.add(comp_type)
                    item.selection_reason = f"リスク分散: {item.risk_score:.1f}リスクレベル"
                    break
        
        # 残り枠があれば最高確率で埋める
        remaining_items = [item for item in portfolio_items if item not in selected]
        for item in remaining_items:
            if len(selected) >= self.resource_constraints.max_concurrent_competitions:
                break
            
            if self.can_add_to_portfolio(selected, item):
                selected.append(item)
                item.selection_reason = "残り枠最適化"
        
        rejected = [item for item in portfolio_items if item not in selected]
        
        return self.create_optimization_result(
            selected, rejected, PortfolioStrategy.RISK_DIVERSIFIED,
            "リスク分散とタイプ多様化による安定化", self.calculate_total_resources(selected)
        )
    
    async def optimize_balanced(
        self, 
        portfolio_items: List[CompetitionPortfolioItem]
    ) -> PortfolioOptimizationResult:
        """バランス型戦略"""
        
        # 複数の組み合わせを評価してバランススコア最高を選択
        best_combination = None
        best_score = -1
        
        # 上位候補から組み合わせ生成
        top_candidates = portfolio_items[:min(8, len(portfolio_items))]
        
        for combination in itertools.combinations(top_candidates, min(3, len(top_candidates))):
            combination_list = list(combination)
            
            # リソース制約チェック
            if not self.meets_resource_constraints(combination_list):
                continue
            
            # バランススコア算出
            balance_score = self.calculate_balance_score(combination_list)
            
            if balance_score > best_score:
                best_score = balance_score
                best_combination = combination_list
        
        # フォールバック: 制約下で最高確率選択
        if not best_combination:
            return await self.optimize_max_probability(portfolio_items)
        
        for item in best_combination:
            item.selection_reason = f"バランス最適化: {best_score:.2f}スコア"
        
        rejected = [item for item in portfolio_items if item not in best_combination]
        
        return self.create_optimization_result(
            best_combination, rejected, PortfolioStrategy.BALANCED,
            "確率・リスク・効率のバランス最適化", self.calculate_total_resources(best_combination)
        )
    
    async def optimize_aggressive(
        self, 
        portfolio_items: List[CompetitionPortfolioItem]
    ) -> PortfolioOptimizationResult:
        """積極型戦略"""
        
        # 高確率・高リスクを含む組み合わせを選好
        # 金メダル確率重視
        portfolio_items_copy = portfolio_items.copy()
        portfolio_items_copy.sort(
            key=lambda x: x.medal_probability_result.gold_probability, 
            reverse=True
        )
        
        selected = []
        for item in portfolio_items_copy:
            if len(selected) >= self.resource_constraints.max_concurrent_competitions:
                break
            
            if self.can_add_to_portfolio(selected, item):
                selected.append(item)
                item.selection_reason = f"積極戦略: 金メダル確率{item.medal_probability_result.gold_probability:.3f}"
        
        rejected = [item for item in portfolio_items if item not in selected]
        
        return self.create_optimization_result(
            selected, rejected, PortfolioStrategy.AGGRESSIVE,
            "金メダル確率最大化による積極的選択", self.calculate_total_resources(selected)
        )
    
    async def optimize_conservative(
        self, 
        portfolio_items: List[CompetitionPortfolioItem]
    ) -> PortfolioOptimizationResult:
        """保守型戦略"""
        
        # 低リスク・安定した確率のコンペを選好
        # リスクスコア昇順でソート
        portfolio_items_copy = portfolio_items.copy()
        portfolio_items_copy.sort(key=lambda x: x.risk_score)
        
        # さらに確率でフィルタ（一定以上のもののみ）
        viable_items = [
            item for item in portfolio_items_copy 
            if item.medal_probability_result.overall_probability >= 0.3
        ]
        
        selected = []
        for item in viable_items:
            if len(selected) >= self.resource_constraints.max_concurrent_competitions:
                break
            
            if self.can_add_to_portfolio(selected, item):
                selected.append(item)
                item.selection_reason = f"保守戦略: {item.risk_score:.1f}低リスク"
        
        rejected = [item for item in portfolio_items if item not in selected]
        
        return self.create_optimization_result(
            selected, rejected, PortfolioStrategy.CONSERVATIVE,
            "低リスク安定型による安全な選択", self.calculate_total_resources(selected)
        )
    
    def can_add_to_portfolio(
        self, 
        current_selection: List[CompetitionPortfolioItem],
        new_item: CompetitionPortfolioItem
    ) -> bool:
        """ポートフォリオ追加可能性チェック"""
        
        test_selection = current_selection + [new_item]
        return self.meets_resource_constraints(test_selection)
    
    def meets_resource_constraints(
        self, 
        portfolio: List[CompetitionPortfolioItem]
    ) -> bool:
        """リソース制約充足チェック"""
        
        total_resources = self.calculate_total_resources(portfolio)
        
        # 各制約をチェック
        if total_resources[ResourceType.TIME] > self.resource_constraints.weekly_time_hours:
            return False
        if total_resources[ResourceType.GPU_HOURS] > self.resource_constraints.gpu_hours_per_week:
            return False
        if total_resources[ResourceType.MENTAL_CAPACITY] > self.resource_constraints.mental_capacity_points:
            return False
        
        return True
    
    def calculate_total_resources(
        self, 
        portfolio: List[CompetitionPortfolioItem]
    ) -> Dict[ResourceType, float]:
        """総リソース使用量計算"""
        
        total = {resource_type: 0.0 for resource_type in ResourceType}
        
        for item in portfolio:
            for resource_type, requirement in item.estimated_resource_requirements.items():
                total[resource_type] += requirement
        
        return total
    
    def calculate_balance_score(
        self, 
        portfolio: List[CompetitionPortfolioItem]
    ) -> float:
        """バランススコア算出"""
        
        if not portfolio:
            return 0.0
        
        # 各要素のスコア算出
        avg_probability = sum(
            item.medal_probability_result.overall_probability for item in portfolio
        ) / len(portfolio)
        
        avg_risk = sum(item.risk_score for item in portfolio) / len(portfolio)
        risk_factor = max(0, 1.0 - avg_risk / 100.0)  # リスク正規化
        
        avg_opportunity = sum(item.opportunity_score for item in portfolio) / len(portfolio)
        opportunity_factor = min(1.0, avg_opportunity / 50.0)  # 機会正規化
        
        # リソース効率
        total_resources = self.calculate_total_resources(portfolio)
        time_efficiency = 1.0 - (total_resources[ResourceType.TIME] / self.resource_constraints.weekly_time_hours)
        gpu_efficiency = 1.0 - (total_resources[ResourceType.GPU_HOURS] / self.resource_constraints.gpu_hours_per_week)
        resource_efficiency = (time_efficiency + gpu_efficiency) / 2
        
        # 多様性スコア
        diversity_score = self.calculate_diversity_score(portfolio)
        
        # 重み付きバランススコア
        balance_score = (
            avg_probability * 0.4 +
            risk_factor * 0.2 +
            opportunity_factor * 0.15 +
            resource_efficiency * 0.15 +
            diversity_score * 0.1
        )
        
        return balance_score
    
    def calculate_diversity_score(
        self, 
        portfolio: List[CompetitionPortfolioItem]
    ) -> float:
        """多様性スコア算出"""
        
        if not portfolio:
            return 0.0
        
        # 競技タイプ多様性
        comp_types = set(item.competition_data.competition_type for item in portfolio)
        type_diversity = len(comp_types) / len(portfolio)  # 最大1.0
        
        # リスクレベル多様性
        risk_levels = [item.risk_score for item in portfolio]
        risk_std = np.std(risk_levels) if len(risk_levels) > 1 else 0
        risk_diversity = min(1.0, risk_std / 50.0)  # 正規化
        
        # 時期多様性
        days_remaining = [item.competition_data.days_remaining for item in portfolio]
        time_std = np.std(days_remaining) if len(days_remaining) > 1 else 0
        time_diversity = min(1.0, time_std / 30.0)  # 正規化
        
        diversity_score = (type_diversity + risk_diversity + time_diversity) / 3
        
        return diversity_score
    
    def create_optimization_result(
        self,
        selected: List[CompetitionPortfolioItem],
        rejected: List[CompetitionPortfolioItem],
        strategy: PortfolioStrategy,
        rationale: str,
        resource_utilization: Dict[ResourceType, float]
    ) -> PortfolioOptimizationResult:
        """最適化結果作成"""
        
        # ポートフォリオメトリクス算出
        portfolio_metrics = self.calculate_portfolio_metrics(selected)
        
        # 期待メダル数
        expected_medal_count = sum(
            item.medal_probability_result.overall_probability for item in selected
        )
        
        # ポートフォリオリスクスコア
        portfolio_risk_score = sum(item.risk_score for item in selected) / max(1, len(selected))
        
        # 多様化スコア
        diversification_score = self.calculate_diversity_score(selected)
        
        # 代替ポートフォリオ提案
        alternative_portfolios = self.generate_alternative_portfolios(selected, rejected)
        
        return PortfolioOptimizationResult(
            selected_competitions=selected,
            rejected_competitions=rejected,
            portfolio_metrics=portfolio_metrics,
            resource_utilization=resource_utilization,
            expected_medal_count=expected_medal_count,
            portfolio_risk_score=portfolio_risk_score,
            diversification_score=diversification_score,
            optimization_strategy=strategy,
            selection_rationale=rationale,
            alternative_portfolios=alternative_portfolios
        )
    
    def calculate_portfolio_metrics(
        self, 
        portfolio: List[CompetitionPortfolioItem]
    ) -> Dict[str, float]:
        """ポートフォリオメトリクス算出"""
        
        if not portfolio:
            return {}
        
        probabilities = [item.medal_probability_result.overall_probability for item in portfolio]
        
        return {
            "avg_medal_probability": sum(probabilities) / len(probabilities),
            "max_medal_probability": max(probabilities),
            "min_medal_probability": min(probabilities),
            "probability_std": np.std(probabilities),
            "total_expected_medals": sum(probabilities),
            "portfolio_size": len(portfolio),
            "avg_gold_probability": sum(item.medal_probability_result.gold_probability for item in portfolio) / len(portfolio),
            "resource_efficiency": self.calculate_resource_efficiency(portfolio)
        }
    
    def calculate_resource_efficiency(
        self, 
        portfolio: List[CompetitionPortfolioItem]
    ) -> float:
        """リソース効率算出"""
        
        if not portfolio:
            return 0.0
        
        total_resources = self.calculate_total_resources(portfolio)
        total_expected_medals = sum(
            item.medal_probability_result.overall_probability for item in portfolio
        )
        
        # 時間あたりの期待メダル数
        time_efficiency = total_expected_medals / max(1, total_resources[ResourceType.TIME])
        
        return time_efficiency
    
    def generate_alternative_portfolios(
        self,
        selected: List[CompetitionPortfolioItem],
        rejected: List[CompetitionPortfolioItem]
    ) -> List[Dict[str, Any]]:
        """代替ポートフォリオ提案生成"""
        
        alternatives = []
        
        # より保守的な選択肢
        if rejected:
            conservative_alternative = self.create_conservative_alternative(selected, rejected)
            if conservative_alternative:
                alternatives.append(conservative_alternative)
        
        # より積極的な選択肢
        aggressive_alternative = self.create_aggressive_alternative(selected, rejected)
        if aggressive_alternative:
            alternatives.append(aggressive_alternative)
        
        return alternatives[:3]  # 最大3つの代替案
    
    def create_conservative_alternative(
        self,
        current_selected: List[CompetitionPortfolioItem],
        rejected: List[CompetitionPortfolioItem]
    ) -> Optional[Dict[str, Any]]:
        """保守的代替案作成"""
        
        # 現在の選択より低リスクな組み合わせを探索
        current_avg_risk = sum(item.risk_score for item in current_selected) / len(current_selected)
        
        low_risk_candidates = [
            item for item in rejected 
            if item.risk_score < current_avg_risk and 
            item.medal_probability_result.overall_probability >= 0.2
        ]
        
        if len(low_risk_candidates) >= 2:
            # 上位2つの低リスク候補を選択
            low_risk_candidates.sort(key=lambda x: x.medal_probability_result.overall_probability, reverse=True)
            alternative_selection = low_risk_candidates[:2]
            
            # 現在の選択から1つ維持
            if current_selected:
                best_current = max(current_selected, key=lambda x: x.medal_probability_result.overall_probability)
                alternative_selection.append(best_current)
            
            alternative_metrics = self.calculate_portfolio_metrics(alternative_selection)
            
            return {
                "type": "conservative",
                "description": "より低リスクな組み合わせ",
                "competitions": [item.competition_data.title for item in alternative_selection],
                "metrics": alternative_metrics,
                "risk_reduction": current_avg_risk - (sum(item.risk_score for item in alternative_selection) / len(alternative_selection))
            }
        
        return None
    
    def create_aggressive_alternative(
        self,
        current_selected: List[CompetitionPortfolioItem],
        rejected: List[CompetitionPortfolioItem]
    ) -> Optional[Dict[str, Any]]:
        """積極的代替案作成"""
        
        # 高確率だが選ばれなかったコンペで代替案作成
        high_prob_rejected = [
            item for item in rejected 
            if item.medal_probability_result.overall_probability >= 0.4
        ]
        
        if high_prob_rejected:
            # 最高確率の候補で現在の選択を一部置換
            best_rejected = max(high_prob_rejected, key=lambda x: x.medal_probability_result.overall_probability)
            
            if current_selected:
                # 現在の選択から最低確率を置換
                worst_current = min(current_selected, key=lambda x: x.medal_probability_result.overall_probability)
                alternative_selection = [item for item in current_selected if item != worst_current]
                alternative_selection.append(best_rejected)
                
                alternative_metrics = self.calculate_portfolio_metrics(alternative_selection)
                
                return {
                    "type": "aggressive",
                    "description": "より高確率な候補での置換",
                    "competitions": [item.competition_data.title for item in alternative_selection],
                    "metrics": alternative_metrics,
                    "probability_gain": best_rejected.medal_probability_result.overall_probability - worst_current.medal_probability_result.overall_probability
                }
        
        return None
    
    async def evaluate_portfolio_transition(
        self,
        current_result: PortfolioOptimizationResult,
        current_portfolio: List[CompetitionData]
    ) -> PortfolioOptimizationResult:
        """現在のポートフォリオからの移行評価"""
        
        # 現在のポートフォリオの評価
        current_items = await self.evaluate_competitions(current_portfolio)
        current_metrics = self.calculate_portfolio_metrics(current_items)
        
        # 移行メリットの評価
        new_expected_medals = current_result.expected_medal_count
        current_expected_medals = current_metrics.get("total_expected_medals", 0)
        
        medal_improvement = new_expected_medals - current_expected_medals
        
        # 移行コスト考慮
        transition_cost = self.calculate_transition_cost(current_portfolio, current_result.selected_competitions)
        
        # 移行推奨閾値
        min_improvement_threshold = 0.1  # 10%以上の改善で移行推奨
        
        if medal_improvement > min_improvement_threshold or transition_cost < 0.2:
            current_result.selection_rationale += f" (現在比+{medal_improvement:.2f}メダル期待値改善)"
        else:
            current_result.selection_rationale += f" (現在比+{medal_improvement:.2f}、移行推奨せず)"
        
        return current_result
    
    def calculate_transition_cost(
        self,
        current_portfolio: List[CompetitionData],
        new_portfolio: List[CompetitionPortfolioItem]
    ) -> float:
        """移行コスト算出"""
        
        current_titles = set(comp.title for comp in current_portfolio)
        new_titles = set(item.competition_data.title for item in new_portfolio)
        
        # 変更が必要なコンペ数
        changes_needed = len(current_titles.symmetric_difference(new_titles))
        
        # 正規化された移行コスト（0-1スケール）
        max_changes = max(len(current_titles), len(new_titles), 1)
        transition_cost = changes_needed / max_changes
        
        return transition_cost