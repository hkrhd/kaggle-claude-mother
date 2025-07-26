"""
メダル確率算出エンジン

多次元確率モデルによるKaggleメダル獲得確率の高精度算出。
参加者数・賞金・専門性マッチング・時間的優位性・過去実績を統合評価。
"""

import asyncio
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CompetitionType(Enum):
    """コンペ種別"""
    TABULAR = "tabular"
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    TIME_SERIES = "time_series"
    AUDIO = "audio"
    GRAPH = "graph"
    MULTI_MODAL = "multi_modal"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


class PrizeType(Enum):
    """賞金種別"""
    MONETARY = "monetary"
    CAREER = "career"
    KNOWLEDGE = "knowledge"
    MIXED = "mixed"


@dataclass
class CompetitionData:
    """コンペティションデータ"""
    competition_id: str
    title: str
    participant_count: int
    total_prize: float
    prize_type: PrizeType
    competition_type: CompetitionType
    days_remaining: int
    data_characteristics: Dict[str, Any]
    skill_requirements: List[str]
    leaderboard_competition: float  # 競争激化度
    historical_similar_comps: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertiseProfile:
    """専門性プロファイル"""
    tabular_skill_level: float = 0.7
    computer_vision_skill: float = 0.6
    nlp_skill_level: float = 0.8
    audio_processing_skill: float = 0.4
    time_series_skill: float = 0.7
    graph_analysis_skill: float = 0.5
    
    # 技術スキル
    deep_learning_expertise: float = 0.8
    classical_ml_expertise: float = 0.9
    feature_engineering_skill: float = 0.8
    ensemble_methods_skill: float = 0.7
    optimization_skill: float = 0.6
    
    # メタスキル
    fast_prototyping: float = 0.8
    debugging_skill: float = 0.7
    competition_strategy: float = 0.6


@dataclass
class MedalProbabilityResult:
    """メダル確率算出結果"""
    overall_medal_probability: float
    bronze_probability: float
    silver_probability: float
    gold_probability: float
    confidence_interval: Tuple[float, float]
    factor_breakdown: Dict[str, float]
    risk_factors: List[str]
    opportunity_factors: List[str]
    recommendation: str
    calculation_metadata: Dict[str, Any]


class MedalProbabilityCalculator:
    """メダル確率算出エンジン"""
    
    def __init__(self):
        self.weight_factors = {
            "participant_count": 0.25,      # 参加者数による競争激化
            "prize_amount": 0.20,           # 賞金による参加者質向上
            "domain_expertise": 0.30,       # 専門分野マッチング度
            "time_remaining": 0.15,         # 残り時間による参入優位性
            "historical_performance": 0.10  # 過去類似コンペでの実績
        }
        
        self.expertise_profile = ExpertiseProfile()
        self.historical_data = self.load_historical_data()
        self.logger = logging.getLogger(__name__)
    
    def load_historical_data(self) -> Dict[str, Any]:
        """過去データ読み込み"""
        return {
            "participant_averages": {
                "tabular": 2500,
                "computer_vision": 1800,
                "nlp": 2200,
                "time_series": 1500,
                "audio": 800,
                "graph": 600
            },
            "prize_quality_correlation": 0.7,  # 賞金と参加者質の相関
            "optimal_entry_patterns": {
                "early_entry_bonus": 0.15,  # 早期参入ボーナス
                "late_entry_penalty": 0.10   # 遅い参入ペナルティ
            },
            "medal_thresholds": {
                "high_competition": {"gold": 0.05, "silver": 0.10, "bronze": 0.20},
                "medium_competition": {"gold": 0.08, "silver": 0.15, "bronze": 0.30},
                "low_competition": {"gold": 0.12, "silver": 0.25, "bronze": 0.40}
            }
        }
    
    async def calculate_medal_probability(
        self, 
        competition_data: CompetitionData
    ) -> MedalProbabilityResult:
        """メダル確率算出メイン処理"""
        
        try:
            self.logger.info(f"メダル確率算出開始: {competition_data.title}")
            
            # 各要因の算出
            participant_factor = self.calculate_participant_impact(competition_data)
            prize_factor = self.calculate_prize_impact(competition_data)
            domain_factor = await self.calculate_domain_matching(competition_data)
            timing_factor = self.calculate_timing_advantage(competition_data)
            historical_factor = await self.calculate_historical_performance_factor(competition_data)
            
            # 重み付き総合確率計算
            medal_probability = (
                participant_factor * self.weight_factors["participant_count"] +
                prize_factor * self.weight_factors["prize_amount"] +
                domain_factor * self.weight_factors["domain_expertise"] +
                timing_factor * self.weight_factors["time_remaining"] +
                historical_factor * self.weight_factors["historical_performance"]
            )
            
            # 信頼区間計算
            confidence_interval = self.calculate_confidence_interval(
                medal_probability, competition_data
            )
            
            # リスク・機会要因分析
            risk_factors, opportunity_factors = self.analyze_risk_opportunity_factors(
                competition_data, {
                    "participant_factor": participant_factor,
                    "prize_factor": prize_factor,
                    "domain_factor": domain_factor,
                    "timing_factor": timing_factor,
                    "historical_factor": historical_factor
                }
            )
            
            # 推奨行動
            recommendation = self.generate_recommendation(
                medal_probability, competition_data, risk_factors, opportunity_factors
            )
            
            result = MedalProbabilityResult(
                overall_medal_probability=medal_probability,
                bronze_probability=medal_probability * 0.6,
                silver_probability=medal_probability * 0.4,
                gold_probability=medal_probability * 0.2,
                confidence_interval=confidence_interval,
                factor_breakdown={
                    "participant_impact": participant_factor,
                    "prize_impact": prize_factor,
                    "domain_matching": domain_factor,
                    "timing_advantage": timing_factor,
                    "historical_performance": historical_factor
                },
                risk_factors=risk_factors,
                opportunity_factors=opportunity_factors,
                recommendation=recommendation,
                calculation_metadata={
                    "calculation_time": datetime.utcnow().isoformat(),
                    "weight_factors_used": self.weight_factors.copy(),
                    "expertise_profile_version": "1.0"
                }
            )
            
            self.logger.info(f"メダル確率算出完了: {medal_probability:.3f}")
            return result
            
        except Exception as e:
            self.logger.error(f"メダル確率算出失敗: {e}")
            raise
    
    def calculate_participant_impact(self, competition_data: CompetitionData) -> float:
        """参加者数による競争激化度算出"""
        
        participant_count = competition_data.participant_count
        comp_type = competition_data.competition_type.value
        
        # 競技別平均参加者数との比較
        avg_participants = self.historical_data["participant_averages"].get(comp_type, 2000)
        
        # 正規化された競争激化度 (0-1スケール)
        if participant_count <= avg_participants * 0.5:
            # 参加者少数：高確率
            factor = 0.8 + (avg_participants * 0.5 - participant_count) / (avg_participants * 0.5) * 0.2
        elif participant_count <= avg_participants:
            # 標準的参加者数：中程度確率  
            factor = 0.5 + (avg_participants - participant_count) / (avg_participants * 0.5) * 0.3
        elif participant_count <= avg_participants * 2:
            # 高競争：低確率
            factor = 0.2 + (avg_participants * 2 - participant_count) / avg_participants * 0.3
        else:
            # 超高競争：極低確率
            factor = max(0.05, 0.2 * (avg_participants * 3 - participant_count) / avg_participants)
        
        return min(1.0, max(0.0, factor))
    
    def calculate_prize_impact(self, competition_data: CompetitionData) -> float:
        """賞金による参加者質・競争激化度算出"""
        
        prize_amount = competition_data.total_prize
        prize_type = competition_data.prize_type
        
        # 賞金タイプ別基本係数
        prize_type_factors = {
            PrizeType.MONETARY: 1.0,
            PrizeType.CAREER: 0.8,
            PrizeType.KNOWLEDGE: 0.6,
            PrizeType.MIXED: 0.9
        }
        
        base_factor = prize_type_factors[prize_type]
        
        # 賞金額による調整
        if prize_amount >= 100000:  # $100k+
            prize_factor = 0.2  # 超高賞金：トップレベル参加者集中
        elif prize_amount >= 50000:   # $50k+
            prize_factor = 0.3
        elif prize_amount >= 25000:   # $25k+
            prize_factor = 0.5
        elif prize_amount >= 10000:   # $10k+
            prize_factor = 0.7
        else:  # $10k未満
            prize_factor = 0.8  # 低賞金：参加者質が相対的に低い
        
        return prize_factor * base_factor
    
    async def calculate_domain_matching(self, competition_data: CompetitionData) -> float:
        """専門分野マッチング度算出"""
        
        comp_type = competition_data.competition_type
        skill_requirements = competition_data.skill_requirements
        
        # データタイプ別得意分野マッチング
        data_type_matching = {
            CompetitionType.TABULAR: self.expertise_profile.tabular_skill_level,
            CompetitionType.COMPUTER_VISION: self.expertise_profile.computer_vision_skill,
            CompetitionType.NLP: self.expertise_profile.nlp_skill_level,
            CompetitionType.TIME_SERIES: self.expertise_profile.time_series_skill,
            CompetitionType.AUDIO: self.expertise_profile.audio_processing_skill,
            CompetitionType.GRAPH: self.expertise_profile.graph_analysis_skill,
            CompetitionType.MULTI_MODAL: (
                self.expertise_profile.computer_vision_skill * 0.4 +
                self.expertise_profile.nlp_skill_level * 0.4 +
                self.expertise_profile.tabular_skill_level * 0.2
            ),
            CompetitionType.REINFORCEMENT_LEARNING: (
                self.expertise_profile.deep_learning_expertise * 0.7 +
                self.expertise_profile.optimization_skill * 0.3
            )
        }
        
        base_matching = data_type_matching.get(comp_type, 0.5)
        
        # 技術要件マッチング
        technique_matching = await self.assess_technique_requirements(skill_requirements)
        
        # 複合スコア計算
        domain_factor = (base_matching * 0.6 + technique_matching * 0.4)
        
        return min(1.0, max(0.0, domain_factor))
    
    async def assess_technique_requirements(self, skill_requirements: List[str]) -> float:
        """技術要件マッチング評価"""
        
        if not skill_requirements:
            return 0.7  # デフォルト値
        
        technique_mapping = {
            "deep_learning": self.expertise_profile.deep_learning_expertise,
            "neural_networks": self.expertise_profile.deep_learning_expertise,
            "cnn": self.expertise_profile.computer_vision_skill,
            "rnn": self.expertise_profile.nlp_skill_level,
            "transformer": self.expertise_profile.nlp_skill_level,
            "ensemble": self.expertise_profile.ensemble_methods_skill,
            "stacking": self.expertise_profile.ensemble_methods_skill,
            "boosting": self.expertise_profile.classical_ml_expertise,
            "feature_engineering": self.expertise_profile.feature_engineering_skill,
            "time_series": self.expertise_profile.time_series_skill,
            "optimization": self.expertise_profile.optimization_skill,
            "clustering": self.expertise_profile.classical_ml_expertise,
            "classification": self.expertise_profile.classical_ml_expertise,
            "regression": self.expertise_profile.classical_ml_expertise
        }
        
        matched_scores = []
        for requirement in skill_requirements:
            requirement_lower = requirement.lower()
            for tech, score in technique_mapping.items():
                if tech in requirement_lower:
                    matched_scores.append(score)
                    break
            else:
                matched_scores.append(0.5)  # 未知技術のデフォルト
        
        return statistics.mean(matched_scores) if matched_scores else 0.5
    
    def calculate_timing_advantage(self, competition_data: CompetitionData) -> float:
        """時間的優位性算出"""
        
        days_remaining = competition_data.days_remaining
        leaderboard_competition = competition_data.leaderboard_competition
        
        # 早期参入ボーナス
        if days_remaining >= 60:  # 2ヶ月以上
            timing_base = 0.9  # 十分な開発時間
        elif days_remaining >= 30:  # 1ヶ月以上
            timing_base = 0.8
        elif days_remaining >= 14:  # 2週間以上
            timing_base = 0.6
        elif days_remaining >= 7:   # 1週間以上
            timing_base = 0.4
        else:  # 1週間未満
            timing_base = 0.2  # 時間不足による低確率
        
        # リーダーボード競争激化による調整
        competition_penalty = leaderboard_competition * 0.2  # 最大20%減少
        
        timing_factor = timing_base * (1 - competition_penalty)
        
        return min(1.0, max(0.1, timing_factor))
    
    async def calculate_historical_performance_factor(
        self, 
        competition_data: CompetitionData
    ) -> float:
        """過去実績による成功予測"""
        
        # 類似コンペの特定
        similar_competitions = await self.find_similar_competitions(competition_data)
        
        if not similar_competitions:
            return 0.6  # デフォルト値
        
        # 過去実績の分析（モック実装）
        # 実際の実装では、データベースから過去の成績を取得
        mock_past_results = {
            "total_competitions": len(similar_competitions),
            "medal_count": max(1, len(similar_competitions) // 3),  # 1/3がメダル獲得と仮定
            "avg_ranking_percentile": 0.25,  # 平均上位25%と仮定
            "recent_trend": "improving"  # 最近の傾向
        }
        
        # 成功率ベース計算
        success_rate = mock_past_results["medal_count"] / max(1, mock_past_results["total_competitions"])
        
        # トレンド調整
        trend_adjustment = {
            "improving": 1.2,
            "stable": 1.0,
            "declining": 0.8
        }
        
        adjusted_factor = success_rate * trend_adjustment.get(
            mock_past_results["recent_trend"], 1.0
        )
        
        return min(1.0, max(0.1, adjusted_factor))
    
    async def find_similar_competitions(self, competition_data: CompetitionData) -> List[str]:
        """類似コンペティション特定"""
        
        # モック実装：実際はコンペデータベースから類似検索
        similar_comps = []
        
        # 同一競技タイプの過去コンペ
        comp_type = competition_data.competition_type.value
        similar_comps.extend([
            f"past_{comp_type}_comp_1",
            f"past_{comp_type}_comp_2", 
            f"past_{comp_type}_comp_3"
        ])
        
        # 参加者数・賞金が類似するコンペ
        similar_comps.extend([
            f"similar_size_comp_1",
            f"similar_prize_comp_1"
        ])
        
        return similar_comps[:10]  # 最大10件
    
    def calculate_confidence_interval(
        self, 
        medal_probability: float, 
        competition_data: CompetitionData
    ) -> Tuple[float, float]:
        """信頼区間算出"""
        
        # 不確実性要因
        uncertainty_factors = []
        
        # 参加者数による不確実性
        if competition_data.participant_count > 3000:
            uncertainty_factors.append(0.15)  # 高い不確実性
        elif competition_data.participant_count < 500:
            uncertainty_factors.append(0.10)  # 中程度の不確実性
        else:
            uncertainty_factors.append(0.05)  # 低い不確実性
        
        # 時間による不確実性
        if competition_data.days_remaining > 90:
            uncertainty_factors.append(0.20)  # 長期間の不確実性
        elif competition_data.days_remaining < 14:
            uncertainty_factors.append(0.15)  # 短期間の情報不足
        else:
            uncertainty_factors.append(0.08)
        
        # 総合不確実性
        total_uncertainty = statistics.mean(uncertainty_factors)
        
        # 95%信頼区間
        margin = medal_probability * total_uncertainty * 1.96
        
        lower_bound = max(0.0, medal_probability - margin)
        upper_bound = min(1.0, medal_probability + margin)
        
        return (lower_bound, upper_bound)
    
    def analyze_risk_opportunity_factors(
        self,
        competition_data: CompetitionData,
        factor_scores: Dict[str, float]
    ) -> Tuple[List[str], List[str]]:
        """リスク・機会要因分析"""
        
        risk_factors = []
        opportunity_factors = []
        
        # 参加者数リスク
        if competition_data.participant_count > 4000:
            risk_factors.append("超高競争：4000人以上参加")
        elif competition_data.participant_count < 800:
            opportunity_factors.append("低競争：参加者800人未満")
        
        # 賞金リスク・機会
        if competition_data.total_prize > 100000:
            risk_factors.append("高賞金による強豪参加者集中")
        elif competition_data.total_prize < 10000:
            opportunity_factors.append("低賞金による参加者質低下")
            
        # 専門性マッチング
        domain_score = factor_scores["domain_factor"]
        if domain_score > 0.8:
            opportunity_factors.append("高い専門性マッチング")
        elif domain_score < 0.4:
            risk_factors.append("専門分野の不一致")
        
        # 時間的要因
        if competition_data.days_remaining < 7:
            risk_factors.append("開発時間不足")
        elif competition_data.days_remaining > 60:
            opportunity_factors.append("十分な開発・実験時間")
        
        # 競争激化度
        if competition_data.leaderboard_competition > 0.8:
            risk_factors.append("リーダーボード激戦")
        elif competition_data.leaderboard_competition < 0.3:
            opportunity_factors.append("リーダーボード余裕")
        
        return risk_factors, opportunity_factors
    
    def generate_recommendation(
        self,
        medal_probability: float,
        competition_data: CompetitionData,
        risk_factors: List[str],
        opportunity_factors: List[str]
    ) -> str:
        """推奨行動生成"""
        
        if medal_probability >= 0.7:
            base_recommendation = "強く推奨：高いメダル獲得可能性"
        elif medal_probability >= 0.5:
            base_recommendation = "推奨：中程度のメダル獲得可能性"
        elif medal_probability >= 0.3:
            base_recommendation = "条件付き推奨：リスク要因を慎重に評価"
        else:
            base_recommendation = "非推奨：メダル獲得確率が低い"
        
        # リスク・機会要因による補足
        additional_notes = []
        
        if len(opportunity_factors) > len(risk_factors):
            additional_notes.append("機会要因が上回る")
        elif len(risk_factors) > len(opportunity_factors):
            additional_notes.append("リスク要因が多い")
        
        if competition_data.days_remaining < 14:
            additional_notes.append("早急な参加判断が必要")
        
        if additional_notes:
            recommendation = f"{base_recommendation} ({', '.join(additional_notes)})"
        else:
            recommendation = base_recommendation
        
        return recommendation
    
    def update_weight_factors(self, performance_feedback: Dict[str, Any]):
        """重み係数の動的更新"""
        
        # 実際の成果に基づく重み調整
        if "actual_results" in performance_feedback:
            actual_medal = performance_feedback["actual_results"].get("medal_achieved", False)
            predicted_probability = performance_feedback.get("predicted_probability", 0.5)
            
            # 予測精度による調整
            if actual_medal and predicted_probability < 0.5:
                # 低予測だったが成功：専門性・過去実績の重み増加
                self.weight_factors["domain_expertise"] *= 1.1
                self.weight_factors["historical_performance"] *= 1.1
            elif not actual_medal and predicted_probability > 0.7:
                # 高予測だったが失敗：参加者数・賞金の重み増加
                self.weight_factors["participant_count"] *= 1.1
                self.weight_factors["prize_amount"] *= 1.1
            
            # 正規化
            total_weight = sum(self.weight_factors.values())
            for key in self.weight_factors:
                self.weight_factors[key] /= total_weight
        
        self.logger.info(f"重み係数更新: {self.weight_factors}")
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """算出統計取得"""
        
        return {
            "current_weight_factors": self.weight_factors.copy(),
            "expertise_profile": {
                "tabular": self.expertise_profile.tabular_skill_level,
                "cv": self.expertise_profile.computer_vision_skill,
                "nlp": self.expertise_profile.nlp_skill_level,
                "ts": self.expertise_profile.time_series_skill
            },
            "historical_data_version": "1.0",
            "last_updated": datetime.utcnow().isoformat()
        }