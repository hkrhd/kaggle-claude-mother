"""
メダル確率算出エンジン

多次元確率モデルによる精密なメダル獲得可能性算出。
plan_planner.md の確率算出アルゴリズム仕様に準拠。
"""

import asyncio
import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import statistics

try:
    import numpy as np
    import pandas as pd
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
        def sqrt(x):
            return math.sqrt(x)
    
    class MockPandas:
        class DataFrame:
            def __init__(self, data):
                self.data = data
            def mean(self):
                return 0.5
            def std(self):
                return 0.1
    
    np = MockNumpy()
    pd = MockPandas()

from ..models.competition import CompetitionInfo, CompetitionType, PrizeType
from ..models.probability import (
    MedalProbability, ProbabilityFactors, ConfidenceLevel,
    calculate_base_probability, calculate_domain_matching,
    calculate_timing_factor, calculate_resource_efficiency
)


class MedalProbabilityCalculator:
    """メダル確率算出エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 個人スキルセット（設定可能）
        self.personal_skills = [
            "python", "pandas", "numpy", "scikit_learn", "xgboost", "lightgbm",
            "feature_engineering", "ensemble", "cross_validation", "hyperparameter_tuning",
            "data_cleaning", "eda", "visualization", "statistics"
        ]
        
        # リソース設定
        self.available_compute_hours = 100.0    # 週当たり利用可能計算時間
        self.available_human_hours = 20.0       # 週当たり利用可能作業時間
        
        # 過去実績データ（学習により更新）
        self.historical_performance = {
            "competitions_participated": 5,
            "medals_won": 1,
            "average_rank_percentile": 0.25,
            "best_rank_percentile": 0.05,
            "favorite_competition_types": [CompetitionType.TABULAR, CompetitionType.TIME_SERIES]
        }
        
        # 計算重み設定
        self.factor_weights = {
            "base_probability": 0.25,
            "domain_matching": 0.20,
            "timing_factor": 0.15,
            "resource_efficiency": 0.15,
            "competition_intensity": 0.10,
            "external_advantages": 0.10,
            "historical_performance": 0.05
        }
    
    async def calculate_medal_probability(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[Dict[str, Any]] = None
    ) -> MedalProbability:
        """メダル確率算出メイン処理"""
        
        start_time = datetime.utcnow()
        self.logger.info(f"メダル確率算出開始: {competition_info.title}")
        
        try:
            # 基本要因計算
            probability_factors = await self._calculate_probability_factors(
                competition_info, current_status
            )
            
            # 多次元確率モデル適用
            medal_probabilities = await self._apply_multidimensional_model(
                competition_info, probability_factors
            )
            
            # 信頼区間・リスク分析
            confidence_analysis = await self._calculate_confidence_analysis(
                competition_info, probability_factors, medal_probabilities
            )
            
            # 期待順位・メダル圏推定
            ranking_analysis = await self._calculate_ranking_analysis(
                competition_info, medal_probabilities
            )
            
            # 結果統合
            result = MedalProbability(
                competition_info=competition_info,
                calculation_timestamp=start_time,
                overall_probability=medal_probabilities["overall"],
                gold_probability=medal_probabilities["gold"],
                silver_probability=medal_probabilities["silver"],
                bronze_probability=medal_probabilities["bronze"],
                confidence_interval=confidence_analysis["interval"],
                confidence_level=confidence_analysis["level"],
                probability_factors=probability_factors,
                factor_breakdown=await self._create_factor_breakdown(probability_factors),
                expected_rank=ranking_analysis["expected_rank"],
                expected_percentile=ranking_analysis["expected_percentile"],
                medal_cutoff_estimates=ranking_analysis["medal_cutoffs"],
                risk_factors=confidence_analysis["risk_factors"],
                uncertainty_sources=confidence_analysis["uncertainty_sources"],
                historical_comparison=await self._create_historical_comparison(competition_info),
                calculation_method="multi_dimensional_model_v1",
                model_version="1.0",
                data_sources=["competition_basic", "historical_data", "personal_profile"],
                calculation_duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            # メダル圏カットオフ更新
            result.update_medal_cutoffs()
            
            self.logger.info(f"メダル確率算出完了: {result.overall_probability:.3f} ({competition_info.title})")
            return result
            
        except Exception as e:
            self.logger.error(f"メダル確率算出失敗: {e}")
            raise
    
    async def _calculate_probability_factors(
        self,
        competition_info: CompetitionInfo,
        current_status: Optional[Dict[str, Any]]
    ) -> ProbabilityFactors:
        """確率要因計算"""
        
        # 基本要因
        participant_factor = await self._calculate_participant_factor(competition_info)
        prize_factor = await self._calculate_prize_factor(competition_info)
        competition_type_factor = await self._calculate_competition_type_factor(competition_info)
        
        # マッチング要因
        domain_matching = calculate_domain_matching(
            competition_info.competition_type,
            competition_info.skill_requirements,
            self.personal_skills
        )
        skill_alignment = await self._calculate_skill_alignment(competition_info)
        experience_factor = await self._calculate_experience_factor(competition_info)
        
        # 時間要因
        timing_factor = calculate_timing_factor(competition_info.days_remaining)
        remaining_time_factor = await self._calculate_remaining_time_factor(competition_info)
        urgency_weight = await self._calculate_urgency_weight(competition_info)
        
        # リソース要因
        estimated_compute = await self._estimate_required_compute(competition_info)
        estimated_human = await self._estimate_required_human_time(competition_info)
        
        resource_efficiency = calculate_resource_efficiency(
            estimated_compute, self.available_compute_hours,
            estimated_human, self.available_human_hours
        )
        compute_availability = min(1.0, self.available_compute_hours / max(1.0, estimated_compute))
        time_availability = min(1.0, self.available_human_hours / max(1.0, estimated_human))
        
        # 競争要因
        competition_intensity = await self._calculate_competition_intensity(competition_info)
        leaderboard_volatility = await self._calculate_leaderboard_volatility(competition_info)
        late_entry_penalty = await self._calculate_late_entry_penalty(competition_info)
        
        # 外部要因
        external_data_advantage = await self._calculate_external_data_advantage(competition_info)
        community_activity = await self._calculate_community_activity(competition_info)
        past_solution_reference = await self._calculate_past_solution_reference(competition_info)
        
        return ProbabilityFactors(
            participant_factor=participant_factor,
            prize_factor=prize_factor,
            competition_type_factor=competition_type_factor,
            domain_matching=domain_matching,
            skill_alignment=skill_alignment,
            experience_factor=experience_factor,
            timing_factor=timing_factor,
            remaining_time_factor=remaining_time_factor,
            urgency_weight=urgency_weight,
            resource_efficiency=resource_efficiency,
            compute_availability=compute_availability,
            time_availability=time_availability,
            competition_intensity=competition_intensity,
            leaderboard_volatility=leaderboard_volatility,
            late_entry_penalty=late_entry_penalty,
            external_data_advantage=external_data_advantage,
            community_activity=community_activity,
            past_solution_reference=past_solution_reference
        )
    
    async def _calculate_participant_factor(self, competition_info: CompetitionInfo) -> float:
        """参加者数要因計算"""
        participant_count = competition_info.participant_count
        
        if participant_count == 0:
            return 0.5  # データ不足時のデフォルト
        
        # 対数スケールでの正規化（参加者数が多いほど困難）
        if participant_count < 50:
            return 0.9  # 小規模コンペは有利
        elif participant_count < 200:
            return 0.8
        elif participant_count < 500:
            return 0.7
        elif participant_count < 1000:
            return 0.6
        elif participant_count < 2000:
            return 0.5
        elif participant_count < 5000:
            return 0.4
        else:
            return 0.3  # 大規模コンペは不利
    
    async def _calculate_prize_factor(self, competition_info: CompetitionInfo) -> float:
        """賞金要因計算"""
        total_prize = competition_info.total_prize
        prize_type = competition_info.prize_type
        
        # 賞金種別による基本係数
        type_factors = {
            PrizeType.MONETARY: 1.0,    # 金銭賞（基準）
            PrizeType.POINTS: 0.8,      # ポイント賞
            PrizeType.KNOWLEDGE: 1.2,   # 学習コンペ（競争緩和）  
            PrizeType.SWAG: 0.9         # グッズ賞
        }
        
        base_factor = type_factors.get(prize_type, 1.0)
        
        if prize_type != PrizeType.MONETARY or total_prize == 0:
            return base_factor * 0.8  # 非金銭賞は競争緩和
        
        # 金銭賞の場合の競争激化係数
        if total_prize >= 100000:
            competition_factor = 0.6  # 高額賞金は競争激化
        elif total_prize >= 50000:
            competition_factor = 0.7
        elif total_prize >= 25000:
            competition_factor = 0.8
        elif total_prize >= 10000:
            competition_factor = 0.9
        elif total_prize >= 5000:
            competition_factor = 1.0
        else:
            competition_factor = 1.1  # 低額賞金は競争緩和
        
        return base_factor * competition_factor
    
    async def _calculate_competition_type_factor(self, competition_info: CompetitionInfo) -> float:
        """コンペ種別要因計算"""
        competition_type = competition_info.competition_type
        
        # 個人の得意分野による重み付け
        expertise_scores = {
            CompetitionType.TABULAR: 0.95,           # 最得意
            CompetitionType.TIME_SERIES: 0.85,       # 得意
            CompetitionType.COMPUTER_VISION: 0.70,   # 標準的
            CompetitionType.NLP: 0.65,               # やや苦手
            CompetitionType.CODE_COMPETITION: 0.60,  # やや苦手
            CompetitionType.GRAPH: 0.55,             # 苦手
            CompetitionType.AUDIO: 0.45,             # 苦手
            CompetitionType.MULTI_MODAL: 0.40,       # かなり苦手
            CompetitionType.REINFORCEMENT_LEARNING: 0.30,  # 最も苦手
            CompetitionType.GETTING_STARTED: 0.90    # 学習コンペは有利
        }
        
        return expertise_scores.get(competition_type, 0.50)
    
    async def _calculate_skill_alignment(self, competition_info: CompetitionInfo) -> float:
        """スキル適合度計算"""
        required_skills = set(competition_info.skill_requirements)
        personal_skills = set(self.personal_skills)
        
        if not required_skills:
            return 0.7  # スキル情報不足時のデフォルト
        
        # スキルマッチ率
        matched_skills = required_skills & personal_skills
        match_ratio = len(matched_skills) / len(required_skills)
        
        # 重要スキルの重み付け
        critical_skills = {
            "python", "pandas", "numpy", "scikit_learn", 
            "feature_engineering", "ensemble", "cross_validation"
        }
        
        critical_matched = required_skills & personal_skills & critical_skills
        critical_required = required_skills & critical_skills
        
        if critical_required:
            critical_ratio = len(critical_matched) / len(critical_required)
            # 重要スキルの重み付け（70%）
            return 0.7 * critical_ratio + 0.3 * match_ratio
        else:
            return match_ratio
    
    async def _calculate_experience_factor(self, competition_info: CompetitionInfo) -> float:
        """経験値要因計算"""
        
        # 過去参加実績
        total_competitions = self.historical_performance["competitions_participated"]
        medals_won = self.historical_performance["medals_won"]
        avg_percentile = self.historical_performance["average_rank_percentile"]
        
        # 基本経験スコア
        if total_competitions == 0:
            experience_score = 0.3  # 初心者
        elif total_competitions < 5:
            experience_score = 0.5  # 初級者
        elif total_competitions < 15:
            experience_score = 0.7  # 中級者
        elif total_competitions < 30:
            experience_score = 0.85 # 上級者
        else:
            experience_score = 0.95 # エキスパート
        
        # メダル獲得実績による調整
        if medals_won > 0:
            medal_bonus = min(0.2, medals_won * 0.05)
            experience_score += medal_bonus
        
        # 平均順位による調整
        if avg_percentile < 0.1:
            rank_bonus = 0.15  # 常に上位10%
        elif avg_percentile < 0.2:
            rank_bonus = 0.10  # 常に上位20%
        elif avg_percentile < 0.5:
            rank_bonus = 0.05  # 平均以上
        else:
            rank_bonus = 0.0
        
        experience_score += rank_bonus
        
        # 同種コンペ経験による調整
        favorite_types = self.historical_performance["favorite_competition_types"]
        if competition_info.competition_type in favorite_types:
            experience_score += 0.1  # 得意分野ボーナス
        
        return min(1.0, experience_score)
    
    async def _calculate_remaining_time_factor(self, competition_info: CompetitionInfo) -> float:
        """残り時間要因計算"""
        days_remaining = competition_info.days_remaining
        
        if days_remaining <= 0:
            return 0.0  # 終了済み
        
        # 残り時間による効果曲線
        if days_remaining >= 60:
            return 1.0    # 十分な時間
        elif days_remaining >= 30:
            return 0.9    # 適度な時間
        elif days_remaining >= 14:
            return 0.75   # やや短い
        elif days_remaining >= 7:
            return 0.6    # 短い
        elif days_remaining >= 3:
            return 0.4    # かなり短い
        else:
            return 0.2    # 極めて短い
    
    async def _calculate_urgency_weight(self, competition_info: CompetitionInfo) -> float:
        """緊急度重み計算"""
        days_remaining = competition_info.days_remaining
        total_teams = competition_info.total_teams
        
        # 基本緊急度（残り時間逆比例）
        if days_remaining <= 0:
            return 0.0
        
        base_urgency = max(0.1, 1.0 / (days_remaining / 7.0))  # 週単位正規化
        
        # 競争激しさによる調整
        if total_teams > 1000:
            competition_urgency = 1.2  # 激戦は早期参加が重要
        elif total_teams > 500:
            competition_urgency = 1.1
        else:
            competition_urgency = 1.0
        
        return min(2.0, base_urgency * competition_urgency)
    
    async def _estimate_required_compute(self, competition_info: CompetitionInfo) -> float:
        """必要計算時間推定"""
        
        # データサイズによる基本推定
        data_size_gb = competition_info.data_size_gb
        if data_size_gb <= 0:
            data_size_gb = 1.0  # デフォルト推定
        
        # コンペ種別による計算コスト
        compute_costs = {
            CompetitionType.TABULAR: 1.0,           # 基準
            CompetitionType.TIME_SERIES: 1.2,       # やや重い
            CompetitionType.COMPUTER_VISION: 3.0,   # 重い（GPU必須）
            CompetitionType.NLP: 2.5,               # 重い（GPU推奨）
            CompetitionType.AUDIO: 2.0,             # やや重い
            CompetitionType.GRAPH: 1.5,             # やや重い
            CompetitionType.MULTI_MODAL: 4.0,       # 非常に重い
            CompetitionType.REINFORCEMENT_LEARNING: 5.0,  # 最も重い
            CompetitionType.CODE_COMPETITION: 0.5,  # 軽い
            CompetitionType.GETTING_STARTED: 0.8    # 軽い
        }
        
        base_cost = compute_costs.get(competition_info.competition_type, 1.0)
        
        # データサイズによる線形スケーリング
        size_factor = math.sqrt(data_size_gb)  # 平方根スケーリング
        
        # 参加者数による競争激化（より精密なモデルが必要）
        participant_factor = 1.0 + (competition_info.participant_count / 10000.0)
        
        estimated_hours = base_cost * size_factor * participant_factor * 10  # 基準10時間
        
        return max(5.0, min(200.0, estimated_hours))  # 5-200時間の範囲
    
    async def _estimate_required_human_time(self, competition_info: CompetitionInfo) -> float:
        """必要人工時間推定"""
        
        # 基本作業時間（コンペ種別別）
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
        
        # データ複雑性による調整
        if competition_info.feature_count > 100:
            complexity_factor = 1.3  # 高次元データ
        elif competition_info.feature_count > 50:
            complexity_factor = 1.1
        else:
            complexity_factor = 1.0
        
        # 評価指標による調整
        if "auc" in competition_info.evaluation_metric.lower():
            metric_factor = 1.0  # 標準的
        elif "log_loss" in competition_info.evaluation_metric.lower():
            metric_factor = 1.2  # 確率較正が必要
        elif "ndcg" in competition_info.evaluation_metric.lower():
            metric_factor = 1.3  # ランキング最適化
        else:
            metric_factor = 1.0
        
        # 競争レベルによる調整
        if competition_info.total_prize > 50000:
            competition_factor = 1.4  # 高賞金は高品質が必要
        elif competition_info.total_prize > 25000:
            competition_factor = 1.2
        else:
            competition_factor = 1.0
        
        estimated_hours = base_time * complexity_factor * metric_factor * competition_factor
        
        return max(5.0, min(80.0, estimated_hours))  # 5-80時間の範囲
    
    async def _calculate_competition_intensity(self, competition_info: CompetitionInfo) -> float:
        """競争激しさ計算"""
        
        # 参加者数による基本強度
        participant_count = competition_info.participant_count
        if participant_count < 100:
            base_intensity = 0.3
        elif participant_count < 500:
            base_intensity = 0.5
        elif participant_count < 1000:
            base_intensity = 0.7
        elif participant_count < 2000:
            base_intensity = 0.8
        else:
            base_intensity = 0.9
        
        # 賞金による調整
        prize_intensity = min(0.3, competition_info.total_prize / 100000.0)
        
        # コミュニティ活動による調整
        discussion_intensity = min(0.2, competition_info.discussion_count / 1000.0)
        notebook_intensity = min(0.2, competition_info.notebook_count / 500.0)
        
        total_intensity = base_intensity + prize_intensity + discussion_intensity + notebook_intensity
        
        return min(1.0, total_intensity)
    
    async def _calculate_leaderboard_volatility(self, competition_info: CompetitionInfo) -> float:
        """リーダーボード変動性計算"""
        
        # コンペ種別による変動性
        volatility_scores = {
            CompetitionType.TABULAR: 0.6,           # 中程度の変動
            CompetitionType.TIME_SERIES: 0.7,       # やや高い変動
            CompetitionType.COMPUTER_VISION: 0.5,   # 比較的安定
            CompetitionType.NLP: 0.8,               # 高い変動
            CompetitionType.AUDIO: 0.7,             # やや高い変動
            CompetitionType.GRAPH: 0.6,             # 中程度の変動
            CompetitionType.MULTI_MODAL: 0.9,       # 非常に高い変動
            CompetitionType.REINFORCEMENT_LEARNING: 0.95,  # 最高変動
            CompetitionType.CODE_COMPETITION: 0.3,  # 低い変動
            CompetitionType.GETTING_STARTED: 0.4    # 低い変動
        }
        
        base_volatility = volatility_scores.get(competition_info.competition_type, 0.6)
        
        # 残り時間による調整（終盤は変動激化）
        if competition_info.days_remaining < 7:
            time_factor = 1.5  # 終盤は変動激化
        elif competition_info.days_remaining < 14:
            time_factor = 1.2  # 後半は変動増加
        else:
            time_factor = 1.0
        
        return min(1.0, base_volatility * time_factor)
    
    async def _calculate_late_entry_penalty(self, competition_info: CompetitionInfo) -> float:
        """遅参ペナルティ計算"""
        
        days_remaining = competition_info.days_remaining
        
        # 推定総期間（デフォルト90日）
        estimated_total_days = 90
        if competition_info.start_time and competition_info.end_time:
            estimated_total_days = (competition_info.end_time - competition_info.start_time).days
        
        if estimated_total_days <= 0:
            return 0.8  # 情報不足時のデフォルトペナルティ
        
        # 残り時間割合
        remaining_ratio = days_remaining / estimated_total_days
        
        # 段階的ペナルティ
        if remaining_ratio > 0.8:
            return 0.0    # 早期参加はペナルティなし
        elif remaining_ratio > 0.6:
            return 0.1    # 軽微なペナルティ
        elif remaining_ratio > 0.4:
            return 0.3    # 中程度のペナルティ  
        elif remaining_ratio > 0.2:
            return 0.5    # 重いペナルティ
        elif remaining_ratio > 0.1:
            return 0.7    # 非常に重いペナルティ
        else:
            return 0.9    # 最終盤はほぼ絶望的
    
    async def _calculate_external_data_advantage(self, competition_info: CompetitionInfo) -> float:
        """外部データ優位性計算"""
        
        if not competition_info.external_data_allowed:
            return 0.0  # 外部データ禁止
        
        # コンペ種別による外部データ活用可能性
        external_data_potential = {
            CompetitionType.TABULAR: 0.8,           # 高い活用可能性
            CompetitionType.TIME_SERIES: 0.7,       # 良い活用可能性
            CompetitionType.COMPUTER_VISION: 0.6,   # 中程度の活用可能性
            CompetitionType.NLP: 0.9,               # 非常に高い活用可能性
            CompetitionType.AUDIO: 0.5,             # 中程度の活用可能性
            CompetitionType.GRAPH: 0.4,             # やや低い活用可能性
            CompetitionType.MULTI_MODAL: 0.7,       # 良い活用可能性
            CompetitionType.REINFORCEMENT_LEARNING: 0.3,  # 低い活用可能性
            CompetitionType.CODE_COMPETITION: 0.2,  # 低い活用可能性
            CompetitionType.GETTING_STARTED: 0.1    # ほぼ不要
        }
        
        return external_data_potential.get(competition_info.competition_type, 0.5)
    
    async def _calculate_community_activity(self, competition_info: CompetitionInfo) -> float:
        """コミュニティ活動度計算"""
        
        # ディスカッション・ノートブック数による活動度
        discussion_activity = min(1.0, competition_info.discussion_count / 100.0)
        notebook_activity = min(1.0, competition_info.notebook_count / 50.0)
        
        # 重み付き平均
        activity_score = 0.6 * discussion_activity + 0.4 * notebook_activity
        
        return activity_score
    
    async def _calculate_past_solution_reference(self, competition_info: CompetitionInfo) -> float:
        """過去解法参照可能性計算"""
        
        # コンペ種別による過去解法の参考価値
        reference_values = {
            CompetitionType.TABULAR: 0.9,           # 非常に高い参考価値
            CompetitionType.TIME_SERIES: 0.8,       # 高い参考価値
            CompetitionType.COMPUTER_VISION: 0.7,   # 良い参考価値
            CompetitionType.NLP: 0.6,               # 中程度の参考価値
            CompetitionType.AUDIO: 0.5,             # やや低い参考価値
            CompetitionType.GRAPH: 0.4,             # やや低い参考価値
            CompetitionType.MULTI_MODAL: 0.3,       # 低い参考価値
            CompetitionType.REINFORCEMENT_LEARNING: 0.2,  # 非常に低い参考価値
            CompetitionType.CODE_COMPETITION: 0.8,  # 高い参考価値
            CompetitionType.GETTING_STARTED: 0.9    # 非常に高い参考価値
        }
        
        return reference_values.get(competition_info.competition_type, 0.5)
    
    async def _apply_multidimensional_model(
        self,
        competition_info: CompetitionInfo,
        factors: ProbabilityFactors
    ) -> Dict[str, float]:
        """多次元確率モデル適用"""
        
        # 基本確率計算
        base_score = calculate_base_probability(
            competition_info.participant_count,
            competition_info.total_prize,
            competition_info.competition_type
        )
        
        # 多次元要因統合
        weighted_score = factors.get_weighted_score(self.factor_weights)
        
        # 最終総合確率
        overall_probability = base_score * weighted_score
        overall_probability = max(0.01, min(0.99, overall_probability))  # 1-99%の範囲
        
        # メダル別確率分布
        gold_probability = overall_probability * 0.3   # 金メダル: 総合確率の30%
        silver_probability = overall_probability * 0.35 # 銀メダル: 総合確率の35%
        bronze_probability = overall_probability * 0.35 # 銅メダル: 総合確率の35%
        
        # 正規化（合計が overall_probability を超えないように）
        total_medal_prob = gold_probability + silver_probability + bronze_probability
        if total_medal_prob > overall_probability:
            normalize_factor = overall_probability / total_medal_prob
            gold_probability *= normalize_factor
            silver_probability *= normalize_factor
            bronze_probability *= normalize_factor
        
        return {
            "overall": overall_probability,
            "gold": gold_probability,
            "silver": silver_probability,
            "bronze": bronze_probability
        }
    
    async def _calculate_confidence_analysis(
        self,
        competition_info: CompetitionInfo,
        factors: ProbabilityFactors,
        probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """信頼度・リスク分析"""
        
        # 基本不確実性要因
        uncertainty_sources = []
        risk_factors = {}
        
        # データ品質による不確実性
        if competition_info.participant_count == 0:
            uncertainty_sources.append("参加者数データ不足")
            risk_factors["data_quality"] = 0.3
        
        if competition_info.total_prize == 0:
            uncertainty_sources.append("賞金情報不足")
            risk_factors["prize_uncertainty"] = 0.2
        
        # コンペ種別による不確実性
        type_uncertainty = {
            CompetitionType.TABULAR: 0.1,
            CompetitionType.TIME_SERIES: 0.15,
            CompetitionType.COMPUTER_VISION: 0.2,
            CompetitionType.NLP: 0.25,
            CompetitionType.AUDIO: 0.3,
            CompetitionType.GRAPH: 0.25,
            CompetitionType.MULTI_MODAL: 0.4,
            CompetitionType.REINFORCEMENT_LEARNING: 0.5,
            CompetitionType.CODE_COMPETITION: 0.15,
            CompetitionType.GETTING_STARTED: 0.05
        }
        
        risk_factors["competition_type"] = type_uncertainty.get(
            competition_info.competition_type, 0.2
        )
        
        # 時間要因による不確実性
        if competition_info.days_remaining < 7:
            uncertainty_sources.append("残り時間不足")
            risk_factors["time_constraint"] = 0.4
        elif competition_info.days_remaining < 14:
            risk_factors["time_constraint"] = 0.2
        else:
            risk_factors["time_constraint"] = 0.1
        
        # リソース制約による不確実性
        if factors.resource_efficiency < 0.5:
            uncertainty_sources.append("リソース不足")
            risk_factors["resource_constraint"] = 0.3
        
        # 競争激しさによる不確実性
        if factors.competition_intensity > 0.8:
            uncertainty_sources.append("激しい競争")
            risk_factors["high_competition"] = 0.25
        
        # 総合不確実性計算
        total_uncertainty = sum(risk_factors.values()) / len(risk_factors) if risk_factors else 0.1
        
        # 信頼区間計算
        margin = total_uncertainty * 1.96  # 95%信頼区間
        lower_bound = max(0.0, probabilities["overall"] - margin)
        upper_bound = min(1.0, probabilities["overall"] + margin)
        
        # 信頼度レベル判定
        if total_uncertainty < 0.1:
            confidence_level = ConfidenceLevel.VERY_HIGH
        elif total_uncertainty < 0.2:
            confidence_level = ConfidenceLevel.HIGH
        elif total_uncertainty < 0.3:
            confidence_level = ConfidenceLevel.MODERATE
        elif total_uncertainty < 0.4:
            confidence_level = ConfidenceLevel.LOW
        else:
            confidence_level = ConfidenceLevel.VERY_LOW
        
        return {
            "interval": (lower_bound, upper_bound),
            "level": confidence_level,
            "risk_factors": risk_factors,
            "uncertainty_sources": uncertainty_sources
        }
    
    async def _calculate_ranking_analysis(
        self,
        competition_info: CompetitionInfo,
        probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """期待順位・メダル圏分析"""
        
        participant_count = competition_info.participant_count
        if participant_count <= 0:
            participant_count = 1000  # デフォルト推定
        
        # 期待順位計算（確率から逆算）
        overall_prob = probabilities["overall"]
        expected_percentile = 1.0 - overall_prob  # 確率が高いほど上位
        expected_rank = int(participant_count * expected_percentile)
        expected_rank = max(1, min(participant_count, expected_rank))
        
        # メダル圏カットオフ推定
        gold_cutoff = max(1, int(participant_count * 0.01))    # 上位1%
        silver_cutoff = max(1, int(participant_count * 0.02))  # 上位2%
        bronze_cutoff = max(1, int(participant_count * 0.03))  # 上位3%
        
        # 小規模コンペの調整
        if participant_count < 100:
            gold_cutoff = max(1, int(participant_count * 0.05))
            silver_cutoff = max(1, int(participant_count * 0.10))
            bronze_cutoff = max(1, int(participant_count * 0.15))
        
        return {
            "expected_rank": expected_rank,
            "expected_percentile": expected_percentile,
            "medal_cutoffs": {
                "gold": gold_cutoff,
                "silver": silver_cutoff,
                "bronze": bronze_cutoff
            }
        }
    
    async def _create_factor_breakdown(self, factors: ProbabilityFactors) -> Dict[str, float]:
        """要因分解作成"""
        
        return {
            "基本要因": (factors.participant_factor + factors.prize_factor + factors.competition_type_factor) / 3,
            "専門性マッチング": factors.domain_matching,
            "スキル適合": factors.skill_alignment,
            "経験・実績": factors.experience_factor,
            "時間効率": (factors.timing_factor + factors.remaining_time_factor) / 2,
            "リソース効率": factors.resource_efficiency,
            "競争環境": (factors.competition_intensity + factors.leaderboard_volatility) / 2,
            "外部優位性": (factors.external_data_advantage + factors.past_solution_reference) / 2
        }
    
    async def _create_historical_comparison(self, competition_info: CompetitionInfo) -> Dict[str, float]:
        """過去実績比較作成"""
        
        # 同種コンペでの過去実績
        historical_data = self.historical_performance
        
        same_type_performance = 0.6  # デフォルト値
        if competition_info.competition_type in historical_data["favorite_competition_types"]:
            same_type_performance = 0.8  # 得意分野は高めに設定
        
        return {
            "同種コンペ平均順位": historical_data["average_rank_percentile"],
            "同種コンペ最高順位": historical_data["best_rank_percentile"],
            "予想相対成績": same_type_performance,
            "改善ポテンシャル": 0.15  # 学習による改善見込み
        }