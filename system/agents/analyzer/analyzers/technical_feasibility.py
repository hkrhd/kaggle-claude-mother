"""
技術実装可能性分析システム

技術の複雑度・GPU要件・時間制約・ライブラリ依存性を分析し、
実装成功確率を定量的に評価するシステム。
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class TechnicalComplexity(Enum):
    """技術複雑度分類"""
    BASIC = "basic"           # 基本的な実装
    INTERMEDIATE = "intermediate"  # 中級レベル
    ADVANCED = "advanced"     # 高度な実装
    RESEARCH = "research"     # 研究レベル


class ResourceRequirement(Enum):
    """リソース要件"""
    LOW = "low"        # CPU + 8GB RAM
    MEDIUM = "medium"  # GPU + 16GB RAM
    HIGH = "high"      # 複数GPU + 32GB RAM
    EXTREME = "extreme" # クラスタ環境


@dataclass
class TechnicalSpecification:
    """技術仕様"""
    name: str
    description: str
    complexity: TechnicalComplexity
    estimated_implementation_hours: int
    required_libraries: List[str]
    gpu_memory_gb: float
    cpu_cores_min: int
    ram_gb_min: int
    disk_space_gb: float
    implementation_difficulty_factors: List[str]
    common_pitfalls: List[str]
    success_indicators: List[str]


@dataclass
class FeasibilityResult:
    """実装可能性評価結果"""
    technique_name: str
    feasibility_score: float  # 0-1
    implementation_probability: float  # 0-1
    estimated_completion_days: int
    resource_compatibility: bool
    risk_factors: List[str]
    mitigation_strategies: List[str]
    recommended_implementation_path: str
    confidence_level: float
    analysis_timestamp: datetime


class TechnicalFeasibilityAnalyzer:
    """技術実装可能性分析エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 利用可能リソース（デフォルト設定）
        self.available_resources = {
            "gpu_memory_gb": 16.0,    # T4 GPU
            "cpu_cores": 4,
            "ram_gb": 16.0,
            "daily_gpu_hours": 8.0,
            "implementation_hours_per_day": 6.0
        }
        
        # ライブラリ依存性データベース
        self.library_compatibility = {
            "scikit-learn": {"complexity": 0.2, "stability": 0.95, "gpu_support": False},
            "xgboost": {"complexity": 0.3, "stability": 0.90, "gpu_support": True},
            "lightgbm": {"complexity": 0.3, "stability": 0.90, "gpu_support": True},
            "catboost": {"complexity": 0.3, "stability": 0.85, "gpu_support": True},
            "tensorflow": {"complexity": 0.7, "stability": 0.85, "gpu_support": True},
            "pytorch": {"complexity": 0.7, "stability": 0.85, "gpu_support": True},
            "transformers": {"complexity": 0.8, "stability": 0.80, "gpu_support": True},
            "optuna": {"complexity": 0.4, "stability": 0.90, "gpu_support": False},
            "hyperopt": {"complexity": 0.5, "stability": 0.80, "gpu_support": False},
            "featuretools": {"complexity": 0.6, "stability": 0.75, "gpu_support": False}
        }
    
    async def analyze_technique_feasibility(
        self,
        technique_spec: TechnicalSpecification,
        available_days: int,
        current_skill_level: float = 0.7  # 0-1スケール
    ) -> FeasibilityResult:
        """技術実装可能性の総合分析"""
        
        try:
            # 各要因の分析
            complexity_score = await self._analyze_implementation_complexity(technique_spec)
            resource_score = await self._analyze_resource_compatibility(technique_spec)
            library_score = await self._analyze_library_dependencies(technique_spec.required_libraries)
            time_score = await self._analyze_time_constraints(technique_spec, available_days)
            skill_score = await self._analyze_skill_requirements(technique_spec, current_skill_level)
            
            # 総合実装可能性スコア算出
            feasibility_score = self._calculate_overall_feasibility(
                complexity_score, resource_score, library_score, time_score, skill_score
            )
            
            # 実装成功確率の推定
            implementation_probability = self._estimate_success_probability(
                feasibility_score, technique_spec.complexity, current_skill_level
            )
            
            # 完了予想日数
            estimated_days = self._estimate_completion_time(
                technique_spec, feasibility_score, current_skill_level
            )
            
            # リスク要因の特定
            risk_factors = await self._identify_risk_factors(
                technique_spec, complexity_score, resource_score, time_score
            )
            
            # 軽減戦略の生成
            mitigation_strategies = await self._generate_mitigation_strategies(
                risk_factors, technique_spec
            )
            
            # 推奨実装パスの決定
            implementation_path = await self._determine_implementation_path(
                technique_spec, feasibility_score, available_days
            )
            
            # 信頼度の算出
            confidence_level = self._calculate_confidence_level(
                [complexity_score, resource_score, library_score, time_score, skill_score]
            )
            
            return FeasibilityResult(
                technique_name=technique_spec.name,
                feasibility_score=feasibility_score,
                implementation_probability=implementation_probability,
                estimated_completion_days=estimated_days,
                resource_compatibility=resource_score > 0.7,
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies,
                recommended_implementation_path=implementation_path,
                confidence_level=confidence_level,
                analysis_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"技術実装可能性分析エラー: {e}")
            return FeasibilityResult(
                technique_name=technique_spec.name,
                feasibility_score=0.0,
                implementation_probability=0.0,
                estimated_completion_days=999,
                resource_compatibility=False,
                risk_factors=["分析エラー"],
                mitigation_strategies=["手動分析による再評価"],
                recommended_implementation_path="conservative",
                confidence_level=0.0,
                analysis_timestamp=datetime.utcnow()
            )
    
    async def _analyze_implementation_complexity(self, spec: TechnicalSpecification) -> float:
        """実装複雑度分析"""
        # 基本複雑度スコア
        complexity_scores = {
            TechnicalComplexity.BASIC: 0.9,
            TechnicalComplexity.INTERMEDIATE: 0.7,
            TechnicalComplexity.ADVANCED: 0.5,
            TechnicalComplexity.RESEARCH: 0.3
        }
        base_score = complexity_scores[spec.complexity]
        
        # 実装時間による調整
        time_penalty = min(spec.estimated_implementation_hours / 80.0, 0.3)  # 80時間超で減点
        
        # 難易度要因による調整
        difficulty_penalty = len(spec.implementation_difficulty_factors) * 0.05
        
        return max(base_score - time_penalty - difficulty_penalty, 0.1)
    
    async def _analyze_resource_compatibility(self, spec: TechnicalSpecification) -> float:
        """リソース互換性分析"""
        compatibility_score = 1.0
        
        # GPU メモリ要件チェック
        if spec.gpu_memory_gb > self.available_resources["gpu_memory_gb"]:
            gpu_ratio = self.available_resources["gpu_memory_gb"] / spec.gpu_memory_gb
            compatibility_score *= max(gpu_ratio, 0.3)
        
        # CPU コア要件チェック
        if spec.cpu_cores_min > self.available_resources["cpu_cores"]:
            cpu_ratio = self.available_resources["cpu_cores"] / spec.cpu_cores_min
            compatibility_score *= max(cpu_ratio, 0.5)
        
        # RAM 要件チェック
        if spec.ram_gb_min > self.available_resources["ram_gb"]:
            ram_ratio = self.available_resources["ram_gb"] / spec.ram_gb_min
            compatibility_score *= max(ram_ratio, 0.4)
        
        return compatibility_score
    
    async def _analyze_library_dependencies(self, required_libraries: List[str]) -> float:
        """ライブラリ依存性分析"""
        if not required_libraries:
            return 0.8  # 依存なしの場合はやや低スコア（実装量増大）
        
        total_complexity = 0.0
        total_stability = 0.0
        unknown_libraries = 0
        
        for lib in required_libraries:
            if lib in self.library_compatibility:
                lib_info = self.library_compatibility[lib]
                total_complexity += lib_info["complexity"]
                total_stability += lib_info["stability"]
            else:
                unknown_libraries += 1
                # 未知ライブラリはリスク要因
                total_complexity += 0.6
                total_stability += 0.7
        
        avg_complexity = total_complexity / len(required_libraries)
        avg_stability = total_stability / len(required_libraries)
        unknown_penalty = unknown_libraries * 0.1
        
        # 複雑度が低く、安定性が高いほど高スコア
        library_score = (1.0 - avg_complexity) * 0.6 + avg_stability * 0.4 - unknown_penalty
        
        return max(library_score, 0.2)
    
    async def _analyze_time_constraints(self, spec: TechnicalSpecification, available_days: int) -> float:
        """時間制約分析"""
        daily_work_hours = self.available_resources["implementation_hours_per_day"]
        total_available_hours = available_days * daily_work_hours
        
        if total_available_hours >= spec.estimated_implementation_hours:
            # 余裕のある場合、バッファ率でスコア決定
            buffer_ratio = total_available_hours / spec.estimated_implementation_hours
            if buffer_ratio >= 2.0:
                return 1.0  # 十分な余裕
            elif buffer_ratio >= 1.5:
                return 0.9  # 適度な余裕
            elif buffer_ratio >= 1.2:
                return 0.8  # 最小限の余裕
            else:
                return 0.7  # ギリギリ
        else:
            # 時間不足の場合
            shortage_ratio = spec.estimated_implementation_hours / total_available_hours
            return max(1.0 / shortage_ratio, 0.1)
    
    async def _analyze_skill_requirements(self, spec: TechnicalSpecification, skill_level: float) -> float:
        """スキル要件分析"""
        # 技術複雑度に応じた必要スキルレベル
        required_skill_levels = {
            TechnicalComplexity.BASIC: 0.3,
            TechnicalComplexity.INTERMEDIATE: 0.5,
            TechnicalComplexity.ADVANCED: 0.7,
            TechnicalComplexity.RESEARCH: 0.9
        }
        
        required_skill = required_skill_levels[spec.complexity]
        
        if skill_level >= required_skill:
            # スキル要件を満たす場合
            skill_surplus = skill_level - required_skill
            return min(0.8 + skill_surplus * 0.4, 1.0)
        else:
            # スキル不足の場合
            skill_ratio = skill_level / required_skill
            return max(skill_ratio * 0.6, 0.2)
    
    def _calculate_overall_feasibility(
        self, complexity: float, resource: float, library: float, time: float, skill: float
    ) -> float:
        """総合実装可能性スコア算出"""
        # 重み付き平均（時間とスキルを重視）
        weights = {
            "complexity": 0.15,
            "resource": 0.20,
            "library": 0.15,
            "time": 0.30,
            "skill": 0.20
        }
        
        weighted_score = (
            complexity * weights["complexity"] +
            resource * weights["resource"] +
            library * weights["library"] +
            time * weights["time"] +
            skill * weights["skill"]
        )
        
        return min(weighted_score, 1.0)
    
    def _estimate_success_probability(
        self, feasibility_score: float, complexity: TechnicalComplexity, skill_level: float
    ) -> float:
        """実装成功確率推定"""
        # 基本成功確率（実装可能性スコアベース）
        base_probability = feasibility_score * 0.8
        
        # 複雑度による調整
        complexity_adjustments = {
            TechnicalComplexity.BASIC: 0.1,
            TechnicalComplexity.INTERMEDIATE: 0.0,
            TechnicalComplexity.ADVANCED: -0.1,
            TechnicalComplexity.RESEARCH: -0.2
        }
        
        complexity_adj = complexity_adjustments[complexity]
        
        # スキルレベルボーナス
        skill_bonus = max((skill_level - 0.5) * 0.2, 0)
        
        success_probability = base_probability + complexity_adj + skill_bonus
        
        return max(min(success_probability, 0.95), 0.05)  # 5%-95%の範囲
    
    def _estimate_completion_time(
        self, spec: TechnicalSpecification, feasibility_score: float, skill_level: float
    ) -> int:
        """完了予想時間算出"""
        base_hours = spec.estimated_implementation_hours
        
        # 実装可能性スコアによる調整（低いほど時間増大）
        feasibility_multiplier = 1.0 + (1.0 - feasibility_score) * 0.5
        
        # スキルレベルによる調整
        skill_multiplier = max(1.0 - (skill_level - 0.5) * 0.3, 0.7)
        
        adjusted_hours = base_hours * feasibility_multiplier * skill_multiplier
        
        daily_hours = self.available_resources["implementation_hours_per_day"]
        estimated_days = int(adjusted_hours / daily_hours) + 1
        
        return min(estimated_days, 30)  # 最大30日
    
    async def _identify_risk_factors(
        self, spec: TechnicalSpecification, complexity_score: float, 
        resource_score: float, time_score: float
    ) -> List[str]:
        """リスク要因特定"""
        risks = []
        
        # 複雑度リスク
        if complexity_score < 0.6:
            risks.append("高い実装複雑度による開発遅延リスク")
        
        # リソースリスク  
        if resource_score < 0.7:
            risks.append("リソース不足による性能劣化リスク")
        
        # 時間制約リスク
        if time_score < 0.7:
            risks.append("時間制約による機能削減リスク")
        
        # 技術固有のリスク
        risks.extend(spec.common_pitfalls)
        
        return risks
    
    async def _generate_mitigation_strategies(
        self, risk_factors: List[str], spec: TechnicalSpecification
    ) -> List[str]:
        """軽減戦略生成"""
        strategies = []
        
        for risk in risk_factors:
            if "複雑度" in risk:
                strategies.append("段階的実装（MVP→フル機能）")
            elif "リソース" in risk:
                strategies.append("軽量化・効率化の優先実装")
            elif "時間制約" in risk:
                strategies.append("並列開発・既存実装の活用")
        
        # 技術固有の戦略
        if spec.complexity in [TechnicalComplexity.ADVANCED, TechnicalComplexity.RESEARCH]:
            strategies.append("プロトタイプによる事前検証")
            strategies.append("フォールバック手法の事前準備")
        
        return list(set(strategies))  # 重複除去
    
    async def _determine_implementation_path(
        self, spec: TechnicalSpecification, feasibility_score: float, available_days: int
    ) -> str:
        """実装パス決定"""
        if feasibility_score >= 0.8 and available_days >= 7:
            return "aggressive"  # 積極的実装
        elif feasibility_score >= 0.6 and available_days >= 5:
            return "balanced"    # バランス実装
        elif feasibility_score >= 0.4:
            return "conservative" # 保守的実装
        else:
            return "minimal"     # 最小限実装
    
    def _calculate_confidence_level(self, factor_scores: List[float]) -> float:
        """信頼度算出"""
        # スコアの分散が小さいほど信頼度が高い
        mean_score = sum(factor_scores) / len(factor_scores)
        variance = sum((score - mean_score) ** 2 for score in factor_scores) / len(factor_scores)
        
        # 分散ベースの信頼度（低分散=高信頼度）
        confidence = max(1.0 - variance * 2.0, 0.3)
        
        # 平均スコアによる調整
        score_confidence = mean_score * 0.5
        
        return min(confidence + score_confidence, 0.95)
    
    async def create_implementation_recommendation(
        self, feasibility_result: FeasibilityResult
    ) -> Dict[str, Any]:
        """実装推奨事項生成"""
        recommendation = {
            "technique": feasibility_result.technique_name,
            "overall_recommendation": self._get_overall_recommendation(feasibility_result),
            "implementation_priority": self._calculate_priority(feasibility_result),
            "resource_requirements": self._format_resource_requirements(feasibility_result),
            "timeline": {
                "estimated_days": feasibility_result.estimated_completion_days,
                "critical_milestones": self._generate_milestones(feasibility_result),
                "buffer_recommendations": f"{feasibility_result.estimated_completion_days * 0.2:.1f}日のバッファ推奨"
            },
            "risk_management": {
                "identified_risks": feasibility_result.risk_factors,
                "mitigation_strategies": feasibility_result.mitigation_strategies,
                "fallback_plan": self._generate_fallback_plan(feasibility_result)
            },
            "success_indicators": self._define_success_metrics(feasibility_result),
            "implementation_path": feasibility_result.recommended_implementation_path
        }
        
        return recommendation
    
    def _get_overall_recommendation(self, result: FeasibilityResult) -> str:
        """総合推奨判定"""
        if result.feasibility_score >= 0.8:
            return "HIGHLY_RECOMMENDED"
        elif result.feasibility_score >= 0.6:
            return "RECOMMENDED_WITH_CAUTION"
        elif result.feasibility_score >= 0.4:
            return "CONDITIONAL_IMPLEMENTATION"
        else:
            return "NOT_RECOMMENDED"
    
    def _calculate_priority(self, result: FeasibilityResult) -> str:
        """実装優先度算出"""
        if result.implementation_probability >= 0.8 and result.estimated_completion_days <= 5:
            return "HIGH"
        elif result.implementation_probability >= 0.6 and result.estimated_completion_days <= 10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _format_resource_requirements(self, result: FeasibilityResult) -> Dict[str, str]:
        """リソース要件整理"""
        return {
            "gpu_compatibility": "必須" if not result.resource_compatibility else "推奨",
            "estimated_gpu_hours": f"{result.estimated_completion_days * 2}時間",
            "memory_requirement": "16GB RAM推奨",
            "storage_requirement": "10GB以上の空き容量"
        }
    
    def _generate_milestones(self, result: FeasibilityResult) -> List[str]:
        """マイルストーン生成"""
        total_days = result.estimated_completion_days
        
        if total_days <= 3:
            return ["プロトタイプ完成 (Day 1)", "最適化・テスト (Day 2)", "最終実装 (Day 3)"]
        elif total_days <= 7:
            return [
                "設計・環境構築 (Day 1-2)",
                "基本実装 (Day 3-4)", 
                "最適化・改良 (Day 5-6)",
                "テスト・統合 (Day 7)"
            ]
        else:
            return [
                "詳細設計・調査 (Week 1)",
                "プロトタイプ開発 (Week 2)", 
                "本格実装 (Week 3)",
                "最適化・テスト (Week 4)"
            ]
    
    def _generate_fallback_plan(self, result: FeasibilityResult) -> str:
        """フォールバックプラン生成"""
        if result.implementation_probability < 0.6:
            return "既存ライブラリによる簡易実装への切り替え"
        elif result.estimated_completion_days > 10:
            return "機能限定版の段階的実装"
        else:
            return "パラメータ調整による軽量化実装"
    
    def _define_success_metrics(self, result: FeasibilityResult) -> List[str]:
        """成功指標定義"""
        return [
            f"実装完了期限: {result.estimated_completion_days}日以内",
            f"動作成功率: {result.implementation_probability * 100:.0f}%以上",
            "メモリ使用量: 利用可能量の80%以下", 
            "実行時間: ベースライン手法の150%以下"
        ]