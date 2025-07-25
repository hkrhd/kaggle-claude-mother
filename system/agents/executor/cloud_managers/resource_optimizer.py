"""
クラウドリソース最適化システム

複数クラウド環境（Kaggle、Colab、Paperspace）の無料リソースを
最大活用する効率的な配分・スケジューリングアルゴリズム。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math

from ..cloud_managers.kaggle_kernel_manager import KaggleKernelManager
from ..cloud_managers.colab_execution_manager import ColabExecutionManager
from ..cloud_managers.paperspace_manager import PaperspaceManager


class CloudPlatform(Enum):
    """クラウドプラットフォーム"""
    KAGGLE = "kaggle"
    COLAB = "colab"
    PAPERSPACE = "paperspace"


class ResourceType(Enum):
    """リソースタイプ"""
    GPU_HOURS = "gpu_hours"
    CPU_HOURS = "cpu_hours"
    STORAGE_GB = "storage_gb"
    MEMORY_GB = "memory_gb"


@dataclass
class CloudResource:
    """クラウドリソース情報"""
    platform: CloudPlatform
    resource_type: ResourceType
    total_limit: float
    current_usage: float
    reset_period: str  # "weekly", "daily", "monthly"
    reset_remaining_hours: float
    cost_per_unit: float = 0.0
    
    @property
    def available(self) -> float:
        """利用可能リソース"""
        return max(0.0, self.total_limit - self.current_usage)
    
    @property
    def utilization_rate(self) -> float:
        """利用率"""
        if self.total_limit == 0:
            return 1.0
        return self.current_usage / self.total_limit
    
    @property
    def efficiency_score(self) -> float:
        """効率スコア（利用可能量 / リセットまでの時間）"""
        if self.reset_remaining_hours <= 0:
            return float('inf')
        return self.available / self.reset_remaining_hours


@dataclass
class ExecutionRequirement:
    """実行要件"""
    technique_name: str
    competition_name: str
    estimated_gpu_hours: float
    estimated_cpu_hours: float
    memory_gb_required: float
    storage_gb_required: float
    deadline_hours: float
    priority_score: float
    complexity_level: float
    
    def __post_init__(self):
        # 最小リソース要件の調整
        self.estimated_gpu_hours = max(0.5, self.estimated_gpu_hours)
        self.estimated_cpu_hours = max(0.2, self.estimated_cpu_hours)


@dataclass
class ResourceAllocation:
    """リソース配分結果"""
    technique_name: str
    platform: CloudPlatform
    allocated_gpu_hours: float
    allocated_cpu_hours: float
    estimated_cost: float
    execution_priority: int
    feasibility_score: float
    alternative_platforms: List[CloudPlatform]
    
    # 実行予測
    estimated_start_time: datetime
    estimated_completion_time: datetime
    success_probability: float


class CloudResourceOptimizer:
    """クラウドリソース最適化エンジン"""
    
    def __init__(
        self,
        kaggle_manager: Optional[KaggleKernelManager] = None,
        colab_manager: Optional[ColabExecutionManager] = None,
        paperspace_manager: Optional[PaperspaceManager] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # クラウドマネージャー
        self.kaggle_manager = kaggle_manager
        self.colab_manager = colab_manager  
        self.paperspace_manager = paperspace_manager
        
        # 最適化パラメータ
        self.optimization_weights = {
            "cost": 0.4,          # コスト重視
            "performance": 0.3,    # 性能重視
            "availability": 0.2,   # 可用性重視
            "efficiency": 0.1      # 効率重視
        }
        
        # プラットフォーム特性
        self.platform_characteristics = {
            CloudPlatform.KAGGLE: {
                "gpu_performance_factor": 1.0,
                "reliability_score": 0.95,
                "setup_overhead_minutes": 5,
                "concurrent_limit": 2
            },
            CloudPlatform.COLAB: {
                "gpu_performance_factor": 0.9,
                "reliability_score": 0.85,
                "setup_overhead_minutes": 3,
                "concurrent_limit": 1
            },
            CloudPlatform.PAPERSPACE: {
                "gpu_performance_factor": 1.1,
                "reliability_score": 0.90,
                "setup_overhead_minutes": 8,
                "concurrent_limit": 2
            }
        }
    
    async def optimize_experiment_allocation(
        self,
        experiments: List[ExecutionRequirement],
        priority_weights: Dict[str, float] = None,
        deadline_constraint: Optional[datetime] = None
    ) -> List[ResourceAllocation]:
        """実験リソース配分最適化"""
        
        self.logger.info(f"リソース配分最適化開始: {len(experiments)}実験")
        
        # 現在のリソース状況取得
        current_resources = await self.get_current_resource_state()
        
        # 実験の優先順位付け
        prioritized_experiments = self._prioritize_experiments(
            experiments, priority_weights or {}
        )
        
        # 配分最適化実行
        allocations = []
        remaining_resources = current_resources.copy()
        
        for experiment in prioritized_experiments:
            allocation = await self._allocate_optimal_resources(
                experiment, remaining_resources, deadline_constraint
            )
            
            if allocation:
                allocations.append(allocation)
                # 配分したリソースを減算
                remaining_resources = self._update_remaining_resources(
                    remaining_resources, allocation
                )
            else:
                self.logger.warning(f"リソース不足: {experiment.technique_name}")
        
        # 配分結果の最適化・調整
        optimized_allocations = await self._optimize_allocations(
            allocations, current_resources
        )
        
        self.logger.info(f"リソース配分最適化完了: {len(optimized_allocations)}配分")
        return optimized_allocations
    
    async def get_current_resource_state(self) -> Dict[CloudPlatform, List[CloudResource]]:
        """現在のリソース状態取得"""
        
        resources = {}
        
        # Kaggle リソース
        if self.kaggle_manager:
            kaggle_status = await self.kaggle_manager.get_resource_status()
            usage = kaggle_status.get("resource_usage", {})
            
            resources[CloudPlatform.KAGGLE] = [
                CloudResource(
                    platform=CloudPlatform.KAGGLE,
                    resource_type=ResourceType.GPU_HOURS,
                    total_limit=usage.get("weekly_limit", 30.0),
                    current_usage=usage.get("used_hours", 0.0),
                    reset_period="weekly",
                    reset_remaining_hours=usage.get("reset_in_days", 3) * 24,
                    cost_per_unit=0.0
                )
            ]
        
        # Google Colab リソース
        if self.colab_manager:
            colab_status = await self.colab_manager.get_resource_status()
            usage = colab_status.get("resource_usage", {})
            
            resources[CloudPlatform.COLAB] = [
                CloudResource(
                    platform=CloudPlatform.COLAB,
                    resource_type=ResourceType.GPU_HOURS,
                    total_limit=usage.get("daily_limit", 12.0),
                    current_usage=usage.get("used_hours", 0.0),
                    reset_period="daily",
                    reset_remaining_hours=usage.get("reset_in_hours", 12),
                    cost_per_unit=0.0
                )
            ]
        
        # Paperspace リソース
        if self.paperspace_manager:
            paperspace_status = await self.paperspace_manager.get_resource_status()
            usage = paperspace_status.get("resource_usage", {})
            
            resources[CloudPlatform.PAPERSPACE] = [
                CloudResource(
                    platform=CloudPlatform.PAPERSPACE,
                    resource_type=ResourceType.GPU_HOURS,
                    total_limit=usage.get("monthly_limit", 6.0),
                    current_usage=usage.get("used_hours", 0.0),
                    reset_period="monthly",
                    reset_remaining_hours=usage.get("reset_in_days", 15) * 24,
                    cost_per_unit=0.0
                )
            ]
        
        return resources
    
    def _prioritize_experiments(
        self,
        experiments: List[ExecutionRequirement],
        priority_weights: Dict[str, float]
    ) -> List[ExecutionRequirement]:
        """実験優先順位付け"""
        
        def calculate_priority_score(exp: ExecutionRequirement) -> float:
            # 基本優先度スコア
            base_score = exp.priority_score
            
            # 締切緊急度（時間が少ないほど高優先度）
            urgency_score = max(0, 1.0 - exp.deadline_hours / 168)  # 1週間を基準
            
            # 複雑度ボーナス（高複雑度は早期実行）
            complexity_bonus = exp.complexity_level * 0.2
            
            # GPU要求量逆相関（軽量なものを優先）
            resource_factor = max(0.1, 1.0 - exp.estimated_gpu_hours / 10.0)
            
            total_score = (
                base_score * 0.4 +
                urgency_score * 0.3 +
                complexity_bonus * 0.2 +
                resource_factor * 0.1
            )
            
            return total_score
        
        # 優先度スコア計算・ソート
        experiments_with_scores = [
            (exp, calculate_priority_score(exp)) for exp in experiments
        ]
        
        experiments_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [exp for exp, score in experiments_with_scores]
    
    async def _allocate_optimal_resources(
        self,
        experiment: ExecutionRequirement,
        available_resources: Dict[CloudPlatform, List[CloudResource]],
        deadline_constraint: Optional[datetime]
    ) -> Optional[ResourceAllocation]:
        """最適リソース配分"""
        
        # 各プラットフォームでの実行可能性・コスト評価
        candidates = []
        
        for platform, resources in available_resources.items():
            gpu_resource = next(
                (r for r in resources if r.resource_type == ResourceType.GPU_HOURS),
                None
            )
            
            if not gpu_resource or gpu_resource.available < experiment.estimated_gpu_hours:
                continue  # リソース不足
            
            # 実行コスト・性能評価
            cost_estimate = await self._estimate_execution_cost(
                experiment, platform, gpu_resource
            )
            
            # 実行可能性スコア計算
            feasibility_score = self._calculate_feasibility_score(
                experiment, platform, gpu_resource, deadline_constraint
            )
            
            candidates.append({
                "platform": platform,
                "gpu_resource": gpu_resource,
                "cost": cost_estimate,
                "feasibility_score": feasibility_score,
                "performance_factor": self.platform_characteristics[platform]["gpu_performance_factor"]
            })
        
        if not candidates:
            return None
        
        # 最適候補選択
        best_candidate = max(candidates, key=lambda x: x["feasibility_score"])
        
        # ResourceAllocation作成
        allocation = ResourceAllocation(
            technique_name=experiment.technique_name,
            platform=best_candidate["platform"],
            allocated_gpu_hours=experiment.estimated_gpu_hours,
            allocated_cpu_hours=experiment.estimated_cpu_hours,
            estimated_cost=best_candidate["cost"]["total_cost"],
            execution_priority=1,  # 後で調整
            feasibility_score=best_candidate["feasibility_score"],
            alternative_platforms=[
                c["platform"] for c in candidates 
                if c["platform"] != best_candidate["platform"]
            ][:2],
            estimated_start_time=datetime.utcnow() + timedelta(minutes=5),
            estimated_completion_time=datetime.utcnow() + timedelta(
                hours=experiment.estimated_gpu_hours + 0.5
            ),
            success_probability=min(0.95, best_candidate["feasibility_score"])
        )
        
        return allocation
    
    async def _estimate_execution_cost(
        self,
        experiment: ExecutionRequirement,
        platform: CloudPlatform,
        gpu_resource: CloudResource
    ) -> Dict[str, Any]:
        """実行コスト見積もり"""
        
        # プラットフォーム別コスト推定
        if platform == CloudPlatform.KAGGLE and self.kaggle_manager:
            return await self.kaggle_manager.estimate_execution_cost(
                technique_complexity=experiment.complexity_level,
                competition_type=experiment.competition_name.split('_')[0],
                dataset_size_gb=experiment.storage_gb_required
            )
        
        elif platform == CloudPlatform.COLAB and self.colab_manager:
            from ..cloud_managers.colab_execution_manager import ColabRuntimeType
            return await self.colab_manager.estimate_execution_cost(
                technique_complexity=experiment.complexity_level,
                competition_type=experiment.competition_name.split('_')[0],
                runtime_type=ColabRuntimeType.GPU
            )
        
        elif platform == CloudPlatform.PAPERSPACE and self.paperspace_manager:
            from ..cloud_managers.paperspace_manager import GradientMachineType
            return await self.paperspace_manager.estimate_execution_cost(
                technique_complexity=experiment.complexity_level,
                competition_type=experiment.competition_name.split('_')[0],
                machine_type=GradientMachineType.FREE_GPU
            )
        
        # フォールバック推定
        return {
            "estimated_gpu_hours": experiment.estimated_gpu_hours,
            "estimated_runtime_hours": experiment.estimated_gpu_hours,
            "estimated_cost_usd": 0.0,
            "feasibility": True,
            "confidence": 0.5
        }
    
    def _calculate_feasibility_score(
        self,
        experiment: ExecutionRequirement,
        platform: CloudPlatform,
        gpu_resource: CloudResource,
        deadline_constraint: Optional[datetime]
    ) -> float:
        """実行可能性スコア計算"""
        
        # 基本スコア要素
        scores = {}
        
        # リソース可用性スコア
        resource_ratio = gpu_resource.available / max(1.0, experiment.estimated_gpu_hours)
        scores["resource_availability"] = min(1.0, resource_ratio)
        
        # プラットフォーム信頼性スコア
        scores["reliability"] = self.platform_characteristics[platform]["reliability_score"]
        
        # パフォーマンス効率スコア
        perf_factor = self.platform_characteristics[platform]["gpu_performance_factor"]
        scores["performance"] = min(1.0, perf_factor)
        
        # 締切制約適合スコア
        if deadline_constraint:
            required_time = experiment.estimated_gpu_hours + 1  # バッファ
            available_time = (deadline_constraint - datetime.utcnow()).total_seconds() / 3600
            scores["deadline_compliance"] = min(1.0, available_time / required_time) if required_time > 0 else 0
        else:
            scores["deadline_compliance"] = 1.0
        
        # リソース効率スコア
        scores["resource_efficiency"] = gpu_resource.efficiency_score / 10.0  # 正規化
        
        # 重み付き総合スコア
        weights = {
            "resource_availability": 0.3,
            "reliability": 0.25,
            "performance": 0.2,
            "deadline_compliance": 0.15,
            "resource_efficiency": 0.1
        }
        
        total_score = sum(
            scores[key] * weight for key, weight in weights.items()
        )
        
        return max(0.0, min(1.0, total_score))
    
    def _update_remaining_resources(
        self,
        resources: Dict[CloudPlatform, List[CloudResource]],
        allocation: ResourceAllocation
    ) -> Dict[CloudPlatform, List[CloudResource]]:
        """残リソース更新"""
        
        updated_resources = {}
        
        for platform, resource_list in resources.items():
            updated_list = []
            
            for resource in resource_list:
                if platform == allocation.platform and resource.resource_type == ResourceType.GPU_HOURS:
                    # 配分されたリソースを減算
                    updated_resource = CloudResource(
                        platform=resource.platform,
                        resource_type=resource.resource_type,
                        total_limit=resource.total_limit,
                        current_usage=resource.current_usage + allocation.allocated_gpu_hours,
                        reset_period=resource.reset_period,
                        reset_remaining_hours=resource.reset_remaining_hours,
                        cost_per_unit=resource.cost_per_unit
                    )
                    updated_list.append(updated_resource)
                else:
                    updated_list.append(resource)
            
            updated_resources[platform] = updated_list
        
        return updated_resources
    
    async def _optimize_allocations(
        self,
        allocations: List[ResourceAllocation],
        total_resources: Dict[CloudPlatform, List[CloudResource]]
    ) -> List[ResourceAllocation]:
        """配分結果最適化"""
        
        # 実行優先度の再計算
        for i, allocation in enumerate(allocations):
            allocation.execution_priority = i + 1
        
        # プラットフォーム分散の最適化
        platform_usage = {}
        for allocation in allocations:
            platform_usage[allocation.platform] = platform_usage.get(allocation.platform, 0) + 1
        
        # 過度な集中がある場合の再配分
        max_usage = max(platform_usage.values()) if platform_usage else 0
        if max_usage > len(allocations) * 0.7:  # 70%以上が1つのプラットフォームに集中
            allocations = await self._rebalance_platform_distribution(allocations)
        
        return allocations
    
    async def _rebalance_platform_distribution(
        self,
        allocations: List[ResourceAllocation]
    ) -> List[ResourceAllocation]:
        """プラットフォーム分散の再調整"""
        
        # 低優先度の配分について代替プラットフォームへの移行を検討
        rebalanced = []
        
        for allocation in allocations:
            if (allocation.execution_priority > 3 and 
                allocation.alternative_platforms and
                allocation.feasibility_score > 0.6):
                
                # 代替プラットフォームでの実行可能性チェック
                for alt_platform in allocation.alternative_platforms:
                    # 簡易代替可能性判定
                    if self._can_migrate_to_platform(allocation, alt_platform):
                        allocation.platform = alt_platform
                        allocation.feasibility_score *= 0.9  # 若干の性能低下を仮定
                        break
            
            rebalanced.append(allocation)
        
        return rebalanced
    
    def _can_migrate_to_platform(
        self,
        allocation: ResourceAllocation,
        target_platform: CloudPlatform
    ) -> bool:
        """プラットフォーム移行可能性判定"""
        
        # 簡易判定ロジック
        # 実際にはより詳細なリソース状況・互換性チェックが必要
        
        platform_limits = {
            CloudPlatform.KAGGLE: 6.0,     # 単一実行の実質制限
            CloudPlatform.COLAB: 12.0,     # 日次制限
            CloudPlatform.PAPERSPACE: 6.0  # 月次制限だが単発実行考慮
        }
        
        return allocation.allocated_gpu_hours <= platform_limits.get(target_platform, 2.0)
    
    async def estimate_implementation_cost(
        self,
        technique_name: str,
        complexity: float,
        gpu_requirement: bool = True
    ) -> Dict[str, Any]:
        """実装コスト推定（各マネージャーで使用）"""
        
        # 技術別基本コスト
        base_costs = {
            "gradient_boosting_ensemble": {"gpu_hours": 2.0, "complexity_factor": 1.2},
            "multi_level_stacking": {"gpu_hours": 3.5, "complexity_factor": 1.8},
            "neural_network": {"gpu_hours": 4.0, "complexity_factor": 2.0},
            "feature_engineering": {"gpu_hours": 1.5, "complexity_factor": 1.0},
            "optuna_optimization": {"gpu_hours": 6.0, "complexity_factor": 1.5}
        }
        
        base_cost = base_costs.get(technique_name, {"gpu_hours": 2.0, "complexity_factor": 1.0})
        
        # 複雑度調整
        adjusted_gpu_hours = base_cost["gpu_hours"] * base_cost["complexity_factor"] * (0.5 + complexity)
        
        # GPU要件調整
        if not gpu_requirement:
            adjusted_gpu_hours *= 0.3  # CPU実行は遅い
        
        return {
            "estimated_gpu_hours": min(adjusted_gpu_hours, 8.0),
            "estimated_cpu_hours": adjusted_gpu_hours * 0.2,
            "estimated_memory_gb": 8.0 + (complexity * 8.0),
            "estimated_storage_gb": 2.0 + (complexity * 3.0),
            "confidence": 0.8
        }
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """最適化レポート生成"""
        
        current_resources = await self.get_current_resource_state()
        
        # 各プラットフォームの効率性分析
        efficiency_analysis = {}
        
        for platform, resources in current_resources.items():
            gpu_resource = next(
                (r for r in resources if r.resource_type == ResourceType.GPU_HOURS),
                None
            )
            
            if gpu_resource:
                efficiency_analysis[platform.value] = {
                    "available_hours": gpu_resource.available,
                    "utilization_rate": gpu_resource.utilization_rate,
                    "efficiency_score": gpu_resource.efficiency_score,
                    "reset_in_hours": gpu_resource.reset_remaining_hours,
                    "recommendation": self._generate_platform_recommendation(gpu_resource)
                }
        
        # 総合推奨事項
        recommendations = self._generate_overall_recommendations(current_resources)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "platform_efficiency": efficiency_analysis,
            "overall_recommendations": recommendations,
            "resource_summary": {
                "total_available_gpu_hours": sum(
                    next((r.available for r in resources if r.resource_type == ResourceType.GPU_HOURS), 0)
                    for resources in current_resources.values()
                ),
                "most_efficient_platform": max(
                    efficiency_analysis.keys(),
                    key=lambda p: efficiency_analysis[p]["efficiency_score"]
                ) if efficiency_analysis else None
            }
        }
    
    def _generate_platform_recommendation(self, resource: CloudResource) -> str:
        """プラットフォーム別推奨事項生成"""
        
        if resource.utilization_rate < 0.3:
            return "積極活用推奨: 十分なリソースが利用可能"
        elif resource.utilization_rate < 0.7:
            return "適度利用推奨: バランス良く活用"
        elif resource.utilization_rate < 0.9:
            return "注意利用推奨: リソース制限近づく"
        else:
            return "利用制限: リソース不足、他プラットフォーム検討"
    
    def _generate_overall_recommendations(
        self,
        resources: Dict[CloudPlatform, List[CloudResource]]
    ) -> List[str]:
        """総合推奨事項生成"""
        
        recommendations = []
        
        # リソース状況に基づく推奨
        total_available = sum(
            next((r.available for r in resource_list if r.resource_type == ResourceType.GPU_HOURS), 0)
            for resource_list in resources.values()
        )
        
        if total_available > 20:
            recommendations.append("高負荷実験の実行に適したタイミング")
        elif total_available > 10:
            recommendations.append("中程度の実験実行に適している")
        elif total_available > 5:
            recommendations.append("軽量実験に限定して実行推奨")
        else:
            recommendations.append("リソース制限中、リセット待機推奨")
        
        # プラットフォーム分散推奨
        available_platforms = len([
            platform for platform, resource_list in resources.items()
            if any(r.available > 1.0 for r in resource_list if r.resource_type == ResourceType.GPU_HOURS)
        ])
        
        if available_platforms >= 3:
            recommendations.append("全プラットフォーム活用で最大効率化可能")
        elif available_platforms >= 2:
            recommendations.append("複数プラットフォーム並列活用推奨")
        else:
            recommendations.append("単一プラットフォーム集中利用")
        
        return recommendations