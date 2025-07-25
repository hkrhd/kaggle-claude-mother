"""
実験設計システム

技術実装に対する最適なハイパーパラメータ探索・実験戦略を設計し、
効率的な実験実行計画を生成するシステム。
"""

import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import math

class ExperimentStrategy(Enum):
    """実験戦略"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    PROGRESSIVE_HALVING = "progressive_halving"
    MULTI_FIDELITY = "multi_fidelity"

class ParameterType(Enum):
    """パラメータタイプ"""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

@dataclass
class ParameterSpace:
    """パラメータ空間定義"""
    name: str
    param_type: ParameterType
    bounds: Any  # (min, max) for numeric, list for categorical
    default: Any
    importance: float = 1.0  # 0.0-1.0
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.param_type.value,
            "bounds": self.bounds,
            "default": self.default,
            "importance": self.importance,
            "description": self.description
        }

@dataclass
class ExperimentConfig:
    """実験設定"""
    technique_name: str
    parameter_spaces: List[ParameterSpace]
    strategy: ExperimentStrategy
    max_trials: int
    max_time_hours: float
    cv_folds: int = 5
    scoring_metric: str = "rmse"
    early_stopping: bool = True
    parallel_jobs: int = 1
    resource_constraints: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "technique_name": self.technique_name,
            "parameter_spaces": [ps.to_dict() for ps in self.parameter_spaces],
            "strategy": self.strategy.value,
            "max_trials": self.max_trials,
            "max_time_hours": self.max_time_hours,
            "cv_folds": self.cv_folds,
            "scoring_metric": self.scoring_metric,
            "early_stopping": self.early_stopping,
            "parallel_jobs": self.parallel_jobs,
            "resource_constraints": self.resource_constraints or {}
        }

@dataclass
class ExperimentPlan:
    """実験計画"""
    experiment_id: str
    configs: List[ExperimentConfig]
    execution_order: List[str]  # technique names in execution order
    estimated_total_time: float
    resource_allocation: Dict[str, Any]
    success_criteria: Dict[str, float]
    fallback_strategies: List[str]
    created_at: datetime

class ExperimentDesigner:
    """実験設計エンジン"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 技術別デフォルトパラメータ空間
        self.default_parameter_spaces = self._initialize_parameter_spaces()
        
        # 実験戦略別設定
        self.strategy_configs = {
            ExperimentStrategy.GRID_SEARCH: {
                "exhaustive": True,
                "parallel_friendly": True,
                "time_predictable": True,
                "memory_efficient": False
            },
            ExperimentStrategy.RANDOM_SEARCH: {
                "exhaustive": False,
                "parallel_friendly": True,
                "time_predictable": True,
                "memory_efficient": True
            },
            ExperimentStrategy.BAYESIAN_OPTIMIZATION: {
                "exhaustive": False,
                "parallel_friendly": False,
                "time_predictable": False,
                "memory_efficient": True
            },
            ExperimentStrategy.PROGRESSIVE_HALVING: {
                "exhaustive": False,
                "parallel_friendly": True,
                "time_predictable": False,
                "memory_efficient": True
            }
        }
    
    def _initialize_parameter_spaces(self) -> Dict[str, List[ParameterSpace]]:
        """技術別デフォルトパラメータ空間初期化"""
        
        spaces = {}
        
        # Gradient Boosting Ensemble
        spaces["gradient_boosting_ensemble"] = [
            ParameterSpace(
                name="n_estimators",
                param_type=ParameterType.INTEGER,
                bounds=(100, 2000),
                default=1000,
                importance=0.8,
                description="Number of boosting rounds"
            ),
            ParameterSpace(
                name="learning_rate",
                param_type=ParameterType.FLOAT,
                bounds=(0.001, 0.3),
                default=0.01,
                importance=0.9,
                description="Learning rate for boosting"
            ),
            ParameterSpace(
                name="max_depth",
                param_type=ParameterType.INTEGER,
                bounds=(3, 12),
                default=6,
                importance=0.7,
                description="Maximum tree depth"
            ),
            ParameterSpace(
                name="subsample",
                param_type=ParameterType.FLOAT,
                bounds=(0.6, 1.0),
                default=0.8,
                importance=0.6,
                description="Subsample ratio for training"
            ),
            ParameterSpace(
                name="colsample_bytree",
                param_type=ParameterType.FLOAT,
                bounds=(0.6, 1.0),
                default=0.8,
                importance=0.5,
                description="Feature sampling ratio per tree"
            ),
            ParameterSpace(
                name="reg_alpha",
                param_type=ParameterType.FLOAT,
                bounds=(0.0, 10.0),
                default=0.0,
                importance=0.4,
                description="L1 regularization term"
            ),
            ParameterSpace(
                name="reg_lambda",
                param_type=ParameterType.FLOAT,
                bounds=(0.0, 10.0),
                default=1.0,
                importance=0.4,
                description="L2 regularization term"
            )
        ]
        
        # Multi-level Stacking
        spaces["multi_level_stacking"] = [
            ParameterSpace(
                name="level1_n_estimators",
                param_type=ParameterType.INTEGER,
                bounds=(100, 1000),
                default=500,
                importance=0.7,
                description="Base learner n_estimators"
            ),
            ParameterSpace(
                name="level1_max_depth",
                param_type=ParameterType.INTEGER,
                bounds=(3, 10),
                default=6,
                importance=0.6,
                description="Base learner max depth"
            ),
            ParameterSpace(
                name="meta_alpha",
                param_type=ParameterType.FLOAT,
                bounds=(0.01, 100.0),
                default=1.0,
                importance=0.8,
                description="Meta learner regularization"
            ),
            ParameterSpace(
                name="cv_folds",
                param_type=ParameterType.INTEGER,
                bounds=(3, 10),
                default=5,
                importance=0.5,
                description="Cross-validation folds for stacking"
            ),
            ParameterSpace(
                name="final_blender_alpha",
                param_type=ParameterType.FLOAT,
                bounds=(0.1, 10.0),
                default=0.5,
                importance=0.6,
                description="Final blender regularization"
            )
        ]
        
        # Feature Engineering
        spaces["feature_engineering_automated"] = [
            ParameterSpace(
                name="max_interactions",
                param_type=ParameterType.INTEGER,
                bounds=(50, 500),
                default=100,
                importance=0.7,
                description="Maximum interaction features to generate"
            ),
            ParameterSpace(
                name="polynomial_degree",
                param_type=ParameterType.INTEGER,
                bounds=(2, 3),
                default=2,
                importance=0.6,
                description="Polynomial features degree"
            ),
            ParameterSpace(
                name="feature_selection_k",
                param_type=ParameterType.INTEGER,
                bounds=(20, 200),
                default=100,
                importance=0.8,
                description="Number of features to select"
            ),
            ParameterSpace(
                name="pca_components",
                param_type=ParameterType.FLOAT,
                bounds=(0.8, 0.99),
                default=0.95,
                importance=0.4,
                description="PCA variance ratio to retain"
            ),
            ParameterSpace(
                name="scaling_method",
                param_type=ParameterType.CATEGORICAL,
                bounds=["standard", "minmax", "robust"],
                default="standard",
                importance=0.3,
                description="Feature scaling method"
            )
        ]
        
        # Neural Network
        spaces["neural_network"] = [
            ParameterSpace(
                name="hidden_layers",
                param_type=ParameterType.INTEGER,
                bounds=(2, 8),
                default=4,
                importance=0.8,
                description="Number of hidden layers"
            ),
            ParameterSpace(
                name="hidden_size",
                param_type=ParameterType.INTEGER,
                bounds=(64, 1024),
                default=256,
                importance=0.7,
                description="Hidden layer size"
            ),
            ParameterSpace(
                name="dropout_rate",
                param_type=ParameterType.FLOAT,
                bounds=(0.0, 0.8),
                default=0.2,
                importance=0.6,
                description="Dropout rate"
            ),
            ParameterSpace(
                name="learning_rate",
                param_type=ParameterType.FLOAT,
                bounds=(1e-5, 1e-1),
                default=1e-3,
                importance=0.9,
                description="Learning rate"
            ),
            ParameterSpace(
                name="batch_size",
                param_type=ParameterType.INTEGER,
                bounds=(32, 512),
                default=128,
                importance=0.5,
                description="Batch size"
            ),
            ParameterSpace(
                name="weight_decay",
                param_type=ParameterType.FLOAT,
                bounds=(1e-6, 1e-2),
                default=1e-4,
                importance=0.4,
                description="Weight decay regularization"
            )
        ]
        
        return spaces
    
    async def design_experiments(
        self,
        techniques: List[Dict[str, Any]],
        notebooks: Dict[str, Dict[str, str]],
        resource_constraints: Dict[str, Any]
    ) -> ExperimentPlan:
        """実験計画設計"""
        
        self.logger.info(f"実験計画設計開始: {len(techniques)}技術")
        
        experiment_id = f"exp-{int(datetime.utcnow().timestamp())}"
        configs = []
        
        # 各技術の実験設定作成
        for technique_info in techniques:
            technique_name = technique_info["technique"]
            
            config = await self._design_technique_experiment(
                technique_name=technique_name,
                technique_info=technique_info,
                resource_constraints=resource_constraints
            )
            
            if config:
                configs.append(config)
        
        # 実行順序決定
        execution_order = self._determine_execution_order(configs, resource_constraints)
        
        # 総実行時間推定
        estimated_total_time = sum(config.max_time_hours for config in configs)
        
        # リソース配分
        resource_allocation = await self._allocate_experiment_resources(
            configs, resource_constraints
        )
        
        # 成功基準設定
        success_criteria = self._define_success_criteria(configs)
        
        # フォールバック戦略
        fallback_strategies = self._design_fallback_strategies(configs)
        
        experiment_plan = ExperimentPlan(
            experiment_id=experiment_id,
            configs=configs,
            execution_order=execution_order,
            estimated_total_time=estimated_total_time,
            resource_allocation=resource_allocation,
            success_criteria=success_criteria,
            fallback_strategies=fallback_strategies,
            created_at=datetime.utcnow()
        )
        
        self.logger.info(f"実験計画設計完了: {len(configs)}設定, 推定{estimated_total_time:.1f}時間")
        return experiment_plan
    
    async def _design_technique_experiment(
        self,
        technique_name: str,
        technique_info: Dict[str, Any],
        resource_constraints: Dict[str, Any]
    ) -> Optional[ExperimentConfig]:
        """技術別実験設計"""
        
        # デフォルトパラメータ空間取得
        default_spaces = self.default_parameter_spaces.get(technique_name, [])
        if not default_spaces:
            self.logger.warning(f"Unknown technique for experiment design: {technique_name}")
            return None
        
        # リソース制約に基づく戦略選択
        strategy = self._select_optimization_strategy(
            technique_name, resource_constraints
        )
        
        # 試行回数・時間制限調整
        max_trials, max_time_hours = self._calculate_experiment_limits(
            technique_name, strategy, resource_constraints
        )
        
        # パラメータ空間調整
        adjusted_spaces = self._adjust_parameter_spaces(
            default_spaces, technique_info, resource_constraints
        )
        
        config = ExperimentConfig(
            technique_name=technique_name,
            parameter_spaces=adjusted_spaces,
            strategy=strategy,
            max_trials=max_trials,
            max_time_hours=max_time_hours,
            cv_folds=5 if resource_constraints.get("max_gpu_hours", 10) > 5 else 3,
            scoring_metric=self._determine_scoring_metric(technique_info),
            early_stopping=True,
            parallel_jobs=min(2, resource_constraints.get("parallel_limit", 1)),
            resource_constraints=resource_constraints
        )
        
        return config
    
    def _select_optimization_strategy(
        self,
        technique_name: str,
        resource_constraints: Dict[str, Any]
    ) -> ExperimentStrategy:
        """最適化戦略選択"""
        
        max_time = resource_constraints.get("max_gpu_hours", 4.0)
        parallel_capability = resource_constraints.get("parallel_limit", 1) > 1
        
        # 時間制約が厳しい場合
        if max_time < 2.0:
            return ExperimentStrategy.RANDOM_SEARCH
        
        # 並列実行可能で時間に余裕がある場合
        elif parallel_capability and max_time > 4.0:
            return ExperimentStrategy.GRID_SEARCH
        
        # 中程度の時間制約
        elif max_time < 6.0:
            return ExperimentStrategy.BAYESIAN_OPTIMIZATION
        
        # 十分な時間とリソース
        else:
            # 複雑な技術にはBayesian Optimization
            if "stacking" in technique_name or "neural" in technique_name:
                return ExperimentStrategy.BAYESIAN_OPTIMIZATION
            else:
                return ExperimentStrategy.PROGRESSIVE_HALVING
    
    def _calculate_experiment_limits(
        self,
        technique_name: str,
        strategy: ExperimentStrategy,
        resource_constraints: Dict[str, Any]
    ) -> Tuple[int, float]:
        """実験制限計算"""
        
        max_gpu_hours = resource_constraints.get("max_gpu_hours", 4.0)
        
        # 技術別基本実行時間
        base_times = {
            "gradient_boosting_ensemble": 0.3,
            "multi_level_stacking": 0.8,
            "feature_engineering_automated": 0.2,
            "neural_network": 1.5
        }
        
        base_time = base_times.get(technique_name, 0.5)
        
        # 戦略別調整
        if strategy == ExperimentStrategy.GRID_SEARCH:
            max_trials = min(50, int(max_gpu_hours / base_time))
            max_time_hours = max_gpu_hours * 0.8
        
        elif strategy == ExperimentStrategy.RANDOM_SEARCH:
            max_trials = min(100, int(max_gpu_hours / (base_time * 0.7)))
            max_time_hours = max_gpu_hours * 0.9
        
        elif strategy == ExperimentStrategy.BAYESIAN_OPTIMIZATION:
            max_trials = min(80, int(max_gpu_hours / base_time))
            max_time_hours = max_gpu_hours * 0.85
        
        else:  # PROGRESSIVE_HALVING
            max_trials = min(150, int(max_gpu_hours / (base_time * 0.5)))
            max_time_hours = max_gpu_hours * 0.9
        
        return max(5, max_trials), max(0.5, max_time_hours)
    
    def _adjust_parameter_spaces(
        self,
        default_spaces: List[ParameterSpace],
        technique_info: Dict[str, Any],
        resource_constraints: Dict[str, Any]
    ) -> List[ParameterSpace]:
        """パラメータ空間調整"""
        
        adjusted_spaces = []
        max_time = resource_constraints.get("max_gpu_hours", 4.0)
        
        for space in default_spaces:
            adjusted_space = ParameterSpace(
                name=space.name,
                param_type=space.param_type,
                bounds=space.bounds,
                default=space.default,
                importance=space.importance,
                description=space.description
            )
            
            # 時間制約による調整
            if max_time < 2.0 and space.name in ["n_estimators", "max_depth"]:
                # 時間制約が厳しい場合は範囲を縮小
                if space.param_type == ParameterType.INTEGER:
                    min_val, max_val = space.bounds
                    adjusted_space.bounds = (min_val, min(max_val, int(max_val * 0.6)))
            
            # 複雑度による調整
            complexity = technique_info.get("integrated_score", 0.5)
            if complexity > 0.8 and space.importance > 0.7:
                # 高複雑度技術では重要パラメータの探索範囲を拡大
                if space.param_type == ParameterType.FLOAT:
                    min_val, max_val = space.bounds
                    range_expansion = (max_val - min_val) * 0.2
                    adjusted_space.bounds = (
                        max(0, min_val - range_expansion),
                        max_val + range_expansion
                    )
            
            adjusted_spaces.append(adjusted_space)
        
        return adjusted_spaces
    
    def _determine_scoring_metric(self, technique_info: Dict[str, Any]) -> str:
        """スコアリング指標決定"""
        
        # 技術名から推定
        technique = technique_info.get("technique", "")
        
        if "classification" in technique.lower():
            return "roc_auc"
        elif "regression" in technique.lower() or "ensemble" in technique.lower():
            return "neg_mean_squared_error"
        else:
            return "neg_mean_squared_error"  # デフォルト
    
    def _determine_execution_order(
        self,
        configs: List[ExperimentConfig],
        resource_constraints: Dict[str, Any]
    ) -> List[str]:
        """実行順序決定"""
        
        # 実行時間と重要度に基づく優先順位付け
        scored_configs = []
        
        for config in configs:
            # 実行時間スコア（短いほど高得点）
            time_score = 1.0 / max(config.max_time_hours, 0.1)
            
            # 技術重要度スコア
            importance_score = {
                "gradient_boosting_ensemble": 0.9,
                "multi_level_stacking": 0.8,
                "feature_engineering_automated": 0.7,
                "neural_network": 0.6
            }.get(config.technique_name, 0.5)
            
            # 並列実行可能性スコア
            parallel_score = 1.0 if self.strategy_configs[config.strategy]["parallel_friendly"] else 0.5
            
            total_score = time_score * 0.4 + importance_score * 0.4 + parallel_score * 0.2
            scored_configs.append((config.technique_name, total_score))
        
        # スコア順にソート
        scored_configs.sort(key=lambda x: x[1], reverse=True)
        
        return [name for name, score in scored_configs]
    
    async def _allocate_experiment_resources(
        self,
        configs: List[ExperimentConfig],
        resource_constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """実験リソース配分"""
        
        total_gpu_hours = sum(config.max_time_hours for config in configs)
        available_gpu_hours = resource_constraints.get("max_gpu_hours", 8.0)
        
        allocation = {
            "total_required_gpu_hours": total_gpu_hours,
            "available_gpu_hours": available_gpu_hours,
            "resource_utilization": min(1.0, total_gpu_hours / available_gpu_hours),
            "technique_allocations": {}
        }
        
        # 各技術への配分
        for config in configs:
            allocation["technique_allocations"][config.technique_name] = {
                "allocated_gpu_hours": config.max_time_hours,
                "max_trials": config.max_trials,
                "parallel_jobs": config.parallel_jobs,
                "strategy": config.strategy.value
            }
        
        return allocation
    
    def _define_success_criteria(self, configs: List[ExperimentConfig]) -> Dict[str, float]:
        """成功基準定義"""
        
        return {
            "min_techniques_completed": max(1, len(configs) * 0.6),
            "min_improvement_threshold": 0.02,  # 2%の改善
            "max_cv_std": 0.05,  # CV標準偏差の上限
            "min_validation_score": 0.7,
            "resource_efficiency_threshold": 0.8
        }
    
    def _design_fallback_strategies(self, configs: List[ExperimentConfig]) -> List[str]:
        """フォールバック戦略設計"""
        
        strategies = []
        
        # 時間制約への対応
        strategies.append("時間超過時: 残り技術を簡易パラメータで実行")
        
        # リソース不足への対応
        strategies.append("GPU不足時: CPU実行への切り替え")
        
        # 実験失敗への対応
        strategies.append("実験失敗時: デフォルトパラメータでの実行")
        
        # 性能不足への対応
        strategies.append("性能不達時: アンサンブル手法の適用")
        
        return strategies
    
    async def generate_optuna_study_config(
        self,
        config: ExperimentConfig
    ) -> Dict[str, Any]:
        """Optuna study設定生成"""
        
        study_config = {
            "study_name": f"{config.technique_name}_optimization",
            "direction": "minimize" if "neg_" in config.scoring_metric else "maximize",
            "sampler": self._get_optuna_sampler_config(config.strategy),
            "pruner": self._get_optuna_pruner_config(config.early_stopping),
            "storage": "sqlite:///optuna_studies.db",
            "load_if_exists": True
        }
        
        # パラメータ空間をOptuna形式に変換
        parameter_suggestions = []
        for param_space in config.parameter_spaces:
            suggestion = self._convert_to_optuna_suggestion(param_space)
            parameter_suggestions.append(suggestion)
        
        study_config["parameter_suggestions"] = parameter_suggestions
        
        return study_config
    
    def _get_optuna_sampler_config(self, strategy: ExperimentStrategy) -> Dict[str, Any]:
        """Optunaサンプラー設定"""
        
        if strategy == ExperimentStrategy.RANDOM_SEARCH:
            return {"type": "RandomSampler", "seed": 42}
        
        elif strategy == ExperimentStrategy.BAYESIAN_OPTIMIZATION:
            return {
                "type": "TPESampler",
                "n_startup_trials": 10,
                "n_ei_candidates": 24,
                "seed": 42
            }
        
        elif strategy == ExperimentStrategy.GRID_SEARCH:
            return {"type": "GridSampler"}
        
        else:  # Default
            return {"type": "TPESampler", "seed": 42}
    
    def _get_optuna_pruner_config(self, early_stopping: bool) -> Optional[Dict[str, Any]]:
        """Optunaプルーナー設定"""
        
        if not early_stopping:
            return None
        
        return {
            "type": "MedianPruner",
            "n_startup_trials": 5,
            "n_warmup_steps": 10,
            "interval_steps": 1
        }
    
    def _convert_to_optuna_suggestion(self, param_space: ParameterSpace) -> Dict[str, Any]:
        """OptunaパラメータサジェストでのAny変換"""
        
        suggestion = {"name": param_space.name}
        
        if param_space.param_type == ParameterType.INTEGER:
            min_val, max_val = param_space.bounds
            suggestion["type"] = "suggest_int"
            suggestion["low"] = min_val
            suggestion["high"] = max_val
        
        elif param_space.param_type == ParameterType.FLOAT:
            min_val, max_val = param_space.bounds
            suggestion["type"] = "suggest_float"
            suggestion["low"] = min_val
            suggestion["high"] = max_val
            suggestion["log"] = param_space.name in ["learning_rate", "reg_alpha", "reg_lambda"]
        
        elif param_space.param_type == ParameterType.CATEGORICAL:
            suggestion["type"] = "suggest_categorical"
            suggestion["choices"] = param_space.bounds
        
        elif param_space.param_type == ParameterType.BOOLEAN:
            suggestion["type"] = "suggest_categorical"
            suggestion["choices"] = [True, False]
        
        return suggestion
    
    async def create_experiment_summary(
        self,
        experiment_plan: ExperimentPlan
    ) -> Dict[str, Any]:
        """実験サマリー作成"""
        
        summary = {
            "experiment_id": experiment_plan.experiment_id,
            "total_techniques": len(experiment_plan.configs),
            "estimated_duration_hours": experiment_plan.estimated_total_time,
            "execution_order": experiment_plan.execution_order,
            "resource_summary": experiment_plan.resource_allocation,
            "success_criteria": experiment_plan.success_criteria,
            "technique_details": []
        }
        
        for config in experiment_plan.configs:
            technique_detail = {
                "technique_name": config.technique_name,
                "optimization_strategy": config.strategy.value,
                "max_trials": config.max_trials,
                "estimated_hours": config.max_time_hours,
                "parameter_count": len(config.parameter_spaces),
                "key_parameters": [
                    ps.name for ps in config.parameter_spaces 
                    if ps.importance > 0.7
                ]
            }
            summary["technique_details"].append(technique_detail)
        
        return summary