"""
ハイパーパラメータ最適化システム

Optuna・Bayesian最適化・Multi-fidelity最適化を統合活用し、
GPU時間制約下での効率的なハイパーパラメータ探索を実現するシステム。
"""

import asyncio
import logging
import json
import pickle
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import uuid
import math

# Optuna（テスト環境では模擬）
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    # テスト環境用の模擬Optuna
    class MockOptuna:
        class Study:
            def __init__(self, direction="minimize"):
                self.direction = direction
                self.trials = []
                self.best_value = 0.0
                self.best_params = {}
            
            def optimize(self, objective, n_trials=100, timeout=None):
                # 模擬最適化
                for i in range(min(10, n_trials)):
                    trial = MockOptuna.Trial(i)
                    value = objective(trial)
                    self.trials.append({"params": trial.params, "value": value})
                    if i == 0 or (self.direction == "minimize" and value < self.best_value) or (self.direction == "maximize" and value > self.best_value):
                        self.best_value = value
                        self.best_params = trial.params.copy()
            
            def enqueue_trial(self, params):
                pass
        
        class Trial:
            def __init__(self, number):
                self.number = number
                self.params = {}
            
            def suggest_int(self, name, low, high, log=False):
                val = low + (high - low) // 2
                self.params[name] = val
                return val
            
            def suggest_float(self, name, low, high, log=False):
                val = (low + high) / 2
                self.params[name] = val
                return val
            
            def suggest_categorical(self, name, choices):
                val = choices[0]
                self.params[name] = val
                return val
        
        def create_study(self, direction="minimize", sampler=None, pruner=None, storage=None, study_name=None, load_if_exists=True):
            return self.Study(direction)
        
        class samplers:
            TPESampler = lambda **kwargs: None
            RandomSampler = lambda **kwargs: None
            CmaEsSampler = lambda **kwargs: None
        
        class pruners:
            MedianPruner = lambda **kwargs: None
            HyperbandPruner = lambda **kwargs: None
    
    optuna = MockOptuna()
    OPTUNA_AVAILABLE = False


class OptimizationStrategy(Enum):
    """最適化戦略"""
    BAYESIAN_TPE = "bayesian_tpe"
    RANDOM_SEARCH = "random_search"
    GRID_SEARCH = "grid_search"
    CMA_ES = "cma_es"
    HYPERBAND = "hyperband"
    MULTI_FIDELITY = "multi_fidelity"


class ParameterType(Enum):
    """パラメータタイプ"""
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class ParameterSpec:
    """パラメータ仕様"""
    name: str
    param_type: ParameterType
    bounds: Any  # (min, max) for numeric, list for categorical
    default: Any
    log_scale: bool = False
    importance: float = 1.0
    description: str = ""
    
    def suggest_value(self, trial) -> Any:
        """Optunaトライアルでの値サジェスト"""
        if self.param_type == ParameterType.INTEGER:
            return trial.suggest_int(self.name, self.bounds[0], self.bounds[1], log=self.log_scale)
        elif self.param_type == ParameterType.FLOAT:
            return trial.suggest_float(self.name, self.bounds[0], self.bounds[1], log=self.log_scale)
        elif self.param_type == ParameterType.CATEGORICAL:
            return trial.suggest_categorical(self.name, self.bounds)
        elif self.param_type == ParameterType.BOOLEAN:
            return trial.suggest_categorical(self.name, [True, False])
        else:
            return self.default


@dataclass
class OptimizationConfig:
    """最適化設定"""
    study_name: str
    technique_name: str
    parameter_specs: List[ParameterSpec]
    strategy: OptimizationStrategy
    max_trials: int
    max_time_hours: float
    cv_folds: int = 5
    scoring_metric: str = "neg_mean_squared_error"
    early_stopping: bool = True
    storage_url: Optional[str] = None
    resource_constraints: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "study_name": self.study_name,
            "technique_name": self.technique_name,
            "parameter_count": len(self.parameter_specs),
            "strategy": self.strategy.value,
            "max_trials": self.max_trials,
            "max_time_hours": self.max_time_hours,
            "cv_folds": self.cv_folds,
            "scoring_metric": self.scoring_metric,
            "early_stopping": self.early_stopping
        }


@dataclass
class OptimizationResult:
    """最適化結果"""
    study_name: str
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    optimization_time_hours: float
    convergence_trial: int
    parameter_importance: Dict[str, float]
    optimization_history: List[Dict[str, Any]]
    
    # メタ情報
    technique_name: str
    strategy_used: OptimizationStrategy
    success: bool = True
    error_message: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class HyperparameterTuner:
    """ハイパーパラメータ最適化エンジン"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # ストレージ設定
        self.storage_path = storage_path or "/tmp/optuna_studies.db"
        self.storage_url = f"sqlite:///{self.storage_path}"
        
        # 最適化履歴
        self.optimization_history: List[OptimizationResult] = []
        self.active_studies: Dict[str, Any] = {}
        
        # 戦略別設定
        self.strategy_configs = {
            OptimizationStrategy.BAYESIAN_TPE: {
                "sampler_class": "TPESampler",
                "sampler_kwargs": {"n_startup_trials": 10, "n_ei_candidates": 24},
                "pruner_class": "MedianPruner",
                "pruner_kwargs": {"n_startup_trials": 5, "n_warmup_steps": 10}
            },
            OptimizationStrategy.RANDOM_SEARCH: {
                "sampler_class": "RandomSampler",
                "sampler_kwargs": {},
                "pruner_class": None,
                "pruner_kwargs": {}
            },
            OptimizationStrategy.CMA_ES: {
                "sampler_class": "CmaEsSampler",
                "sampler_kwargs": {},
                "pruner_class": "MedianPruner",
                "pruner_kwargs": {"n_startup_trials": 10}
            },
            OptimizationStrategy.HYPERBAND: {
                "sampler_class": "TPESampler",
                "sampler_kwargs": {"n_startup_trials": 5},
                "pruner_class": "HyperbandPruner",
                "pruner_kwargs": {"min_resource": 1, "max_resource": 27, "reduction_factor": 3}
            }
        }
    
    async def optimize_hyperparameters(
        self,
        config: OptimizationConfig,
        objective_function: Callable[[Dict[str, Any]], float],
        initial_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """ハイパーパラメータ最適化実行"""
        
        self.logger.info(f"ハイパーパラメータ最適化開始: {config.study_name}")
        
        start_time = datetime.utcnow()
        
        try:
            # Optuna Study作成
            study = self._create_study(config)
            
            # 初期パラメータの設定（もしあれば）
            if initial_params:
                study.enqueue_trial(initial_params)
            
            # 目的関数ラッパー作成
            wrapped_objective = self._create_objective_wrapper(
                config, objective_function
            )
            
            # 最適化実行
            timeout_seconds = config.max_time_hours * 3600
            
            if OPTUNA_AVAILABLE:
                study.optimize(
                    wrapped_objective,
                    n_trials=config.max_trials,
                    timeout=timeout_seconds
                )
            else:
                # モック環境での最適化
                study.optimize(wrapped_objective, n_trials=min(10, config.max_trials))
            
            # 結果処理
            optimization_time = (datetime.utcnow() - start_time).total_seconds() / 3600
            
            result = OptimizationResult(
                study_name=config.study_name,
                best_params=study.best_params,
                best_score=study.best_value,
                n_trials=len(study.trials),
                optimization_time_hours=optimization_time,
                convergence_trial=self._find_convergence_trial(study.trials),
                parameter_importance=self._calculate_parameter_importance(study),
                optimization_history=self._extract_optimization_history(study.trials),
                technique_name=config.technique_name,
                strategy_used=config.strategy
            )
            
            # 履歴に追加
            self.optimization_history.append(result)
            
            self.logger.info(f"最適化完了: {config.study_name}, ベストスコア: {result.best_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"最適化エラー: {config.study_name} - {e}")
            
            optimization_time = (datetime.utcnow() - start_time).total_seconds() / 3600
            
            # エラー結果作成
            error_result = OptimizationResult(
                study_name=config.study_name,
                best_params={},
                best_score=0.0,
                n_trials=0,
                optimization_time_hours=optimization_time,
                convergence_trial=-1,
                parameter_importance={},
                optimization_history=[],
                technique_name=config.technique_name,
                strategy_used=config.strategy,
                success=False,
                error_message=str(e)
            )
            
            return error_result
    
    def _create_study(self, config: OptimizationConfig):
        """Optuna Study作成"""
        
        strategy_config = self.strategy_configs.get(config.strategy, {})
        
        # Sampler作成
        sampler = None
        if OPTUNA_AVAILABLE:
            sampler_class = strategy_config.get("sampler_class")
            sampler_kwargs = strategy_config.get("sampler_kwargs", {})
            
            if sampler_class == "TPESampler":
                sampler = optuna.samplers.TPESampler(**sampler_kwargs)
            elif sampler_class == "RandomSampler":
                sampler = optuna.samplers.RandomSampler(**sampler_kwargs)
            elif sampler_class == "CmaEsSampler":
                sampler = optuna.samplers.CmaEsSampler(**sampler_kwargs)
        
        # Pruner作成
        pruner = None
        if OPTUNA_AVAILABLE and config.early_stopping:
            pruner_class = strategy_config.get("pruner_class")
            pruner_kwargs = strategy_config.get("pruner_kwargs", {})
            
            if pruner_class == "MedianPruner":
                pruner = optuna.pruners.MedianPruner(**pruner_kwargs)
            elif pruner_class == "HyperbandPruner":
                pruner = optuna.pruners.HyperbandPruner(**pruner_kwargs)
        
        # Study作成
        direction = "minimize" if "neg_" in config.scoring_metric else "maximize"
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.storage_url if OPTUNA_AVAILABLE else None,
            study_name=config.study_name,
            load_if_exists=True
        )
        
        # アクティブStudyに登録
        self.active_studies[config.study_name] = study
        
        return study
    
    def _create_objective_wrapper(
        self,
        config: OptimizationConfig,
        objective_function: Callable[[Dict[str, Any]], float]
    ) -> Callable:
        """目的関数ラッパー作成"""
        
        def objective(trial):
            try:
                # パラメータサジェスト
                params = {}
                for param_spec in config.parameter_specs:
                    params[param_spec.name] = param_spec.suggest_value(trial)
                
                # 目的関数実行
                score = objective_function(params)
                
                return score
                
            except Exception as e:
                self.logger.error(f"目的関数実行エラー: {e}")
                # エラー時のペナルティスコア
                return float('inf') if "minimize" in config.scoring_metric else -float('inf')
        
        return objective
    
    def _find_convergence_trial(self, trials: List[Any]) -> int:
        """収束トライアル特定"""
        
        if len(trials) < 10:
            return len(trials) - 1
        
        # 最近10トライアルでの改善チェック
        best_values = []
        for i, trial in enumerate(trials):
            if i == 0:
                best_values.append(trial.get("value", 0))
            else:
                prev_best = best_values[-1]
                current_value = trial.get("value", 0)
                
                # 最小化の場合の改善判定
                if current_value < prev_best:
                    best_values.append(current_value)
                else:
                    best_values.append(prev_best)
        
        # 10トライアル連続で改善がない場合を収束とみなす
        for i in range(len(best_values) - 10, 0, -1):
            if best_values[i] != best_values[i + 9]:
                return i + 9
        
        return len(trials) - 1
    
    def _calculate_parameter_importance(self, study) -> Dict[str, float]:
        """パラメータ重要度計算"""
        
        importance = {}
        
        if OPTUNA_AVAILABLE and hasattr(study, 'trials') and len(study.trials) > 10:
            try:
                # Optuna標準の重要度計算
                param_importance = optuna.importance.get_param_importances(study)
                for param, imp in param_importance.items():
                    importance[param] = imp
            except:
                # フォールバック: 分散による重要度推定
                pass
        
        # フォールバック実装
        if not importance:
            param_values = {}
            param_scores = {}
            
            for trial in study.trials:
                trial_params = trial.get("params", {})
                trial_value = trial.get("value", 0)
                
                for param, value in trial_params.items():
                    if param not in param_values:
                        param_values[param] = []
                        param_scores[param] = []
                    
                    param_values[param].append(value)
                    param_scores[param].append(trial_value)
            
            # 相関による重要度推定
            for param in param_values:
                if len(param_values[param]) > 1:
                    correlation = self._calculate_correlation(
                        param_values[param], param_scores[param]
                    )
                    importance[param] = abs(correlation)
                else:
                    importance[param] = 0.0
        
        return importance
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """相関係数計算"""
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _extract_optimization_history(self, trials: List[Any]) -> List[Dict[str, Any]]:
        """最適化履歴抽出"""
        
        history = []
        
        for i, trial in enumerate(trials):
            history.append({
                "trial_number": i,
                "params": trial.get("params", {}),
                "value": trial.get("value", 0.0),
                "state": "COMPLETE"  # 簡略化
            })
        
        return history
    
    async def create_technique_optimization_config(
        self,
        technique_name: str,
        competition_type: str = "tabular",
        resource_constraints: Dict[str, Any] = None
    ) -> OptimizationConfig:
        """技術別最適化設定作成"""
        
        # 技術別パラメータ空間定義
        parameter_specs = self._get_technique_parameter_specs(technique_name)
        
        # リソース制約に基づく設定調整
        constraints = resource_constraints or {}
        max_time_hours = constraints.get("max_gpu_hours", 4.0)
        
        # 戦略選択
        strategy = self._select_optimization_strategy(
            technique_name, max_time_hours, len(parameter_specs)
        )
        
        # 試行回数計算
        max_trials = self._calculate_max_trials(strategy, max_time_hours, len(parameter_specs))
        
        config = OptimizationConfig(
            study_name=f"{technique_name}_{competition_type}_{int(datetime.utcnow().timestamp())}",
            technique_name=technique_name,
            parameter_specs=parameter_specs,
            strategy=strategy,
            max_trials=max_trials,
            max_time_hours=max_time_hours * 0.8,  # 80%をハイパーパラメータ最適化に割り当て
            cv_folds=5 if max_time_hours > 3 else 3,
            scoring_metric=self._determine_scoring_metric(competition_type),
            early_stopping=True,
            storage_url=self.storage_url,
            resource_constraints=constraints
        )
        
        return config
    
    def _get_technique_parameter_specs(self, technique_name: str) -> List[ParameterSpec]:
        """技術別パラメータ仕様取得"""
        
        specs_map = {
            "gradient_boosting_ensemble": [
                ParameterSpec("n_estimators", ParameterType.INTEGER, (100, 2000), 1000, importance=0.8),
                ParameterSpec("learning_rate", ParameterType.FLOAT, (0.001, 0.3), 0.01, log_scale=True, importance=0.9),
                ParameterSpec("max_depth", ParameterType.INTEGER, (3, 12), 6, importance=0.7),
                ParameterSpec("subsample", ParameterType.FLOAT, (0.6, 1.0), 0.8, importance=0.6),
                ParameterSpec("colsample_bytree", ParameterType.FLOAT, (0.6, 1.0), 0.8, importance=0.5),
                ParameterSpec("reg_alpha", ParameterType.FLOAT, (0.0, 10.0), 0.0, importance=0.4),
                ParameterSpec("reg_lambda", ParameterType.FLOAT, (0.0, 10.0), 1.0, importance=0.4)
            ],
            "multi_level_stacking": [
                ParameterSpec("level1_n_estimators", ParameterType.INTEGER, (100, 1000), 500, importance=0.7),
                ParameterSpec("level1_max_depth", ParameterType.INTEGER, (3, 10), 6, importance=0.6),
                ParameterSpec("meta_alpha", ParameterType.FLOAT, (0.01, 100.0), 1.0, log_scale=True, importance=0.8),
                ParameterSpec("cv_folds", ParameterType.INTEGER, (3, 10), 5, importance=0.5),
                ParameterSpec("final_blender_alpha", ParameterType.FLOAT, (0.1, 10.0), 0.5, importance=0.6)
            ],
            "neural_network": [
                ParameterSpec("hidden_layers", ParameterType.INTEGER, (2, 8), 4, importance=0.8),
                ParameterSpec("hidden_size", ParameterType.INTEGER, (64, 1024), 256, importance=0.7),
                ParameterSpec("dropout_rate", ParameterType.FLOAT, (0.0, 0.8), 0.2, importance=0.6),
                ParameterSpec("learning_rate", ParameterType.FLOAT, (1e-5, 1e-1), 1e-3, log_scale=True, importance=0.9),
                ParameterSpec("batch_size", ParameterType.INTEGER, (32, 512), 128, importance=0.5),
                ParameterSpec("weight_decay", ParameterType.FLOAT, (1e-6, 1e-2), 1e-4, log_scale=True, importance=0.4)
            ],
            "feature_engineering_automated": [
                ParameterSpec("max_interactions", ParameterType.INTEGER, (50, 500), 100, importance=0.7),
                ParameterSpec("polynomial_degree", ParameterType.INTEGER, (2, 3), 2, importance=0.6),
                ParameterSpec("feature_selection_k", ParameterType.INTEGER, (20, 200), 100, importance=0.8),
                ParameterSpec("pca_components", ParameterType.FLOAT, (0.8, 0.99), 0.95, importance=0.4),
                ParameterSpec("scaling_method", ParameterType.CATEGORICAL, ["standard", "minmax", "robust"], "standard", importance=0.3)
            ]
        }
        
        return specs_map.get(technique_name, [
            ParameterSpec("default_param", ParameterType.FLOAT, (0.1, 10.0), 1.0, importance=1.0)
        ])
    
    def _select_optimization_strategy(
        self,
        technique_name: str,
        max_time_hours: float,
        param_count: int
    ) -> OptimizationStrategy:
        """最適化戦略選択"""
        
        # 時間制約による戦略選択
        if max_time_hours < 1.0:
            return OptimizationStrategy.RANDOM_SEARCH
        elif max_time_hours < 2.0:
            return OptimizationStrategy.BAYESIAN_TPE
        elif max_time_hours < 4.0:
            if param_count > 10:
                return OptimizationStrategy.HYPERBAND
            else:
                return OptimizationStrategy.BAYESIAN_TPE
        else:
            # 十分な時間がある場合
            if "neural" in technique_name:
                return OptimizationStrategy.HYPERBAND
            elif param_count > 15:
                return OptimizationStrategy.CMA_ES
            else:
                return OptimizationStrategy.BAYESIAN_TPE
    
    def _calculate_max_trials(
        self,
        strategy: OptimizationStrategy,
        max_time_hours: float,
        param_count: int
    ) -> int:
        """最大試行回数計算"""
        
        # 戦略別の基本試行回数
        base_trials = {
            OptimizationStrategy.RANDOM_SEARCH: int(max_time_hours * 30),
            OptimizationStrategy.BAYESIAN_TPE: int(max_time_hours * 20),
            OptimizationStrategy.CMA_ES: int(max_time_hours * 15),
            OptimizationStrategy.HYPERBAND: int(max_time_hours * 40)
        }
        
        trials = base_trials.get(strategy, int(max_time_hours * 20))
        
        # パラメータ数による調整
        if param_count > 10:
            trials = int(trials * 1.5)
        elif param_count < 5:
            trials = int(trials * 0.7)
        
        return max(10, min(trials, 500))  # 10〜500の範囲に制限
    
    def _determine_scoring_metric(self, competition_type: str) -> str:
        """スコアリング指標決定"""
        
        metric_map = {
            "tabular": "neg_mean_squared_error",
            "classification": "roc_auc",
            "regression": "neg_mean_squared_error",
            "time_series": "neg_mean_absolute_error",
            "computer_vision": "accuracy",
            "nlp": "f1_score"
        }
        
        return metric_map.get(competition_type, "neg_mean_squared_error")
    
    async def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化サマリー取得"""
        
        if not self.optimization_history:
            return {"message": "最適化履歴がありません"}
        
        # 最近の最適化統計
        recent_optimizations = self.optimization_history[-10:]
        
        successful_opts = [opt for opt in recent_optimizations if opt.success]
        total_time = sum(opt.optimization_time_hours for opt in recent_optimizations)
        avg_trials = sum(opt.n_trials for opt in recent_optimizations) / len(recent_optimizations)
        
        # 技術別統計
        technique_stats = {}
        for opt in recent_optimizations:
            tech = opt.technique_name
            if tech not in technique_stats:
                technique_stats[tech] = {
                    "optimizations": 0,
                    "success_rate": 0.0,
                    "avg_best_score": 0.0,
                    "avg_trials": 0.0
                }
            
            stats = technique_stats[tech]
            stats["optimizations"] += 1
            stats["avg_best_score"] += opt.best_score
            stats["avg_trials"] += opt.n_trials
            
            if opt.success:
                stats["success_rate"] += 1
        
        # 平均計算
        for tech, stats in technique_stats.items():
            count = stats["optimizations"]
            stats["success_rate"] /= count
            stats["avg_best_score"] /= count
            stats["avg_trials"] /= count
        
        return {
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "success_rate": len(successful_opts) / len(recent_optimizations),
            "total_optimization_hours": total_time,
            "average_trials_per_optimization": avg_trials,
            "technique_statistics": technique_stats,
            "active_studies": len(self.active_studies),
            "storage_path": self.storage_path
        }
    
    async def load_optimization_result(self, study_name: str) -> Optional[OptimizationResult]:
        """最適化結果読み込み"""
        
        # 履歴から検索
        for result in self.optimization_history:
            if result.study_name == study_name:
                return result
        
        # ストレージから読み込み（実装省略）
        return None
    
    async def save_optimization_result(self, result: OptimizationResult, file_path: str):
        """最適化結果保存"""
        
        try:
            result_data = {
                "study_name": result.study_name,
                "best_params": result.best_params,
                "best_score": result.best_score,
                "n_trials": result.n_trials,
                "optimization_time_hours": result.optimization_time_hours,
                "technique_name": result.technique_name,
                "strategy_used": result.strategy_used.value,
                "parameter_importance": result.parameter_importance,
                "optimization_history": result.optimization_history,
                "success": result.success,
                "created_at": result.created_at.isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"最適化結果保存完了: {file_path}")
            
        except Exception as e:
            self.logger.error(f"最適化結果保存エラー: {e}")
            raise