"""
高度実行エージェント

analyzerの技術分析結果を受けて、複数クラウド環境での
並列実験実行・GPU最適化・自動モデル構築を担当するメインエージェント。
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# クラウド実行管理
from .cloud_managers.kaggle_kernel_manager import KaggleKernelManager
from .cloud_managers.colab_execution_manager import ColabExecutionManager
from .cloud_managers.paperspace_manager import PaperspaceManager
from .cloud_managers.resource_optimizer import CloudResourceOptimizer

# コード生成・実験設計
from .code_generators.notebook_generator import NotebookGenerator
from .code_generators.experiment_designer import ExperimentDesigner

# 実行オーケストレーション
from .execution_orchestrator.parallel_executor import ParallelExecutor

# 最適化エンジン
from .optimization.hyperparameter_tuner import HyperparameterTuner

# GitHub Issue安全システム
from ...issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ...issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations

class CloudEnvironment(Enum):
    """クラウド実行環境"""
    KAGGLE_KERNELS = "kaggle_kernels"
    GOOGLE_COLAB = "google_colab"
    PAPERSPACE_GRADIENT = "paperspace_gradient"

class ExecutionPhase(Enum):
    """実行フェーズ"""
    INITIALIZATION = "initialization"
    RESOURCE_PLANNING = "resource_planning"
    CODE_GENERATION = "code_generation"
    PARALLEL_EXECUTION = "parallel_execution"
    OPTIMIZATION = "optimization"
    RESULT_COLLECTION = "result_collection"
    SUBMISSION = "submission"
    COMPLETED = "completed"

class ExecutionPriority(Enum):
    """実行優先度"""
    MEDAL_CRITICAL = "medal_critical"  # メダル獲得重要
    HIGH = "high"                      # 高優先度
    STANDARD = "standard"              # 標準
    EXPERIMENTAL = "experimental"      # 実験的

@dataclass
class ExecutionRequest:
    """実行リクエスト"""
    competition_name: str
    analyzer_issue_number: int  # 分析エージェントからのIssue番号
    techniques_to_implement: List[Dict[str, Any]]
    priority: ExecutionPriority
    deadline_days: int
    resource_constraints: Dict[str, Any] = None
    special_requirements: List[str] = None

@dataclass
class ExecutionResult:
    """実行結果"""
    request: ExecutionRequest
    execution_id: str
    
    # 各実行環境の結果
    kaggle_results: List[Dict[str, Any]]
    colab_results: List[Dict[str, Any]]
    paperspace_results: List[Dict[str, Any]]
    
    # 統合結果
    best_score: float
    best_model_config: Dict[str, Any]
    ensemble_performance: Dict[str, Any]
    submission_ready: bool
    
    # 実行統計
    total_gpu_hours_used: float
    total_experiments_run: int
    success_rate: float
    resource_efficiency: float
    
    # メタデータ
    execution_duration: float
    phases_completed: List[ExecutionPhase]
    created_at: datetime


class ExecutorAgent:
    """高度実行エージェント - メインクラス"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        
        # エージェント情報
        self.agent_id = f"executor-{uuid.uuid4().hex[:8]}"
        self.agent_version = "1.0.0"
        self.start_time = datetime.utcnow()
        
        # GitHub Issue連携
        self.github_wrapper = GitHubApiWrapper(
            access_token=github_token,
            repo_name=repo_name
        )
        self.atomic_operations = AtomicIssueOperations(
            github_client=self.github_wrapper.github,
            repo_name=repo_name
        )
        
        # クラウド実行管理
        self.kaggle_manager = KaggleKernelManager()
        self.colab_manager = ColabExecutionManager()
        self.paperspace_manager = PaperspaceManager()
        self.resource_optimizer = CloudResourceOptimizer()
        
        # コード生成・実験設計
        self.notebook_generator = NotebookGenerator()
        self.experiment_designer = ExperimentDesigner()
        
        # 実行オーケストレーション
        self.parallel_executor = ParallelExecutor(
            kaggle_manager=self.kaggle_manager,
            colab_manager=self.colab_manager,
            paperspace_manager=self.paperspace_manager
        )
        
        # 最適化エンジン
        self.hyperparameter_tuner = HyperparameterTuner()
        
        # 実行履歴
        self.execution_history: List[ExecutionResult] = []
        self.current_execution: Optional[ExecutionResult] = None
        
        # 設定
        self.max_concurrent_executions = 3
        self.max_gpu_hours_per_competition = 20
    
    async def execute_technical_implementation(
        self,
        request: ExecutionRequest
    ) -> ExecutionResult:
        """技術実装メイン実行"""
        
        execution_start = datetime.utcnow()
        execution_id = f"execution-{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"技術実装実行開始: {request.competition_name} (ID: {execution_id})")
        
        try:
            # 実行結果オブジェクト初期化
            result = ExecutionResult(
                request=request,
                execution_id=execution_id,
                kaggle_results=[],
                colab_results=[],
                paperspace_results=[],
                best_score=0.0,
                best_model_config={},
                ensemble_performance={},
                submission_ready=False,
                total_gpu_hours_used=0.0,
                total_experiments_run=0,
                success_rate=0.0,
                resource_efficiency=0.0,
                execution_duration=0.0,
                phases_completed=[],
                created_at=execution_start
            )
            
            self.current_execution = result
            
            # 実行フェーズ順次実行
            await self._execute_phase(result, ExecutionPhase.INITIALIZATION)
            await self._execute_phase(result, ExecutionPhase.RESOURCE_PLANNING)
            await self._execute_phase(result, ExecutionPhase.CODE_GENERATION)
            await self._execute_phase(result, ExecutionPhase.PARALLEL_EXECUTION)
            await self._execute_phase(result, ExecutionPhase.OPTIMIZATION)
            await self._execute_phase(result, ExecutionPhase.RESULT_COLLECTION)
            
            # 提出判断・実行
            if result.best_score > 0 and result.success_rate > 0.5:
                await self._execute_phase(result, ExecutionPhase.SUBMISSION)
            
            # 完了処理
            result.execution_duration = (datetime.utcnow() - execution_start).total_seconds()
            result.phases_completed.append(ExecutionPhase.COMPLETED)
            
            # 実行履歴に追加
            self.execution_history.append(result)
            self.current_execution = None
            
            self.logger.info(f"技術実装実行完了: {result.execution_duration:.1f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"技術実装実行エラー: {e}")
            # エラー時も基本情報は返す
            result.execution_duration = (datetime.utcnow() - execution_start).total_seconds()
            return result
    
    async def _execute_phase(self, result: ExecutionResult, phase: ExecutionPhase):
        """実行フェーズ実行"""
        
        phase_start = datetime.utcnow()
        self.logger.info(f"フェーズ開始: {phase.value}")
        
        try:
            if phase == ExecutionPhase.INITIALIZATION:
                await self._phase_initialization(result)
            
            elif phase == ExecutionPhase.RESOURCE_PLANNING:
                await self._phase_resource_planning(result)
            
            elif phase == ExecutionPhase.CODE_GENERATION:
                await self._phase_code_generation(result)
            
            elif phase == ExecutionPhase.PARALLEL_EXECUTION:
                await self._phase_parallel_execution(result)
            
            elif phase == ExecutionPhase.OPTIMIZATION:
                await self._phase_optimization(result)
            
            elif phase == ExecutionPhase.RESULT_COLLECTION:
                await self._phase_result_collection(result)
            
            elif phase == ExecutionPhase.SUBMISSION:
                await self._phase_submission(result)
            
            # フェーズ完了記録
            result.phases_completed.append(phase)
            phase_duration = (datetime.utcnow() - phase_start).total_seconds()
            self.logger.info(f"フェーズ完了: {phase.value} ({phase_duration:.1f}秒)")
            
        except Exception as e:
            self.logger.error(f"フェーズエラー: {phase.value} - {e}")
    
    async def _phase_initialization(self, result: ExecutionResult):
        """初期化フェーズ"""
        
        request = result.request
        
        # analyzerからの技術指針を取得
        analyzer_issue = await self.github_wrapper.get_issue_safely(
            request.analyzer_issue_number
        )
        
        if analyzer_issue.success:
            # Issue本文から技術推奨を解析
            result.analyzer_recommendations = await self._parse_analyzer_recommendations(
                analyzer_issue.issue.body
            )
        
        # リソース制約の正規化
        result.resource_constraints = {
            "max_gpu_hours": min(request.deadline_days * 2, self.max_gpu_hours_per_competition),
            "preferred_environments": self._determine_preferred_environments(request.priority),
            "parallel_limit": 3 if request.priority == ExecutionPriority.MEDAL_CRITICAL else 2
        }
        
        self.logger.info(f"初期化完了: {request.competition_name}")
    
    async def _phase_resource_planning(self, result: ExecutionResult):
        """リソース計画フェーズ"""
        
        # 技術実装の計算コスト推定
        implementation_costs = []
        for technique in result.request.techniques_to_implement:
            cost = await self.resource_optimizer.estimate_implementation_cost(
                technique_name=technique["technique"],
                complexity=technique.get("integrated_score", 0.5),
                gpu_requirement=technique.get("gpu_required", False)
            )
            implementation_costs.append(cost)
        
        # 最適なリソース配分を決定
        result.resource_allocation = await self.resource_optimizer.optimize_experiment_allocation(
            techniques=result.request.techniques_to_implement,
            costs=implementation_costs,
            priority=result.request.priority,
            max_gpu_hours=result.resource_constraints["max_gpu_hours"]
        )
        
        self.logger.info(f"リソース計画完了: {len(result.resource_allocation)}技術配分")
    
    async def _phase_code_generation(self, result: ExecutionResult):
        """コード生成フェーズ"""
        
        # 各技術の実装用ノートブック生成
        result.generated_notebooks = {}
        
        for technique in result.request.techniques_to_implement:
            # 環境別最適化ノートブック生成
            notebooks = await self.notebook_generator.generate_technique_notebooks(
                technique_name=technique["technique"],
                competition_type=result.request.competition_name.split('_')[0],  # 簡易タイプ推定
                target_environments=result.resource_allocation.get(technique["technique"], {}).get("environments", ["kaggle"])
            )
            
            result.generated_notebooks[technique["technique"]] = notebooks
        
        # 実験設計・パラメータ探索空間定義
        result.experiment_designs = await self.experiment_designer.design_experiments(
            techniques=result.request.techniques_to_implement,
            notebooks=result.generated_notebooks,
            resource_constraints=result.resource_constraints
        )
        
        self.logger.info(f"コード生成完了: {len(result.generated_notebooks)}技術ノートブック")
    
    async def _phase_parallel_execution(self, result: ExecutionResult):
        """並列実行フェーズ"""
        
        # 複数環境での並列実験実行
        execution_tasks = []
        
        for technique, allocation in result.resource_allocation.items():
            for env in allocation.get("environments", ["kaggle"]):
                task = self.parallel_executor.execute_technique(
                    technique_name=technique,
                    notebook=result.generated_notebooks[technique][env],
                    environment=CloudEnvironment(env),
                    resource_limit=allocation.get("gpu_hours", 2.0)
                )
                execution_tasks.append(task)
        
        # 並列実行（例外は個別に処理）
        execution_results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # 結果を環境別に分類
        for i, exec_result in enumerate(execution_results):
            if isinstance(exec_result, Exception):
                self.logger.error(f"実行エラー {i}: {exec_result}")
                continue
            
            env = exec_result.get("environment", "unknown")
            if env == "kaggle_kernels":
                result.kaggle_results.append(exec_result)
            elif env == "google_colab":
                result.colab_results.append(exec_result)
            elif env == "paperspace_gradient":
                result.paperspace_results.append(exec_result)
        
        # 実行統計更新
        all_results = result.kaggle_results + result.colab_results + result.paperspace_results
        result.total_experiments_run = len(all_results)
        result.success_rate = len([r for r in all_results if r.get("success", False)]) / max(len(all_results), 1)
        result.total_gpu_hours_used = sum(r.get("gpu_hours_used", 0) for r in all_results)
        
        self.logger.info(f"並列実行完了: {result.total_experiments_run}実験, 成功率{result.success_rate:.1%}")
    
    async def _phase_optimization(self, result: ExecutionResult):
        """最適化フェーズ"""
        
        if not result.kaggle_results and not result.colab_results and not result.paperspace_results:
            self.logger.warning("最適化対象の結果がありません")
            return
        
        # 最高性能モデル特定
        all_results = result.kaggle_results + result.colab_results + result.paperspace_results
        successful_results = [r for r in all_results if r.get("success", False)]
        
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.get("score", 0))
            result.best_score = best_result.get("score", 0)
            result.best_model_config = best_result.get("model_config", {})
        
        # アンサンブル性能評価
        if len(successful_results) >= 2:
            result.ensemble_performance = await self.model_selection.evaluate_ensemble_potential(
                successful_results
            )
        
        # リソース効率算出
        if result.total_gpu_hours_used > 0:
            result.resource_efficiency = result.best_score / result.total_gpu_hours_used
        
        self.logger.info(f"最適化完了: 最高スコア{result.best_score:.4f}")
    
    async def _phase_result_collection(self, result: ExecutionResult):
        """結果収集フェーズ"""
        
        # 結果の統合・レポート生成
        result.execution_summary = {
            "total_techniques_attempted": len(result.request.techniques_to_implement),
            "successful_implementations": len([r for r in (result.kaggle_results + result.colab_results + result.paperspace_results) if r.get("success", False)]),
            "resource_utilization": {
                "gpu_hours_used": result.total_gpu_hours_used,
                "gpu_hours_allocated": result.resource_constraints["max_gpu_hours"],
                "utilization_rate": result.total_gpu_hours_used / result.resource_constraints["max_gpu_hours"]
            },
            "performance_metrics": {
                "best_score": result.best_score,
                "success_rate": result.success_rate,
                "resource_efficiency": result.resource_efficiency
            }
        }
        
        # 提出準備判定
        result.submission_ready = (
            result.best_score > 0 and
            result.success_rate > 0.3 and
            len(result.phases_completed) >= 5
        )
        
        self.logger.info(f"結果収集完了: 提出準備{'完了' if result.submission_ready else '未完'}")
    
    async def _phase_submission(self, result: ExecutionResult):
        """提出フェーズ"""
        
        if not result.submission_ready:
            self.logger.warning("提出準備未完のため提出フェーズをスキップ")
            return
        
        # 最高性能モデルでの提出実行（模擬）
        submission_result = {
            "submitted": True,
            "model_config": result.best_model_config,
            "final_score": result.best_score,
            "submission_timestamp": datetime.utcnow()
        }
        
        result.submission_info = submission_result
        
        self.logger.info(f"提出完了: スコア{result.best_score:.4f}")
    
    async def _parse_analyzer_recommendations(self, issue_body: str) -> Dict[str, Any]:
        """analyzerからの技術推奨解析"""
        
        # 簡易的な推奨技術解析（実際にはより高度なパース処理）
        recommendations = {
            "priority_techniques": [],
            "implementation_timeline": [],
            "resource_requirements": {},
            "risk_factors": []
        }
        
        # Issue本文から技術名を抽出（例）
        if "gradient_boosting" in issue_body.lower():
            recommendations["priority_techniques"].append("gradient_boosting_ensemble")
        if "stacking" in issue_body.lower():
            recommendations["priority_techniques"].append("multi_level_stacking")
        
        return recommendations
    
    def _determine_preferred_environments(self, priority: ExecutionPriority) -> List[str]:
        """優先度に基づく推奨環境決定"""
        
        if priority == ExecutionPriority.MEDAL_CRITICAL:
            return ["kaggle_kernels", "google_colab", "paperspace_gradient"]
        elif priority == ExecutionPriority.HIGH:
            return ["kaggle_kernels", "google_colab"]
        else:
            return ["kaggle_kernels"]
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """エージェント状態取得"""
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "uptime_hours": uptime,
            "execution_history_count": len(self.execution_history),
            "current_execution_active": self.current_execution is not None,
            "resource_status": await self.resource_optimizer.get_current_quotas(),
            "last_execution": {
                "competition_name": self.execution_history[-1].request.competition_name if self.execution_history else None,
                "best_score": self.execution_history[-1].best_score if self.execution_history else None,
                "success_rate": self.execution_history[-1].success_rate if self.execution_history else None
            } if self.execution_history else None
        }
    
    async def execute_quick_test(
        self,
        competition_name: str = "test_competition",
        techniques: List[str] = None
    ) -> Dict[str, Any]:
        """クイックテスト実行（開発・デバッグ用）"""
        
        techniques = techniques or ["gradient_boosting_ensemble"]
        
        quick_request = ExecutionRequest(
            competition_name=competition_name,
            analyzer_issue_number=12345,  # テスト用
            techniques_to_implement=[
                {"technique": tech, "integrated_score": 0.8, "gpu_required": False}
                for tech in techniques
            ],
            priority=ExecutionPriority.EXPERIMENTAL,
            deadline_days=1
        )
        
        result = await self.execute_technical_implementation(quick_request)
        
        return {
            "execution_id": result.execution_id,
            "success_rate": result.success_rate,
            "total_experiments": result.total_experiments_run,
            "gpu_hours_used": result.total_gpu_hours_used,
            "execution_duration": result.execution_duration,
            "submission_ready": result.submission_ready
        }