"""
並列実行オーケストレーター

複数クラウド環境（Kaggle、Colab、Paperspace）での
並列実験実行・結果収集・障害復旧を統合管理するシステム。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid

from ..cloud_managers.kaggle_kernel_manager import KaggleKernelManager
from ..cloud_managers.colab_execution_manager import ColabExecutionManager
from ..cloud_managers.paperspace_manager import PaperspaceManager


class ExecutionStatus(Enum):
    """実行ステータス"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CloudEnvironment(Enum):
    """クラウド環境"""
    KAGGLE_KERNELS = "kaggle_kernels"
    GOOGLE_COLAB = "google_colab"
    PAPERSPACE_GRADIENT = "paperspace_gradient"


@dataclass
class ExecutionTask:
    """実行タスク"""
    task_id: str
    technique_name: str
    notebook_code: str
    environment: CloudEnvironment
    competition_name: str
    estimated_runtime_hours: float
    priority: int = 1
    retry_count: int = 0
    max_retries: int = 2
    
    # 実行状態
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # 結果
    execution_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # メタデータ
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class ParallelExecution:
    """並列実行状態"""
    execution_id: str
    tasks: List[ExecutionTask]
    start_time: datetime
    completion_time: Optional[datetime] = None
    
    # 統計
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    
    # 結果
    results: List[Dict[str, Any]] = None
    success_rate: float = 0.0
    total_gpu_hours_used: float = 0.0
    
    def __post_init__(self):
        if self.results is None:
            self.results = []
        self.total_tasks = len(self.tasks)


class ParallelExecutor:
    """並列実行オーケストレーター"""
    
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
        
        # 実行管理
        self.active_executions: Dict[str, ParallelExecution] = {}
        self.execution_queue: List[ExecutionTask] = []
        self.execution_history: List[ParallelExecution] = []
        
        # 設定
        self.max_concurrent_tasks = 5
        self.task_timeout_hours = 3.0
        self.retry_delay_minutes = 5
        
        # 環境別制限
        self.environment_limits = {
            CloudEnvironment.KAGGLE_KERNELS: 2,
            CloudEnvironment.GOOGLE_COLAB: 1,
            CloudEnvironment.PAPERSPACE_GRADIENT: 2
        }
        
        # 実行中タスク追跡
        self.running_tasks: Dict[CloudEnvironment, List[ExecutionTask]] = {
            env: [] for env in CloudEnvironment
        }
    
    async def execute_parallel_techniques(
        self,
        technique_configs: List[Dict[str, Any]],
        notebook_codes: Dict[str, Dict[str, str]],
        execution_settings: Optional[Dict[str, Any]] = None
    ) -> ParallelExecution:
        """並列技術実行"""
        
        execution_id = f"parallel-{uuid.uuid4().hex[:8]}"
        self.logger.info(f"並列実行開始: {execution_id}, {len(technique_configs)}技術")
        
        # 実行タスク作成
        tasks = self._create_execution_tasks(
            technique_configs, notebook_codes, execution_settings or {}
        )
        
        # 並列実行オブジェクト作成
        parallel_execution = ParallelExecution(
            execution_id=execution_id,
            tasks=tasks,
            start_time=datetime.utcnow()
        )
        
        # 実行管理に登録
        self.active_executions[execution_id] = parallel_execution
        
        try:
            # 並列実行実行
            await self._execute_parallel_tasks(parallel_execution)
            
            # 結果収集・統計計算
            await self._collect_execution_results(parallel_execution)
            
            # 完了処理
            parallel_execution.completion_time = datetime.utcnow()
            
            self.logger.info(f"並列実行完了: {execution_id}, 成功率: {parallel_execution.success_rate:.1%}")
            
        except Exception as e:
            self.logger.error(f"並列実行エラー: {execution_id} - {e}")
            parallel_execution.completion_time = datetime.utcnow()
        
        finally:
            # アクティブ実行から削除、履歴に追加
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.execution_history.append(parallel_execution)
        
        return parallel_execution
    
    def _create_execution_tasks(
        self,
        technique_configs: List[Dict[str, Any]],
        notebook_codes: Dict[str, Dict[str, str]],
        execution_settings: Dict[str, Any]
    ) -> List[ExecutionTask]:
        """実行タスク作成"""
        
        tasks = []
        
        for i, config in enumerate(technique_configs):
            technique_name = config["technique"]
            
            # 利用可能環境取得
            available_environments = self._get_available_environments(config)
            
            # 各環境でのタスク作成
            for env_name in available_environments:
                try:
                    environment = CloudEnvironment(env_name)
                    
                    # ノートブックコード取得
                    notebook_code = notebook_codes.get(technique_name, {}).get(env_name)
                    if not notebook_code:
                        self.logger.warning(f"ノートブックコードなし: {technique_name} @ {env_name}")
                        continue
                    
                    # タスク作成
                    task = ExecutionTask(
                        task_id=f"task-{technique_name}-{env_name}-{uuid.uuid4().hex[:6]}",
                        technique_name=technique_name,
                        notebook_code=notebook_code,
                        environment=environment,
                        competition_name=execution_settings.get("competition_name", "default"),
                        estimated_runtime_hours=config.get("estimated_runtime_hours", 1.5),
                        priority=len(technique_configs) - i,  # 順序に基づく優先度
                        max_retries=execution_settings.get("max_retries", 2)
                    )
                    
                    tasks.append(task)
                    
                except ValueError:
                    self.logger.warning(f"Unknown environment: {env_name}")
                    continue
        
        # 優先度順ソート
        tasks.sort(key=lambda t: t.priority, reverse=True)
        
        self.logger.info(f"実行タスク作成完了: {len(tasks)}タスク")
        return tasks
    
    def _get_available_environments(self, config: Dict[str, Any]) -> List[str]:
        """利用可能環境取得"""
        
        # 設定から環境リスト取得
        preferred_environments = config.get("preferred_environments", [])
        if preferred_environments:
            return preferred_environments
        
        # デフォルト環境選択
        available = []
        
        if self.kaggle_manager:
            available.append("kaggle_kernels")
        if self.colab_manager:
            available.append("google_colab")
        if self.paperspace_manager:
            available.append("paperspace_gradient")
        
        return available
    
    async def _execute_parallel_tasks(self, parallel_execution: ParallelExecution):
        """並列タスク実行"""
        
        tasks = parallel_execution.tasks
        
        # タスクを実行キューに追加
        self.execution_queue.extend(tasks)
        
        # 並列実行制御
        semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # 各タスクの実行コルーチン作成
        task_coroutines = [
            self._execute_single_task(task, semaphore, parallel_execution)
            for task in tasks
        ]
        
        # 並列実行（例外は個別処理）
        await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        self.logger.info(f"並列実行完了: {parallel_execution.execution_id}")
    
    async def _execute_single_task(
        self,
        task: ExecutionTask,
        semaphore: asyncio.Semaphore,
        parallel_execution: ParallelExecution
    ):
        """単一タスク実行"""
        
        async with semaphore:
            try:
                # 環境容量チェック
                await self._wait_for_environment_capacity(task.environment)
                
                # タスク実行開始
                task.status = ExecutionStatus.RUNNING
                task.start_time = datetime.utcnow()
                
                # 実行中タスクに追加
                self.running_tasks[task.environment].append(task)
                
                self.logger.info(f"タスク実行開始: {task.task_id} @ {task.environment.value}")
                
                # 環境別実行
                result = await self._execute_task_in_environment(task)
                
                # 成功処理
                task.status = ExecutionStatus.COMPLETED
                task.completion_time = datetime.utcnow()
                task.execution_result = result
                
                parallel_execution.completed_tasks += 1
                
                self.logger.info(f"タスク実行完了: {task.task_id}, スコア: {result.get('score', 0):.4f}")
                
            except asyncio.TimeoutError:
                # タイムアウト処理
                task.status = ExecutionStatus.TIMEOUT
                task.completion_time = datetime.utcnow()
                task.error_message = "Execution timeout"
                
                parallel_execution.failed_tasks += 1
                
                self.logger.warning(f"タスクタイムアウト: {task.task_id}")
                
                # リトライ判定
                if task.retry_count < task.max_retries:
                    await self._schedule_task_retry(task, parallel_execution)
                
            except Exception as e:
                # エラー処理
                task.status = ExecutionStatus.FAILED
                task.completion_time = datetime.utcnow()
                task.error_message = str(e)
                
                parallel_execution.failed_tasks += 1
                
                self.logger.error(f"タスク実行エラー: {task.task_id} - {e}")
                
                # リトライ判定
                if task.retry_count < task.max_retries:
                    await self._schedule_task_retry(task, parallel_execution)
            
            finally:
                # 実行中タスクから削除
                if task in self.running_tasks[task.environment]:
                    self.running_tasks[task.environment].remove(task)
    
    async def _wait_for_environment_capacity(self, environment: CloudEnvironment):
        """環境容量待機"""
        
        limit = self.environment_limits.get(environment, 1)
        
        while len(self.running_tasks[environment]) >= limit:
            self.logger.debug(f"環境容量待機: {environment.value} ({len(self.running_tasks[environment])}/{limit})")
            await asyncio.sleep(10)  # 10秒待機
    
    async def _execute_task_in_environment(self, task: ExecutionTask) -> Dict[str, Any]:
        """環境別タスク実行"""
        
        timeout_seconds = task.estimated_runtime_hours * 3600
        
        try:
            if task.environment == CloudEnvironment.KAGGLE_KERNELS:
                return await asyncio.wait_for(
                    self._execute_kaggle_task(task),
                    timeout=timeout_seconds
                )
            
            elif task.environment == CloudEnvironment.GOOGLE_COLAB:
                return await asyncio.wait_for(
                    self._execute_colab_task(task),
                    timeout=timeout_seconds
                )
            
            elif task.environment == CloudEnvironment.PAPERSPACE_GRADIENT:
                return await asyncio.wait_for(
                    self._execute_paperspace_task(task),
                    timeout=timeout_seconds
                )
            
            else:
                raise ValueError(f"Unknown environment: {task.environment}")
                
        except asyncio.TimeoutError:
            # タイムアウトの場合はキャンセル処理
            await self._cancel_task_execution(task)
            raise
    
    async def _execute_kaggle_task(self, task: ExecutionTask) -> Dict[str, Any]:
        """Kaggleタスク実行"""
        
        if not self.kaggle_manager:
            raise Exception("Kaggle manager not available")
        
        # Kernel実行
        kernel_execution = await self.kaggle_manager.execute_technique_kernel(
            technique_name=task.technique_name,
            notebook_code=task.notebook_code,
            competition_name=task.competition_name,
            estimated_runtime_hours=task.estimated_runtime_hours
        )
        
        # 実行監視
        result = await self.kaggle_manager.monitor_kernel_execution(kernel_execution.kernel_id)
        
        return {
            "environment": task.environment.value,
            "kernel_id": result.kernel_id,
            "success": result.success,
            "score": result.score,
            "gpu_hours_used": result.gpu_hours_used,
            "execution_log": result.execution_log[:1000],  # ログの最初の1000文字
            "output_files": result.output_files or []
        }
    
    async def _execute_colab_task(self, task: ExecutionTask) -> Dict[str, Any]:
        """Colabタスク実行"""
        
        if not self.colab_manager:
            raise Exception("Colab manager not available")
        
        from ..cloud_managers.colab_execution_manager import ColabRuntimeType
        
        # ノートブック実行
        execution = await self.colab_manager.execute_technique_notebook(
            technique_name=task.technique_name,
            notebook_code=task.notebook_code,
            competition_name=task.competition_name,
            estimated_runtime_hours=task.estimated_runtime_hours,
            runtime_type=ColabRuntimeType.GPU
        )
        
        # 実行監視
        result = await self.colab_manager.monitor_execution(execution.execution_id)
        
        return {
            "environment": task.environment.value,
            "execution_id": result.execution_id,
            "success": result.success,
            "score": result.score,
            "gpu_hours_used": result.gpu_hours_used,
            "output_data": result.output_data or {},
            "drive_file_id": result.drive_file_id
        }
    
    async def _execute_paperspace_task(self, task: ExecutionTask) -> Dict[str, Any]:
        """Paperspaceタスク実行"""
        
        if not self.paperspace_manager:
            raise Exception("Paperspace manager not available")
        
        from ..cloud_managers.paperspace_manager import GradientMachineType
        
        # Job実行
        job_execution = await self.paperspace_manager.execute_technique_job(
            technique_name=task.technique_name,
            notebook_code=task.notebook_code,
            competition_name=task.competition_name,
            estimated_runtime_hours=task.estimated_runtime_hours,
            machine_type=GradientMachineType.FREE_GPU
        )
        
        # 実行監視
        result = await self.paperspace_manager.monitor_job_execution(job_execution.job_id)
        
        return {
            "environment": task.environment.value,
            "job_id": result.job_id,
            "success": result.success,
            "score": result.score,
            "gpu_hours_used": result.gpu_hours_used,
            "artifacts": result.artifacts or []
        }
    
    async def _cancel_task_execution(self, task: ExecutionTask):
        """タスク実行キャンセル"""
        
        try:
            if task.environment == CloudEnvironment.KAGGLE_KERNELS and self.kaggle_manager:
                # Kaggle Kernelキャンセル（実装依存）
                pass
            
            elif task.environment == CloudEnvironment.GOOGLE_COLAB and self.colab_manager:
                # Colab実行終了
                if task.execution_result and "execution_id" in task.execution_result:
                    await self.colab_manager.terminate_execution(task.execution_result["execution_id"])
            
            elif task.environment == CloudEnvironment.PAPERSPACE_GRADIENT and self.paperspace_manager:
                # Paperspace Jobキャンセル
                if task.execution_result and "job_id" in task.execution_result:
                    await self.paperspace_manager.cancel_job(task.execution_result["job_id"])
            
            self.logger.info(f"タスクキャンセル完了: {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"タスクキャンセル失敗: {task.task_id} - {e}")
    
    async def _schedule_task_retry(
        self,
        task: ExecutionTask,
        parallel_execution: ParallelExecution
    ):
        """タスクリトライスケジュール"""
        
        task.retry_count += 1
        task.status = ExecutionStatus.PENDING
        
        self.logger.info(f"タスクリトライスケジュール: {task.task_id} (試行{task.retry_count}/{task.max_retries})")
        
        # 遅延後に再実行
        await asyncio.sleep(self.retry_delay_minutes * 60)
        
        # セマフォ制御下で再実行
        semaphore = asyncio.Semaphore(1)
        await self._execute_single_task(task, semaphore, parallel_execution)
    
    async def _collect_execution_results(self, parallel_execution: ParallelExecution):
        """実行結果収集"""
        
        results = []
        total_gpu_hours = 0.0
        successful_tasks = 0
        
        for task in parallel_execution.tasks:
            if task.execution_result:
                results.append({
                    "task_id": task.task_id,
                    "technique_name": task.technique_name,
                    "environment": task.environment.value,
                    "status": task.status.value,
                    "score": task.execution_result.get("score", 0.0),
                    "gpu_hours_used": task.execution_result.get("gpu_hours_used", 0.0),
                    "success": task.execution_result.get("success", False),
                    "execution_time_hours": (
                        (task.completion_time - task.start_time).total_seconds() / 3600
                        if task.completion_time and task.start_time else 0.0
                    )
                })
                
                total_gpu_hours += task.execution_result.get("gpu_hours_used", 0.0)
                
                if task.execution_result.get("success", False):
                    successful_tasks += 1
        
        # 統計更新
        parallel_execution.results = results
        parallel_execution.total_gpu_hours_used = total_gpu_hours
        parallel_execution.success_rate = successful_tasks / max(len(parallel_execution.tasks), 1)
        
        self.logger.info(f"結果収集完了: {len(results)}結果, GPU時間: {total_gpu_hours:.1f}h")
    
    async def execute_technique(
        self,
        technique_name: str,
        notebook: str,
        environment: CloudEnvironment,
        resource_limit: float
    ) -> Dict[str, Any]:
        """単一技術実行（ParallelExecutorからの呼び出し用）"""
        
        task = ExecutionTask(
            task_id=f"single-{technique_name}-{uuid.uuid4().hex[:6]}",
            technique_name=technique_name,
            notebook_code=notebook,
            environment=environment,
            competition_name="single_execution",
            estimated_runtime_hours=resource_limit
        )
        
        try:
            result = await self._execute_task_in_environment(task)
            return result
            
        except Exception as e:
            self.logger.error(f"単一技術実行エラー: {technique_name} - {e}")
            return {
                "environment": environment.value,
                "success": False,
                "error": str(e),
                "score": 0.0,
                "gpu_hours_used": 0.0
            }
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """実行状態取得"""
        
        # アクティブ実行から検索
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            
            return {
                "execution_id": execution_id,
                "status": "running",
                "progress": {
                    "total_tasks": execution.total_tasks,
                    "completed_tasks": execution.completed_tasks,
                    "failed_tasks": execution.failed_tasks,
                    "running_tasks": execution.total_tasks - execution.completed_tasks - execution.failed_tasks
                },
                "elapsed_time_hours": (datetime.utcnow() - execution.start_time).total_seconds() / 3600,
                "success_rate": execution.success_rate
            }
        
        # 履歴から検索
        for execution in self.execution_history:
            if execution.execution_id == execution_id:
                return {
                    "execution_id": execution_id,
                    "status": "completed",
                    "results": execution.results,
                    "success_rate": execution.success_rate,
                    "total_gpu_hours_used": execution.total_gpu_hours_used,
                    "duration_hours": (
                        (execution.completion_time - execution.start_time).total_seconds() / 3600
                        if execution.completion_time else 0.0
                    )
                }
        
        return None
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """実行キャンセル"""
        
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        
        try:
            # 実行中タスクのキャンセル
            for task in execution.tasks:
                if task.status == ExecutionStatus.RUNNING:
                    await self._cancel_task_execution(task)
                    task.status = ExecutionStatus.CANCELLED
                    execution.cancelled_tasks += 1
            
            # 実行完了処理
            execution.completion_time = datetime.utcnow()
            del self.active_executions[execution_id]
            self.execution_history.append(execution)
            
            self.logger.info(f"実行キャンセル完了: {execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"実行キャンセル失敗: {execution_id} - {e}")
            return False
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """性能サマリー取得"""
        
        if not self.execution_history:
            return {"message": "実行履歴がありません"}
        
        # 最近の実行統計
        recent_executions = self.execution_history[-10:]  # 最新10件
        
        total_tasks = sum(ex.total_tasks for ex in recent_executions)
        total_successful = sum(ex.completed_tasks for ex in recent_executions)
        total_gpu_hours = sum(ex.total_gpu_hours_used for ex in recent_executions)
        
        # 環境別統計
        environment_stats = {}
        for env in CloudEnvironment:
            env_results = []
            for execution in recent_executions:
                env_tasks = [r for r in execution.results if r["environment"] == env.value]
                env_results.extend(env_tasks)
            
            if env_results:
                success_count = len([r for r in env_results if r["success"]])
                avg_score = sum(r["score"] for r in env_results) / len(env_results)
                
                environment_stats[env.value] = {
                    "total_tasks": len(env_results),
                    "success_rate": success_count / len(env_results),
                    "average_score": avg_score,
                    "gpu_hours_used": sum(r["gpu_hours_used"] for r in env_results)
                }
        
        return {
            "recent_executions": len(recent_executions),
            "overall_statistics": {
                "total_tasks": total_tasks,
                "success_rate": total_successful / max(total_tasks, 1),
                "total_gpu_hours_used": total_gpu_hours,
                "average_gpu_hours_per_task": total_gpu_hours / max(total_tasks, 1)
            },
            "environment_statistics": environment_stats,
            "active_executions": len(self.active_executions),
            "queue_size": len(self.execution_queue)
        }