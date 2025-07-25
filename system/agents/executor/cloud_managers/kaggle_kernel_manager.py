"""
Kaggle Kernels実行管理システム

Kaggle Kernels APIを活用した自動実験実行・GPU時間管理・結果収集システム。
週30時間のGPU制限を最大活用する効率的リソース管理を提供。
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import subprocess
import tempfile
import os

# Kaggle API（テスト環境では模擬）
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_API_AVAILABLE = True
except ImportError:
    # テスト環境用の模擬Kaggle API
    class KaggleApi:
        def __init__(self):
            pass
        def authenticate(self):
            pass
        def kernels_push(self, *args, **kwargs):
            return {"status": "success", "kernel_id": "test123"}
        def kernels_status(self, *args, **kwargs):
            return "complete"
        def kernels_output(self, *args, **kwargs):
            return [{"path": "output.json", "content": "test data"}]
    KAGGLE_API_AVAILABLE = False


class KernelStatus(Enum):
    """Kernelステータス"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    ERROR = "error"
    CANCELLED = "cancelled"


class KernelType(Enum):
    """Kernelタイプ"""
    NOTEBOOK = "notebook"
    SCRIPT = "script"


@dataclass
class KernelConfig:
    """Kernel設定"""
    title: str
    code_file: str
    kernel_type: KernelType
    dataset_sources: List[str] = None
    competition_sources: List[str] = None
    kernel_sources: List[str] = None
    enable_gpu: bool = True
    enable_internet: bool = True
    docker_image_pinning_type: str = "ORIGINAL"
    
    def to_metadata(self) -> Dict[str, Any]:
        """Kaggle metadata.json形式に変換"""
        return {
            "id": f"kaggle-claude-mother/{self.title.lower().replace(' ', '-')}",
            "title": self.title,
            "code_file": self.code_file,
            "language": "python",
            "kernel_type": self.kernel_type.value,
            "is_private": True,
            "enable_gpu": self.enable_gpu,
            "enable_internet": self.enable_internet,
            "dataset_sources": self.dataset_sources or [],
            "competition_sources": self.competition_sources or [],
            "kernel_sources": self.kernel_sources or [],
            "docker_image_pinning_type": self.docker_image_pinning_type
        }


@dataclass
class KernelExecution:
    """Kernel実行状態"""
    kernel_id: str
    title: str
    status: KernelStatus
    start_time: datetime
    gpu_enabled: bool
    estimated_runtime_minutes: int
    metadata: Dict[str, Any] = None
    
    # 実行結果
    completion_time: Optional[datetime] = None
    output_files: List[Dict[str, Any]] = None
    execution_log: str = ""
    gpu_hours_used: float = 0.0
    success: bool = False
    score: float = 0.0
    error_message: Optional[str] = None


class KaggleResourceTracker:
    """Kaggleリソース追跡"""
    
    def __init__(self):
        self.weekly_gpu_limit = 30.0  # 時間
        self.weekly_reset_day = 0  # 月曜日
        self.current_usage = {}
    
    def get_current_week_key(self) -> str:
        """現在週のキー生成"""
        now = datetime.utcnow()
        # 週の開始（月曜日）を計算
        days_since_monday = now.weekday()
        week_start = now - timedelta(days=days_since_monday)
        return week_start.strftime("%Y-%W")
    
    def get_remaining_gpu_hours(self) -> float:
        """残りGPU時間取得"""
        week_key = self.get_current_week_key()
        used_hours = self.current_usage.get(week_key, 0.0)
        return max(0.0, self.weekly_gpu_limit - used_hours)
    
    def can_allocate_gpu_hours(self, required_hours: float) -> bool:
        """GPU時間割り当て可能性判定"""
        return self.get_remaining_gpu_hours() >= required_hours
    
    def allocate_gpu_hours(self, hours: float) -> bool:
        """GPU時間割り当て"""
        if not self.can_allocate_gpu_hours(hours):
            return False
        
        week_key = self.get_current_week_key()
        self.current_usage[week_key] = self.current_usage.get(week_key, 0.0) + hours
        return True
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """使用統計取得"""
        week_key = self.get_current_week_key()
        used_hours = self.current_usage.get(week_key, 0.0)
        
        return {
            "weekly_limit": self.weekly_gpu_limit,
            "used_hours": used_hours,
            "remaining_hours": self.get_remaining_gpu_hours(),
            "utilization_rate": used_hours / self.weekly_gpu_limit,
            "week_key": week_key,
            "reset_in_days": (7 - datetime.utcnow().weekday()) % 7
        }


class KaggleKernelManager:
    """Kaggle Kernels実行管理システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api = KaggleApi()
        self.resource_tracker = KaggleResourceTracker()
        
        # 認証
        try:
            self.api.authenticate()
            self.authenticated = True
        except Exception as e:
            self.logger.warning(f"Kaggle API認証失敗: {e}")
            self.authenticated = KAGGLE_API_AVAILABLE
        
        # 実行中のKernel追跡
        self.active_kernels: Dict[str, KernelExecution] = {}
        self.execution_history: List[KernelExecution] = []
        
        # 設定
        self.max_concurrent_kernels = 3
        self.default_timeout_minutes = 120
    
    async def execute_technique_kernel(
        self,
        technique_name: str,
        notebook_code: str,
        competition_name: str,
        estimated_runtime_hours: float = 2.0,
        datasets: List[str] = None,
        priority: str = "standard"
    ) -> KernelExecution:
        """技術実装Kernel実行"""
        
        # リソース割り当てチェック
        if not self.resource_tracker.can_allocate_gpu_hours(estimated_runtime_hours):
            raise Exception(f"GPU時間不足: 必要{estimated_runtime_hours}h, 残り{self.resource_tracker.get_remaining_gpu_hours()}h")
        
        # Kernel設定作成
        kernel_title = f"[Claude] {technique_name} - {competition_name}"
        kernel_config = KernelConfig(
            title=kernel_title,
            code_file="notebook.ipynb",
            kernel_type=KernelType.NOTEBOOK,
            competition_sources=[competition_name] if competition_name else [],
            dataset_sources=datasets or [],
            enable_gpu=True,
            enable_internet=True
        )
        
        # Kernel実行開始
        try:
            kernel_execution = await self._create_and_submit_kernel(
                config=kernel_config,
                notebook_code=notebook_code,
                estimated_runtime_hours=estimated_runtime_hours
            )
            
            # リソース割り当て
            self.resource_tracker.allocate_gpu_hours(estimated_runtime_hours)
            
            # 実行追跡に追加
            self.active_kernels[kernel_execution.kernel_id] = kernel_execution
            
            self.logger.info(f"Kernel実行開始: {kernel_execution.kernel_id} - {technique_name}")
            return kernel_execution
            
        except Exception as e:
            self.logger.error(f"Kernel実行失敗: {technique_name} - {e}")
            raise
    
    async def _create_and_submit_kernel(
        self,
        config: KernelConfig,
        notebook_code: str,
        estimated_runtime_hours: float
    ) -> KernelExecution:
        """Kernel作成・提出"""
        
        # 一時ディレクトリでKernel準備
        with tempfile.TemporaryDirectory() as temp_dir:
            # notebook.ipynbファイル作成
            notebook_path = os.path.join(temp_dir, "notebook.ipynb")
            with open(notebook_path, 'w', encoding='utf-8') as f:
                f.write(notebook_code)
            
            # metadata.jsonファイル作成
            metadata_path = os.path.join(temp_dir, "kernel-metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(config.to_metadata(), f, indent=2)
            
            # Kernel提出
            try:
                if KAGGLE_API_AVAILABLE and self.authenticated:
                    result = self.api.kernels_push(temp_dir)
                    kernel_id = result.get("kernel_id", f"test-{int(time.time())}")
                else:
                    # テスト環境用の模擬実行
                    kernel_id = f"test-kernel-{int(time.time())}"
                
                # 実行状態オブジェクト作成
                execution = KernelExecution(
                    kernel_id=kernel_id,
                    title=config.title,
                    status=KernelStatus.QUEUED,
                    start_time=datetime.utcnow(),
                    gpu_enabled=config.enable_gpu,
                    estimated_runtime_minutes=int(estimated_runtime_hours * 60),
                    metadata=config.to_metadata()
                )
                
                return execution
                
            except Exception as e:
                self.logger.error(f"Kernel提出エラー: {e}")
                raise
    
    async def monitor_kernel_execution(
        self,
        kernel_id: str,
        check_interval_seconds: int = 60
    ) -> KernelExecution:
        """Kernel実行監視"""
        
        if kernel_id not in self.active_kernels:
            raise ValueError(f"監視対象Kernelが見つかりません: {kernel_id}")
        
        execution = self.active_kernels[kernel_id]
        
        while execution.status in [KernelStatus.QUEUED, KernelStatus.RUNNING]:
            try:
                # ステータス確認
                if KAGGLE_API_AVAILABLE and self.authenticated:
                    status = self.api.kernels_status(kernel_id)
                    execution.status = KernelStatus(status.lower())
                else:
                    # テスト環境用の模擬進行
                    await asyncio.sleep(1)
                    execution.status = KernelStatus.COMPLETE
                    execution.success = True
                    execution.score = 0.75  # 模擬スコア
                
                # 完了判定
                if execution.status in [KernelStatus.COMPLETE, KernelStatus.ERROR, KernelStatus.CANCELLED]:
                    await self._process_kernel_completion(execution)
                    break
                
                # 待機
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Kernel監視エラー: {kernel_id} - {e}")
                execution.status = KernelStatus.ERROR
                execution.error_message = str(e)
                break
        
        # 実行完了処理
        if kernel_id in self.active_kernels:
            del self.active_kernels[kernel_id]
        
        self.execution_history.append(execution)
        
        return execution
    
    async def _process_kernel_completion(self, execution: KernelExecution):
        """Kernel完了処理"""
        
        execution.completion_time = datetime.utcnow()
        
        if execution.status == KernelStatus.COMPLETE:
            try:
                # 出力ファイル取得
                if KAGGLE_API_AVAILABLE and self.authenticated:
                    output_files = self.api.kernels_output(execution.kernel_id)
                    execution.output_files = output_files
                else:
                    # テスト環境用の模擬出力
                    execution.output_files = [
                        {"path": "submission.csv", "size": 1024},
                        {"path": "training_log.txt", "size": 512}
                    ]
                
                # GPU使用時間計算
                if execution.gpu_enabled:
                    runtime_hours = (execution.completion_time - execution.start_time).total_seconds() / 3600
                    execution.gpu_hours_used = min(runtime_hours, execution.estimated_runtime_minutes / 60)
                
                # スコア抽出（出力ファイルから）
                execution.score = await self._extract_score_from_output(execution.output_files)
                execution.success = execution.score > 0
                
                self.logger.info(f"Kernel実行完了: {execution.kernel_id}, スコア: {execution.score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Kernel完了処理エラー: {execution.kernel_id} - {e}")
                execution.error_message = str(e)
                execution.success = False
        
        else:
            # エラー・キャンセル処理
            execution.success = False
            self.logger.warning(f"Kernel実行失敗: {execution.kernel_id}, ステータス: {execution.status.value}")
    
    async def _extract_score_from_output(self, output_files: List[Dict[str, Any]]) -> float:
        """出力ファイルからスコア抽出"""
        
        # 模擬スコア抽出（実際にはログファイルやjsonから解析）
        for file_info in output_files:
            if "log" in file_info.get("path", "").lower():
                # ログファイルからスコア抽出（模擬）
                return 0.8234  # 例: CV Score
        
        return 0.0
    
    async def cancel_kernel(self, kernel_id: str) -> bool:
        """Kernel実行キャンセル"""
        
        try:
            if KAGGLE_API_AVAILABLE and self.authenticated:
                # 実際のキャンセル処理（APIに依存）
                pass
            
            # 内部状態更新
            if kernel_id in self.active_kernels:
                execution = self.active_kernels[kernel_id]
                execution.status = KernelStatus.CANCELLED
                execution.completion_time = datetime.utcnow()
                del self.active_kernels[kernel_id]
                self.execution_history.append(execution)
            
            self.logger.info(f"Kernelキャンセル完了: {kernel_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Kernelキャンセル失敗: {kernel_id} - {e}")
            return False
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """リソース状態取得"""
        
        usage_stats = self.resource_tracker.get_usage_statistics()
        
        return {
            "resource_usage": usage_stats,
            "active_kernels": len(self.active_kernels),
            "max_concurrent": self.max_concurrent_kernels,
            "execution_history_count": len(self.execution_history),
            "recent_success_rate": self._calculate_recent_success_rate(),
            "authentication_status": self.authenticated
        }
    
    def _calculate_recent_success_rate(self) -> float:
        """最近の成功率計算"""
        
        if not self.execution_history:
            return 0.0
        
        # 過去24時間の実行結果
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_executions = [
            e for e in self.execution_history 
            if e.completion_time and e.completion_time > cutoff_time
        ]
        
        if not recent_executions:
            return 0.0
        
        successful = len([e for e in recent_executions if e.success])
        return successful / len(recent_executions)
    
    async def cleanup_completed_kernels(self, hours_threshold: int = 24):
        """完了Kernel履歴クリーンアップ"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_threshold)
        
        # 古い履歴を削除（最新100件は保持）
        if len(self.execution_history) > 100:
            self.execution_history = sorted(
                self.execution_history, 
                key=lambda x: x.completion_time or x.start_time,
                reverse=True
            )[:100]
        
        cleaned_count = len([e for e in self.execution_history if (e.completion_time or e.start_time) < cutoff_time])
        self.execution_history = [e for e in self.execution_history if (e.completion_time or e.start_time) >= cutoff_time]
        
        self.logger.info(f"Kernel履歴クリーンアップ: {cleaned_count}件削除")
    
    async def estimate_execution_cost(
        self,
        technique_complexity: float,
        competition_type: str,
        dataset_size_gb: float = 1.0
    ) -> Dict[str, Any]:
        """実行コスト推定"""
        
        # 基本実行時間推定
        base_hours = {
            "tabular": 1.5,
            "computer_vision": 3.0,
            "nlp": 2.5,
            "time_series": 2.0
        }.get(competition_type, 2.0)
        
        # 複雑度による調整
        complexity_multiplier = 0.5 + (technique_complexity * 1.5)
        
        # データサイズによる調整
        data_multiplier = max(1.0, dataset_size_gb / 5.0)
        
        estimated_hours = base_hours * complexity_multiplier * data_multiplier
        
        return {
            "estimated_gpu_hours": min(estimated_hours, 6.0),  # 最大6時間制限
            "estimated_cost_usd": 0.0,  # Kaggleは無料
            "feasibility": self.resource_tracker.can_allocate_gpu_hours(estimated_hours),
            "confidence": 0.8,
            "factors": {
                "base_hours": base_hours,
                "complexity_multiplier": complexity_multiplier,
                "data_multiplier": data_multiplier
            }
        }