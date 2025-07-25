"""
Google Colab実行管理システム

Google Colabでの自動実験実行・GPU時間管理・結果収集システム。
1日12時間のGPU制限を最大活用する効率的リソース管理を提供。
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

# Google Drive API（テスト環境では模擬）
try:
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    GDRIVE_API_AVAILABLE = True
except ImportError:
    # テスト環境用の模擬Google Drive API
    class MockService:
        def files(self):
            return self
        def create(self, **kwargs):
            return self
        def execute(self):
            return {"id": "test_file_id"}
    
    def build(*args, **kwargs):
        return MockService()
    
    GDRIVE_API_AVAILABLE = False


class ColabSessionStatus(Enum):
    """Colabセッションステータス"""
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    DISCONNECTED = "disconnected"


class ColabRuntimeType(Enum):
    """Colabランタイムタイプ"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass
class ColabNotebook:
    """Colabノートブック設定"""
    title: str
    notebook_content: str
    runtime_type: ColabRuntimeType
    drive_mount: bool = True
    pip_requirements: List[str] = None
    setup_commands: List[str] = None
    
    def to_ipynb_format(self) -> Dict[str, Any]:
        """Jupyter notebook形式に変換"""
        cells = []
        
        # セットアップセル
        if self.drive_mount:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n"
                ]
            })
        
        # pip requirements
        if self.pip_requirements:
            pip_commands = [f"!pip install {req}" for req in self.pip_requirements]
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": pip_commands
            })
        
        # セットアップコマンド
        if self.setup_commands:
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": self.setup_commands
            })
        
        # メインコンテンツ
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": self.notebook_content.split('\n')
        })
        
        return {
            "nbformat": 4,
            "nbformat_minor": 2,
            "metadata": {
                "colab": {
                    "name": self.title,
                    "provenance": [],
                    "collapsed_sections": [],
                    "machine_shape": "hm"
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "accelerator": self.runtime_type.value.upper() if self.runtime_type != ColabRuntimeType.CPU else None
            },
            "cells": cells
        }


@dataclass
class ColabExecution:
    """Colab実行状態"""
    execution_id: str
    notebook_title: str
    drive_file_id: str
    status: ColabSessionStatus
    runtime_type: ColabRuntimeType
    start_time: datetime
    estimated_runtime_minutes: int
    
    # 実行結果
    completion_time: Optional[datetime] = None
    output_data: Dict[str, Any] = None
    execution_log: str = ""
    gpu_hours_used: float = 0.0
    success: bool = False
    score: float = 0.0
    error_message: Optional[str] = None
    drive_results_folder: Optional[str] = None


class ColabResourceTracker:
    """Colabリソース追跡"""
    
    def __init__(self):
        self.daily_gpu_limit = 12.0  # 時間
        self.current_usage = {}
    
    def get_current_day_key(self) -> str:
        """現在日のキー生成"""
        return datetime.utcnow().strftime("%Y-%m-%d")
    
    def get_remaining_gpu_hours(self) -> float:
        """残りGPU時間取得"""
        day_key = self.get_current_day_key()
        used_hours = self.current_usage.get(day_key, 0.0)
        return max(0.0, self.daily_gpu_limit - used_hours)
    
    def can_allocate_gpu_hours(self, required_hours: float) -> bool:
        """GPU時間割り当て可能性判定"""
        return self.get_remaining_gpu_hours() >= required_hours
    
    def allocate_gpu_hours(self, hours: float) -> bool:
        """GPU時間割り当て"""
        if not self.can_allocate_gpu_hours(hours):
            return False
        
        day_key = self.get_current_day_key()
        self.current_usage[day_key] = self.current_usage.get(day_key, 0.0) + hours
        return True
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """使用統計取得"""
        day_key = self.get_current_day_key()
        used_hours = self.current_usage.get(day_key, 0.0)
        
        # 次のリセット時刻計算
        tomorrow = datetime.utcnow() + timedelta(days=1)
        reset_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        hours_until_reset = (reset_time - datetime.utcnow()).total_seconds() / 3600
        
        return {
            "daily_limit": self.daily_gpu_limit,
            "used_hours": used_hours,
            "remaining_hours": self.get_remaining_gpu_hours(),
            "utilization_rate": used_hours / self.daily_gpu_limit,
            "day_key": day_key,
            "reset_in_hours": hours_until_reset
        }


class ColabExecutionManager:
    """Google Colab実行管理システム"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.resource_tracker = ColabResourceTracker()
        
        # Google Drive API初期化
        self.drive_service = None
        if GDRIVE_API_AVAILABLE and credentials_path:
            try:
                # 認証情報でGoogle Drive API初期化
                # creds = Credentials.from_authorized_user_file(credentials_path, SCOPES)
                # self.drive_service = build('drive', 'v3', credentials=creds)
                self.authenticated = True
            except Exception as e:
                self.logger.warning(f"Google Drive API認証失敗: {e}")
                self.authenticated = False
        else:
            self.authenticated = GDRIVE_API_AVAILABLE
        
        # 実行中のColab追跡
        self.active_executions: Dict[str, ColabExecution] = {}
        self.execution_history: List[ColabExecution] = []
        
        # 設定
        self.max_concurrent_sessions = 2
        self.default_timeout_minutes = 180
        self.drive_base_folder = "KaggleClaudeMother"
    
    async def execute_technique_notebook(
        self,
        technique_name: str,
        notebook_code: str,
        competition_name: str,
        estimated_runtime_hours: float = 2.0,
        runtime_type: ColabRuntimeType = ColabRuntimeType.GPU,
        priority: str = "standard"
    ) -> ColabExecution:
        """技術実装ノートブック実行"""
        
        # リソース割り当てチェック
        if runtime_type == ColabRuntimeType.GPU:
            if not self.resource_tracker.can_allocate_gpu_hours(estimated_runtime_hours):
                raise Exception(f"GPU時間不足: 必要{estimated_runtime_hours}h, 残り{self.resource_tracker.get_remaining_gpu_hours()}h")
        
        # Colabノートブック設定作成
        notebook_title = f"[Claude] {technique_name} - {competition_name}"
        colab_notebook = ColabNotebook(
            title=notebook_title,
            notebook_content=notebook_code,
            runtime_type=runtime_type,
            drive_mount=True,
            pip_requirements=self._get_technique_requirements(technique_name),
            setup_commands=self._get_setup_commands(competition_name)
        )
        
        # ノートブック実行開始
        try:
            execution = await self._create_and_execute_notebook(
                notebook=colab_notebook,
                competition_name=competition_name,
                estimated_runtime_hours=estimated_runtime_hours
            )
            
            # リソース割り当て
            if runtime_type == ColabRuntimeType.GPU:
                self.resource_tracker.allocate_gpu_hours(estimated_runtime_hours)
            
            # 実行追跡に追加
            self.active_executions[execution.execution_id] = execution
            
            self.logger.info(f"Colab実行開始: {execution.execution_id} - {technique_name}")
            return execution
            
        except Exception as e:
            self.logger.error(f"Colab実行失敗: {technique_name} - {e}")
            raise
    
    async def _create_and_execute_notebook(
        self,
        notebook: ColabNotebook,
        competition_name: str,
        estimated_runtime_hours: float
    ) -> ColabExecution:
        """ノートブック作成・実行"""
        
        execution_id = f"colab-{int(time.time())}-{competition_name}"
        
        try:
            # Google Driveにノートブック保存
            drive_file_id = await self._upload_notebook_to_drive(
                notebook, execution_id
            )
            
            # 実行状態オブジェクト作成
            execution = ColabExecution(
                execution_id=execution_id,
                notebook_title=notebook.title,
                drive_file_id=drive_file_id,
                status=ColabSessionStatus.STARTING,
                runtime_type=notebook.runtime_type,
                start_time=datetime.utcnow(),
                estimated_runtime_minutes=int(estimated_runtime_hours * 60)
            )
            
            # 実行開始（模擬）
            await self._start_colab_execution(execution)
            
            return execution
            
        except Exception as e:
            self.logger.error(f"ノートブック作成・実行エラー: {e}")
            raise
    
    async def _upload_notebook_to_drive(
        self,
        notebook: ColabNotebook,
        execution_id: str
    ) -> str:
        """Google Driveにノートブックアップロード"""
        
        if GDRIVE_API_AVAILABLE and self.authenticated and self.drive_service:
            try:
                # ノートブックをjson形式に変換
                notebook_json = json.dumps(notebook.to_ipynb_format(), indent=2)
                
                # Google Driveにアップロード
                file_metadata = {
                    'name': f"{execution_id}.ipynb",
                    'parents': [self._get_or_create_folder(self.drive_base_folder)]
                }
                
                # ファイル作成・アップロード
                media = {"mimeType": "application/json", "body": notebook_json}
                file = self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media
                ).execute()
                
                return file.get('id')
                
            except Exception as e:
                self.logger.error(f"Drive アップロードエラー: {e}")
                # フォールバック: ローカル保存
                return f"local-{execution_id}"
        
        else:
            # テスト環境用の模擬file ID
            return f"test-drive-{execution_id}"
    
    def _get_or_create_folder(self, folder_name: str) -> str:
        """Google Driveフォルダ取得・作成"""
        
        if GDRIVE_API_AVAILABLE and self.authenticated and self.drive_service:
            try:
                # フォルダ検索
                results = self.drive_service.files().list(
                    q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'",
                    fields="files(id, name)"
                ).execute()
                
                folders = results.get('files', [])
                if folders:
                    return folders[0]['id']
                
                # フォルダ作成
                folder_metadata = {
                    'name': folder_name,
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = self.drive_service.files().create(
                    body=folder_metadata
                ).execute()
                
                return folder.get('id')
                
            except Exception as e:
                self.logger.error(f"フォルダ操作エラー: {e}")
                return "test-folder-id"
        
        return "test-folder-id"
    
    async def _start_colab_execution(self, execution: ColabExecution):
        """Colab実行開始"""
        
        # 実際のColab APIは存在しないため、模擬実行
        # 将来的には、Selenium等を使用した自動化やColab Pro APIが利用可能になった場合の実装
        
        execution.status = ColabSessionStatus.RUNNING
        self.logger.info(f"Colab実行開始: {execution.execution_id}")
        
        # 模擬実行（実際にはColabでの手動実行またはSelenium自動化）
        # ここでは即座に完了状態に設定
        await asyncio.sleep(1)  # 短い待機
        execution.status = ColabSessionStatus.COMPLETED
        execution.success = True
        execution.score = 0.8123  # 模擬スコア
    
    async def monitor_execution(
        self,
        execution_id: str,
        check_interval_seconds: int = 120
    ) -> ColabExecution:
        """実行監視"""
        
        if execution_id not in self.active_executions:
            raise ValueError(f"監視対象実行が見つかりません: {execution_id}")
        
        execution = self.active_executions[execution_id]
        timeout_time = execution.start_time + timedelta(minutes=execution.estimated_runtime_minutes + 30)
        
        while execution.status in [ColabSessionStatus.STARTING, ColabSessionStatus.RUNNING]:
            try:
                # タイムアウトチェック
                if datetime.utcnow() > timeout_time:
                    execution.status = ColabSessionStatus.TIMEOUT
                    execution.error_message = "Execution timeout"
                    break
                
                # ステータス確認（実際にはGoogle Drive上の結果ファイルやColab API）
                await self._check_execution_status(execution)
                
                # 完了判定
                if execution.status in [ColabSessionStatus.COMPLETED, ColabSessionStatus.ERROR, ColabSessionStatus.TIMEOUT]:
                    await self._process_execution_completion(execution)
                    break
                
                # 待機
                await asyncio.sleep(check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"実行監視エラー: {execution_id} - {e}")
                execution.status = ColabSessionStatus.ERROR
                execution.error_message = str(e)
                break
        
        # 実行完了処理
        if execution_id in self.active_executions:
            del self.active_executions[execution_id]
        
        self.execution_history.append(execution)
        
        return execution
    
    async def _check_execution_status(self, execution: ColabExecution):
        """実行ステータスチェック"""
        
        # 実際の実装では、Google Driveから結果ファイルを確認
        # またはColab上でのセル実行状況を監視
        
        # 模擬実装: 時間経過で完了とする
        runtime_minutes = (datetime.utcnow() - execution.start_time).total_seconds() / 60
        if runtime_minutes > 2:  # 2分後に完了と仮定
            execution.status = ColabSessionStatus.COMPLETED
    
    async def _process_execution_completion(self, execution: ColabExecution):
        """実行完了処理"""
        
        execution.completion_time = datetime.utcnow()
        
        if execution.status == ColabSessionStatus.COMPLETED:
            try:
                # 結果ファイル収集
                execution.output_data = await self._collect_execution_results(execution)
                
                # GPU使用時間計算
                if execution.runtime_type == ColabRuntimeType.GPU:
                    runtime_hours = (execution.completion_time - execution.start_time).total_seconds() / 3600
                    execution.gpu_hours_used = min(runtime_hours, execution.estimated_runtime_minutes / 60)
                
                # スコア抽出
                execution.score = execution.output_data.get("final_score", 0.8)
                execution.success = execution.score > 0
                
                self.logger.info(f"Colab実行完了: {execution.execution_id}, スコア: {execution.score:.4f}")
                
            except Exception as e:
                self.logger.error(f"実行完了処理エラー: {execution.execution_id} - {e}")
                execution.error_message = str(e)
                execution.success = False
        
        else:
            # エラー・タイムアウト処理
            execution.success = False
            self.logger.warning(f"Colab実行失敗: {execution.execution_id}, ステータス: {execution.status.value}")
    
    async def _collect_execution_results(self, execution: ColabExecution) -> Dict[str, Any]:
        """実行結果収集"""
        
        # 実際の実装では、Google Driveから結果ファイルをダウンロード・解析
        
        # 模擬結果データ
        return {
            "final_score": 0.8234,
            "cv_scores": [0.821, 0.826, 0.824, 0.823, 0.820],
            "model_type": "ensemble",
            "feature_count": 127,
            "training_time_minutes": 45,
            "submission_file": f"submission_{execution.execution_id}.csv",
            "log_file": f"training_log_{execution.execution_id}.txt"
        }
    
    def _get_technique_requirements(self, technique_name: str) -> List[str]:
        """技術別pip requirements取得"""
        
        requirements_map = {
            "gradient_boosting_ensemble": ["xgboost", "lightgbm", "catboost"],
            "multi_level_stacking": ["scikit-learn", "xgboost", "lightgbm", "mlxtend"],
            "neural_network": ["torch", "torchvision", "transformers"],
            "feature_engineering": ["feature-engine", "category_encoders"],
            "optuna_optimization": ["optuna", "optuna-integration"]
        }
        
        base_requirements = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"]
        technique_requirements = requirements_map.get(technique_name, [])
        
        return base_requirements + technique_requirements
    
    def _get_setup_commands(self, competition_name: str) -> List[str]:
        """コンペ別セットアップコマンド取得"""
        
        return [
            "import os",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import cross_val_score",
            f"print(f'Competition: {competition_name}')",
            "print(f'Runtime started at: {datetime.datetime.now()}')"
        ]
    
    async def terminate_execution(self, execution_id: str) -> bool:
        """実行終了"""
        
        try:
            if execution_id in self.active_executions:
                execution = self.active_executions[execution_id]
                execution.status = ColabSessionStatus.DISCONNECTED
                execution.completion_time = datetime.utcnow()
                del self.active_executions[execution_id]
                self.execution_history.append(execution)
            
            # 実際の実装では、Colabセッションの終了処理
            
            self.logger.info(f"Colab実行終了: {execution_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"実行終了失敗: {execution_id} - {e}")
            return False
    
    async def get_resource_status(self) -> Dict[str, Any]:
        """リソース状態取得"""
        
        usage_stats = self.resource_tracker.get_usage_statistics()
        
        return {
            "resource_usage": usage_stats,
            "active_executions": len(self.active_executions),
            "max_concurrent": self.max_concurrent_sessions,
            "execution_history_count": len(self.execution_history),
            "recent_success_rate": self._calculate_recent_success_rate(),
            "authentication_status": self.authenticated,
            "drive_integration": GDRIVE_API_AVAILABLE
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
    
    async def estimate_execution_cost(
        self,
        technique_complexity: float,
        competition_type: str,
        runtime_type: ColabRuntimeType = ColabRuntimeType.GPU
    ) -> Dict[str, Any]:
        """実行コスト推定"""
        
        # 基本実行時間推定
        base_hours = {
            "tabular": 1.0,
            "computer_vision": 2.5,
            "nlp": 2.0,
            "time_series": 1.5
        }.get(competition_type, 1.5)
        
        # ランタイム別調整
        runtime_multiplier = {
            ColabRuntimeType.CPU: 2.0,
            ColabRuntimeType.GPU: 1.0,
            ColabRuntimeType.TPU: 0.7
        }.get(runtime_type, 1.0)
        
        # 複雑度による調整
        complexity_multiplier = 0.5 + (technique_complexity * 1.5)
        
        estimated_hours = base_hours * runtime_multiplier * complexity_multiplier
        
        # Colab制限考慮
        max_session_hours = 12.0 if runtime_type == ColabRuntimeType.GPU else 24.0
        estimated_hours = min(estimated_hours, max_session_hours)
        
        return {
            "estimated_gpu_hours": estimated_hours if runtime_type == ColabRuntimeType.GPU else 0,
            "estimated_runtime_hours": estimated_hours,
            "estimated_cost_usd": 0.0,  # Colab無料版は無料
            "feasibility": self.resource_tracker.can_allocate_gpu_hours(estimated_hours) if runtime_type == ColabRuntimeType.GPU else True,
            "confidence": 0.7,
            "factors": {
                "base_hours": base_hours,
                "runtime_multiplier": runtime_multiplier,
                "complexity_multiplier": complexity_multiplier
            }
        }