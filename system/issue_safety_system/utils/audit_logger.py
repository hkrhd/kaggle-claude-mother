"""
監査ログ・操作追跡システム

GitHub API操作の完全な監査証跡を提供。
セキュリティ・コンプライアンス・デバッグを支援。
"""

import json
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import uuid
from pathlib import Path


class AuditLevel(Enum):
    """監査レベル"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEventType(Enum):
    """監査イベント種別"""
    OPERATION_START = "operation_start"
    OPERATION_COMPLETE = "operation_complete"
    OPERATION_ERROR = "operation_error"
    RATE_LIMIT_HIT = "rate_limit_hit"
    AUTHENTICATION_ERROR = "authentication_error"
    PERMISSION_DENIED = "permission_denied"
    DUPLICATE_DETECTION = "duplicate_detection"
    RETRY_ATTEMPT = "retry_attempt"
    SYSTEM_EVENT = "system_event"


@dataclass
class AuditEvent:
    """監査イベント"""
    event_id: str
    event_type: AuditEventType
    operation: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    level: AuditLevel = AuditLevel.INFO
    context: Dict[str, Any] = field(default_factory=dict)
    user: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "context": self.context,
            "user": self.user,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "duration": self.duration,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class OperationContext:
    """操作コンテキスト"""
    operation_id: str
    operation_name: str
    start_time: datetime
    context_data: Dict[str, Any]
    session_id: Optional[str] = None
    parent_operation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "context_data": self.context_data,
            "session_id": self.session_id,
            "parent_operation_id": self.parent_operation_id
        }


class AuditLogger:
    """監査ログ・操作追跡システム"""
    
    def __init__(
        self,
        log_directory: str = "logs/audit",
        max_log_size_mb: int = 100,
        retention_days: int = 30,
        session_id: Optional[str] = None
    ):
        self.log_directory = Path(log_directory)
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        self.retention_days = retention_days
        self.session_id = session_id or self.generate_session_id()
        
        # ログディレクトリ作成
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # ログファイルパス
        self.current_log_file = self.log_directory / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # 内部状態
        self.active_operations: Dict[str, OperationContext] = {}
        self.event_count = 0
        self.start_time = datetime.utcnow()
        self.lock = threading.Lock()
        
        # Python標準ログ
        self.logger = logging.getLogger(__name__)
        
        # 初期化イベント
        self.log_event(
            AuditEvent(
                event_id=self.generate_event_id(),
                event_type=AuditEventType.SYSTEM_EVENT,
                operation="audit_logger_init",
                level=AuditLevel.INFO,
                context={
                    "session_id": self.session_id,
                    "log_directory": str(self.log_directory),
                    "retention_days": self.retention_days
                }
            )
        )
    
    def generate_event_id(self) -> str:
        """イベントID生成"""
        return f"evt_{uuid.uuid4().hex[:12]}"
    
    def generate_operation_id(self) -> str:
        """操作ID生成"""
        return f"op_{uuid.uuid4().hex[:12]}"
    
    def generate_session_id(self) -> str:
        """セッションID生成"""
        return f"sess_{uuid.uuid4().hex[:16]}"
    
    def start_operation(
        self,
        operation_name: str,
        context_data: Dict[str, Any],
        parent_operation_id: Optional[str] = None
    ) -> OperationContext:
        """操作開始・コンテキスト作成"""
        
        operation_id = self.generate_operation_id()
        operation_context = OperationContext(
            operation_id=operation_id,
            operation_name=operation_name,
            start_time=datetime.utcnow(),
            context_data=context_data,
            session_id=self.session_id,
            parent_operation_id=parent_operation_id
        )
        
        with self.lock:
            self.active_operations[operation_id] = operation_context
        
        # 開始イベントログ
        start_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.OPERATION_START,
            operation=operation_name,
            level=AuditLevel.INFO,
            context=context_data,
            session_id=self.session_id,
            metadata={
                "operation_id": operation_id,
                "parent_operation_id": parent_operation_id
            }
        )
        
        self.log_event(start_event)
        
        return operation_context
    
    def complete_operation(
        self,
        operation_context: OperationContext,
        success: bool,
        result_data: Optional[Dict[str, Any]] = None
    ):
        """操作完了・結果記録"""
        
        operation_id = operation_context.operation_id
        end_time = datetime.utcnow()
        duration = (end_time - operation_context.start_time).total_seconds()
        
        # 完了イベントログ
        complete_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.OPERATION_COMPLETE,
            operation=operation_context.operation_name,
            level=AuditLevel.INFO if success else AuditLevel.ERROR,
            context=operation_context.context_data,
            session_id=self.session_id,
            duration=duration,
            success=success,
            metadata={
                "operation_id": operation_id,
                "result_data": result_data or {},
                "end_time": end_time.isoformat()
            }
        )
        
        self.log_event(complete_event)
        
        # アクティブ操作から削除
        with self.lock:
            self.active_operations.pop(operation_id, None)
    
    def log_error(
        self,
        operation_context: OperationContext,
        error: Exception,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """エラーログ記録"""
        
        error_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.OPERATION_ERROR,
            operation=operation_context.operation_name,
            level=AuditLevel.ERROR,
            context=operation_context.context_data,
            session_id=self.session_id,
            success=False,
            error_message=str(error),
            metadata={
                "operation_id": operation_context.operation_id,
                "error_type": type(error).__name__,
                "additional_context": additional_context or {}
            }
        )
        
        self.log_event(error_event)
    
    def log_rate_limit_event(
        self,
        operation: str,
        rate_limit_info: Dict[str, Any],
        action_taken: str
    ):
        """レート制限イベントログ"""
        
        rate_limit_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.RATE_LIMIT_HIT,
            operation=operation,
            level=AuditLevel.WARNING,
            context=rate_limit_info,
            session_id=self.session_id,
            metadata={
                "action_taken": action_taken,
                "rate_limit_reset_time": rate_limit_info.get("reset_time"),
                "remaining_requests": rate_limit_info.get("remaining")
            }
        )
        
        self.log_event(rate_limit_event)
    
    def log_duplicate_detection(
        self,
        operation: str,
        duplicate_info: Dict[str, Any]
    ):
        """重複検出イベントログ"""
        
        duplicate_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.DUPLICATE_DETECTION,
            operation=operation,
            level=AuditLevel.INFO,
            context=duplicate_info,
            session_id=self.session_id,
            metadata={
                "detection_method": duplicate_info.get("method"),
                "similarity_score": duplicate_info.get("similarity_score")
            }
        )
        
        self.log_event(duplicate_event)
    
    def log_retry_attempt(
        self,
        operation: str,
        attempt_number: int,
        error: Exception,
        delay_seconds: float
    ):
        """リトライ試行ログ"""
        
        retry_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.RETRY_ATTEMPT,
            operation=operation,
            level=AuditLevel.WARNING,
            context={
                "attempt_number": attempt_number,
                "error_message": str(error),
                "delay_seconds": delay_seconds
            },
            session_id=self.session_id,
            metadata={
                "error_type": type(error).__name__,
                "retry_strategy": "exponential_backoff"
            }
        )
        
        self.log_event(retry_event)
    
    def log_event(self, event: AuditEvent):
        """イベントログ出力"""
        
        try:
            # ファイルサイズチェック・ローテーション
            self.check_log_rotation()
            
            # JSONLines形式で出力
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                f.write(event.to_json() + '\n')
            
            # イベントカウント更新
            with self.lock:
                self.event_count += 1
            
            # 標準ログにも出力（レベル別）
            log_message = f"[{event.event_type.value}] {event.operation}: {event.context}"
            
            if event.level == AuditLevel.DEBUG:
                self.logger.debug(log_message)
            elif event.level == AuditLevel.INFO:
                self.logger.info(log_message)
            elif event.level == AuditLevel.WARNING:
                self.logger.warning(log_message)
            elif event.level == AuditLevel.ERROR:
                self.logger.error(log_message)
            elif event.level == AuditLevel.CRITICAL:
                self.logger.critical(log_message)
                
        except Exception as e:
            # ログ出力失敗は標準ログのみ
            self.logger.error(f"監査ログ出力失敗: {e}")
    
    def check_log_rotation(self):
        """ログローテーション確認"""
        
        if not self.current_log_file.exists():
            return
        
        # ファイルサイズチェック
        file_size = self.current_log_file.stat().st_size
        if file_size > self.max_log_size_bytes:
            # 新しいファイル名生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rotated_file = self.log_directory / f"audit_{timestamp}.jsonl"
            
            # ファイル移動
            self.current_log_file.rename(rotated_file)
            
            # 新しいログファイル作成
            today = datetime.now().strftime('%Y%m%d')
            self.current_log_file = self.log_directory / f"audit_{today}.jsonl"
            
            self.logger.info(f"ログローテーション実行: {rotated_file}")
    
    def cleanup_old_logs(self):
        """古いログファイル削除"""
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        for log_file in self.log_directory.glob("audit_*.jsonl"):
            try:
                file_stat = log_file.stat()
                file_date = datetime.fromtimestamp(file_stat.st_mtime)
                
                if file_date < cutoff_date:
                    log_file.unlink()
                    deleted_count += 1
                    
            except Exception as e:
                self.logger.warning(f"ログファイル削除失敗: {log_file} - {e}")
        
        if deleted_count > 0:
            self.logger.info(f"古いログファイル削除: {deleted_count}件")
    
    def get_statistics(self) -> Dict[str, Any]:
        """監査統計取得"""
        
        uptime = datetime.utcnow() - self.start_time
        
        # アクティブ操作統計
        with self.lock:
            active_operations_count = len(self.active_operations)
            active_operations_list = [
                {
                    "operation_id": op.operation_id,
                    "operation_name": op.operation_name,
                    "duration_seconds": (datetime.utcnow() - op.start_time).total_seconds()
                }
                for op in self.active_operations.values()
            ]
        
        # ログファイル統計
        log_files = list(self.log_directory.glob("audit_*.jsonl"))
        total_log_size = sum(f.stat().st_size for f in log_files if f.exists())
        
        return {
            "session_id": self.session_id,
            "uptime_seconds": uptime.total_seconds(),
            "total_events": self.event_count,
            "events_per_minute": (self.event_count / max(uptime.total_seconds() / 60, 1)),
            "active_operations": {
                "count": active_operations_count,
                "operations": active_operations_list
            },
            "log_files": {
                "count": len(log_files),
                "total_size_mb": round(total_log_size / (1024 * 1024), 2)
            },
            "current_log_file": str(self.current_log_file),
            "retention_days": self.retention_days
        }
    
    def search_events(
        self,
        operation_name: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """イベント検索"""
        
        matching_events = []
        
        # 検索対象ファイル特定
        search_files = []
        if start_time or end_time:
            # 日付範囲指定時は該当ファイルのみ
            for log_file in self.log_directory.glob("audit_*.jsonl"):
                try:
                    file_date_str = log_file.stem.split('_')[1][:8]  # YYYYMMDD
                    file_date = datetime.strptime(file_date_str, '%Y%m%d')
                    
                    if start_time and file_date < start_time.replace(hour=0, minute=0, second=0):
                        continue
                    if end_time and file_date > end_time.replace(hour=23, minute=59, second=59):
                        continue
                    
                    search_files.append(log_file)
                except:
                    # ファイル名パース失敗時は検索対象に含める
                    search_files.append(log_file)
        else:
            # 全ファイル検索
            search_files = list(self.log_directory.glob("audit_*.jsonl"))
        
        # ファイル内検索
        for log_file in sorted(search_files, reverse=True):  # 新しいファイルから検索
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(matching_events) >= limit:
                            break
                        
                        try:
                            event_data = json.loads(line.strip())
                            
                            # フィルタ適用
                            if operation_name and event_data.get('operation') != operation_name:
                                continue
                            if event_type and event_data.get('event_type') != event_type.value:
                                continue
                            if success is not None and event_data.get('success') != success:
                                continue
                            
                            # 時間範囲フィルタ
                            event_time = datetime.fromisoformat(event_data['timestamp'])
                            if start_time and event_time < start_time:
                                continue
                            if end_time and event_time > end_time:
                                continue
                            
                            matching_events.append(event_data)
                            
                        except json.JSONDecodeError:
                            continue
                        
            except Exception as e:
                self.logger.warning(f"ログファイル検索失敗: {log_file} - {e}")
        
        return matching_events[:limit]
    
    def export_audit_report(
        self,
        start_time: datetime,
        end_time: datetime,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """監査レポート出力"""
        
        # 期間内イベント取得
        events = self.search_events(
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # 統計計算
        total_events = len(events)
        operations = {}
        event_types = {}
        success_count = 0
        error_count = 0
        
        for event in events:
            # 操作別統計
            operation = event.get('operation', 'unknown')
            operations[operation] = operations.get(operation, 0) + 1
            
            # イベント種別統計
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # 成功/失敗統計
            if event.get('success') is True:
                success_count += 1
            elif event.get('success') is False:
                error_count += 1
        
        # レポート作成
        report = {
            "report_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": (end_time - start_time).total_seconds() / 3600
            },
            "summary": {
                "total_events": total_events,
                "success_count": success_count,
                "error_count": error_count,
                "unknown_count": total_events - success_count - error_count,
                "success_rate_percent": (success_count / total_events * 100) if total_events > 0 else 0
            },
            "operations": operations,
            "event_types": event_types,
            "session_id": self.session_id,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # ファイル出力（指定時）
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"監査レポート出力: {output_file}")
                report["output_file"] = output_file
                
            except Exception as e:
                self.logger.error(f"レポート出力失敗: {e}")
                report["output_error"] = str(e)
        
        return report
    
    def __enter__(self):
        """コンテキストマネージャー開始"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャー終了"""
        
        # 終了イベントログ
        shutdown_event = AuditEvent(
            event_id=self.generate_event_id(),
            event_type=AuditEventType.SYSTEM_EVENT,
            operation="audit_logger_shutdown",
            level=AuditLevel.INFO,
            context={
                "session_id": self.session_id,
                "total_events": self.event_count,
                "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds()
            }
        )
        
        self.log_event(shutdown_event)
        
        # 古いログクリーンアップ
        self.cleanup_old_logs()
    
    def force_log_rotation(self):
        """強制ログローテーション"""
        
        if self.current_log_file.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rotated_file = self.log_directory / f"audit_{timestamp}_manual.jsonl"
            
            self.current_log_file.rename(rotated_file)
            
            today = datetime.now().strftime('%Y%m%d')
            self.current_log_file = self.log_directory / f"audit_{today}.jsonl"
            
            self.logger.info(f"手動ログローテーション実行: {rotated_file}")
            return str(rotated_file)
        
        return None