"""
高度モニタリングエージェント

実行中のKaggle競技における全エージェント活動・実験進捗・リソース使用状況・
パフォーマンス指標をリアルタイム監視し、最適化提案を行うメインエージェント。
"""

import asyncio
import logging
import json
import uuid
import subprocess
import psutil
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics
from pathlib import Path

# GitHub Issue安全システム
from ...issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ...issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations

# 他エージェントの状態監視
from ..analyzer.analyzer_agent import AnalyzerAgent
from ..executor.executor_agent import ExecutorAgent

# LLMベース異常診断統合
from ..shared.llm_decision_base import ClaudeClient, LLMDecisionEngine, LLMDecisionRequest, LLMDecisionResponse, LLMDecisionType
from ...prompts.prompt_manager import PromptManager, PromptType


class MonitoringLevel(Enum):
    """監視レベル"""
    CRITICAL = "critical"    # 重要: エラー・異常のみ
    STANDARD = "standard"    # 標準: 通常動作+警告
    DETAILED = "detailed"    # 詳細: 全活動監視
    DEBUG = "debug"         # デバッグ: 最大詳細


class AlertSeverity(Enum):
    """アラート重要度"""
    EMERGENCY = "emergency"    # 緊急: 即座対応必要
    WARNING = "warning"       # 警告: 注意が必要
    INFO = "info"            # 情報: 参考情報
    SUCCESS = "success"      # 成功: 正常完了


class SystemHealth(Enum):
    """システム健全性"""
    HEALTHY = "healthy"        # 正常
    DEGRADED = "degraded"     # 性能低下
    CRITICAL = "critical"     # 重大問題
    OFFLINE = "offline"       # オフライン


@dataclass
class PerformanceMetrics:
    """パフォーマンス指標"""
    timestamp: datetime
    agent_type: str
    
    # 実行統計
    tasks_completed: int
    tasks_failed: int
    success_rate: float
    
    # リソース使用量
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: float
    gpu_memory_mb: float
    
    # API使用量
    api_calls_count: int
    api_rate_limit_remaining: int
    
    # カスタムメトリクス
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CompetitionProgress:
    """競技進捗状況"""
    competition_name: str
    start_time: datetime
    deadline: datetime
    
    # 分析進捗
    analysis_completion: float  # 0.0-1.0
    techniques_identified: int
    techniques_implemented: int
    
    # 実行進捗
    experiments_total: int
    experiments_completed: int
    experiments_running: int
    experiments_failed: int
    
    # スコア進捗
    current_best_score: float
    target_score: float
    score_improvement_rate: float
    
    # リソース消費
    gpu_hours_used: float
    gpu_hours_remaining: float
    budget_utilization: float
    
    # 予測
    estimated_completion_time: datetime
    medal_probability: float


@dataclass
class ServiceHealthCheck:
    """サービス健全性チェック結果"""
    timestamp: datetime
    service_active: bool
    service_status: str
    error_patterns_detected: List[str]
    memory_usage_percent: float
    disk_usage_percent: float
    system_load: float
    auto_repair_applied: bool = False
    repair_actions: List[str] = field(default_factory=list)


@dataclass
class SystemAlert:
    """システムアラート"""
    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    source_agent: str
    title: str
    description: str
    
    # コンテキスト情報
    competition_name: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    metrics_snapshot: Optional[Dict[str, Any]] = None
    
    # 対応状況
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "source_agent": self.source_agent,
            "title": self.title,
            "description": self.description,
            "competition_name": self.competition_name,
            "affected_components": self.affected_components,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


@dataclass
class MonitoringReport:
    """監視レポート"""
    report_id: str
    generated_at: datetime
    monitoring_period_hours: float
    
    # システム状態
    overall_health: SystemHealth
    agent_health_status: Dict[str, SystemHealth]
    
    # パフォーマンスサマリー
    performance_summary: Dict[str, PerformanceMetrics]
    
    # 進捗サマリー
    competition_progress: List[CompetitionProgress]
    
    # アラート統計
    alerts_summary: Dict[str, int]  # severity -> count
    active_alerts: List[SystemAlert]
    
    # 推奨事項
    recommendations: List[str]
    
    # 予測・トレンド
    trend_analysis: Dict[str, Any]


class MonitorAgent:
    """高度モニタリングエージェント - メインクラス"""
    
    def __init__(self, github_token: str, repo_name: str, service_name: str = "kaggle-claude-mother"):
        self.logger = logging.getLogger(__name__)
        
        # エージェント情報
        self.agent_id = f"monitor-{uuid.uuid4().hex[:8]}"
        self.agent_version = "3.0.0"  # LLMベース異常診断統合版
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
        
        # LLMベース異常診断統合
        self.claude_client = ClaudeClient()
        self.prompt_manager = PromptManager()
        self.llm_enabled = True
        
        # サービス監視設定
        self.service_name = service_name
        self.service_monitoring_enabled = True
        self.auto_repair_enabled = True
        
        # 監視対象エージェント（実際には参照で取得）
        self.monitored_agents: Dict[str, Any] = {}
        
        # 監視設定
        self.monitoring_level = MonitoringLevel.STANDARD
        self.monitoring_interval_seconds = 30  # エージェント監視間隔
        self.service_check_interval_seconds = 60  # サービス監視間隔
        self.alert_thresholds = {
            "success_rate_threshold": 0.8,
            "gpu_usage_threshold": 0.9,
            "memory_usage_threshold_mb": 8000,
            "memory_usage_threshold_percent": 90,
            "disk_usage_threshold_percent": 80,
            "api_rate_remaining_threshold": 100,
            "error_threshold_count": 3
        }
        
        # 監視データ
        self.performance_history: List[PerformanceMetrics] = []
        self.active_alerts: List[SystemAlert] = []
        self.competition_tracking: Dict[str, CompetitionProgress] = {}
        self.service_health_history: List[ServiceHealthCheck] = []
        
        # エラー・修復統計
        self.error_count = 0
        self.last_restart_time = 0
        self.total_auto_repairs = 0
        
        # 統計
        self.monitoring_cycles_completed = 0
        self.total_alerts_generated = 0
        
        # 監視タスク
        self.monitoring_task: Optional[asyncio.Task] = None
        self.service_monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring_active = False
        
        # 主要監視Issue番号（Issue重複回避）
        self.main_monitoring_issue: Optional[int] = None
    
    async def start_monitoring(self, target_agents: Dict[str, Any]):
        """監視開始"""
        
        self.logger.info(f"モニタリング開始: {len(target_agents)}エージェント監視")
        
        # 監視対象エージェント設定
        self.monitored_agents = target_agents
        
        # GitHub Issue作成: 監視開始通知
        monitoring_start_issue = await self._create_monitoring_issue(
            title=f"🔍 システム監視開始 - {self.agent_id}",
            description=f"""
## 監視システム起動

**監視エージェントID**: `{self.agent_id}`
**開始時刻**: {self.start_time.isoformat()}
**監視レベル**: {self.monitoring_level.value}
**監視間隔**: {self.monitoring_interval_seconds}秒

### 監視対象エージェント
{chr(10).join([f"- **{name}**: {agent.__class__.__name__}" for name, agent in target_agents.items()])}

### 監視項目
- パフォーマンス指標（CPU/GPU/メモリ使用量）
- 実験実行進捗・成功率
- API使用量・レート制限
- システム健全性・異常検出
- 競技進捗・スコア改善状況

このIssueで継続的な監視状況をレポートします。
            """,
            labels=["monitor", "system-status", "active"]
        )
        
        # 監視ループ開始
        self.is_monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info(f"監視タスク開始: Issue #{monitoring_start_issue}")
        
        return monitoring_start_issue
    
    async def stop_monitoring(self):
        """監視停止"""
        
        self.logger.info("モニタリング停止中...")
        
        self.is_monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 監視停止レポート生成
        final_report = await self.generate_monitoring_report()
        
        # GitHub Issue更新: 監視停止通知
        await self._post_monitoring_update(
            title="🛑 システム監視停止",
            content=f"""
## 監視システム停止

**停止時刻**: {datetime.utcnow().isoformat()}
**総監視サイクル**: {self.monitoring_cycles_completed}
**生成アラート数**: {self.total_alerts_generated}

### 最終監視レポート
```json
{json.dumps(final_report.__dict__, indent=2, default=str)}
```

監視システムは正常に停止しました。
            """
        )
        
        self.logger.info("モニタリング停止完了")
    
    async def _monitoring_loop(self):
        """メイン監視ループ"""
        
        while self.is_monitoring_active:
            try:
                cycle_start = datetime.utcnow()
                
                # 1. 全エージェントのメトリクス収集
                current_metrics = await self._collect_all_metrics()
                
                # 2. システム健全性評価
                health_status = await self._evaluate_system_health(current_metrics)
                
                # 3. アラート生成・処理
                new_alerts = await self._process_alerts(current_metrics, health_status)
                
                # 4. 競技進捗更新
                await self._update_competition_progress()
                
                # 5. パフォーマンス履歴更新
                self.performance_history.extend(current_metrics)
                
                # 6. 定期レポート生成（1時間毎）
                if self.monitoring_cycles_completed % 120 == 0:  # 30秒x120 = 1時間
                    report = await self.generate_monitoring_report()
                    await self._post_monitoring_report(report)
                
                # 7. 統計更新
                self.monitoring_cycles_completed += 1
                self.total_alerts_generated += len(new_alerts)
                
                # 8. 次回監視まで待機
                cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
                sleep_time = max(0, self.monitoring_interval_seconds - cycle_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"監視ループエラー: {e}")
                await asyncio.sleep(self.monitoring_interval_seconds)
    
    async def _collect_all_metrics(self) -> List[PerformanceMetrics]:
        """全エージェントメトリクス収集"""
        
        metrics = []
        timestamp = datetime.utcnow()
        
        for agent_name, agent in self.monitored_agents.items():
            try:
                # エージェント固有メトリクス収集
                agent_metrics = await self._collect_agent_metrics(agent_name, agent, timestamp)
                metrics.append(agent_metrics)
                
            except Exception as e:
                self.logger.error(f"メトリクス収集失敗 {agent_name}: {e}")
                
                # エラー時のデフォルトメトリクス
                error_metrics = PerformanceMetrics(
                    timestamp=timestamp,
                    agent_type=agent_name,
                    tasks_completed=0,
                    tasks_failed=1,
                    success_rate=0.0,
                    cpu_usage_percent=0.0,
                    memory_usage_mb=0.0,
                    gpu_usage_percent=0.0,
                    gpu_memory_mb=0.0,
                    api_calls_count=0,
                    api_rate_limit_remaining=0,
                    custom_metrics={"collection_error": 1.0}
                )
                metrics.append(error_metrics)
        
        return metrics
    
    async def _collect_agent_metrics(
        self,
        agent_name: str,
        agent: Any,
        timestamp: datetime
    ) -> PerformanceMetrics:
        """エージェント別メトリクス収集"""
        
        # 基本メトリクス初期化
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            agent_type=agent_name,
            tasks_completed=0,
            tasks_failed=0,
            success_rate=1.0,
            cpu_usage_percent=0.0,
            memory_usage_mb=0.0,
            gpu_usage_percent=0.0,
            gpu_memory_mb=0.0,
            api_calls_count=0,
            api_rate_limit_remaining=5000
        )
        
        # エージェント固有メトリクス収集
        if agent_name == "analyzer":
            if hasattr(agent, 'analysis_history'):
                completed = len([a for a in agent.analysis_history if a.success])
                failed = len([a for a in agent.analysis_history if not a.success])
                metrics.tasks_completed = completed
                metrics.tasks_failed = failed
                metrics.success_rate = completed / max(1, completed + failed)
            
            if hasattr(agent, 'github_api_calls_count'):
                metrics.api_calls_count = getattr(agent, 'github_api_calls_count', 0)
        
        elif agent_name == "executor":
            if hasattr(agent, 'execution_history') and agent.execution_history:
                completed = len([e for e in agent.execution_history if e.success_rate > 0.5])
                failed = len([e for e in agent.execution_history if e.success_rate <= 0.5])
                metrics.tasks_completed = completed
                metrics.tasks_failed = failed
                metrics.success_rate = completed / max(1, completed + failed)
            else:
                # 実行履歴なしの場合は待機状態として扱う
                metrics.tasks_completed = 0
                metrics.tasks_failed = 0
                metrics.success_rate = 1.0  # 待機状態は健全として扱う
            
            # GPU使用量集計
            if hasattr(agent, 'execution_history') and agent.execution_history:
                total_gpu_hours = sum(e.total_gpu_hours_used for e in agent.execution_history)
                metrics.custom_metrics["total_gpu_hours_used"] = total_gpu_hours
            else:
                metrics.custom_metrics["total_gpu_hours_used"] = 0.0
        
        elif agent_name == "planner":
            if hasattr(agent, 'planning_history'):
                completed = len([p for p in agent.planning_history if p.success])
                failed = len([p for p in agent.planning_history if not p.success])
                metrics.tasks_completed = completed
                metrics.tasks_failed = failed
                metrics.success_rate = completed / max(1, completed + failed)
        
        # システムリソース監視（模擬値）
        import random
        metrics.cpu_usage_percent = random.uniform(20, 80)
        metrics.memory_usage_mb = random.uniform(2000, 6000)
        metrics.gpu_usage_percent = random.uniform(10, 90)
        metrics.gpu_memory_mb = random.uniform(1000, 8000)
        
        return metrics
    
    async def _evaluate_system_health(
        self,
        current_metrics: List[PerformanceMetrics]
    ) -> Dict[str, SystemHealth]:
        """システム健全性評価"""
        
        health_status = {}
        
        for metrics in current_metrics:
            agent_health = SystemHealth.HEALTHY
            
            # 成功率チェック
            if metrics.success_rate < 0.5:
                agent_health = SystemHealth.CRITICAL
            elif metrics.success_rate < 0.8:
                agent_health = SystemHealth.DEGRADED
            
            # リソース使用率チェック
            if (metrics.cpu_usage_percent > 90 or 
                metrics.memory_usage_mb > 7000 or
                metrics.gpu_usage_percent > 95):
                if agent_health == SystemHealth.HEALTHY:
                    agent_health = SystemHealth.DEGRADED
                elif agent_health == SystemHealth.DEGRADED:
                    agent_health = SystemHealth.CRITICAL
            
            # API制限チェック
            if metrics.api_rate_limit_remaining < 100:
                if agent_health == SystemHealth.HEALTHY:
                    agent_health = SystemHealth.DEGRADED
            
            health_status[metrics.agent_type] = agent_health
        
        return health_status
    
    async def _process_alerts(
        self,
        current_metrics: List[PerformanceMetrics],
        health_status: Dict[str, SystemHealth]
    ) -> List[SystemAlert]:
        """アラート処理"""
        
        new_alerts = []
        
        for metrics in current_metrics:
            agent_name = metrics.agent_type
            agent_health = health_status.get(agent_name, SystemHealth.HEALTHY)
            
            # 健全性アラート
            if agent_health == SystemHealth.CRITICAL:
                alert = SystemAlert(
                    alert_id=f"health-{agent_name}-{uuid.uuid4().hex[:6]}",
                    timestamp=datetime.utcnow(),
                    severity=AlertSeverity.EMERGENCY,
                    source_agent=agent_name,
                    title=f"🚨 {agent_name}エージェント重大問題",
                    description=f"成功率: {metrics.success_rate:.1%}, CPU: {metrics.cpu_usage_percent:.1f}%",
                    affected_components=[agent_name],
                    metrics_snapshot=metrics.__dict__
                )
                new_alerts.append(alert)
            
            elif agent_health == SystemHealth.DEGRADED:
                alert = SystemAlert(
                    alert_id=f"perf-{agent_name}-{uuid.uuid4().hex[:6]}",
                    timestamp=datetime.utcnow(),
                    severity=AlertSeverity.WARNING,
                    source_agent=agent_name,
                    title=f"⚠️ {agent_name}エージェント性能低下",
                    description=f"成功率: {metrics.success_rate:.1%}, リソース使用量高",
                    affected_components=[agent_name],
                    metrics_snapshot=metrics.__dict__
                )
                new_alerts.append(alert)
            
            # GPU使用量アラート
            if metrics.gpu_usage_percent > 95:
                alert = SystemAlert(
                    alert_id=f"gpu-{agent_name}-{uuid.uuid4().hex[:6]}",
                    timestamp=datetime.utcnow(),
                    severity=AlertSeverity.WARNING,
                    source_agent=agent_name,
                    title=f"🎮 GPU使用量高警告",
                    description=f"GPU使用率: {metrics.gpu_usage_percent:.1f}%",
                    affected_components=[agent_name],
                    metrics_snapshot={"gpu_usage": metrics.gpu_usage_percent}
                )
                new_alerts.append(alert)
        
        # 新しいアラートをアクティブリストに追加
        self.active_alerts.extend(new_alerts)
        
        # アラート通知
        for alert in new_alerts:
            await self._send_alert_notification(alert)
            
            # 重要アラートに対してLLM診断実行
            if alert.severity in [AlertSeverity.EMERGENCY, AlertSeverity.WARNING] and self.llm_enabled:
                try:
                    await self._perform_llm_anomaly_diagnosis(alert, current_metrics)
                except Exception as e:
                    self.logger.warning(f"LLM異常診断失敗: {e}")
        
        return new_alerts
    
    async def _send_alert_notification(self, alert: SystemAlert):
        """アラート通知送信"""
        
        # 重要度に応じたアイコン
        severity_icons = {
            AlertSeverity.EMERGENCY: "🚨",
            AlertSeverity.WARNING: "⚠️", 
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.SUCCESS: "✅"
        }
        
        icon = severity_icons.get(alert.severity, "📊")
        
        # GitHub Issue コメント作成
        await self._post_monitoring_update(
            title=f"{icon} アラート: {alert.title}",
            content=f"""
## {alert.severity.value.upper()} - {alert.title}

**発生時刻**: {alert.timestamp.isoformat()}
**ソースエージェント**: {alert.source_agent}
**アラートID**: `{alert.alert_id}`

### 詳細
{alert.description}

### 影響を受けるコンポーネント
{', '.join(alert.affected_components) if alert.affected_components else 'なし'}

### メトリクススナップショット
```json
{json.dumps(alert.metrics_snapshot, indent=2, default=str) if alert.metrics_snapshot else 'データなし'}
```

---
*自動生成アラート - Monitor Agent {self.agent_id}*
            """
        )
        
        self.logger.warning(f"アラート送信: {alert.title} [{alert.severity.value}]")
    
    async def _perform_llm_anomaly_diagnosis(
        self, 
        alert: SystemAlert, 
        current_metrics: List[PerformanceMetrics]
    ):
        """LLMベース異常診断実行"""
        
        try:
            # 診断コンテキスト準備
            diagnosis_context = self._prepare_diagnosis_context(alert, current_metrics)
            
            # LLMプロンプト取得・実行
            prompt = self.prompt_manager.get_optimized_prompt(
                prompt_type=PromptType.ANOMALY_DIAGNOSIS,
                context_data=diagnosis_context,
                agent_name="monitor"
            )
            
            # Claude API呼び出し
            llm_response = await self.claude_client.complete(
                prompt=prompt,
                max_tokens=3000,
                temperature=0.1  # 診断は保守的に
            )
            
            # LLM診断結果解析
            diagnosis_result = self._parse_llm_diagnosis_response(llm_response)
            
            # 診断結果をアラートに追加
            alert.llm_diagnosis = diagnosis_result
            
            # 診断結果レポート投稿
            await self._post_llm_diagnosis_report(alert, diagnosis_result)
            
            self.logger.info(f"LLM異常診断完了: {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"LLM異常診断エラー: {e}")
    
    def _prepare_diagnosis_context(
        self, 
        alert: SystemAlert, 
        current_metrics: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """LLM診断用コンテキスト準備"""
        
        # 基本異常情報
        context = {
            "detection_timestamp": alert.timestamp.isoformat(),
            "affected_systems": [alert.source_agent],
            "error_messages": [alert.description],
            "urgency_level": "critical" if alert.severity == AlertSeverity.EMERGENCY else "high"
        }
        
        # 競技情報（利用可能な場合）
        if self.competition_tracking:
            active_competition = list(self.competition_tracking.values())[0]
            context["competition_name"] = active_competition.competition_name
            hours_remaining = (active_competition.deadline - datetime.utcnow()).total_seconds() / 3600
            context["hours_until_deadline"] = max(0, hours_remaining)
        else:
            context["competition_name"] = "Unknown Competition"
            context["hours_until_deadline"] = 24  # デフォルト
        
        # パフォーマンス指標
        affected_metrics = [m for m in current_metrics if m.agent_type == alert.source_agent]
        if affected_metrics:
            latest_metrics = affected_metrics[-1]
            context["performance_metrics"] = {
                "success_rate": f"{latest_metrics.success_rate:.1%}",
                "cpu_usage": f"{latest_metrics.cpu_usage_percent:.1f}%",
                "memory_usage": f"{latest_metrics.memory_usage_mb:.1f}MB",
                "gpu_usage": f"{latest_metrics.gpu_usage_percent:.1f}%"
            }
        else:
            context["performance_metrics"] = {"note": "指標データ不足"}
        
        # リソース使用状況
        if current_metrics:
            avg_cpu = sum(m.cpu_usage_percent for m in current_metrics) / len(current_metrics)
            avg_memory = sum(m.memory_usage_mb for m in current_metrics) / len(current_metrics)
            context["resource_usage"] = {
                "average_cpu_usage": f"{avg_cpu:.1f}%",
                "average_memory_usage": f"{avg_memory:.1f}MB",
                "total_monitored_agents": len(current_metrics)
            }
        
        # API状況
        api_metrics = [m for m in current_metrics if m.api_calls_count > 0]
        if api_metrics:
            total_api_calls = sum(m.api_calls_count for m in api_metrics)
            min_rate_limit = min(m.api_rate_limit_remaining for m in api_metrics)
            context["api_status"] = {
                "total_api_calls": total_api_calls,
                "minimum_rate_limit_remaining": min_rate_limit
            }
        
        # 実行中断情報（アラート内容から推定）
        context["execution_interruption_points"] = []
        if "失敗" in alert.description or "エラー" in alert.description:
            context["execution_interruption_points"].append(alert.source_agent)
        
        # 競技への影響評価
        context["interruption_duration"] = "5分"  # 推定値
        context["lost_experiments"] = 1 if alert.severity == AlertSeverity.EMERGENCY else 0
        context["submission_impact"] = "軽微" if alert.severity == AlertSeverity.WARNING else "中程度"
        context["medal_probability_impact"] = 0.05 if alert.severity == AlertSeverity.WARNING else 0.1
        
        # 実行中の重要処理
        context["active_critical_processes"] = [
            proc for proc in self.monitored_agents.keys() 
            if any(m.agent_type == proc and m.tasks_completed > 0 for m in current_metrics)
        ]
        
        return context
    
    def _parse_llm_diagnosis_response(self, llm_response: str) -> Dict[str, Any]:
        """LLM診断応答解析・構造化"""
        
        try:
            import json
            
            # JSON抽出・パース
            diagnosis_result = json.loads(llm_response)
            
            # 必須フィールド検証
            required_fields = ["diagnosis_summary", "severity_assessment", "immediate_actions"]
            for field in required_fields:
                if field not in diagnosis_result:
                    raise ValueError(f"必須フィールド不足: {field}")
            
            self.logger.info(f"LLM診断応答解析完了: {diagnosis_result['diagnosis_summary']['primary_cause']}")
            return diagnosis_result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"LLM診断応答解析失敗: {e}")
            # フォールバック用の最小構造
            return {
                "diagnosis_summary": {
                    "primary_cause": "LLM診断解析失敗",
                    "confidence_level": 0.3,
                    "diagnosis_certainty": "UNCERTAIN"
                },
                "severity_assessment": {
                    "criticality_level": "MEDIUM",
                    "estimated_recovery_time": "不明"
                },
                "immediate_actions": [
                    {
                        "action": "手動診断実行",
                        "priority": 1,
                        "estimated_time": "15分"
                    }
                ]
            }
    
    async def _post_llm_diagnosis_report(
        self, 
        alert: SystemAlert, 
        diagnosis_result: Dict[str, Any]
    ):
        """LLM診断結果レポート投稿"""
        
        diagnosis_summary = diagnosis_result.get("diagnosis_summary", {})
        severity_assessment = diagnosis_result.get("severity_assessment", {})
        immediate_actions = diagnosis_result.get("immediate_actions", [])
        
        await self._post_monitoring_update(
            title=f"🤖 LLM異常診断: {alert.title}",
            content=f"""
## 🔍 LLM異常診断結果
**アラートID**: `{alert.alert_id}`
**診断時刻**: {datetime.utcnow().isoformat()}

### 📋 診断サマリー
- **主原因**: {diagnosis_summary.get('primary_cause', 'N/A')}
- **信頼度**: {diagnosis_summary.get('confidence_level', 0):.2f}
- **診断確実性**: {diagnosis_summary.get('diagnosis_certainty', 'UNCERTAIN')}

### ⚠️ 影響度評価
- **重要度レベル**: {severity_assessment.get('criticality_level', 'MEDIUM')}
- **推定復旧時間**: {severity_assessment.get('estimated_recovery_time', '不明')}
- **メダル獲得影響**: {severity_assessment.get('medal_impact_score', 0):.2f}

### 🚀 推奨即時対応
{chr(10).join([f"**{i+1}. {action['action']}** (優先度: {action['priority']}, 推定時間: {action['estimated_time']})" for i, action in enumerate(immediate_actions[:3])])}

### 📊 根本原因解決策
{chr(10).join([f"- {solution['solution']}" for solution in diagnosis_result.get('root_cause_resolution', [])[:2]])}

### 🔄 代替戦略
{chr(10).join([f"- {strategy['strategy']}" for strategy in diagnosis_result.get('alternative_strategies', [])[:2]])}

---
*LLMベース異常診断 - Monitor Agent {self.agent_id}*
            """
        )
    
    def enable_llm_diagnosis(self, enabled: bool = True):
        """LLM診断の有効/無効切り替え"""
        self.llm_enabled = enabled
        self.logger.info(f"LLM異常診断: {'有効' if enabled else '無効'}")
    
    def get_llm_diagnosis_stats(self) -> Dict[str, Any]:
        """LLM診断統計情報"""
        
        llm_diagnosed_alerts = len([
            alert for alert in self.active_alerts 
            if hasattr(alert, 'llm_diagnosis') and alert.llm_diagnosis
        ])
        
        return {
            "llm_diagnosis_enabled": self.llm_enabled,
            "total_alerts": len(self.active_alerts),
            "llm_diagnosed_alerts": llm_diagnosed_alerts,
            "llm_diagnosis_rate": llm_diagnosed_alerts / max(1, len(self.active_alerts)),
            "prompt_manager_stats": self.prompt_manager.get_prompt_stats()
        }
    
    async def _update_competition_progress(self):
        """競技進捗更新"""
        
        # 実際の実装では、各エージェントから競技進捗を収集
        # ここでは模擬的な進捗更新
        
        for comp_name in ["test-competition-1", "test-competition-2"]:
            if comp_name not in self.competition_tracking:
                # 新しい競技の進捗初期化
                progress = CompetitionProgress(
                    competition_name=comp_name,
                    start_time=datetime.utcnow() - timedelta(hours=24),
                    deadline=datetime.utcnow() + timedelta(days=30),
                    analysis_completion=0.0,
                    techniques_identified=0,
                    techniques_implemented=0,
                    experiments_total=0,
                    experiments_completed=0,
                    experiments_running=0,
                    experiments_failed=0,
                    current_best_score=0.0,
                    target_score=0.9,
                    score_improvement_rate=0.0,
                    gpu_hours_used=0.0,
                    gpu_hours_remaining=50.0,
                    budget_utilization=0.0,
                    estimated_completion_time=datetime.utcnow() + timedelta(days=25),
                    medal_probability=0.3
                )
                self.competition_tracking[comp_name] = progress
            
            # 進捗更新（模擬）
            progress = self.competition_tracking[comp_name]
            progress.analysis_completion = min(1.0, progress.analysis_completion + 0.01)
            progress.current_best_score = min(0.95, progress.current_best_score + 0.001)
            progress.gpu_hours_used = min(50.0, progress.gpu_hours_used + 0.1)
            progress.budget_utilization = progress.gpu_hours_used / 50.0
    
    async def generate_monitoring_report(self) -> MonitoringReport:
        """監視レポート生成"""
        
        report_start = datetime.utcnow() - timedelta(hours=1)
        recent_metrics = [m for m in self.performance_history if m.timestamp > report_start]
        
        # 全体健全性評価
        if recent_metrics:
            avg_success_rate = statistics.mean([m.success_rate for m in recent_metrics])
            if avg_success_rate > 0.9:
                overall_health = SystemHealth.HEALTHY
            elif avg_success_rate > 0.7:
                overall_health = SystemHealth.DEGRADED
            else:
                overall_health = SystemHealth.CRITICAL
        else:
            overall_health = SystemHealth.OFFLINE
        
        # エージェント別健全性
        agent_health = {}
        for agent_name in self.monitored_agents.keys():
            agent_metrics = [m for m in recent_metrics if m.agent_type == agent_name]
            if agent_metrics:
                avg_success = statistics.mean([m.success_rate for m in agent_metrics])
                agent_health[agent_name] = (
                    SystemHealth.HEALTHY if avg_success > 0.8 else
                    SystemHealth.DEGRADED if avg_success > 0.5 else
                    SystemHealth.CRITICAL
                )
            else:
                agent_health[agent_name] = SystemHealth.OFFLINE
        
        # パフォーマンスサマリー
        perf_summary = {}
        for agent_name in self.monitored_agents.keys():
            agent_metrics = [m for m in recent_metrics if m.agent_type == agent_name]
            if agent_metrics:
                perf_summary[agent_name] = agent_metrics[-1]  # 最新メトリクス
        
        # アラート統計
        alerts_summary = {}
        for severity in AlertSeverity:
            alerts_summary[severity.value] = len([
                a for a in self.active_alerts 
                if a.severity == severity and not a.resolved
            ])
        
        # 推奨事項生成
        recommendations = []
        if overall_health == SystemHealth.CRITICAL:
            recommendations.append("🚨 緊急対応が必要です。失敗率の高いエージェントを確認してください。")
        if any(health == SystemHealth.DEGRADED for health in agent_health.values()):
            recommendations.append("⚠️ 一部エージェントで性能低下が発生しています。リソース配分を見直してください。")
        
        recent_gpu_usage = [m.gpu_usage_percent for m in recent_metrics if m.gpu_usage_percent > 0]
        if recent_gpu_usage and statistics.mean(recent_gpu_usage) > 80:
            recommendations.append("🎮 GPU使用率が高い状態が続いています。ワークロード分散を検討してください。")
        
        if not recommendations:
            recommendations.append("✅ システムは正常に動作しています。")
        
        # レポート作成
        report = MonitoringReport(
            report_id=f"report-{uuid.uuid4().hex[:8]}",
            generated_at=datetime.utcnow(),
            monitoring_period_hours=1.0,
            overall_health=overall_health,
            agent_health_status=agent_health,
            performance_summary=perf_summary,
            competition_progress=list(self.competition_tracking.values()),
            alerts_summary=alerts_summary,
            active_alerts=[a for a in self.active_alerts if not a.resolved],
            recommendations=recommendations,
            trend_analysis={
                "monitoring_cycles": self.monitoring_cycles_completed,
                "total_alerts": self.total_alerts_generated,
                "avg_success_rate": statistics.mean([m.success_rate for m in recent_metrics]) if recent_metrics else 0.0
            }
        )
        
        return report
    
    async def _post_monitoring_report(self, report: MonitoringReport):
        """監視レポート投稿"""
        
        health_icons = {
            SystemHealth.HEALTHY: "✅",
            SystemHealth.DEGRADED: "⚠️",
            SystemHealth.CRITICAL: "🚨",
            SystemHealth.OFFLINE: "💀"
        }
        
        overall_icon = health_icons.get(report.overall_health, "❓")
        
        content = f"""
## {overall_icon} システム監視レポート - {report.generated_at.strftime('%Y-%m-%d %H:%M')}

### 📊 システム全体状況
- **全体健全性**: {overall_icon} {report.overall_health.value}
- **監視サイクル**: {report.trend_analysis.get('monitoring_cycles', 0)}
- **平均成功率**: {report.trend_analysis.get('avg_success_rate', 0):.1%}

### 🔧 エージェント別状況
{chr(10).join([f"- **{name}**: {health_icons.get(health, '❓')} {health.value}" for name, health in report.agent_health_status.items()])}

### 🏆 競技進捗
{chr(10).join([f"- **{prog.competition_name}**: 分析{prog.analysis_completion:.1%}, 実験{prog.experiments_completed}/{prog.experiments_total}, ベストスコア{prog.current_best_score:.4f}" for prog in report.competition_progress]) if report.competition_progress else "- 進行中の競技なし"}

### 🚨 アクティブアラート
{chr(10).join([f"- **{alert.severity.value}**: {alert.title}" for alert in report.active_alerts[:5]]) if report.active_alerts else "- アラートなし"}

### 💡 推奨事項
{chr(10).join([f"- {rec}" for rec in report.recommendations])}

### 📈 リソース使用状況
{chr(10).join([f"- **{name}**: GPU {metrics.gpu_usage_percent:.1f}%, メモリ {metrics.memory_usage_mb:.0f}MB" for name, metrics in report.performance_summary.items()]) if report.performance_summary else "- データなし"}

---
*レポートID: {report.report_id} | Monitor Agent {self.agent_id}*
        """
        
        await self._post_monitoring_update(
            title=f"{overall_icon} 定期監視レポート",
            content=content
        )
    
    async def _create_monitoring_issue(self, title: str, description: str, labels: List[str] = None) -> int:
        """監視用Issue作成"""
        
        try:
            issue_data = await self.atomic_operations.create_issue(
                title=title,
                description=description,
                labels=labels or ["monitor"]
            )
            return issue_data["number"]
            
        except Exception as e:
            self.logger.error(f"監視Issue作成失敗: {e}")
            return -1
    
    async def _post_monitoring_update(self, title: str, content: str):
        """監視更新投稿"""
        
        try:
            # 最新の監視Issueにコメント投稿
            # 実際の実装では、監視用のIssue番号を管理する
            await self.atomic_operations.create_comment(
                issue_number=1,  # 仮のIssue番号
                comment_body=f"## {title}\n\n{content}"
            )
            
        except Exception as e:
            self.logger.error(f"監視更新投稿失敗: {e}")
    
    async def acknowledge_alert(self, alert_id: str, notes: str = "") -> bool:
        """アラート確認"""
        
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.resolution_notes = notes
                
                await self._post_monitoring_update(
                    title="✅ アラート確認",
                    content=f"アラート `{alert_id}` が確認されました。\n\n**確認メモ**: {notes}"
                )
                
                self.logger.info(f"アラート確認: {alert_id}")
                return True
        
        return False
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str) -> bool:
        """アラート解決"""
        
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_notes = resolution_notes
                
                await self._post_monitoring_update(
                    title="🔧 アラート解決",
                    content=f"アラート `{alert_id}` が解決されました。\n\n**解決内容**: {resolution_notes}"
                )
                
                self.logger.info(f"アラート解決: {alert_id}")
                return True
        
        return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得（サービス監視統合版）"""
        
        current_time = datetime.utcnow()
        uptime_hours = (current_time - self.start_time).total_seconds() / 3600
        
        # サービス監視統計追加
        service_health_stats = {}
        if self.service_health_history:
            recent_checks = self.service_health_history[-10:]  # 最新10回
            service_health_stats = {
                "service_active_rate": sum(1 for c in recent_checks if c.service_active) / len(recent_checks),
                "auto_repairs_applied": sum(1 for c in recent_checks if c.auto_repair_applied),
                "avg_memory_usage": sum(c.memory_usage_percent for c in recent_checks) / len(recent_checks),
                "avg_disk_usage": sum(c.disk_usage_percent for c in recent_checks) / len(recent_checks)
            }
        
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "uptime_hours": uptime_hours,
            "monitoring_active": self.is_monitoring_active,
            "monitoring_cycles_completed": self.monitoring_cycles_completed,
            "total_alerts_generated": self.total_alerts_generated,
            "active_alerts_count": len([a for a in self.active_alerts if not a.resolved]),
            "monitored_agents": list(self.monitored_agents.keys()),
            "competition_tracking_count": len(self.competition_tracking),
            "performance_history_count": len(self.performance_history),
            "service_monitoring_enabled": self.service_monitoring_enabled,
            "auto_repair_enabled": self.auto_repair_enabled,
            "total_auto_repairs": self.total_auto_repairs,
            "service_health_checks_count": len(self.service_health_history),
            "service_health_stats": service_health_stats,
            "main_monitoring_issue": self.main_monitoring_issue
        }
    
    async def _perform_service_health_check(self) -> ServiceHealthCheck:
        """サービス健全性チェック実行（動的監視）"""
        
        timestamp = datetime.utcnow()
        
        try:
            # 1. systemctl status チェック
            service_active = await self._check_service_status()
            service_status = "active" if service_active else "inactive"
            
            # 2. プロセス生存チェック
            process_alive = await self._check_process_alive()
            if not process_alive and service_active:
                service_status = "active-but-no-process"
                service_active = False
            
            # 3. ログエラーパターン検出
            error_patterns = await self._detect_log_errors()
            
            # 4. システムリソース監視
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage('/').percent
            system_load = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
            
            health_check = ServiceHealthCheck(
                timestamp=timestamp,
                service_active=service_active,
                service_status=service_status,
                error_patterns_detected=error_patterns,
                memory_usage_percent=memory_usage,
                disk_usage_percent=disk_usage,
                system_load=system_load
            )
            
            # エラー統計更新
            if error_patterns or not service_active:
                self.error_count += 1
            else:
                self.error_count = max(0, self.error_count - 1)  # 成功時は減らす
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"サービス健全性チェック失敗: {e}")
            
            # エラー時のデフォルト値
            return ServiceHealthCheck(
                timestamp=timestamp,
                service_active=False,
                service_status="check-error",
                error_patterns_detected=["health_check_failed"],
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                system_load=0.0
            )
    
    async def _check_service_status(self) -> bool:
        """systemctl サービス状態チェック"""
        
        try:
            # systemctl is-active コマンドで状態確認
            result = await asyncio.create_subprocess_exec(
                'systemctl', 'is-active', self.service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            status = stdout.decode().strip()
            is_active = (status == "active")
            
            if not is_active:
                self.logger.warning(f"サービス非アクティブ: {self.service_name} - {status}")
            
            return is_active
            
        except Exception as e:
            self.logger.error(f"サービス状態チェック失敗: {e}")
            return False
    
    async def _check_process_alive(self) -> bool:
        """メインプロセス生存確認"""
        
        try:
            # プロセス一覧から main.py を検索
            for process in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = process.info['cmdline']
                    if cmdline and any('main.py' in cmd for cmd in cmdline):
                        # プロセスが応答可能かチェック
                        if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.logger.warning("main.py プロセスが見つかりません")
            return False
            
        except Exception as e:
            self.logger.error(f"プロセス生存確認失敗: {e}")
            return False
    
    async def _detect_log_errors(self) -> List[str]:
        """ログエラーパターン検出（動的ログ解析）"""
        
        error_patterns_found = []
        
        try:
            # journalctl で最新のログを取得
            result = await asyncio.create_subprocess_exec(
                'journalctl', '-u', self.service_name, '--since', '2 minutes ago', 
                '--no-pager', '-n', '50',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            recent_logs = stdout.decode()
            
            # エラーパターン定義
            error_patterns = {
                "python_exception": ["Exception", "Traceback", "Error:"],
                "import_error": ["ModuleNotFoundError", "ImportError"],
                "connection_error": ["ConnectionError", "TimeoutError", "API rate limit"],
                "memory_error": ["MemoryError", "out of memory"],
                "datetime_error": ["can't subtract offset-naive and offset-aware datetimes"],
                "attribute_error": ["object has no attribute"],
                "system_error": ["failed", "FAILED", "CRITICAL"]
            }
            
            # パターンマッチング
            for category, patterns in error_patterns.items():
                for pattern in patterns:
                    if pattern in recent_logs:
                        error_patterns_found.append(f"{category}:{pattern}")
                        break
            
            if error_patterns_found:
                self.logger.warning(f"エラーパターン検出: {error_patterns_found}")
            
            return error_patterns_found
            
        except Exception as e:
            self.logger.error(f"ログエラー検出失敗: {e}")
            return ["log_analysis_failed"]
    
    async def _perform_auto_repair(self, health_check: ServiceHealthCheck):
        """自動修復実行（インテリジェント修復戦略）"""
        
        if self.error_count < self.alert_thresholds["error_threshold_count"]:
            self.logger.info(f"エラー閾値未達成: {self.error_count}/{self.alert_thresholds['error_threshold_count']}")
            return
        
        repair_actions = []
        
        try:
            self.logger.info(f"自動修復開始: エラー数{self.error_count}, パターン{health_check.error_patterns_detected}")
            
            # 1. エラーパターン別修復
            for error_pattern in health_check.error_patterns_detected:
                if "import_error" in error_pattern or "ModuleNotFoundError" in error_pattern:
                    await self._repair_dependencies()
                    repair_actions.append("dependencies_sync")
                
                elif "memory_error" in error_pattern or health_check.memory_usage_percent > 90:
                    await self._repair_memory_usage()
                    repair_actions.append("memory_optimization")
                
                elif "datetime_error" in error_pattern:
                    self.logger.info("datetime修正は既にコードに適用済み")
                    repair_actions.append("datetime_fix_applied")
                
                elif "connection_error" in error_pattern:
                    await self._repair_network_issues()
                    repair_actions.append("network_retry")
            
            # 2. サービス状態修復
            if not health_check.service_active:
                await self._repair_service_restart()
                repair_actions.append("service_restart")
            
            # 3. ディスク容量修復
            if health_check.disk_usage_percent > 80:
                await self._repair_disk_cleanup()
                repair_actions.append("disk_cleanup")
            
            # 修復完了記録
            health_check.auto_repair_applied = True
            health_check.repair_actions = repair_actions
            self.total_auto_repairs += 1
            self.error_count = 0  # 修復後はエラーカウントリセット
            
            # 修復レポート
            await self._post_monitoring_update(
                title="🔧 自動修復実行",
                content=f"""
## 自動修復完了

**修復時刻**: {datetime.utcnow().isoformat()}
**エラー数**: {self.error_count}回連続
**検出パターン**: {', '.join(health_check.error_patterns_detected)}

### 実行された修復アクション
{chr(10).join([f"- ✅ {action}" for action in repair_actions])}

### システム状態改善
- **サービス状態**: {health_check.service_status}
- **メモリ使用量**: {health_check.memory_usage_percent:.1f}%
- **ディスク使用量**: {health_check.disk_usage_percent:.1f}%

**総自動修復回数**: {self.total_auto_repairs}回
                """
            )
            
            self.logger.info(f"自動修復完了: {len(repair_actions)}個のアクション実行")
            
        except Exception as e:
            self.logger.error(f"自動修復失敗: {e}")
            repair_actions.append("repair_failed")
    
    async def _repair_dependencies(self):
        """依存関係修復"""
        
        try:
            self.logger.info("依存関係修復実行中...")
            
            # プロジェクトディレクトリに移動してuv sync実行
            project_dir = Path.cwd()
            
            result = await asyncio.create_subprocess_exec(
                'uv', 'sync', '--upgrade',
                cwd=project_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                self.logger.info("依存関係修復成功")
            else:
                self.logger.error(f"依存関係修復失敗: {stderr.decode()}")
            
        except Exception as e:
            self.logger.error(f"依存関係修復エラー: {e}")
    
    async def _repair_memory_usage(self):
        """メモリ使用量最適化"""
        
        try:
            self.logger.info("メモリ最適化実行中...")
            
            # システムキャッシュクリア
            await asyncio.create_subprocess_exec(
                'sync',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # ページキャッシュクリア（要root権限）
            try:
                result = await asyncio.create_subprocess_exec(
                    'sudo', 'sysctl', 'vm.drop_caches=1',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await result.communicate()
                
                if result.returncode == 0:
                    self.logger.info("システムキャッシュクリア成功")
                
            except Exception:
                self.logger.warning("システムキャッシュクリアはroot権限が必要")
            
            # Pythonガベージコレクション強制実行
            import gc
            collected = gc.collect()
            self.logger.info(f"Python GC実行: {collected}オブジェクト解放")
            
        except Exception as e:
            self.logger.error(f"メモリ最適化エラー: {e}")
    
    async def _repair_network_issues(self):
        """ネットワーク問題修復"""
        
        try:
            self.logger.info("ネットワーク問題修復中...")
            
            # GitHub API接続テスト
            try:
                repo = self.github_wrapper.github.get_repo(self.github_wrapper.repo_name)
                repo.get_issues(state="open", per_page=1)
                self.logger.info("GitHub API接続正常")
                
            except Exception as e:
                self.logger.warning(f"GitHub API接続問題: {e}")
                
                # APIレート制限チェック
                try:
                    rate_limit = self.github_wrapper.github.get_rate_limit()
                    remaining = rate_limit.core.remaining
                    
                    if remaining < 100:
                        self.logger.warning(f"GitHub APIレート制限近接: {remaining}残り")
                        # レート制限回復まで待機
                        reset_time = rate_limit.core.reset
                        wait_minutes = (reset_time - datetime.utcnow()).total_seconds() / 60
                        
                        if wait_minutes > 0 and wait_minutes < 60:
                            self.logger.info(f"APIレート制限回復まで{wait_minutes:.1f}分待機")
                            await asyncio.sleep(min(300, wait_minutes * 60))  # 最大5分待機
                
                except Exception:
                    self.logger.error("APIレート制限チェック失敗")
            
        except Exception as e:
            self.logger.error(f"ネットワーク修復エラー: {e}")
    
    async def _repair_service_restart(self):
        """サービス再起動実行"""
        
        try:
            current_time = datetime.utcnow().timestamp()
            
            # 再起動間隔制限（10分以内は再起動しない）
            if current_time - self.last_restart_time < 600:
                self.logger.warning("再起動間隔が短すぎるためスキップ")
                return
            
            self.logger.info(f"サービス再起動実行: {self.service_name}")
            
            # サービス停止
            result = await asyncio.create_subprocess_exec(
                'sudo', 'systemctl', 'stop', self.service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            
            # 5秒待機
            await asyncio.sleep(5)
            
            # サービス開始
            result = await asyncio.create_subprocess_exec(
                'sudo', 'systemctl', 'start', self.service_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            
            self.last_restart_time = current_time
            
            # 起動確認（10秒待機）
            await asyncio.sleep(10)
            
            is_active = await self._check_service_status()
            if is_active:
                self.logger.info("✅ サービス再起動成功")
            else:
                self.logger.error("❌ サービス再起動失敗")
            
        except Exception as e:
            self.logger.error(f"サービス再起動エラー: {e}")
    
    async def _repair_disk_cleanup(self):
        """ディスク容量クリーンアップ"""
        
        try:
            self.logger.info("ディスククリーンアップ実行中...")
            
            # ログファイルクリーンアップ
            log_dir = Path.cwd() / "logs"
            if log_dir.exists():
                # 7日以上古いログファイル削除
                old_logs = [f for f in log_dir.glob("*.log") 
                           if (datetime.utcnow() - datetime.fromtimestamp(f.stat().st_mtime)).days > 7]
                
                for log_file in old_logs:
                    log_file.unlink()
                    self.logger.info(f"古いログファイル削除: {log_file.name}")
            
            # 一時ファイルクリーンアップ
            temp_patterns = [
                "/tmp/*kaggle*",
                "competitions/*/cache/*",
                "**/__pycache__",
                "**/*.pyc"
            ]
            
            for pattern in temp_patterns:
                try:
                    temp_files = list(Path.cwd().glob(pattern))
                    for temp_file in temp_files[:10]:  # 安全のため最大10個まで
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                    
                    if temp_files:
                        self.logger.info(f"一時ファイル削除: {len(temp_files)}個")
                        
                except Exception as e:
                    self.logger.warning(f"一時ファイル削除失敗 {pattern}: {e}")
            
        except Exception as e:
            self.logger.error(f"ディスククリーンアップエラー: {e}")
    
    async def _check_system_resources(self, health_check: ServiceHealthCheck):
        """システムリソース監視・警告"""
        
        # メモリ使用量警告
        if health_check.memory_usage_percent > self.alert_thresholds["memory_usage_threshold_percent"]:
            alert = SystemAlert(
                alert_id=f"memory-{uuid.uuid4().hex[:6]}",
                timestamp=datetime.utcnow(),
                severity=AlertSeverity.WARNING,
                source_agent="service_monitor",
                title=f"🧠 メモリ使用量高警告",
                description=f"メモリ使用率: {health_check.memory_usage_percent:.1f}%",
                affected_components=["system-memory"],
                metrics_snapshot={"memory_usage_percent": health_check.memory_usage_percent}
            )
            self.active_alerts.append(alert)
            await self._send_alert_notification(alert)
        
        # ディスク使用量警告
        if health_check.disk_usage_percent > self.alert_thresholds["disk_usage_threshold_percent"]:
            alert = SystemAlert(
                alert_id=f"disk-{uuid.uuid4().hex[:6]}",
                timestamp=datetime.utcnow(),
                severity=AlertSeverity.WARNING,
                source_agent="service_monitor",
                title=f"💾 ディスク使用量高警告",
                description=f"ディスク使用率: {health_check.disk_usage_percent:.1f}%",
                affected_components=["system-disk"],
                metrics_snapshot={"disk_usage_percent": health_check.disk_usage_percent}
            )
            self.active_alerts.append(alert)
            await self._send_alert_notification(alert)
    
    async def _post_service_status_report(self, health_check: ServiceHealthCheck):
        """サービス状態定期レポート"""
        
        # システム稼働統計
        recent_checks = self.service_health_history[-30:] if len(self.service_health_history) >= 30 else self.service_health_history
        
        if recent_checks:
            uptime_rate = sum(1 for c in recent_checks if c.service_active) / len(recent_checks) * 100
            avg_memory = sum(c.memory_usage_percent for c in recent_checks) / len(recent_checks)
            avg_disk = sum(c.disk_usage_percent for c in recent_checks) / len(recent_checks)
            repair_count = sum(1 for c in recent_checks if c.auto_repair_applied)
        else:
            uptime_rate = avg_memory = avg_disk = repair_count = 0
        
        status_icon = "✅" if health_check.service_active else "❌"
        
        await self._post_monitoring_update(
            title=f"{status_icon} サービス状態レポート",
            content=f"""
## サービス監視状況 - {health_check.timestamp.strftime('%Y-%m-%d %H:%M')}

### 📊 現在の状態
- **サービス状態**: {status_icon} {health_check.service_status}
- **メモリ使用量**: {health_check.memory_usage_percent:.1f}%
- **ディスク使用量**: {health_check.disk_usage_percent:.1f}%
- **システム負荷**: {health_check.system_load:.2f}

### 📈 30分間統計
- **稼働率**: {uptime_rate:.1f}%
- **平均メモリ使用量**: {avg_memory:.1f}%
- **平均ディスク使用量**: {avg_disk:.1f}%
- **自動修復実行回数**: {repair_count}回

### 🔍 検出されたエラーパターン
{chr(10).join([f"- ⚠️ {pattern}" for pattern in health_check.error_patterns_detected]) if health_check.error_patterns_detected else "- ✅ エラーなし"}

### 🔧 自動修復状況
- **自動修復有効**: {'✅' if self.auto_repair_enabled else '❌'}
- **総修復回数**: {self.total_auto_repairs}回
- **エラー連続回数**: {self.error_count}回

---
*自動生成レポート - 統合Monitor Agent {self.agent_id}*
            """
        )