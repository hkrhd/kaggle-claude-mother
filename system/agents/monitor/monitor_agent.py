"""
高度モニタリングエージェント

実行中のKaggle競技における全エージェント活動・実験進捗・リソース使用状況・
パフォーマンス指標をリアルタイム監視し、最適化提案を行うメインエージェント。
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics

# GitHub Issue安全システム
from ...issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ...issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations

# 他エージェントの状態監視
from ..analyzer.analyzer_agent import AnalyzerAgent
from ..executor.executor_agent import ExecutorAgent


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
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        
        # エージェント情報
        self.agent_id = f"monitor-{uuid.uuid4().hex[:8]}"
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
        
        # 監視対象エージェント（実際には参照で取得）
        self.monitored_agents: Dict[str, Any] = {}
        
        # 監視設定
        self.monitoring_level = MonitoringLevel.STANDARD
        self.monitoring_interval_seconds = 30
        self.alert_thresholds = {
            "success_rate_threshold": 0.8,
            "gpu_usage_threshold": 0.9,
            "memory_usage_threshold_mb": 8000,
            "api_rate_remaining_threshold": 100
        }
        
        # 監視データ
        self.performance_history: List[PerformanceMetrics] = []
        self.active_alerts: List[SystemAlert] = []
        self.competition_tracking: Dict[str, CompetitionProgress] = {}
        
        # 統計
        self.monitoring_cycles_completed = 0
        self.total_alerts_generated = 0
        
        # 監視タスク
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring_active = False
    
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
            if hasattr(agent, 'execution_history'):
                completed = len([e for e in agent.execution_history if e.success_rate > 0.5])
                failed = len([e for e in agent.execution_history if e.success_rate <= 0.5])
                metrics.tasks_completed = completed
                metrics.tasks_failed = failed
                metrics.success_rate = completed / max(1, completed + failed)
                
                # GPU使用量集計
                total_gpu_hours = sum(e.total_gpu_hours_used for e in agent.execution_history)
                metrics.custom_metrics["total_gpu_hours_used"] = total_gpu_hours
        
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
        """システム状態取得"""
        
        current_time = datetime.utcnow()
        uptime_hours = (current_time - self.start_time).total_seconds() / 3600
        
        return {
            "agent_id": self.agent_id,
            "uptime_hours": uptime_hours,
            "monitoring_active": self.is_monitoring_active,
            "monitoring_cycles_completed": self.monitoring_cycles_completed,
            "total_alerts_generated": self.total_alerts_generated,
            "active_alerts_count": len([a for a in self.active_alerts if not a.resolved]),
            "monitored_agents": list(self.monitored_agents.keys()),
            "competition_tracking_count": len(self.competition_tracking),
            "performance_history_count": len(self.performance_history)
        }