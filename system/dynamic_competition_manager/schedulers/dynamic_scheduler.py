"""
動的スケジューラー・自動実行システム

週2回（火・金 7:00）の動的最適化実行・エージェント起動通知。
crontab連携による完全自動化と異常時自動復旧機能。
"""

import asyncio
import os
import subprocess
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import signal

from croniter import croniter
from apscheduler.schedulers.asyncio import AsyncIOScheduler  
from apscheduler.triggers.cron import CronTrigger
from apscheduler.executors.asyncio import AsyncIOExecutor

from ..medal_probability_calculators.medal_probability_calculator import (
    CompetitionData, CompetitionType, PrizeType
)
from ..portfolio_optimizers.competition_portfolio_optimizer import (
    CompetitionPortfolioOptimizer, PortfolioStrategy
)
from ..decision_engines.withdrawal_decision_maker import (
    WithdrawalDecisionMaker, CompetitionStatus, CompetitionPhase
)


class ScheduleType(Enum):
    """スケジュール種別"""
    DYNAMIC_OPTIMIZATION = "dynamic_optimization"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    WITHDRAWAL_CHECK = "withdrawal_check"
    HEALTH_CHECK = "health_check"
    DATA_COLLECTION = "data_collection"


class ExecutionStatus(Enum):
    """実行状態"""
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ScheduledTask:
    """スケジュールタスク"""
    task_id: str
    task_type: ScheduleType
    cron_expression: str
    description: str
    enabled: bool = True
    last_execution: Optional[datetime] = None
    last_status: ExecutionStatus = ExecutionStatus.SCHEDULED
    execution_count: int = 0
    failure_count: int = 0
    average_duration: float = 0.0
    next_run: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """実行結果"""
    task_id: str
    status: ExecutionStatus
    start_time: datetime
    end_time: datetime
    duration: float
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class DynamicScheduler:
    """動的スケジューラー・自動実行システム"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler(
            executors={'default': AsyncIOExecutor()},
            timezone='UTC'
        )
        
        self.portfolio_optimizer = CompetitionPortfolioOptimizer()
        self.withdrawal_maker = WithdrawalDecisionMaker()
        
        self.logger = logging.getLogger(__name__)
        
        # スケジュールタスク定義
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # システム状態
        self.is_running = False
        self.current_portfolio: List[CompetitionData] = []
        self.active_competitions: Dict[str, CompetitionStatus] = {}
        
        # 設定
        self.max_retry_count = 3
        self.execution_timeout = 3600  # 1時間
        self.health_check_interval = 300  # 5分
    
    def setup_default_schedules(self):
        """デフォルトスケジュール設定"""
        
        # 週2回（火・金 7:00）動的最適化
        self.add_scheduled_task(ScheduledTask(
            task_id="dynamic_optimization_tue_fri",
            task_type=ScheduleType.DYNAMIC_OPTIMIZATION,
            cron_expression="0 7 * * TUE,FRI",  # 毎週火・金 7:00 UTC
            description="週2回の動的コンペポートフォリオ最適化",
            metadata={
                "strategy": PortfolioStrategy.BALANCED.value,
                "max_competitions": 3,
                "notification_enabled": True
            }
        ))
        
        # 日次撤退チェック
        self.add_scheduled_task(ScheduledTask(
            task_id="daily_withdrawal_check",
            task_type=ScheduleType.WITHDRAWAL_CHECK,
            cron_expression="0 8 * * *",  # 毎日 8:00 UTC
            description="日次撤退判断チェック",
            metadata={
                "urgency_threshold": 0.7,
                "auto_withdrawal": False  # 人間確認必要
            }
        ))
        
        # システムヘルスチェック
        self.add_scheduled_task(ScheduledTask(
            task_id="system_health_check",
            task_type=ScheduleType.HEALTH_CHECK,
            cron_expression="*/15 * * * *",  # 15分間隔
            description="システム稼働状況チェック",
            metadata={
                "check_agents": True,
                "check_resources": True,
                "alert_threshold": 0.8
            }
        ))
        
        # データ収集（4時間間隔）
        self.add_scheduled_task(ScheduledTask(
            task_id="competition_data_collection",
            task_type=ScheduleType.DATA_COLLECTION,
            cron_expression="0 */4 * * *",  # 4時間間隔
            description="コンペティションデータ収集・更新",
            metadata={
                "data_sources": ["kaggle_api", "discussion_scraping"],
                "update_probabilities": True
            }
        ))
        
        self.logger.info("デフォルトスケジュール設定完了")
    
    def add_scheduled_task(self, task: ScheduledTask):
        """スケジュールタスク追加"""
        
        self.scheduled_tasks[task.task_id] = task
        
        # 次回実行時刻計算
        cron = croniter(task.cron_expression, datetime.utcnow())
        task.next_run = cron.get_next(datetime)
        
        # スケジューラーに登録
        if task.enabled:
            self.scheduler.add_job(
                func=self.execute_scheduled_task,
                args=[task.task_id],
                trigger=CronTrigger.from_crontab(task.cron_expression),
                id=task.task_id,
                replace_existing=True,
                max_instances=1  # 重複実行防止
            )
        
        self.logger.info(f"スケジュールタスク追加: {task.task_id} ({task.cron_expression})")
    
    async def start_scheduler(self):
        """スケジューラー開始"""
        
        if self.is_running:
            self.logger.warning("スケジューラーは既に実行中です")
            return
        
        try:
            # デフォルトスケジュール設定
            self.setup_default_schedules()
            
            # スケジューラー開始
            self.scheduler.start()
            self.is_running = True
            
            # シグナルハンドラー設定
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            self.logger.info("動的スケジューラー開始")
            
            # 初回実行（即座）
            await self.execute_scheduled_task("dynamic_optimization_tue_fri")
            
            # 実行ループ
            await self.run_scheduler_loop()
            
        except Exception as e:
            self.logger.error(f"スケジューラー開始失敗: {e}")
            raise
    
    async def stop_scheduler(self):
        """スケジューラー停止"""
        
        if not self.is_running:
            return
        
        try:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            
            self.logger.info("動的スケジューラー停止")
            
        except Exception as e:
            self.logger.error(f"スケジューラー停止失敗: {e}")
    
    def signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        
        self.logger.info(f"シグナル受信: {signum}")
        asyncio.create_task(self.stop_scheduler())
    
    async def run_scheduler_loop(self):
        """スケジューラー実行ループ"""
        
        while self.is_running:
            try:
                # 実行待機
                await asyncio.sleep(60)  # 1分間隔でチェック
                
                # 手動実行チェック
                await self.check_manual_triggers()
                
                # 実行状況監視
                await self.monitor_executions()
                
            except Exception as e:
                self.logger.error(f"スケジューラーループエラー: {e}")
                await asyncio.sleep(5)  # エラー時は短時間待機
    
    async def execute_scheduled_task(self, task_id: str) -> ExecutionResult:
        """スケジュールタスク実行"""
        
        if task_id not in self.scheduled_tasks:
            raise ValueError(f"未知のタスクID: {task_id}")
        
        task = self.scheduled_tasks[task_id]
        start_time = datetime.utcnow()
        
        self.logger.info(f"スケジュールタスク実行開始: {task_id}")
        
        try:
            # 実行タイムアウト設定
            result_data = await asyncio.wait_for(
                self.dispatch_task_execution(task),
                timeout=self.execution_timeout
            )
            
            # 成功処理
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            task.last_execution = end_time
            task.last_status = ExecutionStatus.COMPLETED
            task.execution_count += 1
            task.average_duration = (task.average_duration * (task.execution_count - 1) + duration) / task.execution_count
            
            result = ExecutionResult(
                task_id=task_id,
                status=ExecutionStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                result_data=result_data
            )
            
            self.execution_history.append(result)
            self.logger.info(f"スケジュールタスク実行完了: {task_id} ({duration:.1f}秒)")
            
            return result
            
        except asyncio.TimeoutError:
            # タイムアウト処理
            task.last_status = ExecutionStatus.FAILED
            task.failure_count += 1
            
            result = ExecutionResult(
                task_id=task_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.utcnow(),
                duration=(datetime.utcnow() - start_time).total_seconds(),
                error_message="実行タイムアウト"
            )
            
            self.execution_history.append(result)
            self.logger.error(f"スケジュールタスクタイムアウト: {task_id}")
            
            return result
            
        except Exception as e:
            # エラー処理
            task.last_status = ExecutionStatus.FAILED
            task.failure_count += 1
            
            result = ExecutionResult(
                task_id=task_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.utcnow(),
                duration=(datetime.utcnow() - start_time).total_seconds(),
                error_message=str(e)
            )
            
            self.execution_history.append(result)
            self.logger.error(f"スケジュールタスク実行失敗: {task_id} - {e}")
            
            # リトライ判定
            if task.failure_count <= self.max_retry_count:
                await self.schedule_retry(task_id, result.retry_count + 1)
            
            return result
    
    async def dispatch_task_execution(self, task: ScheduledTask) -> Dict[str, Any]:
        """タスク種別による実行ディスパッチ"""
        
        if task.task_type == ScheduleType.DYNAMIC_OPTIMIZATION:
            return await self.execute_dynamic_optimization(task)
        elif task.task_type == ScheduleType.WITHDRAWAL_CHECK:
            return await self.execute_withdrawal_check(task)
        elif task.task_type == ScheduleType.HEALTH_CHECK:
            return await self.execute_health_check(task)
        elif task.task_type == ScheduleType.DATA_COLLECTION:
            return await self.execute_data_collection(task)
        else:
            raise ValueError(f"未対応のタスク種別: {task.task_type}")
    
    async def execute_dynamic_optimization(self, task: ScheduledTask) -> Dict[str, Any]:
        """動的最適化実行"""
        
        self.logger.info("動的コンペポートフォリオ最適化実行")
        
        # 利用可能なコンペ取得（モック実装）
        available_competitions = await self.collect_available_competitions()
        
        # ポートフォリオ最適化
        strategy = PortfolioStrategy(task.metadata.get("strategy", "balanced"))
        optimization_result = await self.portfolio_optimizer.optimize_portfolio(
            available_competitions=available_competitions,
            strategy=strategy,
            current_portfolio=self.current_portfolio
        )
        
        # 結果の比較・変更判定
        changes_needed = self.analyze_portfolio_changes(optimization_result)
        
        # エージェント起動通知
        notifications_sent = []
        if changes_needed:
            notifications_sent = await self.send_agent_notifications(optimization_result)
            
            # 現在のポートフォリオ更新
            self.current_portfolio = [
                item.competition_data for item in optimization_result.selected_competitions
            ]
        
        return {
            "optimization_completed": True,
            "portfolio_size": len(optimization_result.selected_competitions),
            "expected_medal_count": optimization_result.expected_medal_count,
            "changes_made": len(changes_needed),
            "notifications_sent": len(notifications_sent),
            "strategy_used": strategy.value,
            "execution_timestamp": datetime.utcnow().isoformat()
        }
    
    async def execute_withdrawal_check(self, task: ScheduledTask) -> Dict[str, Any]:
        """撤退チェック実行"""
        
        self.logger.info("日次撤退判断チェック実行")
        
        withdrawal_analyses = []
        withdrawal_recommendations = []
        
        for comp_id, status in self.active_competitions.items():
            # 撤退分析実行
            analysis = await self.withdrawal_maker.analyze_withdrawal_decision(
                competition_status=status,
                available_alternatives=await self.collect_available_competitions()
            )
            
            withdrawal_analyses.append({
                "competition": status.competition_data.title,
                "should_withdraw": analysis.should_withdraw,
                "withdrawal_score": analysis.withdrawal_score,
                "primary_reason": analysis.primary_reason.value,
                "urgency": analysis.withdrawal_urgency
            })
            
            # 緊急度の高い撤退推奨
            if analysis.should_withdraw and analysis.withdrawal_urgency >= task.metadata.get("urgency_threshold", 0.7):
                withdrawal_recommendations.append({
                    "competition": status.competition_data.title,
                    "recommendation": analysis.recommended_action,
                    "urgency": analysis.withdrawal_urgency
                })
        
        # 緊急撤退通知
        urgent_notifications = []
        if withdrawal_recommendations:
            urgent_notifications = await self.send_withdrawal_notifications(withdrawal_recommendations)
        
        return {
            "analyses_completed": len(withdrawal_analyses),
            "withdrawal_recommendations": len(withdrawal_recommendations),
            "urgent_notifications": len(urgent_notifications),
            "analyses": withdrawal_analyses
        }
    
    async def execute_health_check(self, task: ScheduledTask) -> Dict[str, Any]:
        """システムヘルスチェック実行"""
        
        health_status = {
            "scheduler_running": self.is_running,
            "active_competitions": len(self.active_competitions),
            "recent_executions": len([
                r for r in self.execution_history[-10:] 
                if r.status == ExecutionStatus.COMPLETED
            ]),
            "system_resources": await self.check_system_resources(),
            "agent_status": await self.check_agent_status(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # アラート判定
        alerts = []
        if health_status["recent_executions"] < 5:
            alerts.append("最近の実行成功率が低下")
        
        if health_status["system_resources"]["memory_usage"] > 0.9:
            alerts.append("メモリ使用量が高い")
        
        if alerts:
            await self.send_health_alerts(alerts)
        
        return {
            "health_check_completed": True,
            "overall_status": "healthy" if not alerts else "warning",
            "alerts": alerts,
            "details": health_status
        }
    
    async def execute_data_collection(self, task: ScheduledTask) -> Dict[str, Any]:
        """データ収集実行"""
        
        self.logger.info("コンペティションデータ収集実行")
        
        # データソース別収集
        collection_results = {}
        
        for source in task.metadata.get("data_sources", []):
            try:
                if source == "kaggle_api":
                    result = await self.collect_kaggle_api_data()
                elif source == "discussion_scraping":
                    result = await self.collect_discussion_data()
                else:
                    result = {"status": "unknown_source"}
                
                collection_results[source] = result
                
            except Exception as e:
                collection_results[source] = {"status": "failed", "error": str(e)}
        
        # メダル確率更新
        probability_updates = 0
        if task.metadata.get("update_probabilities", False):
            probability_updates = await self.update_medal_probabilities()
        
        return {
            "data_collection_completed": True,
            "sources_processed": len(collection_results),
            "probability_updates": probability_updates,
            "collection_results": collection_results
        }
    
    async def collect_available_competitions(self) -> List[CompetitionData]:
        """利用可能なコンペ収集（モック実装）"""
        
        # 実際の実装では Kaggle API からデータ取得
        mock_competitions = [
            CompetitionData(
                competition_id="comp_1",
                title="Tabular Playground Series",
                participant_count=2500,
                total_prize=25000,
                prize_type=PrizeType.MONETARY,
                competition_type=CompetitionType.TABULAR,
                days_remaining=45,
                data_characteristics={"rows": 100000, "features": 20},
                skill_requirements=["feature_engineering", "ensemble"],
                leaderboard_competition=0.6
            ),
            CompetitionData(
                competition_id="comp_2", 
                title="Computer Vision Challenge",
                participant_count=1800,
                total_prize=50000,
                prize_type=PrizeType.MONETARY,
                competition_type=CompetitionType.COMPUTER_VISION,
                days_remaining=60,
                data_characteristics={"images": 50000, "classes": 10},
                skill_requirements=["cnn", "deep_learning"],
                leaderboard_competition=0.7
            ),
            CompetitionData(
                competition_id="comp_3",
                title="NLP Text Classification",
                participant_count=2200,
                total_prize=30000,
                prize_type=PrizeType.MONETARY,
                competition_type=CompetitionType.NLP,
                days_remaining=30,
                data_characteristics={"texts": 80000, "languages": 3},
                skill_requirements=["transformer", "nlp"],
                leaderboard_competition=0.5
            )
        ]
        
        return mock_competitions
    
    def analyze_portfolio_changes(self, optimization_result) -> List[Dict[str, Any]]:
        """ポートフォリオ変更分析"""
        
        current_titles = set(comp.title for comp in self.current_portfolio)
        new_titles = set(
            item.competition_data.title 
            for item in optimization_result.selected_competitions
        )
        
        changes = []
        
        # 追加されたコンペ
        added = new_titles - current_titles
        for title in added:
            changes.append({"type": "add", "competition": title})
        
        # 削除されたコンペ
        removed = current_titles - new_titles
        for title in removed:
            changes.append({"type": "remove", "competition": title})
        
        return changes
    
    async def send_agent_notifications(self, optimization_result) -> List[str]:
        """エージェント起動通知送信"""
        
        notifications = []
        
        for item in optimization_result.selected_competitions:
            # GitHub Issue 作成（実際の実装）
            notification = await self.create_agent_issue(
                competition=item.competition_data,
                selection_reason=item.selection_reason,
                expected_probability=item.medal_probability_result.overall_probability
            )
            
            notifications.append(notification)
        
        return notifications
    
    async def create_agent_issue(
        self, 
        competition: CompetitionData, 
        selection_reason: str,
        expected_probability: float
    ) -> str:
        """エージェント用Issue作成"""
        
        # Issue安全連携システムを使用
        from ...issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
        
        # モック実装
        issue_title = f"[{competition.title}] planner: Medal Strategy Analysis - {competition.participant_count} participants"
        
        issue_body = f"""
# 戦略プランニングエージェント実行指示

## 役割
あなたは Kaggle メダル確率算出・戦略策定エージェントです。

## 現在のタスク
GitHub Issue: "{issue_title}" の戦略分析を実行してください。

## 実行コンテキスト
- 作業ディレクトリ: competitions/{competition.competition_id}/
- 対象コンペ: {competition.title}
- 動的管理システムからの選択理由: {selection_reason}
- 期待メダル確率: {expected_probability:.3f}

## 実行手順
1. コンペ基本情報の詳細収集・分析
2. メダル確率の多次元算出・検証
3. 専門性マッチング評価・強み活用戦略
4. 撤退条件・リスク管理設定
5. analyzer エージェント向け戦略Issue作成

## 完了後アクション
GitHub Issue更新 + analyzer エージェント起動通知
"""
        
        # 実際の実装では GitHub API 呼び出し
        self.logger.info(f"エージェントIssue作成: {issue_title}")
        
        return f"issue_created_{competition.competition_id}"
    
    async def send_withdrawal_notifications(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """撤退通知送信"""
        
        notifications = []
        
        for rec in recommendations:
            # 緊急撤退 Issue 作成
            notification = await self.create_withdrawal_issue(rec)
            notifications.append(notification)
        
        return notifications
    
    async def create_withdrawal_issue(self, recommendation: Dict[str, Any]) -> str:
        """撤退Issue作成"""
        
        issue_title = f"[URGENT] Withdrawal Recommendation: {recommendation['competition']}"
        
        issue_body = f"""
# 緊急撤退推奨

## コンペティション
{recommendation['competition']}

## 推奨アクション
{recommendation['recommendation']}

## 緊急度
{recommendation['urgency']:.2f} (1.0が最高)

## 必要な対応
1. 現在の進捗状況確認
2. 撤退判断の最終決定
3. 代替機会の検討
4. リソース再配分

**注意**: 高い緊急度のため早急な対応が必要です。
"""
        
        self.logger.warning(f"撤退推奨Issue作成: {recommendation['competition']}")
        
        return f"withdrawal_issue_{recommendation['competition']}"
    
    async def send_health_alerts(self, alerts: List[str]):
        """ヘルスアラート送信"""
        
        for alert in alerts:
            self.logger.warning(f"システムアラート: {alert}")
            # 実際の実装では通知システム連携
    
    async def check_system_resources(self) -> Dict[str, float]:
        """システムリソースチェック"""
        
        import psutil
        
        return {
            "cpu_usage": psutil.cpu_percent(interval=1) / 100.0,
            "memory_usage": psutil.virtual_memory().percent / 100.0,
            "disk_usage": psutil.disk_usage('/').percent / 100.0
        }
    
    async def check_agent_status(self) -> Dict[str, Any]:
        """エージェント状態チェック"""
        
        # Issue安全連携システムから状態取得
        return {
            "active_agents": len(self.active_competitions),
            "pending_issues": 0,  # 実装依存
            "failed_agents": 0
        }
    
    async def collect_kaggle_api_data(self) -> Dict[str, Any]:
        """Kaggle API データ収集"""
        
        # 実際の実装では kaggle API 使用
        return {
            "status": "success",
            "competitions_updated": 15,
            "last_update": datetime.utcnow().isoformat()
        }
    
    async def collect_discussion_data(self) -> Dict[str, Any]:
        """ディスカッションデータ収集"""
        
        # 実際の実装では web scraping
        return {
            "status": "success", 
            "discussions_scraped": 50,
            "insights_extracted": 12
        }
    
    async def update_medal_probabilities(self) -> int:
        """メダル確率更新"""
        
        # 現在のポートフォリオの確率を再計算
        updates = 0
        
        for competition in self.current_portfolio:
            # 最新データで確率再算出
            # 実装は MedalProbabilityCalculator を使用
            updates += 1
        
        return updates
    
    async def check_manual_triggers(self):
        """手動トリガーチェック"""
        
        # 実際の実装では外部ファイル・API監視
        trigger_file = "/tmp/kaggle_scheduler_trigger"
        
        if os.path.exists(trigger_file):
            try:
                with open(trigger_file, 'r') as f:
                    trigger_data = json.load(f)
                
                task_id = trigger_data.get("task_id")
                if task_id in self.scheduled_tasks:
                    await self.execute_scheduled_task(task_id)
                
                # トリガーファイル削除
                os.remove(trigger_file)
                
            except Exception as e:
                self.logger.error(f"手動トリガー処理失敗: {e}")
    
    async def monitor_executions(self):
        """実行状況監視"""
        
        # 長時間実行の検出
        current_time = datetime.utcnow()
        
        for task in self.scheduled_tasks.values():
            if (task.last_execution and 
                task.last_status == ExecutionStatus.RUNNING and
                (current_time - task.last_execution).total_seconds() > self.execution_timeout):
                
                self.logger.warning(f"長時間実行検出: {task.task_id}")
                # 必要に応じて強制終了・再開
    
    async def schedule_retry(self, task_id: str, retry_count: int):
        """リトライスケジュール"""
        
        # 指数バックオフでリトライ
        delay = min(300, 60 * (2 ** retry_count))  # 最大5分
        
        self.logger.info(f"タスクリトライスケジュール: {task_id} ({delay}秒後)")
        
        await asyncio.sleep(delay)
        await self.execute_scheduled_task(task_id)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """スケジューラー状態取得"""
        
        return {
            "is_running": self.is_running,
            "scheduled_tasks": len(self.scheduled_tasks),
            "execution_history_size": len(self.execution_history),
            "active_competitions": len(self.active_competitions),
            "current_portfolio_size": len(self.current_portfolio),
            "next_executions": {
                task_id: task.next_run.isoformat() if task.next_run else None
                for task_id, task in self.scheduled_tasks.items()
            },
            "recent_execution_success_rate": self.calculate_success_rate(),
            "status_timestamp": datetime.utcnow().isoformat()
        }
    
    def calculate_success_rate(self) -> float:
        """最近の成功率計算"""
        
        recent_executions = self.execution_history[-20:]  # 最新20件
        
        if not recent_executions:
            return 1.0
        
        success_count = sum(
            1 for result in recent_executions 
            if result.status == ExecutionStatus.COMPLETED
        )
        
        return success_count / len(recent_executions)