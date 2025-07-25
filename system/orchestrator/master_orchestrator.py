"""
マスターオーケストレーター

全エージェント（Planner・Analyzer・Executor・Monitor・Retrospective）を統合し、
Kaggle競技の自動参加から振り返りまでの完全自動化を実現するメインシステム。
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import traceback

# GitHub Issue安全システム
from ..issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ..issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations

# 全エージェントインポート
from ..agents.planner.planner_agent import PlannerAgent
from ..agents.analyzer.analyzer_agent import AnalyzerAgent
from ..agents.executor.executor_agent import ExecutorAgent
from ..agents.monitor.monitor_agent import MonitorAgent
from ..agents.retrospective.retrospective_agent import RetrospectiveAgent

# 動的競技管理システム
from ..dynamic_competition_manager import DynamicCompetitionManager


class SystemPhase(Enum):
    """システム実行フェーズ"""
    INITIALIZATION = "initialization"
    PLANNING = "planning"
    ANALYSIS = "analysis"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    RETROSPECTIVE = "retrospective"
    COMPLETED = "completed"
    ERROR = "error"


class OrchestrationMode(Enum):
    """オーケストレーション方式"""
    SEQUENTIAL = "sequential"    # 順次実行
    PARALLEL = "parallel"       # 並列実行
    ADAPTIVE = "adaptive"       # 適応的実行
    EMERGENCY = "emergency"     # 緊急モード


@dataclass
class CompetitionContext:
    """競技実行コンテキスト"""
    competition_id: str
    competition_name: str
    competition_type: str
    start_time: datetime
    deadline: datetime
    priority: str
    
    # 実行設定
    orchestration_mode: OrchestrationMode
    resource_budget: Dict[str, float]
    target_performance: Dict[str, float]
    
    # 状態追跡
    current_phase: SystemPhase
    phase_start_time: datetime
    agent_states: Dict[str, str] = field(default_factory=dict)
    
    # 結果蓄積
    planning_result: Optional[Any] = None
    analysis_result: Optional[Any] = None
    execution_result: Optional[Any] = None
    monitoring_data: List[Any] = field(default_factory=list)
    retrospective_result: Optional[Any] = None


@dataclass
class OrchestrationResult:
    """オーケストレーション結果"""
    orchestration_id: str
    competition_context: CompetitionContext
    start_time: datetime
    completion_time: datetime
    
    # 全体結果
    success: bool
    final_phase: SystemPhase
    overall_score: float
    
    # フェーズ別結果
    phase_results: Dict[str, Any]
    phase_durations: Dict[str, float]
    
    # エージェント別パフォーマンス
    agent_performance: Dict[str, Dict[str, Any]]
    
    # リソース使用量
    resource_consumption: Dict[str, float]
    
    # エラー・警告
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # メタデータ
    system_version: str = "1.0.0"
    total_duration_hours: float = 0.0


class MasterOrchestrator:
    """マスターオーケストレーター - メインクラス"""
    
    def __init__(self, github_token: str, repo_name: str, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        
        # システム情報
        self.orchestrator_id = f"master-{uuid.uuid4().hex[:8]}"
        self.system_version = "1.0.0"
        self.start_time = datetime.utcnow()
        
        # 設定
        self.config = config or {}
        self.github_token = github_token
        self.repo_name = repo_name
        
        # GitHub Issue連携
        self.github_wrapper = GitHubApiWrapper(
            access_token=github_token,
            repo_name=repo_name
        )
        self.atomic_operations = AtomicIssueOperations(
            github_client=self.github_wrapper.github,
            repo_name=repo_name
        )
        
        # エージェント初期化
        self.agents = {}
        self.initialize_agents()
        
        # 競技管理システム
        self.competition_manager = DynamicCompetitionManager(
            github_token=github_token,
            repo_name=repo_name
        )
        
        # 実行コンテキスト管理
        self.active_contexts: Dict[str, CompetitionContext] = {}
        self.orchestration_history: List[OrchestrationResult] = []
        
        # 統計・監視
        self.total_competitions_handled = 0
        self.successful_competitions = 0
        self.system_health_status = "healthy"
    
    def initialize_agents(self):
        """全エージェント初期化"""
        
        try:
            self.agents = {
                "planner": PlannerAgent(
                    repo_owner=self.repo_name.split('/')[0] if '/' in self.repo_name else "hkrhd",
                    repo_name=self.repo_name.split('/')[1] if '/' in self.repo_name else self.repo_name
                ),
                "analyzer": AnalyzerAgent(),
                "executor": ExecutorAgent(
                    github_token=self.github_token,
                    repo_name=self.repo_name
                ),
                "monitor": MonitorAgent(
                    github_token=self.github_token,
                    repo_name=self.repo_name
                ),
                "retrospective": RetrospectiveAgent(
                    github_token=self.github_token,
                    repo_name=self.repo_name
                )
            }
            
            self.logger.info(f"全エージェント初期化完了: {len(self.agents)}エージェント")
            
        except Exception as e:
            self.logger.error(f"エージェント初期化失敗: {e}")
            raise
    
    async def orchestrate_competition(
        self,
        competition_data: Dict[str, Any],
        orchestration_mode: OrchestrationMode = OrchestrationMode.ADAPTIVE
    ) -> OrchestrationResult:
        """競技完全自動実行"""
        
        orchestration_id = f"orch-{uuid.uuid4().hex[:8]}"
        start_time = datetime.utcnow()
        
        self.logger.info(f"🚀 競技オーケストレーション開始: {competition_data.get('name', 'Unknown')} (ID: {orchestration_id})")
        
        # 実行コンテキスト作成
        context = CompetitionContext(
            competition_id=competition_data.get("id", str(uuid.uuid4())),
            competition_name=competition_data.get("name", "Unknown Competition"),
            competition_type=competition_data.get("type", "tabular"),
            start_time=start_time,
            deadline=datetime.fromisoformat(competition_data.get("deadline", (datetime.utcnow() + timedelta(days=30)).isoformat())),
            priority=competition_data.get("priority", "standard"),
            orchestration_mode=orchestration_mode,
            resource_budget=competition_data.get("resource_budget", {
                "max_gpu_hours": 50.0,
                "max_api_calls": 10000,
                "max_execution_time_hours": 72.0
            }),
            target_performance=competition_data.get("target_performance", {
                "min_score_improvement": 0.05,
                "target_ranking_percentile": 0.1
            }),
            current_phase=SystemPhase.INITIALIZATION,
            phase_start_time=start_time
        )
        
        self.active_contexts[orchestration_id] = context
        
        # オーケストレーション実行
        result = await self._execute_orchestration(orchestration_id, context)
        
        # 結果記録
        self.orchestration_history.append(result)
        self.total_competitions_handled += 1
        if result.success:
            self.successful_competitions += 1
        
        # アクティブコンテキストからクリア
        if orchestration_id in self.active_contexts:
            del self.active_contexts[orchestration_id]
        
        self.logger.info(f"🏁 競技オーケストレーション完了: {orchestration_id} (成功: {result.success})")
        
        return result
    
    async def _execute_orchestration(
        self,
        orchestration_id: str,
        context: CompetitionContext
    ) -> OrchestrationResult:
        """オーケストレーション実行"""
        
        phase_results = {}
        phase_durations = {}
        agent_performance = {}
        errors = []
        warnings = []
        
        # GitHub Issue作成: オーケストレーション開始
        orchestration_issue = await self._create_orchestration_issue(context)
        
        try:
            # Phase 1: 初期化・計画
            context.current_phase = SystemPhase.PLANNING
            context.phase_start_time = datetime.utcnow()
            
            await self._post_phase_update(orchestration_issue, context.current_phase, "開始")
            
            planning_result = await self._execute_planning_phase(context)
            phase_results["planning"] = planning_result
            phase_durations["planning"] = (datetime.utcnow() - context.phase_start_time).total_seconds() / 3600
            
            if not planning_result.get("success", False):
                errors.append("計画フェーズ失敗")
                context.current_phase = SystemPhase.ERROR
            else:
                context.planning_result = planning_result
            
            await self._post_phase_update(orchestration_issue, context.current_phase, "完了", planning_result)
            
            # Phase 2: 分析
            if context.current_phase != SystemPhase.ERROR:
                context.current_phase = SystemPhase.ANALYSIS
                context.phase_start_time = datetime.utcnow()
                
                await self._post_phase_update(orchestration_issue, context.current_phase, "開始")
                
                analysis_result = await self._execute_analysis_phase(context)
                phase_results["analysis"] = analysis_result
                phase_durations["analysis"] = (datetime.utcnow() - context.phase_start_time).total_seconds() / 3600
                
                if not analysis_result.get("success", False):
                    errors.append("分析フェーズ失敗")
                    context.current_phase = SystemPhase.ERROR
                else:
                    context.analysis_result = analysis_result
                
                await self._post_phase_update(orchestration_issue, context.current_phase, "完了", analysis_result)
            
            # Phase 3: 実行（監視と並列）
            if context.current_phase != SystemPhase.ERROR:
                context.current_phase = SystemPhase.EXECUTION
                context.phase_start_time = datetime.utcnow()
                
                await self._post_phase_update(orchestration_issue, context.current_phase, "開始")
                
                # 実行と監視を並列実行
                execution_result, monitoring_data = await self._execute_execution_monitoring_phase(context)
                
                phase_results["execution"] = execution_result
                phase_results["monitoring"] = {"monitoring_cycles": len(monitoring_data)}
                phase_durations["execution"] = (datetime.utcnow() - context.phase_start_time).total_seconds() / 3600
                
                if not execution_result.get("success", False):
                    errors.append("実行フェーズ失敗")
                    context.current_phase = SystemPhase.ERROR
                else:
                    context.execution_result = execution_result
                    context.monitoring_data = monitoring_data
                
                await self._post_phase_update(orchestration_issue, context.current_phase, "完了", execution_result)
            
            # Phase 4: 振り返り
            if context.current_phase != SystemPhase.ERROR:
                context.current_phase = SystemPhase.RETROSPECTIVE
                context.phase_start_time = datetime.utcnow()
                
                await self._post_phase_update(orchestration_issue, context.current_phase, "開始")
                
                retrospective_result = await self._execute_retrospective_phase(context)
                phase_results["retrospective"] = retrospective_result
                phase_durations["retrospective"] = (datetime.utcnow() - context.phase_start_time).total_seconds() / 3600
                
                context.retrospective_result = retrospective_result
                
                await self._post_phase_update(orchestration_issue, context.current_phase, "完了", retrospective_result)
                
                context.current_phase = SystemPhase.COMPLETED
            
            # エージェント別パフォーマンス収集
            agent_performance = await self._collect_agent_performance()
            
            # 総合評価
            overall_success = context.current_phase == SystemPhase.COMPLETED
            overall_score = self._calculate_overall_score(phase_results)
            
            # 結果オブジェクト作成
            result = OrchestrationResult(
                orchestration_id=orchestration_id,
                competition_context=context,
                start_time=context.start_time,
                completion_time=datetime.utcnow(),
                success=overall_success,
                final_phase=context.current_phase,
                overall_score=overall_score,
                phase_results=phase_results,
                phase_durations=phase_durations,
                agent_performance=agent_performance,
                resource_consumption=self._calculate_resource_consumption(phase_results),
                errors=errors,
                warnings=warnings,
                total_duration_hours=(datetime.utcnow() - context.start_time).total_seconds() / 3600
            )
            
            # 最終レポート投稿
            await self._post_final_report(orchestration_issue, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"オーケストレーション実行エラー: {e}")
            errors.append(f"システムエラー: {str(e)}")
            
            # エラー結果作成
            error_result = OrchestrationResult(
                orchestration_id=orchestration_id,
                competition_context=context,
                start_time=context.start_time,
                completion_time=datetime.utcnow(),
                success=False,
                final_phase=SystemPhase.ERROR,
                overall_score=0.0,
                phase_results=phase_results,
                phase_durations=phase_durations,
                agent_performance=agent_performance,
                resource_consumption={},
                errors=errors + [traceback.format_exc()],
                warnings=warnings,
                total_duration_hours=(datetime.utcnow() - context.start_time).total_seconds() / 3600
            )
            
            await self._post_error_report(orchestration_issue, error_result)
            
            return error_result
    
    async def _execute_planning_phase(self, context: CompetitionContext) -> Dict[str, Any]:
        """計画フェーズ実行"""
        
        try:
            planner = self.agents["planner"]
            
            # 計画エージェント実行
            planning_result = await planner.create_competition_plan(
                competition_name=context.competition_name,
                competition_type=context.competition_type,
                deadline_days=(context.deadline - context.start_time).days,
                resource_constraints=context.resource_budget
            )
            
            self.logger.info(f"計画フェーズ完了: {context.competition_name}")
            
            return {
                "success": True,
                "plan_id": planning_result.plan_id,
                "total_phases": len(planning_result.execution_phases),
                "estimated_duration_hours": planning_result.estimated_total_duration_hours,
                "resource_allocation": planning_result.resource_allocation
            }
            
        except Exception as e:
            self.logger.error(f"計画フェーズエラー: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_analysis_phase(self, context: CompetitionContext) -> Dict[str, Any]:
        """分析フェーズ実行"""
        
        try:
            analyzer = self.agents["analyzer"]
            
            # 分析要求作成
            analysis_request = {
                "competition_name": context.competition_name,
                "competition_type": context.competition_type,
                "analysis_depth": "comprehensive",
                "planning_context": context.planning_result
            }
            
            # 分析エージェント実行
            analysis_result = await analyzer.analyze_competition(analysis_request)
            
            self.logger.info(f"分析フェーズ完了: {len(analysis_result.recommended_techniques)}技術特定")
            
            return {
                "success": True,
                "analysis_id": analysis_result.analysis_id,
                "techniques_identified": len(analysis_result.recommended_techniques),
                "grandmaster_patterns": len(analysis_result.grandmaster_patterns),
                "feasibility_scores": [t.feasibility_score for t in analysis_result.recommended_techniques]
            }
            
        except Exception as e:
            self.logger.error(f"分析フェーズエラー: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _execute_execution_monitoring_phase(self, context: CompetitionContext) -> Tuple[Dict[str, Any], List[Any]]:
        """実行・監視フェーズ並列実行"""
        
        try:
            executor = self.agents["executor"]
            monitor = self.agents["monitor"]
            
            # 実行要求作成
            from ..agents.executor.executor_agent import ExecutionRequest, ExecutionPriority
            
            execution_request = ExecutionRequest(
                competition_name=context.competition_name,
                analyzer_issue_number=1,  # 実際には分析結果から取得
                techniques_to_implement=context.analysis_result.get("techniques", []),
                priority=ExecutionPriority.HIGH,
                deadline_days=(context.deadline - datetime.utcnow()).days,
                resource_constraints=context.resource_budget
            )
            
            # 監視開始
            monitor_task = asyncio.create_task(
                monitor.start_monitoring({"executor": executor})
            )
            
            # 実行開始
            execution_result = await executor.execute_technical_implementation(execution_request)
            
            # 監視停止
            await monitor.stop_monitoring()
            
            # 監視データ収集
            monitoring_data = []  # 実際には監視エージェントから収集
            
            self.logger.info(f"実行・監視フェーズ完了: 成功率{execution_result.success_rate:.1%}")
            
            execution_summary = {
                "success": execution_result.success_rate > 0.5,
                "execution_id": execution_result.execution_id,
                "techniques_attempted": len(execution_result.kaggle_results + execution_result.colab_results + execution_result.paperspace_results),
                "best_score": execution_result.best_score,
                "gpu_hours_used": execution_result.total_gpu_hours_used,
                "success_rate": execution_result.success_rate
            }
            
            return execution_summary, monitoring_data
            
        except Exception as e:
            self.logger.error(f"実行・監視フェーズエラー: {e}")
            return {
                "success": False,
                "error": str(e)
            }, []
    
    async def _execute_retrospective_phase(self, context: CompetitionContext) -> Dict[str, Any]:
        """振り返りフェーズ実行"""
        
        try:
            retrospective = self.agents["retrospective"]
            
            # 振り返り用競技データ作成
            competition_data = {
                "name": context.competition_name,
                "type": context.competition_type,
                "start_date": context.start_time.isoformat(),
                "end_date": datetime.utcnow().isoformat(),
                "deadline": context.deadline.isoformat()
            }
            
            # エージェント履歴収集
            agent_histories = {
                "planner": [context.planning_result] if context.planning_result else [],
                "analyzer": [context.analysis_result] if context.analysis_result else [],
                "executor": [context.execution_result] if context.execution_result else [],
                "monitor": context.monitoring_data
            }
            
            # 振り返り実行
            retrospective_result = await retrospective.conduct_competition_retrospective(
                competition_data=competition_data,
                agent_histories=agent_histories
            )
            
            self.logger.info(f"振り返りフェーズ完了: {len(retrospective_result.new_insights)}新知見獲得")
            
            return {
                "success": True,
                "report_id": retrospective_result.report_id,
                "overall_performance": retrospective_result.overall_performance,
                "new_insights": len(retrospective_result.new_insights),
                "improvement_recommendations": len(retrospective_result.immediate_improvements)
            }
            
        except Exception as e:
            self.logger.error(f"振り返りフェーズエラー: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _collect_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """エージェント別パフォーマンス収集"""
        
        performance = {}
        
        for agent_name, agent in self.agents.items():
            try:
                if hasattr(agent, 'get_performance_summary'):
                    perf_data = await agent.get_performance_summary()
                    performance[agent_name] = perf_data
                else:
                    performance[agent_name] = {"status": "active", "metrics": "not_available"}
                    
            except Exception as e:
                performance[agent_name] = {"status": "error", "error": str(e)}
        
        return performance
    
    def _calculate_overall_score(self, phase_results: Dict[str, Any]) -> float:
        """総合スコア計算"""
        
        scores = []
        
        # 各フェーズの成功度を評価
        if phase_results.get("planning", {}).get("success", False):
            scores.append(0.2)
        
        if phase_results.get("analysis", {}).get("success", False):
            techniques_count = phase_results["analysis"].get("techniques_identified", 0)
            scores.append(0.2 + min(0.1, techniques_count * 0.02))
        
        if phase_results.get("execution", {}).get("success", False):
            success_rate = phase_results["execution"].get("success_rate", 0)
            scores.append(0.3 + success_rate * 0.2)
        
        if phase_results.get("retrospective", {}).get("success", False):
            scores.append(0.1)
        
        return sum(scores)
    
    def _calculate_resource_consumption(self, phase_results: Dict[str, Any]) -> Dict[str, float]:
        """リソース消費量計算"""
        
        consumption = {
            "total_gpu_hours": 0.0,
            "total_api_calls": 0.0,
            "total_execution_time_hours": 0.0
        }
        
        # 実行フェーズのリソース消費
        if "execution" in phase_results:
            exec_result = phase_results["execution"]
            consumption["total_gpu_hours"] = exec_result.get("gpu_hours_used", 0.0)
        
        # 各フェーズの時間消費
        total_time = sum(phase_results.get("phase_durations", {}).values())
        consumption["total_execution_time_hours"] = total_time
        
        return consumption
    
    async def _create_orchestration_issue(self, context: CompetitionContext) -> int:
        """オーケストレーションIssue作成"""
        
        title = f"🎯 完全自動競技実行: {context.competition_name}"
        
        description = f"""
## Kaggle競技完全自動実行

**競技名**: {context.competition_name}
**競技タイプ**: {context.competition_type}
**オーケストレーターID**: `{self.orchestrator_id}`
**実行モード**: {context.orchestration_mode.value}
**開始時刻**: {context.start_time.isoformat()}
**締切**: {context.deadline.isoformat()}

### リソース予算
- **最大GPU時間**: {context.resource_budget.get('max_gpu_hours', 0)}時間
- **最大API呼び出し**: {context.resource_budget.get('max_api_calls', 0)}回
- **最大実行時間**: {context.resource_budget.get('max_execution_time_hours', 0)}時間

### 目標性能
- **最小スコア改善**: {context.target_performance.get('min_score_improvement', 0)}
- **目標順位パーセンタイル**: {context.target_performance.get('target_ranking_percentile', 0)}

### 実行フェーズ
1. 🎯 **計画フェーズ**: 競技戦略・リソース配分計画
2. 🔍 **分析フェーズ**: グランドマスター手法調査・技術選定
3. ⚡ **実行フェーズ**: 並列実験実行・リアルタイム監視
4. 📊 **振り返りフェーズ**: 学習知見抽出・改善提案

このIssueで全フェーズの進捗と結果をリアルタイム更新します。
        """
        
        try:
            issue_data = await self.atomic_operations.create_issue(
                title=title,
                description=description,
                labels=["orchestration", "competition", "automation", "active"]
            )
            return issue_data["number"]
            
        except Exception as e:
            self.logger.error(f"オーケストレーションIssue作成失敗: {e}")
            return -1
    
    async def _post_phase_update(
        self,
        issue_number: int,
        phase: SystemPhase,
        status: str,
        result: Dict[str, Any] = None
    ):
        """フェーズ更新投稿"""
        
        phase_icons = {
            SystemPhase.PLANNING: "🎯",
            SystemPhase.ANALYSIS: "🔍",
            SystemPhase.EXECUTION: "⚡",
            SystemPhase.MONITORING: "📊",
            SystemPhase.RETROSPECTIVE: "🔄",
            SystemPhase.COMPLETED: "✅",
            SystemPhase.ERROR: "❌"
        }
        
        icon = phase_icons.get(phase, "📋")
        
        content = f"""
## {icon} {phase.value.upper()}フェーズ {status}

**時刻**: {datetime.utcnow().isoformat()}
"""
        
        if result:
            content += f"""
### 結果サマリー
```json
{json.dumps(result, indent=2, default=str, ensure_ascii=False)}
```
"""
        
        content += f"""
---
*Master Orchestrator {self.orchestrator_id}*
        """
        
        try:
            await self.atomic_operations.create_comment(
                issue_number=issue_number,
                comment_body=content
            )
        except Exception as e:
            self.logger.error(f"フェーズ更新投稿失敗: {e}")
    
    async def _post_final_report(self, issue_number: int, result: OrchestrationResult):
        """最終レポート投稿"""
        
        success_icon = "🏆" if result.success else "❌"
        
        content = f"""
## {success_icon} 競技実行完了 - 最終レポート

### 📊 実行サマリー
- **成功状況**: {success_icon} {'成功' if result.success else '失敗'}
- **最終フェーズ**: {result.final_phase.value}
- **総合スコア**: {result.overall_score:.2f}/1.0
- **総実行時間**: {result.total_duration_hours:.1f}時間

### ⏱️ フェーズ別実行時間
{chr(10).join([f"- **{phase}**: {duration:.1f}時間" for phase, duration in result.phase_durations.items()]) if result.phase_durations else "- データなし"}

### 🔧 エージェント別パフォーマンス
{chr(10).join([f"- **{agent}**: {perf.get('status', 'unknown')}" for agent, perf in result.agent_performance.items()]) if result.agent_performance else "- データなし"}

### 📈 リソース消費
- **GPU時間**: {result.resource_consumption.get('total_gpu_hours', 0):.1f}時間
- **実行時間**: {result.resource_consumption.get('total_execution_time_hours', 0):.1f}時間

### ⚠️ エラー・警告
{chr(10).join([f"- ERROR: {error}" for error in result.errors]) if result.errors else "- エラーなし"}
{chr(10).join([f"- WARNING: {warning}" for warning in result.warnings]) if result.warnings else ""}

### 🎯 競技結果
- **競技名**: {result.competition_context.competition_name}
- **競技タイプ**: {result.competition_context.competition_type}
- **実行結果**: {'成功' if result.success else '失敗'}

---
*オーケストレーションID: {result.orchestration_id} | System Version: {result.system_version}*
        """
        
        try:
            await self.atomic_operations.create_comment(
                issue_number=issue_number,
                comment_body=content
            )
        except Exception as e:
            self.logger.error(f"最終レポート投稿失敗: {e}")
    
    async def _post_error_report(self, issue_number: int, result: OrchestrationResult):
        """エラーレポート投稿"""
        
        content = f"""
## ❌ システムエラー発生

**エラー発生時刻**: {datetime.utcnow().isoformat()}
**最終到達フェーズ**: {result.final_phase.value}

### エラー詳細
{chr(10).join([f"```{error}```" for error in result.errors]) if result.errors else "エラー情報なし"}

### 部分実行結果
{chr(10).join([f"- **{phase}**: {'成功' if res.get('success', False) else '失敗'}" for phase, res in result.phase_results.items()]) if result.phase_results else "- 実行結果なし"}

システム管理者による確認が必要です。

---
*Master Orchestrator {self.orchestrator_id}*
        """
        
        try:
            await self.atomic_operations.create_comment(
                issue_number=issue_number,
                comment_body=content
            )
        except Exception as e:
            self.logger.error(f"エラーレポート投稿失敗: {e}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        
        uptime_hours = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "system_version": self.system_version,
            "uptime_hours": uptime_hours,
            "system_health": self.system_health_status,
            "total_competitions_handled": self.total_competitions_handled,
            "successful_competitions": self.successful_competitions,
            "success_rate": self.successful_competitions / max(self.total_competitions_handled, 1),
            "active_contexts_count": len(self.active_contexts),
            "orchestration_history_count": len(self.orchestration_history),
            "agents_status": {
                name: "active" for name in self.agents.keys()
            }
        }
    
    async def start_autonomous_mode(self):
        """自律実行モード開始"""
        
        self.logger.info("🤖 自律実行モード開始")
        
        # 競技管理システムから新競技を定期チェック
        while True:
            try:
                # 新競技検出
                new_competitions = await self.competition_manager.scan_new_competitions()
                
                for competition in new_competitions:
                    # 自動実行判定
                    if self._should_auto_execute(competition):
                        self.logger.info(f"自動実行開始: {competition.get('name', 'Unknown')}")
                        
                        # バックグラウンドで実行
                        asyncio.create_task(
                            self.orchestrate_competition(competition, OrchestrationMode.ADAPTIVE)
                        )
                
                # 次回チェックまで待機
                await asyncio.sleep(3600)  # 1時間間隔
                
            except Exception as e:
                self.logger.error(f"自律実行モードエラー: {e}")
                await asyncio.sleep(600)  # エラー時は10分待機
    
    def _should_auto_execute(self, competition: Dict[str, Any]) -> bool:
        """自動実行判定"""
        
        # 基本判定ロジック
        deadline = datetime.fromisoformat(competition.get("deadline", datetime.utcnow().isoformat()))
        days_remaining = (deadline - datetime.utcnow()).days
        
        # 十分な時間がある競技のみ自動実行
        if days_remaining < 7:
            return False
        
        # リソース予算内での実行可能性
        if self.total_competitions_handled >= 10:  # 同時実行制限
            return False
        
        # 競技タイプフィルタ
        supported_types = ["tabular", "computer-vision", "nlp"]
        if competition.get("type", "") not in supported_types:
            return False
        
        return True