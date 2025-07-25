"""
プランニングエージェント・メインクラス

戦略的コンペ選択・撤退判断・GitHub Issue連携の統合エージェント。
plan_planner.md の実装仕様に準拠した核心エージェント。
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import os

from .models.competition import CompetitionInfo, CompetitionStatus, AnalysisResult
from .models.probability import MedalProbability
from .calculators.medal_probability import MedalProbabilityCalculator
from .calculators.competition_scanner import CompetitionScanner
from .strategies.selection_strategy import CompetitionSelectionStrategy, SelectionStrategy
from .strategies.withdrawal_strategy import WithdrawalStrategy
from .utils.github_issues import GitHubIssueManager
from .utils.kaggle_api import KaggleApiClient

# 模擬的な計画結果クラス
@dataclass
class MockPlanResult:
    plan_id: str
    execution_phases: List[Dict[str, Any]]
    estimated_total_duration_hours: float
    resource_allocation: Dict[str, Any]


class PlannerAgent:
    """プランニングエージェント・メインクラス"""
    
    def __init__(
        self,
        repo_owner: str = "hkrhd",
        repo_name: str = "kaggle-claude-mother",
        kaggle_credentials_path: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        
        # エージェント基本情報
        self.agent_id = "planner"
        self.agent_version = "1.0"
        self.startup_time = datetime.now(timezone.utc)
        
        # コンポーネント初期化
        self.probability_calculator = MedalProbabilityCalculator()
        self.competition_scanner = CompetitionScanner(kaggle_credentials_path)
        self.selection_strategy = CompetitionSelectionStrategy()
        self.withdrawal_strategy = WithdrawalStrategy()
        self.github_manager = GitHubIssueManager(repo_owner, repo_name)
        self.kaggle_client = KaggleApiClient(kaggle_credentials_path)
        
        # 動的管理システム連携
        self.max_portfolio_size = 3
        self.analysis_frequency_hours = 12  # 12時間間隔での分析
        
        # 状態管理
        self.current_portfolio: List[CompetitionInfo] = []
        self.analysis_history: List[AnalysisResult] = []
        self.active_issues: Dict[str, int] = {}  # competition_id -> issue_number
        
        # 設定
        self.auto_mode = True  # 自動実行モード
        self.selection_strategy_type = SelectionStrategy.BALANCED
        
    async def initialize(self) -> bool:
        """エージェント初期化"""
        
        self.logger.info("プランニングエージェント初期化開始")
        
        try:
            # 外部システム初期化
            if not await self.kaggle_client.initialize():
                self.logger.error("Kaggle API クライアント初期化失敗")
                return False
            
            # 現在のポートフォリオ復元
            await self._restore_current_portfolio()
            
            self.logger.info("プランニングエージェント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"プランニングエージェント初期化失敗: {e}")
            return False
    
    async def execute_strategy_analysis(
        self,
        competition_info: CompetitionInfo,
        selection_rationale: str = "",
        issue_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """戦略分析実行"""
        
        self.logger.info(f"戦略分析実行開始: {competition_info.title}")
        
        try:
            # 分析実行
            analysis_result = await self.selection_strategy.analyze_competition_for_selection(
                competition_info=competition_info,
                strategy=self.selection_strategy_type,
                current_portfolio=self.current_portfolio
            )
            
            # メダル確率詳細算出
            medal_probability = await self.probability_calculator.calculate_medal_probability(
                competition_info
            )
            
            # 結果記録
            self.analysis_history.append(analysis_result)
            
            # GitHub Issue 作成/更新
            if self.auto_mode:
                issue_result = await self._handle_issue_creation_update(
                    competition_info, analysis_result, medal_probability, 
                    selection_rationale, issue_number
                )
            else:
                issue_result = {"success": True, "message": "手動モード - Issue操作スキップ"}
            
            # ポートフォリオ更新判定
            portfolio_action = await self._evaluate_portfolio_action(
                analysis_result, competition_info
            )
            
            execution_result = {
                "success": True,
                "competition_id": competition_info.competition_id,
                "analysis_result": analysis_result,
                "medal_probability": medal_probability,
                "recommended_action": analysis_result.recommended_action,
                "action_confidence": analysis_result.action_confidence,
                "portfolio_action": portfolio_action,
                "issue_result": issue_result,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"戦略分析実行完了: {analysis_result.recommended_action} ({competition_info.title})")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"戦略分析実行失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "competition_id": competition_info.competition_id
            }
    
    async def execute_portfolio_optimization(
        self,
        available_competitions: Optional[List[CompetitionInfo]] = None,
        strategy: SelectionStrategy = SelectionStrategy.BALANCED
    ) -> Dict[str, Any]:
        """ポートフォリオ最適化実行"""
        
        self.logger.info("ポートフォリオ最適化実行開始")
        
        try:
            # 利用可能コンペ取得
            if available_competitions is None:
                available_competitions = await self.competition_scanner.scan_active_competitions()
            
            # 最適ポートフォリオ推奨
            portfolio_recommendation = await self.selection_strategy.recommend_optimal_portfolio(
                available_competitions=available_competitions,
                strategy=strategy,
                current_portfolio=self.current_portfolio
            )
            
            # ポートフォリオ変更判定・実行
            changes_made = await self._execute_portfolio_changes(portfolio_recommendation)
            
            # 結果統合
            optimization_result = {
                "success": True,
                "portfolio_recommendation": portfolio_recommendation,
                "changes_made": changes_made,
                "new_portfolio_size": len(portfolio_recommendation.recommended_competitions),
                "expected_medal_count": portfolio_recommendation.expected_medal_count,
                "portfolio_score": portfolio_recommendation.portfolio_score,
                "optimization_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"ポートフォリオ最適化完了: {len(changes_made)}件の変更")
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ最適化失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_withdrawal_analysis(
        self,
        competition_statuses: List[Tuple[CompetitionInfo, CompetitionStatus]],
        available_alternatives: Optional[List[CompetitionInfo]] = None
    ) -> Dict[str, Any]:
        """撤退分析実行"""
        
        self.logger.info(f"撤退分析実行開始: {len(competition_statuses)}件")
        
        try:
            withdrawal_analyses = {}
            alert_issues_created = []
            
            for comp_info, status in competition_statuses:
                # 撤退分析
                analysis = await self.withdrawal_strategy.analyze_withdrawal_decision(
                    competition_info=comp_info,
                    current_status=status,
                    available_alternatives=available_alternatives
                )
                
                withdrawal_analyses[comp_info.competition_id] = analysis
                
                # 緊急撤退アラート
                if analysis.should_withdraw and analysis.withdrawal_urgency.value in ["immediate", "urgent"]:
                    if self.auto_mode:
                        alert_result = await self.github_manager.create_withdrawal_alert_issue(
                            comp_info,
                            analysis.__dict__,
                            analysis.withdrawal_urgency.value
                        )
                        
                        if alert_result["success"]:
                            alert_issues_created.append({
                                "competition_id": comp_info.competition_id,
                                "issue_number": alert_result["issue_number"],
                                "urgency": analysis.withdrawal_urgency.value
                            })
            
            # 撤退実行（immediate の場合）
            immediate_withdrawals = await self._execute_immediate_withdrawals(withdrawal_analyses)
            
            analysis_result = {
                "success": True,
                "total_analyzed": len(competition_statuses),
                "withdrawal_recommended": sum(1 for a in withdrawal_analyses.values() if a.should_withdraw),
                "immediate_withdrawals": len(immediate_withdrawals),
                "alert_issues_created": len(alert_issues_created),
                "withdrawal_analyses": {
                    comp_id: {
                        "should_withdraw": analysis.should_withdraw,
                        "urgency": analysis.withdrawal_urgency.value,
                        "primary_reason": analysis.primary_reason.value if analysis.primary_reason else None,
                        "withdrawal_score": analysis.withdrawal_score,
                        "recommended_action": analysis.recommended_action
                    }
                    for comp_id, analysis in withdrawal_analyses.items()
                },
                "alert_issues": alert_issues_created,
                "immediate_withdrawals": immediate_withdrawals,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"撤退分析実行完了: {analysis_result['withdrawal_recommended']}件の撤退推奨")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"撤退分析実行失敗: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_continuous_monitoring(
        self,
        monitoring_interval_hours: int = 12
    ) -> None:
        """継続監視実行"""
        
        self.logger.info(f"継続監視開始: {monitoring_interval_hours}時間間隔")
        
        try:
            while True:
                # 定期分析実行
                monitoring_result = await self._execute_monitoring_cycle()
                
                self.logger.info(f"監視サイクル完了: {monitoring_result.get('summary', 'N/A')}")
                
                # 次回実行まで待機
                await asyncio.sleep(monitoring_interval_hours * 3600)
                
        except asyncio.CancelledError:
            self.logger.info("継続監視停止")
        except Exception as e:
            self.logger.error(f"継続監視エラー: {e}")
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """エージェント状態取得"""
        
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "startup_time": self.startup_time.isoformat(),
            "uptime_hours": (datetime.utcnow() - self.startup_time).total_seconds() / 3600,
            "auto_mode": self.auto_mode,
            "selection_strategy": self.selection_strategy_type.value,
            "current_portfolio_size": len(self.current_portfolio),
            "total_analyses_performed": len(self.analysis_history),
            "active_issues_count": len(self.active_issues),
            "last_activity": max(
                [analysis.analysis_timestamp for analysis in self.analysis_history] + [self.startup_time]
            ).isoformat() if self.analysis_history else self.startup_time.isoformat()
        }
    
    async def create_competition_plan(
        self,
        competition_name: str,
        competition_type: str,
        deadline_days: int,
        resource_constraints: Dict[str, Any]
    ):
        """競技計画作成（MasterOrchestrator互換）"""
        
        self.logger.info(f"競技計画作成開始: {competition_name}")
        
        try:
            # 模擬的な計画作成（実際の実装では詳細な戦略策定）
            plan_result = MockPlanResult(
                plan_id=f"plan-{competition_name.lower().replace(' ', '-')}-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
                execution_phases=[
                    {"phase": "data_analysis", "duration_hours": 8},
                    {"phase": "feature_engineering", "duration_hours": 16},
                    {"phase": "model_development", "duration_hours": 24},
                    {"phase": "optimization", "duration_hours": 12}
                ],
                estimated_total_duration_hours=60,
                resource_allocation={
                    "gpu_hours": min(40, resource_constraints.get("max_gpu_hours", 50)),
                    "api_calls": min(5000, resource_constraints.get("max_api_calls", 10000))
                }
            )
            
            self.logger.info(f"競技計画作成完了: {plan_result.plan_id}")
            return plan_result
            
        except Exception as e:
            self.logger.error(f"競技計画作成エラー: {e}")
            raise

    async def _handle_issue_creation_update(
        self,
        competition_info: CompetitionInfo,
        analysis_result: AnalysisResult,
        medal_probability: MedalProbability,
        selection_rationale: str,
        existing_issue_number: Optional[int]
    ) -> Dict[str, Any]:
        """Issue作成・更新処理"""
        
        try:
            if existing_issue_number:
                # 既存Issue更新
                progress_update = {
                    "status": "analysis_completed",
                    "progress_percentage": 1.0,
                    "current_task": "戦略分析完了",
                    "achievements": [
                        f"メダル確率算出: {medal_probability.overall_probability:.1%}",
                        f"推奨アクション: {analysis_result.recommended_action}",
                        f"戦略スコア: {analysis_result.strategic_score:.2f}"
                    ],
                    "next_steps": [
                        "analyzer エージェント起動",
                        "詳細技術分析実行"
                    ]
                }
                
                update_result = await self.github_manager.update_issue_with_progress(
                    existing_issue_number, progress_update, "analyzer-ready"
                )
                
                return {
                    "success": update_result["success"],
                    "action": "updated",
                    "issue_number": existing_issue_number
                }
            else:
                # 新規Issue作成
                creation_result = await self.github_manager.create_strategy_analysis_issue(
                    competition_info, analysis_result, medal_probability, selection_rationale
                )
                
                if creation_result["success"]:
                    # アクティブIssue記録
                    self.active_issues[competition_info.competition_id] = creation_result["issue_number"]
                
                return {
                    "success": creation_result["success"],
                    "action": "created",
                    "issue_number": creation_result.get("issue_number"),
                    "issue_url": creation_result.get("issue_url")
                }
                
        except Exception as e:
            self.logger.error(f"Issue処理エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def _evaluate_portfolio_action(
        self,
        analysis_result: AnalysisResult,
        competition_info: CompetitionInfo
    ) -> Dict[str, Any]:
        """ポートフォリオアクション評価"""
        
        action = {"type": "none", "reason": ""}
        
        if analysis_result.recommended_action == "participate":
            if len(self.current_portfolio) < self.max_portfolio_size:
                # 現在のポートフォリオに追加可能
                action = {"type": "add", "reason": "高スコア・空きスロットあり"}
            else:
                # 入れ替え検討
                lowest_score_comp = min(
                    self.current_portfolio,
                    key=lambda comp: self._calculate_current_score(comp)
                )
                current_min_score = self._calculate_current_score(lowest_score_comp)
                
                if analysis_result.strategic_score > current_min_score * 1.2:  # 20%以上向上
                    action = {
                        "type": "replace",
                        "reason": f"既存最低スコア({current_min_score:.2f})との入れ替え",
                        "replace_target": lowest_score_comp.competition_id
                    }
        elif analysis_result.recommended_action == "skip":
            action = {"type": "skip", "reason": "低スコアによるスキップ"}
        
        return action
    
    def _calculate_current_score(self, competition_info: CompetitionInfo) -> float:
        """現在のコンペスコア計算（簡易版）"""
        
        # 最新の分析結果から該当コンペのスコアを取得
        for analysis in reversed(self.analysis_history):
            if analysis.competition_info.competition_id == competition_info.competition_id:
                return analysis.strategic_score
        
        return 0.5  # デフォルトスコア
    
    async def _execute_portfolio_changes(
        self,
        portfolio_recommendation
    ) -> List[Dict[str, Any]]:
        """ポートフォリオ変更実行"""
        
        changes_made = []
        
        # 推奨ポートフォリオとの差分分析
        current_comp_ids = set(comp.competition_id for comp in self.current_portfolio)
        recommended_comp_ids = set(
            score.competition_info.competition_id 
            for score in portfolio_recommendation.recommended_competitions
        )
        
        # 追加すべきコンペ
        to_add = recommended_comp_ids - current_comp_ids
        for comp_id in to_add:
            comp_info = next(
                score.competition_info 
                for score in portfolio_recommendation.recommended_competitions
                if score.competition_info.competition_id == comp_id
            )
            
            # 実際の追加処理（Kaggle参加など）
            add_result = await self._add_competition_to_portfolio(comp_info)
            if add_result["success"]:
                changes_made.append({
                    "type": "added",
                    "competition_id": comp_id,
                    "competition_title": comp_info.title
                })
        
        # 削除すべきコンペ
        to_remove = current_comp_ids - recommended_comp_ids
        for comp_id in to_remove:
            comp_info = next(
                comp for comp in self.current_portfolio
                if comp.competition_id == comp_id
            )
            
            # 実際の削除処理
            remove_result = await self._remove_competition_from_portfolio(comp_info)
            if remove_result["success"]:
                changes_made.append({
                    "type": "removed",
                    "competition_id": comp_id,
                    "competition_title": comp_info.title
                })
        
        return changes_made
    
    async def _add_competition_to_portfolio(self, competition_info: CompetitionInfo) -> Dict[str, Any]:
        """ポートフォリオにコンペ追加"""
        
        try:
            # Kaggle コンペ参加
            join_result = await self.kaggle_client.join_competition(competition_info.competition_id)
            
            if join_result["success"]:
                # ローカルポートフォリオ更新
                self.current_portfolio.append(competition_info)
                
                # 戦略分析Issue作成
                analysis_result = await self.selection_strategy.analyze_competition_for_selection(
                    competition_info, self.selection_strategy_type, self.current_portfolio
                )
                
                medal_probability = await self.probability_calculator.calculate_medal_probability(
                    competition_info
                )
                
                await self.github_manager.create_strategy_analysis_issue(
                    competition_info, analysis_result, medal_probability,
                    "ポートフォリオ最適化による追加"
                )
                
                self.logger.info(f"ポートフォリオにコンペ追加成功: {competition_info.title}")
                return {"success": True}
            else:
                return {"success": False, "error": join_result.get("error", "参加失敗")}
                
        except Exception as e:
            self.logger.error(f"コンペ追加エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def _remove_competition_from_portfolio(self, competition_info: CompetitionInfo) -> Dict[str, Any]:
        """ポートフォリオからコンペ削除"""
        
        try:
            # ローカルポートフォリオから削除
            self.current_portfolio = [
                comp for comp in self.current_portfolio
                if comp.competition_id != competition_info.competition_id
            ]
            
            # 撤退アラートIssue作成
            withdrawal_analysis = {
                "withdrawal_score": 0.8,
                "primary_reason": "portfolio_optimization",
                "urgency": "moderate",
                "recommended_action": "ポートフォリオ最適化による撤退",
                "action_timeline": "即座に実行",
                "alternative_competitions": []
            }
            
            await self.github_manager.create_withdrawal_alert_issue(
                competition_info, withdrawal_analysis, "moderate"
            )
            
            # アクティブIssue クローズ
            if competition_info.competition_id in self.active_issues:
                issue_number = self.active_issues[competition_info.competition_id]
                await self.github_manager.close_completed_issue(
                    issue_number,
                    {"final_result": "ポートフォリオ最適化による撤退"}
                )
                del self.active_issues[competition_info.competition_id]
            
            self.logger.info(f"ポートフォリオからコンペ削除成功: {competition_info.title}")
            return {"success": True}
            
        except Exception as e:
            self.logger.error(f"コンペ削除エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_immediate_withdrawals(
        self,
        withdrawal_analyses: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """即座撤退実行"""
        
        immediate_withdrawals = []
        
        for comp_id, analysis in withdrawal_analyses.items():
            if analysis.should_withdraw and analysis.withdrawal_urgency.value == "immediate":
                # 対象コンペ取得
                target_comp = next(
                    (comp for comp in self.current_portfolio if comp.competition_id == comp_id),
                    None
                )
                
                if target_comp:
                    withdrawal_result = await self._remove_competition_from_portfolio(target_comp)
                    if withdrawal_result["success"]:
                        immediate_withdrawals.append({
                            "competition_id": comp_id,
                            "competition_title": target_comp.title,
                            "withdrawal_reason": analysis.primary_reason.value if analysis.primary_reason else "unknown"
                        })
        
        return immediate_withdrawals
    
    async def _execute_monitoring_cycle(self) -> Dict[str, Any]:
        """監視サイクル実行"""
        
        try:
            cycle_start = datetime.utcnow()
            
            # 1. アクティブコンペスキャン
            active_competitions = await self.competition_scanner.scan_active_competitions()
            
            # 2. 現在ポートフォリオの撤退分析
            portfolio_statuses = []
            for comp in self.current_portfolio:
                # 仮のステータス作成（実際は外部から取得）
                status = CompetitionStatus(competition_info=comp)
                portfolio_statuses.append((comp, status))
            
            withdrawal_result = await self.execute_withdrawal_analysis(
                portfolio_statuses, active_competitions
            )
            
            # 3. ポートフォリオ最適化
            optimization_result = await self.execute_portfolio_optimization(
                active_competitions, self.selection_strategy_type
            )
            
            cycle_duration = (datetime.utcnow() - cycle_start).total_seconds()
            
            return {
                "success": True,
                "cycle_start": cycle_start.isoformat(),
                "cycle_duration_seconds": cycle_duration,
                "active_competitions_found": len(active_competitions),
                "withdrawal_analysis": withdrawal_result,
                "portfolio_optimization": optimization_result,
                "summary": f"{len(active_competitions)}件スキャン、{withdrawal_result.get('withdrawal_recommended', 0)}件撤退推奨、{len(optimization_result.get('changes_made', []))}件ポートフォリオ変更"
            }
            
        except Exception as e:
            self.logger.error(f"監視サイクルエラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def _restore_current_portfolio(self):
        """現在のポートフォリオ復元"""
        
        try:
            # Kaggle から参加中コンペ取得
            user_competitions = await self.kaggle_client.get_user_competitions()
            
            restored_portfolio = []
            for comp_data in user_competitions:
                if comp_data.get("teamCount", 0) > 0:  # 参加中
                    comp_details = await self.kaggle_client.get_competition_details(
                        comp_data.get("ref", "")
                    )
                    if comp_details:
                        restored_portfolio.append(comp_details)
            
            self.current_portfolio = restored_portfolio
            self.logger.info(f"ポートフォリオ復元完了: {len(self.current_portfolio)}件")
            
        except Exception as e:
            self.logger.warning(f"ポートフォリオ復元失敗: {e}")
            self.current_portfolio = []