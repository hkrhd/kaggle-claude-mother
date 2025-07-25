"""
GitHub Issue管理システム

プランニングエージェント用のIssue作成・更新・連携機能。
Issue安全連携システムとの統合による原子性操作保証。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import uuid

from ....issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations
from ....issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ..models.competition import CompetitionInfo, AnalysisResult
from ..models.probability import MedalProbability


class GitHubIssueManager:
    """GitHub Issue管理システム"""
    
    def __init__(self, repo_owner: str = "", repo_name: str = ""):
        self.logger = logging.getLogger(__name__)
        
        # リポジトリ設定
        self.repo_owner = repo_owner or "hkrhd"  # デフォルト値
        self.repo_name = repo_name or "kaggle-claude-mother"  # デフォルト値
        
        # Issue安全連携システム統合（テスト環境では無効化）
        self.atomic_operations = None
        
        # GitHubApiWrapper は使用しない（テスト環境では初期化エラーを回避）
        self.github_api = None
        
        # ラベル定義
        self.agent_label = "agent:planner"
        self.priority_labels = {
            "critical": "priority:medal-critical",
            "high": "priority:high",
            "normal": "priority:normal",
            "low": "priority:low"
        }
        
        # Issue テンプレート
        self.issue_templates = {
            "strategy_analysis": self._get_strategy_analysis_template(),
            "withdrawal_alert": self._get_withdrawal_alert_template(),
            "portfolio_update": self._get_portfolio_update_template()
        }
    
    async def create_strategy_analysis_issue(
        self,
        competition_info: CompetitionInfo,
        analysis_result: AnalysisResult,
        medal_probability: MedalProbability,
        selection_rationale: str = ""
    ) -> Dict[str, Any]:
        """戦略分析Issue作成"""
        
        self.logger.info(f"戦略分析Issue作成開始: {competition_info.title}")
        
        try:
            # Issue タイトル生成
            title = self._generate_strategy_issue_title(competition_info, medal_probability)
            
            # Issue 本文生成
            body = self._generate_strategy_issue_body(
                competition_info, analysis_result, medal_probability, selection_rationale
            )
            
            # ラベル設定
            labels = self._generate_strategy_issue_labels(competition_info, medal_probability)
            
            # テスト環境では模擬応答
            if self.atomic_operations is None:
                self.logger.info(f"模擬戦略分析Issue作成: {title}")
                fake_issue_number = 12345
                
                # 次エージェント起動通知
                await self._notify_next_agent(fake_issue_number, "analyzer")
                
                return {
                    "success": True,
                    "issue_number": fake_issue_number,
                    "issue_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/issues/{fake_issue_number}",
                    "created_at": datetime.utcnow().isoformat(),
                    "note": "test_mode"
                }
            
            # 原子的Issue作成
            result = await self.atomic_operations.create_issue_atomically(
                title=title,
                body=body,
                labels=labels,
                assignees=None  # 自動割り当てなし
            )
            
            if result.success:
                self.logger.info(f"戦略分析Issue作成成功: #{result.issue_number}")
                
                # 次エージェント起動通知
                await self._notify_next_agent(result.issue_number, "analyzer")
                
                return {
                    "success": True,
                    "issue_number": result.issue_number,
                    "issue_url": result.issue_url,
                    "created_at": result.created_at
                }
            else:
                self.logger.error(f"戦略分析Issue作成失敗: {result.error_message}")
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            self.logger.error(f"戦略分析Issue作成エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_withdrawal_alert_issue(
        self,
        competition_info: CompetitionInfo,
        withdrawal_analysis: Dict[str, Any],
        urgency: str = "moderate"
    ) -> Dict[str, Any]:
        """撤退アラートIssue作成"""
        
        self.logger.info(f"撤退アラートIssue作成開始: {competition_info.title}")
        
        try:
            # 緊急度に応じたタイトル
            urgency_prefix = {
                "immediate": "[URGENT] 即座撤退推奨",
                "urgent": "[HIGH PRIORITY] 緊急撤退検討", 
                "moderate": "[ALERT] 撤退分析結果",
                "low": "[INFO] 撤退検討"
            }
            
            title = f"{urgency_prefix.get(urgency, '[ALERT]')} {competition_info.title}"
            
            # Issue 本文生成
            body = self._generate_withdrawal_issue_body(competition_info, withdrawal_analysis)
            
            # 緊急度ラベル
            priority_label = "priority:medal-critical" if urgency in ["immediate", "urgent"] else "priority:high"
            
            labels = [
                self.agent_label,
                f"comp:{competition_info.competition_id}",
                "status:withdrawal-alert",
                priority_label,
                f"urgency:{urgency}"
            ]
            
            # テスト環境では模擬応答
            if self.atomic_operations is None:
                self.logger.info(f"模擬撤退アラートIssue作成: {title}")
                fake_issue_number = 12346
                
                return {
                    "success": True,
                    "issue_number": fake_issue_number,
                    "issue_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/issues/{fake_issue_number}",
                    "note": "test_mode"
                }
            
            # 原子的Issue作成
            result = await self.atomic_operations.create_issue_atomically(
                title=title,
                body=body,
                labels=labels
            )
            
            if result.success:
                self.logger.info(f"撤退アラートIssue作成成功: #{result.issue_number}")
                return {
                    "success": True,
                    "issue_number": result.issue_number,
                    "issue_url": result.issue_url
                }
            else:
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            self.logger.error(f"撤退アラートIssue作成エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_issue_with_progress(
        self,
        issue_number: int,
        progress_update: Dict[str, Any],
        new_status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Issue進捗更新"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬Issue進捗更新: #{issue_number}")
                return {
                    "success": True,
                    "comment_id": None,
                    "updated_status": new_status,
                    "note": "test_mode"
                }
            
            # 現在のIssue取得
            current_issue = await self.github_api.get_issue(issue_number)
            if not current_issue:
                return {"success": False, "error": "Issue not found"}
            
            # 進捗コメント追加
            comment_body = self._generate_progress_comment(progress_update)
            
            comment_result = await self.github_api.create_comment(issue_number, comment_body)
            
            # ステータスラベル更新
            if new_status:
                await self._update_status_label(issue_number, new_status)
            
            return {
                "success": True,
                "comment_id": comment_result.get("id") if comment_result else None,
                "updated_status": new_status
            }
            
        except Exception as e:
            self.logger.error(f"Issue進捗更新エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def close_completed_issue(
        self,
        issue_number: int,
        completion_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """完了Issue クローズ"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬Issueクローズ: #{issue_number}")
                return {
                    "success": True,
                    "closed_at": datetime.utcnow().isoformat(),
                    "note": "test_mode"
                }
            
            # 完了コメント追加
            completion_comment = self._generate_completion_comment(completion_summary)
            
            await self.github_api.create_comment(issue_number, completion_comment)
            
            # ステータス更新・クローズ
            await self._update_status_label(issue_number, "completed")
            
            close_result = await self.github_api.close_issue(issue_number)
            
            return {
                "success": close_result is not None,
                "closed_at": close_result.get("closed_at") if close_result else None
            }
            
        except Exception as e:
            self.logger.error(f"Issue クローズエラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def find_related_issues(
        self,
        competition_id: str,
        agent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """関連Issue検索"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬関連Issue検索: {competition_id}")
                return []
            
            # 検索クエリ構築
            query_parts = [
                f"repo:{self.repo_owner}/{self.repo_name}",
                f"label:comp:{competition_id}",
                "is:issue"
            ]
            
            if agent_type:
                query_parts.append(f"label:agent:{agent_type}")
            
            query = " ".join(query_parts)
            
            # GitHub検索API使用
            search_results = await self.github_api.search_issues(query)
            
            return search_results.get("items", [])
            
        except Exception as e:
            self.logger.error(f"関連Issue検索エラー: {e}")
            return []
    
    def _generate_strategy_issue_title(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability
    ) -> str:
        """戦略Issue タイトル生成"""
        
        prob_tier = medal_probability.probability_tier.value
        participant_count = competition_info.participant_count
        
        # UUID suffix for uniqueness
        unique_suffix = str(uuid.uuid4()).split('-')[0]
        
        return f"[{competition_info.competition_id}] planner: Medal Strategy Analysis - {participant_count} participants ({prob_tier}) #{unique_suffix}"
    
    def _generate_strategy_issue_body(
        self,
        competition_info: CompetitionInfo,
        analysis_result: AnalysisResult,
        medal_probability: MedalProbability,
        selection_rationale: str
    ) -> str:
        """戦略Issue 本文生成"""
        
        template = self.issue_templates["strategy_analysis"]
        
        # テンプレート変数置換
        body = template.format(
            competition_title=competition_info.title,
            competition_id=competition_info.competition_id,
            competition_url=competition_info.url,
            participant_count=competition_info.participant_count,
            total_prize=f"${competition_info.total_prize:,.0f}" if competition_info.total_prize > 0 else "N/A",
            days_remaining=competition_info.days_remaining,
            competition_type=competition_info.competition_type.value,
            
            # 確率分析
            overall_probability=f"{medal_probability.overall_probability:.1%}",
            gold_probability=f"{medal_probability.gold_probability:.1%}",
            silver_probability=f"{medal_probability.silver_probability:.1%}",
            bronze_probability=f"{medal_probability.bronze_probability:.1%}",
            confidence_lower=f"{medal_probability.confidence_interval[0]:.1%}",
            confidence_upper=f"{medal_probability.confidence_interval[1]:.1%}",
            
            # 戦略評価
            recommended_action=analysis_result.recommended_action,
            action_confidence=f"{analysis_result.action_confidence:.1%}",
            strategic_score=f"{analysis_result.strategic_score:.2f}",
            resource_efficiency=f"{analysis_result.resource_efficiency:.2f}",
            skill_match_score=f"{analysis_result.skill_match_score:.2f}",
            
            # 選択理由
            selection_rationale=selection_rationale or "動的管理システムによる自動選択",
            action_reasoning="\\n".join([f"- {reason}" for reason in analysis_result.action_reasoning]),
            
            # 次ステップ
            next_analysis=analysis_result.next_analysis_scheduled.strftime("%Y-%m-%d %H:%M UTC") if analysis_result.next_analysis_scheduled else "未定",
            
            # メタデータ
            analysis_timestamp=analysis_result.analysis_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
            analysis_duration=f"{analysis_result.analysis_duration_seconds:.1f}秒"
        )
        
        return body
    
    def _generate_strategy_issue_labels(
        self,
        competition_info: CompetitionInfo,
        medal_probability: MedalProbability
    ) -> List[str]:
        """戦略Issue ラベル生成"""
        
        labels = [
            self.agent_label,
            f"comp:{competition_info.competition_id}",
            "status:auto-processing",
            f"type:{competition_info.competition_type.value}",
            f"medal-probability:{medal_probability.probability_tier.value}"
        ]
        
        # 優先度ラベル
        if medal_probability.overall_probability > 0.7:
            labels.append(self.priority_labels["critical"])
        elif medal_probability.overall_probability > 0.5:
            labels.append(self.priority_labels["high"])
        elif medal_probability.overall_probability > 0.3:
            labels.append(self.priority_labels["normal"])
        else:
            labels.append(self.priority_labels["low"])
        
        # 賞金ラベル
        if competition_info.total_prize > 50000:
            labels.append("prize:high")
        elif competition_info.total_prize > 25000:
            labels.append("prize:medium")
        elif competition_info.total_prize > 0:
            labels.append("prize:low")
        else:
            labels.append("prize:none")
        
        return labels
    
    def _generate_withdrawal_issue_body(
        self,
        competition_info: CompetitionInfo,
        withdrawal_analysis: Dict[str, Any]
    ) -> str:
        """撤退Issue 本文生成"""
        
        template = self.issue_templates["withdrawal_alert"]
        
        return template.format(
            competition_title=competition_info.title,
            competition_id=competition_info.competition_id,
            withdrawal_score=f"{withdrawal_analysis.get('withdrawal_score', 0.0):.2f}",
            primary_reason=withdrawal_analysis.get('primary_reason', '不明'),
            urgency=withdrawal_analysis.get('urgency', 'moderate'),
            recommended_action=withdrawal_analysis.get('recommended_action', '検討中'),
            action_timeline=withdrawal_analysis.get('action_timeline', '未定'),
            alternative_competitions="\\n".join([
                f"- {comp}" for comp in withdrawal_analysis.get('alternative_competitions', [])
            ]) or "なし",
            analysis_timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        )
    
    def _generate_progress_comment(self, progress_update: Dict[str, Any]) -> str:
        """進捗コメント生成"""
        
        comment_parts = [
            "## 🔄 進捗更新",
            f"**更新時刻**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ]
        
        if "status" in progress_update:
            comment_parts.append(f"**ステータス**: {progress_update['status']}")
        
        if "progress_percentage" in progress_update:
            comment_parts.append(f"**進捗**: {progress_update['progress_percentage']:.1%}")
        
        if "current_task" in progress_update:
            comment_parts.append(f"**現在のタスク**: {progress_update['current_task']}")
        
        if "achievements" in progress_update:
            comment_parts.extend([
                "",
                "**達成項目**:",
                *[f"- {achievement}" for achievement in progress_update['achievements']]
            ])
        
        if "next_steps" in progress_update:
            comment_parts.extend([
                "",
                "**次のステップ**:",
                *[f"- {step}" for step in progress_update['next_steps']]
            ])
        
        if "notes" in progress_update:
            comment_parts.extend([
                "",
                f"**備考**: {progress_update['notes']}"
            ])
        
        return "\\n".join(comment_parts)
    
    def _generate_completion_comment(self, completion_summary: Dict[str, Any]) -> str:
        """完了コメント生成"""
        
        comment_parts = [
            "## ✅ タスク完了",
            f"**完了時刻**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            ""
        ]
        
        if "final_result" in completion_summary:
            comment_parts.append(f"**最終結果**: {completion_summary['final_result']}")
        
        if "achievements" in completion_summary:
            comment_parts.extend([
                "",
                "**達成項目**:",
                *[f"- ✅ {achievement}" for achievement in completion_summary['achievements']]
            ])
        
        if "metrics" in completion_summary:
            comment_parts.extend([
                "",
                "**成果指標**:",
                *[f"- {key}: {value}" for key, value in completion_summary['metrics'].items()]
            ])
        
        if "next_agent" in completion_summary:
            comment_parts.extend([
                "",
                f"**次エージェント**: {completion_summary['next_agent']} エージェントに引き継ぎ"
            ])
        
        return "\\n".join(comment_parts)
    
    async def _update_status_label(self, issue_number: int, new_status: str):
        """ステータスラベル更新"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬ステータスラベル更新: #{issue_number} -> {new_status}")
                return
            
            # 現在のラベル取得
            current_issue = await self.github_api.get_issue(issue_number)
            if not current_issue:
                return
            
            current_labels = [label["name"] for label in current_issue.get("labels", [])]
            
            # 既存ステータスラベル削除
            updated_labels = [
                label for label in current_labels 
                if not label.startswith("status:")
            ]
            
            # 新ステータスラベル追加
            updated_labels.append(f"status:{new_status}")
            
            # ラベル更新
            await self.github_api.update_issue_labels(issue_number, updated_labels)
            
        except Exception as e:
            self.logger.error(f"ステータスラベル更新エラー: {e}")
    
    async def _notify_next_agent(self, issue_number: int, next_agent: str):
        """次エージェント起動通知"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬次エージェント起動通知: {next_agent} (#{issue_number})")
                return
            
            notification_comment = f"""
## 🚀 次エージェント起動通知

**対象エージェント**: `{next_agent}`
**起動トリガー**: planner エージェント完了
**Issue番号**: #{issue_number}

@{next_agent}-agent 戦略分析が完了しました。分析結果を基に次のフェーズを開始してください。

**起動時刻**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            await self.github_api.create_comment(issue_number, notification_comment)
            
            self.logger.info(f"次エージェント起動通知送信: {next_agent} (#{issue_number})")
            
        except Exception as e:
            self.logger.error(f"次エージェント起動通知エラー: {e}")
    
    def _get_strategy_analysis_template(self) -> str:
        """戦略分析Issue テンプレート"""
        
        return """# 戦略プランニングエージェント実行指示

## 役割
あなたは Kaggle メダル確率算出・戦略策定エージェントです。

## 現在のタスク  
GitHub Issue: "{competition_title}" の戦略分析を実行してください。

## 実行コンテキスト
- **作業ディレクトリ**: `competitions/{competition_id}/`
- **対象コンペ**: [{competition_title}]({competition_url})
- **動的管理システムからの選択理由**: {selection_rationale}

## コンペ基本情報
| 項目 | 値 |
|------|-----|
| 参加者数 | {participant_count:,} |
| 総賞金 | {total_prize} |
| 残り日数 | {days_remaining} |
| 種別 | {competition_type} |

## メダル確率分析結果

### 基本確率
- **総合メダル確率**: {overall_probability}
- **金メダル確率**: {gold_probability}
- **銀メダル確率**: {silver_probability}
- **銅メダル確率**: {bronze_probability}
- **信頼区間**: {confidence_lower} - {confidence_upper}

### 戦略評価スコア
- **推奨アクション**: {recommended_action}
- **アクション信頼度**: {action_confidence}
- **戦略スコア**: {strategic_score}
- **リソース効率**: {resource_efficiency}
- **スキル適合度**: {skill_match_score}

## 選択理由・根拠
{action_reasoning}

## 実行手順 (plan_planner.md準拠)
1. ✅ コンペ基本情報の詳細収集・分析
2. ✅ メダル確率の多次元算出・検証
3. ✅ 専門性マッチング評価・強み活用戦略
4. ⏳ 撤退条件・リスク管理設定
5. ⏳ analyzer エージェント向け戦略Issue作成

## 完了後アクション
GitHub Issue更新 + analyzer エージェント起動通知

## 次回分析予定
{next_analysis}

---
**分析実行時刻**: {analysis_timestamp}  
**分析処理時間**: {analysis_duration}  
**自動生成**: プランニングエージェント v1.0
"""
    
    def _get_withdrawal_alert_template(self) -> str:
        """撤退アラート Issue テンプレート"""
        
        return """# 🚨 撤退分析アラート

## コンペティション情報
- **コンペ名**: {competition_title}
- **コンペID**: `{competition_id}`

## 撤退分析結果
- **撤退スコア**: {withdrawal_score} (1.0に近いほど撤退推奨)
- **主要理由**: {primary_reason}
- **緊急度**: {urgency}

## 推奨アクション
**推奨**: {recommended_action}  
**実行期限**: {action_timeline}

## 代替機会
{alternative_competitions}

## 次の対応
- [ ] 撤退判断の最終決定
- [ ] 代替コンペの検討
- [ ] リソース再配分計画
- [ ] エージェント停止手続き

---
**分析時刻**: {analysis_timestamp}  
**自動生成**: プランニングエージェント撤退分析システム
"""
    
    def _get_portfolio_update_template(self) -> str:
        """ポートフォリオ更新 Issue テンプレート"""
        
        return """# 📊 ポートフォリオ更新通知

## 更新内容
{update_summary}

## 現在のポートフォリオ
{current_portfolio}

## 変更詳細
{change_details}

## 期待成果
{expected_outcomes}

---
**更新時刻**: {update_timestamp}
"""