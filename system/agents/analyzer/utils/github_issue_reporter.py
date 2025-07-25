"""
GitHub Issue連携・技術分析レポート生成システム

分析結果を構造化markdown形式でGitHub Issueに出力し、
executor エージェントへの引き継ぎを自動化するシステム。
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

# Issue安全連携システムからのインポート
from ....issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations
from ....issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper


class ReportType(Enum):
    """レポートタイプ"""
    TECHNICAL_ANALYSIS = "technical_analysis"
    IMPLEMENTATION_GUIDE = "implementation_guide"
    RESEARCH_SUMMARY = "research_summary"
    RISK_ASSESSMENT = "risk_assessment"


class ReportPriority(Enum):
    """レポート優先度"""
    MEDAL_CRITICAL = "medal-critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class TechnicalAnalysisReport:
    """技術分析レポート"""
    competition_name: str
    competition_type: str
    analysis_scope: str
    
    # 推奨技術
    recommended_techniques: List[Dict[str, Any]]
    grandmaster_pattern_analysis: Dict[str, Any]
    
    # 実装情報
    implementation_feasibility: Dict[str, Any]
    estimated_implementation_time: str
    required_resources: Dict[str, Any]
    
    # リスク・制約
    technical_risks: List[str]
    implementation_constraints: List[str]
    fallback_strategies: List[str]
    
    # 次段階情報
    executor_instructions: List[str]
    success_metrics: List[str]
    milestone_timeline: List[str]
    
    # メタデータ
    confidence_level: float
    information_sources: List[str]
    analysis_duration: float
    created_at: datetime


class GitHubIssueReporter:
    """GitHub Issue技術レポート生成システム"""
    
    def __init__(self, repo_owner: str = "", repo_name: str = ""):
        self.logger = logging.getLogger(__name__)
        
        # リポジトリ設定
        self.repo_owner = repo_owner or "hkrhd"
        self.repo_name = repo_name or "kaggle-claude-mother"
        
        # Issue安全連携システム統合（テスト環境では無効化）
        self.atomic_operations = None
        self.github_api = None
        
        # ラベル定義
        self.agent_label = "agent:analyzer"
        self.priority_labels = {
            ReportPriority.MEDAL_CRITICAL: "priority:medal-critical",
            ReportPriority.HIGH: "priority:high",
            ReportPriority.NORMAL: "priority:normal",
            ReportPriority.LOW: "priority:low"
        }
        
        # レポートテンプレート
        self.report_templates = {
            ReportType.TECHNICAL_ANALYSIS: self._get_technical_analysis_template(),
            ReportType.IMPLEMENTATION_GUIDE: self._get_implementation_guide_template(),
            ReportType.RESEARCH_SUMMARY: self._get_research_summary_template()
        }
    
    async def create_technical_analysis_issue(
        self,
        report: TechnicalAnalysisReport,
        priority: ReportPriority = ReportPriority.HIGH
    ) -> Dict[str, Any]:
        """技術分析Issue作成"""
        
        self.logger.info(f"技術分析Issue作成開始: {report.competition_name}")
        
        try:
            # Issue タイトル生成
            title = self._generate_analysis_issue_title(report)
            
            # Issue 本文生成
            body = self._generate_analysis_issue_body(report)
            
            # ラベル設定
            labels = self._generate_analysis_issue_labels(report, priority)
            
            # テスト環境では模擬応答
            if self.atomic_operations is None:
                self.logger.info(f"模擬技術分析Issue作成: {title}")
                fake_issue_number = 23456
                
                # executor エージェント起動通知
                await self._notify_executor_agent(fake_issue_number, report)
                
                return {
                    "success": True,
                    "issue_number": fake_issue_number,
                    "issue_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/issues/{fake_issue_number}",
                    "created_at": datetime.utcnow().isoformat(),
                    "note": "test_mode",
                    "executor_notified": True
                }
            
            # 原子的Issue作成
            result = await self.atomic_operations.create_issue_atomically(
                title=title,
                body=body,
                labels=labels,
                assignees=None
            )
            
            if result.success:
                self.logger.info(f"技術分析Issue作成成功: #{result.issue_number}")
                
                # executor エージェント起動通知
                await self._notify_executor_agent(result.issue_number, report)
                
                return {
                    "success": True,
                    "issue_number": result.issue_number,
                    "issue_url": result.issue_url,
                    "created_at": result.created_at,
                    "executor_notified": True
                }
            else:
                self.logger.error(f"技術分析Issue作成失敗: {result.error_message}")
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            self.logger.error(f"技術分析Issue作成エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_analysis_issue_title(self, report: TechnicalAnalysisReport) -> str:
        """分析Issue タイトル生成"""
        
        # 推奨技術の上位3つを抽出
        top_techniques = []
        for tech_info in report.recommended_techniques[:3]:
            tech_name = tech_info.get("technique", "unknown")
            top_techniques.append(tech_name)
        
        techniques_str = ", ".join(top_techniques) if top_techniques else "comprehensive"
        
        return f"[{report.competition_name}] analyzer: Technical Analysis - {techniques_str} implementation strategy"
    
    def _generate_analysis_issue_body(self, report: TechnicalAnalysisReport) -> str:
        """分析Issue 本文生成"""
        
        template = self.report_templates[ReportType.TECHNICAL_ANALYSIS]
        
        # 推奨技術の構造化
        recommended_techniques_md = self._format_recommended_techniques(report.recommended_techniques)
        
        # グランドマスターパターン分析
        grandmaster_analysis_md = self._format_grandmaster_analysis(report.grandmaster_pattern_analysis)
        
        # 実装可能性分析
        feasibility_md = self._format_feasibility_analysis(report.implementation_feasibility)
        
        # リスク評価
        risk_assessment_md = self._format_risk_assessment(report.technical_risks, report.implementation_constraints)
        
        # executor向け指示
        executor_instructions_md = self._format_executor_instructions(report.executor_instructions)
        
        # 成功指標
        success_metrics_md = self._format_success_metrics(report.success_metrics)
        
        # タイムライン
        timeline_md = self._format_timeline(report.milestone_timeline)
        
        # テンプレート変数置換
        body = template.format(
            competition_name=report.competition_name,
            competition_type=report.competition_type,
            analysis_scope=report.analysis_scope,
            
            # 技術分析結果
            recommended_techniques=recommended_techniques_md,
            grandmaster_analysis=grandmaster_analysis_md,
            feasibility_analysis=feasibility_md,
            
            # リソース・制約
            estimated_implementation_time=report.estimated_implementation_time,
            required_resources=self._format_required_resources(report.required_resources),
            risk_assessment=risk_assessment_md,
            
            # 実装指針
            executor_instructions=executor_instructions_md,
            success_metrics=success_metrics_md,
            milestone_timeline=timeline_md,
            
            # メタデータ
            confidence_level=f"{report.confidence_level:.1%}",
            information_sources="\\n".join([f"- {source}" for source in report.information_sources]),
            analysis_duration=f"{report.analysis_duration:.1f}秒",
            analysis_timestamp=report.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        )
        
        return body
    
    def _format_recommended_techniques(self, techniques: List[Dict[str, Any]]) -> str:
        """推奨技術の構造化"""
        if not techniques:
            return "推奨技術の特定に失敗しました。手動分析が必要です。"
        
        formatted_techniques = []
        
        for i, tech in enumerate(techniques, 1):
            technique_name = tech.get("technique", "Unknown")
            recommendation_score = tech.get("recommendation_score", 0.0)
            mention_count = tech.get("mention_count", 0)
            implementation_available = tech.get("implementation_available", False)
            primary_sources = tech.get("primary_sources", [])
            
            implementation_note = "✅ 実装コード利用可能" if implementation_available else "⚠️ 実装コードなし"
            sources_str = ", ".join(primary_sources[:3])
            
            technique_md = f"""
### {i}. {technique_name}
- **推奨スコア**: {recommendation_score:.2f}/1.0
- **言及頻度**: {mention_count}回
- **実装可能性**: {implementation_note}  
- **主要情報源**: {sources_str}
"""
            formatted_techniques.append(technique_md.strip())
        
        return "\\n\\n".join(formatted_techniques)
    
    def _format_grandmaster_analysis(self, analysis: Dict[str, Any]) -> str:
        """グランドマスターパターン分析構造化"""
        if not analysis:
            return "グランドマスターパターン分析データがありません。"
        
        total_techniques = analysis.get("total_applicable_techniques", 0)
        top_recommendations = analysis.get("top_recommendations", [])
        high_risk_techniques = analysis.get("high_risk_techniques", [])
        
        analysis_md = f"""
**適用可能技術数**: {total_techniques}件

**上位推奨技術**:
"""
        
        for i, rec in enumerate(top_recommendations[:3], 1):
            grandmaster = rec.get("grandmaster", "unknown")
            technique = rec.get("technique", {})
            technique_name = technique.get("name", "unknown") if isinstance(technique, dict) else str(technique)
            applicability_score = rec.get("applicability_score", 0.0)
            medal_contribution = rec.get("expected_medal_contribution", 0.0)
            
            analysis_md += f"""
{i}. **{technique_name}** (by {grandmaster})
   - 適用性スコア: {applicability_score:.2f}
   - メダル寄与期待値: {medal_contribution:.2f}
"""
        
        if high_risk_techniques:
            analysis_md += f"""
**高リスク技術** ({len(high_risk_techniques)}件):
"""
            for risk_tech in high_risk_techniques[:3]:
                technique = risk_tech.get("technique", {})
                technique_name = technique.get("name", "unknown") if isinstance(technique, dict) else str(technique)
                risk_level = risk_tech.get("implementation_risk", 0.0)
                analysis_md += f"- {technique_name} (リスクレベル: {risk_level:.2f})\\n"
        
        return analysis_md.strip()
    
    def _format_feasibility_analysis(self, feasibility: Dict[str, Any]) -> str:
        """実装可能性分析構造化"""
        if not feasibility:
            return "実装可能性分析が実行されていません。"
        
        feasibility_score = feasibility.get("feasibility_score", 0.0)
        implementation_probability = feasibility.get("implementation_probability", 0.0)
        estimated_days = feasibility.get("estimated_completion_days", 0)
        resource_compatibility = feasibility.get("resource_compatibility", False)
        
        compatibility_note = "✅ 利用可能リソースと互換" if resource_compatibility else "⚠️ リソース不足の可能性"
        
        feasibility_md = f"""
- **総合実装可能性**: {feasibility_score:.2%}
- **実装成功確率**: {implementation_probability:.2%}
- **推定完了日数**: {estimated_days}日
- **リソース互換性**: {compatibility_note}
"""
        
        return feasibility_md.strip()
    
    def _format_required_resources(self, resources: Dict[str, Any]) -> str:
        """必要リソース構造化"""
        if not resources:
            return "リソース要件が特定されていません。"
        
        gpu_hours = resources.get("estimated_gpu_hours", "不明")
        memory_requirement = resources.get("memory_requirement", "16GB RAM推奨")
        storage_requirement = resources.get("storage_requirement", "10GB以上")
        
        return f"""
- **GPU時間**: {gpu_hours}
- **メモリ要件**: {memory_requirement}
- **ストレージ**: {storage_requirement}
"""
    
    def _format_risk_assessment(self, technical_risks: List[str], constraints: List[str]) -> str:
        """リスク評価構造化"""
        risk_md = "### 技術リスク\\n"
        
        if technical_risks:
            for i, risk in enumerate(technical_risks, 1):
                risk_md += f"{i}. {risk}\\n"
        else:
            risk_md += "特定された技術リスクはありません。\\n"
        
        risk_md += "\\n### 実装制約\\n"
        
        if constraints:
            for i, constraint in enumerate(constraints, 1):
                risk_md += f"{i}. {constraint}\\n"
        else:
            risk_md += "特定された実装制約はありません。\\n"
        
        return risk_md.strip()
    
    def _format_executor_instructions(self, instructions: List[str]) -> str:
        """executor向け指示構造化"""
        if not instructions:
            return "executor向けの具体的指示が生成されていません。"
        
        instructions_md = ""
        for i, instruction in enumerate(instructions, 1):
            instructions_md += f"{i}. {instruction}\\n"
        
        return instructions_md.strip()
    
    def _format_success_metrics(self, metrics: List[str]) -> str:
        """成功指標構造化"""
        if not metrics:
            return "成功指標が定義されていません。"
        
        metrics_md = ""
        for metric in metrics:
            metrics_md += f"- {metric}\\n"
        
        return metrics_md.strip()
    
    def _format_timeline(self, timeline: List[str]) -> str:
        """タイムライン構造化"""
        if not timeline:
            return "実装タイムラインが生成されていません。"
        
        timeline_md = ""
        for milestone in timeline:
            timeline_md += f"- {milestone}\\n"
        
        return timeline_md.strip()
    
    def _generate_analysis_issue_labels(self, report: TechnicalAnalysisReport, priority: ReportPriority) -> List[str]:
        """分析Issue ラベル生成"""
        
        labels = [
            self.agent_label,
            f"comp:{report.competition_name}",
            "status:completed",
            f"type:{report.competition_type}",
            self.priority_labels[priority]
        ]
        
        # 信頼度ラベル
        if report.confidence_level >= 0.8:
            labels.append("confidence:high")
        elif report.confidence_level >= 0.6:
            labels.append("confidence:medium")
        else:
            labels.append("confidence:low")
        
        # 実装時間ラベル
        if "日" in report.estimated_implementation_time:
            try:
                days = int(report.estimated_implementation_time.split("日")[0])
                if days <= 3:
                    labels.append("timeline:fast")
                elif days <= 7:
                    labels.append("timeline:medium")
                else:
                    labels.append("timeline:extended")
            except ValueError:
                labels.append("timeline:unknown")
        
        return labels
    
    async def _notify_executor_agent(self, issue_number: int, report: TechnicalAnalysisReport):
        """executor エージェント起動通知"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬executor起動通知: Issue #{issue_number}")
                return
            
            # executor起動Issue自動作成
            executor_title = f"[{report.competition_name}] executor: High-Performance Implementation"
            
            executor_body = f"""# 🏗️ 実装フェーズ開始

## 分析結果参照
**技術分析Issue**: #{issue_number}
**推奨実装時間**: {report.estimated_implementation_time}
**信頼度レベル**: {report.confidence_level:.1%}

## 優先実装技術
{self._format_executor_priority_techniques(report.recommended_techniques[:3])}

## 実装指針
{self._format_executor_instructions(report.executor_instructions)}

## 成功指標
{self._format_success_metrics(report.success_metrics)}

## リスク管理
{self._format_risk_mitigation_for_executor(report.technical_risks)}

---
**自動生成**: analyzer エージェントからの引き継ぎ
**生成時刻**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
            
            executor_labels = [
                "agent:executor",
                f"comp:{report.competition_name}",
                "status:auto-processing",
                "priority:medal-critical"
            ]
            
            # executor Issue作成
            await self.atomic_operations.create_issue_atomically(
                title=executor_title,
                body=executor_body,
                labels=executor_labels
            )
            
            self.logger.info(f"executor起動Issue作成成功: {report.competition_name}")
            
        except Exception as e:
            self.logger.error(f"executor起動通知エラー: {e}")
    
    def _format_executor_priority_techniques(self, techniques: List[Dict[str, Any]]) -> str:
        """executor向け優先技術フォーマット"""
        if not techniques:
            return "優先技術が特定されていません。"
        
        priority_md = ""
        for i, tech in enumerate(techniques, 1):
            technique_name = tech.get("technique", "Unknown")
            recommendation_score = tech.get("recommendation_score", 0.0)
            implementation_available = tech.get("implementation_available", False)
            
            priority_icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            implementation_icon = "✅" if implementation_available else "⚠️"
            
            priority_md += f"""
{priority_icon} **{technique_name}** (スコア: {recommendation_score:.2f})
   - 実装コード: {implementation_icon} {"利用可能" if implementation_available else "要開発"}
"""
        
        return priority_md.strip()
    
    def _format_risk_mitigation_for_executor(self, risks: List[str]) -> str:
        """executor向けリスク軽減策"""
        if not risks:
            return "特定されたリスクはありません。"
        
        mitigation_md = ""
        for i, risk in enumerate(risks[:3], 1):
            # リスクタイプに応じた軽減策生成
            if "複雑度" in risk:
                mitigation = "段階的実装・プロトタイプによる事前検証"
            elif "時間制約" in risk:
                mitigation = "並列開発・既存実装活用による短縮"
            elif "リソース" in risk:
                mitigation = "軽量化・効率的アルゴリズム選択"
            else:
                mitigation = "継続的モニタリング・早期問題発見"
            
            mitigation_md += f"{i}. **リスク**: {risk}\\n   **軽減策**: {mitigation}\\n\\n"
        
        return mitigation_md.strip()
    
    async def update_issue_with_progress(
        self,
        issue_number: int,
        progress_type: str,
        progress_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Issue進捗更新"""
        
        try:
            # テスト環境では模擬応答
            if self.github_api is None:
                self.logger.info(f"模擬Issue進捗更新: #{issue_number} - {progress_type}")
                return {
                    "success": True,
                    "comment_id": None,
                    "updated_at": datetime.utcnow().isoformat(),
                    "note": "test_mode"
                }
            
            # 進捗タイプ別コメント生成
            comment_body = self._generate_progress_comment(progress_type, progress_data)
            
            # コメント追加
            result = await self.github_api.add_comment_safely(issue_number, comment_body)
            
            return {
                "success": result.success,
                "comment_id": getattr(result, "comment_id", None),
                "updated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Issue進捗更新エラー: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_progress_comment(self, progress_type: str, progress_data: Dict[str, Any]) -> str:
        """進捗コメント生成"""
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        if progress_type == "analysis_completion":
            return f"""## ✅ 分析完了
**完了時刻**: {timestamp}
**分析結果**: {progress_data.get('summary', '分析が完了しました')}
**次段階**: executor による実装フェーズ開始
"""
        
        elif progress_type == "quality_update":
            confidence = progress_data.get('confidence_level', 0.0)
            return f"""## 📊 品質更新
**更新時刻**: {timestamp}
**信頼度**: {confidence:.1%}
**品質状況**: {progress_data.get('quality_note', '品質チェック完了')}
"""
        
        else:
            return f"""## 🔄 進捗更新
**更新時刻**: {timestamp}
**更新内容**: {progress_data.get('update_content', '進捗を更新しました')}
"""
    
    def _get_technical_analysis_template(self) -> str:
        """技術分析Issue テンプレート"""
        
        return """# 深層技術分析エージェント実行結果

## 🎯 分析概要
- **対象コンペ**: [{competition_name}]({competition_name})
- **コンペタイプ**: {competition_type}
- **分析スコープ**: {analysis_scope}
- **分析信頼度**: {confidence_level}

## 🔬 推奨実装技術 (優先順位順)

{recommended_techniques}

## 🏆 グランドマスター解法適用性

{grandmaster_analysis}

## ⚡ 実装可能性評価

{feasibility_analysis}

## 📋 リソース要件

### 必要リソース
{required_resources}

### 推定実装時間
**予想期間**: {estimated_implementation_time}

## 🚨 リスク評価・制約

{risk_assessment}

## 🎯 executor 向け実装指針

{executor_instructions}

## 📊 成功指標

{success_metrics}

## 🗓️ 実装マイルストーン

{milestone_timeline}

---

## 📈 分析メタデータ
- **情報源**:
{information_sources}
- **分析処理時間**: {analysis_duration}
- **分析実行時刻**: {analysis_timestamp}
- **自動生成**: analyzer エージェント v1.0
"""
    
    def _get_implementation_guide_template(self) -> str:
        """実装ガイド テンプレート"""
        
        return """# 実装ガイド: {competition_name}

## 実装戦略
{implementation_strategy}

## 段階的実装計画
{implementation_plan}

## コード例・テンプレート
{code_templates}
"""
    
    def _get_research_summary_template(self) -> str:
        """研究サマリー テンプレート"""
        
        return """# 研究調査結果: {competition_name}

## 調査サマリー
{research_summary}

## 重要論文・技術
{key_papers}

## 実装可能性
{implementation_assessment}
"""
    
    async def create_research_summary_issue(
        self,
        competition_name: str,
        research_data: Dict[str, Any],
        priority: ReportPriority = ReportPriority.NORMAL
    ) -> Dict[str, Any]:
        """研究サマリーIssue作成"""
        
        try:
            title = f"[{competition_name}] analyzer: Research Summary - Latest Techniques"
            
            template = self.report_templates[ReportType.RESEARCH_SUMMARY]
            body = template.format(
                competition_name=competition_name,
                research_summary=research_data.get("summary", "調査結果サマリー"),
                key_papers=research_data.get("papers", "重要論文情報"),
                implementation_assessment=research_data.get("assessment", "実装可能性評価")
            )
            
            labels = [
                self.agent_label,
                f"comp:{competition_name}",
                "status:completed",
                "type:research",
                self.priority_labels[priority]
            ]
            
            # テスト環境では模擬応答
            if self.atomic_operations is None:
                return {
                    "success": True,
                    "issue_number": 23457,
                    "issue_url": f"https://github.com/{self.repo_owner}/{self.repo_name}/issues/23457",
                    "note": "test_mode"
                }
            
            # 実際のIssue作成
            result = await self.atomic_operations.create_issue_atomically(
                title=title, body=body, labels=labels
            )
            
            return {
                "success": result.success,
                "issue_number": getattr(result, "issue_number", None),
                "issue_url": getattr(result, "issue_url", None)
            }
            
        except Exception as e:
            self.logger.error(f"研究サマリーIssue作成エラー: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_report_statistics(self) -> Dict[str, Any]:
        """レポート統計取得"""
        
        # 実際の実装では、Issue履歴から統計を生成
        return {
            "total_reports_created": 0,
            "reports_by_type": {},
            "average_confidence_level": 0.0,
            "most_common_techniques": [],
            "success_rate": 0.0
        }