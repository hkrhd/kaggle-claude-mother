"""
高度反省エージェント

Kaggle競技終了後の全活動振り返り・成功失敗分析・改善点特定・
知識蓄積によるシステム継続学習を実現するメインエージェント。
"""

import asyncio
import logging
import json
import uuid
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import re

# GitHub Issue安全システム
from ...issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ...issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations

# 他エージェントとの連携
from ..analyzer.analyzer_agent import AnalyzerAgent
from ..executor.executor_agent import ExecutorAgent
from ..monitor.monitor_agent import MonitorAgent


class RetrospectiveDepth(Enum):
    """振り返り深度"""
    SURFACE = "surface"      # 表面的：基本統計のみ
    STANDARD = "standard"    # 標準：主要パターン分析
    DEEP = "deep"           # 深層：詳細因子分析
    COMPREHENSIVE = "comprehensive"  # 包括的：全面的分析


class LearningCategory(Enum):
    """学習カテゴリ"""
    TECHNIQUE_EFFECTIVENESS = "technique_effectiveness"  # 技術有効性
    RESOURCE_OPTIMIZATION = "resource_optimization"     # リソース最適化
    TIME_MANAGEMENT = "time_management"                  # 時間管理
    ERROR_PATTERNS = "error_patterns"                    # エラーパターン
    SUCCESS_FACTORS = "success_factors"                  # 成功要因
    COMPETITION_PATTERNS = "competition_patterns"        # 競技パターン


class ImprovementPriority(Enum):
    """改善優先度"""
    CRITICAL = "critical"    # 重要：次回必須改善
    HIGH = "high"           # 高：早期改善推奨
    MEDIUM = "medium"       # 中：中期改善検討
    LOW = "low"            # 低：長期改善候補


@dataclass
class CompetitionSummary:
    """競技サマリー"""
    competition_name: str
    competition_type: str
    start_date: datetime
    end_date: datetime
    deadline: datetime
    
    # 参加状況
    participants_count: int
    final_ranking: Optional[int] = None
    final_percentile: Optional[float] = None
    medal_achieved: Optional[str] = None  # gold, silver, bronze, None
    
    # スコア推移
    initial_score: float = 0.0
    best_score: float = 0.0
    final_score: float = 0.0
    score_improvement: float = 0.0
    
    # リソース使用
    total_gpu_hours_used: float = 0.0
    total_api_calls: int = 0
    total_experiments: int = 0
    
    # 実装状況
    techniques_identified: int = 0
    techniques_attempted: int = 0
    techniques_successful: int = 0


@dataclass
class TechniqueAnalysis:
    """技術分析結果"""
    technique_name: str
    category: str
    
    # 実行統計
    attempts_count: int
    success_count: int
    failure_count: int
    success_rate: float
    
    # パフォーマンス
    avg_score_improvement: float
    best_score_achieved: float
    avg_execution_time_hours: float
    total_gpu_hours_used: float
    
    # 有効性評価
    effectiveness_score: float  # 0.0-1.0
    difficulty_score: float     # 0.0-1.0
    roi_score: float           # return on investment
    
    # 失敗要因
    common_failures: List[str] = field(default_factory=list)
    implementation_challenges: List[str] = field(default_factory=list)
    
    # 成功要因
    success_factors: List[str] = field(default_factory=list)
    optimal_conditions: List[str] = field(default_factory=list)


@dataclass
class LearningInsight:
    """学習知見"""
    insight_id: str
    category: LearningCategory
    title: str
    description: str
    
    # エビデンス
    supporting_data: Dict[str, Any]
    confidence_level: float  # 0.0-1.0
    statistical_significance: float
    
    # 適用範囲
    applicable_competition_types: List[str]
    applicable_contexts: List[str]
    
    # 改善提案
    improvement_recommendations: List[str]
    priority: ImprovementPriority
    
    # メタ情報
    discovered_at: datetime
    validated: bool = False
    applied_count: int = 0


@dataclass
class RetrospectiveReport:
    """振り返りレポート"""
    report_id: str
    competition_summary: CompetitionSummary
    generated_at: datetime
    analysis_depth: RetrospectiveDepth
    
    # 全体評価
    overall_performance: str  # excellent, good, fair, poor
    key_achievements: List[str]
    major_failures: List[str]
    
    # 技術分析
    technique_analyses: List[TechniqueAnalysis]
    
    # 学習知見
    new_insights: List[LearningInsight]
    validated_insights: List[LearningInsight]
    
    # リソース分析
    resource_efficiency: Dict[str, float]
    time_allocation_analysis: Dict[str, float]
    
    # 改善提案
    immediate_improvements: List[str]
    strategic_improvements: List[str]
    
    # 次回適用事項
    lessons_for_next_competition: List[str]
    updated_best_practices: List[str]


class RetrospectiveAgent:
    """高度反省エージェント - メインクラス"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        
        # エージェント情報
        self.agent_id = f"retrospective-{uuid.uuid4().hex[:8]}"
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
        
        # 知識蓄積データベース
        self.historical_insights: List[LearningInsight] = []
        self.technique_knowledge: Dict[str, TechniqueAnalysis] = {}
        self.competition_patterns: Dict[str, Any] = {}
        
        # 分析設定
        self.default_analysis_depth = RetrospectiveDepth.STANDARD
        self.confidence_threshold = 0.7
        self.statistical_significance_threshold = 0.05
        
        # 振り返り履歴
        self.retrospective_history: List[RetrospectiveReport] = []
    
    async def conduct_competition_retrospective(
        self,
        competition_data: Dict[str, Any],
        agent_histories: Dict[str, List[Any]],
        analysis_depth: RetrospectiveDepth = None
    ) -> RetrospectiveReport:
        """競技振り返り実行"""
        
        analysis_depth = analysis_depth or self.default_analysis_depth
        
        self.logger.info(f"競技振り返り開始: {competition_data.get('name', 'Unknown')} (深度: {analysis_depth.value})")
        
        # GitHub Issue作成: 振り返り開始通知
        retrospective_issue = await self._create_retrospective_issue(
            competition_data, analysis_depth
        )
        
        try:
            # 1. 競技サマリー生成
            competition_summary = await self._generate_competition_summary(
                competition_data, agent_histories
            )
            
            # 2. 技術分析実行
            technique_analyses = await self._analyze_techniques(
                agent_histories.get("executor", []), analysis_depth
            )
            
            # 3. 学習知見抽出
            new_insights = await self._extract_learning_insights(
                competition_summary, technique_analyses, agent_histories, analysis_depth
            )
            
            # 4. 既存知見検証
            validated_insights = await self._validate_existing_insights(
                competition_summary, technique_analyses
            )
            
            # 5. リソース効率性分析
            resource_analysis = await self._analyze_resource_efficiency(
                agent_histories, competition_summary
            )
            
            # 6. 改善提案生成
            improvements = await self._generate_improvement_recommendations(
                competition_summary, technique_analyses, new_insights
            )
            
            # 7. 振り返りレポート作成
            report = RetrospectiveReport(
                report_id=f"retro-{uuid.uuid4().hex[:8]}",
                competition_summary=competition_summary,
                generated_at=datetime.utcnow(),
                analysis_depth=analysis_depth,
                overall_performance=self._evaluate_overall_performance(competition_summary),
                key_achievements=improvements["achievements"],
                major_failures=improvements["failures"],
                technique_analyses=technique_analyses,
                new_insights=new_insights,
                validated_insights=validated_insights,
                resource_efficiency=resource_analysis["efficiency"],
                time_allocation_analysis=resource_analysis["time_allocation"],
                immediate_improvements=improvements["immediate"],
                strategic_improvements=improvements["strategic"],
                lessons_for_next_competition=improvements["lessons"],
                updated_best_practices=improvements["best_practices"]
            )
            
            # 8. 知識ベース更新
            await self._update_knowledge_base(report)
            
            # 9. レポート投稿
            await self._post_retrospective_report(report, retrospective_issue)
            
            # 10. 履歴に追加
            self.retrospective_history.append(report)
            
            self.logger.info(f"競技振り返り完了: {report.report_id}")
            return report
            
        except Exception as e:
            self.logger.error(f"競技振り返りエラー: {e}")
            await self._post_error_report(e, retrospective_issue)
            raise
    
    async def _generate_competition_summary(
        self,
        competition_data: Dict[str, Any],
        agent_histories: Dict[str, List[Any]]
    ) -> CompetitionSummary:
        """競技サマリー生成"""
        
        # 基本情報抽出
        name = competition_data.get("name", "Unknown Competition")
        comp_type = competition_data.get("type", "tabular")
        start_date = datetime.fromisoformat(competition_data.get("start_date", datetime.utcnow().isoformat()))
        end_date = datetime.fromisoformat(competition_data.get("end_date", datetime.utcnow().isoformat()))
        deadline = datetime.fromisoformat(competition_data.get("deadline", datetime.utcnow().isoformat()))
        
        # 実行履歴から統計抽出
        executor_history = agent_histories.get("executor", [])
        analyzer_history = agent_histories.get("analyzer", [])
        
        # スコア統計
        all_scores = []
        total_gpu_hours = 0.0
        total_experiments = 0
        
        for execution in executor_history:
            if hasattr(execution, 'best_score') and execution.best_score > 0:
                all_scores.append(execution.best_score)
            if hasattr(execution, 'total_gpu_hours_used'):
                total_gpu_hours += execution.total_gpu_hours_used
            if hasattr(execution, 'total_experiments_run'):
                total_experiments += execution.total_experiments_run
        
        # 技術統計
        techniques_identified = 0
        techniques_attempted = 0
        techniques_successful = 0
        
        for analysis in analyzer_history:
            if hasattr(analysis, 'recommended_techniques'):
                techniques_identified += len(analysis.recommended_techniques)
        
        for execution in executor_history:
            if hasattr(execution, 'techniques_to_implement'):
                techniques_attempted += len(execution.techniques_to_implement)
            if hasattr(execution, 'success_rate') and execution.success_rate > 0.5:
                techniques_successful += 1
        
        summary = CompetitionSummary(
            competition_name=name,
            competition_type=comp_type,
            start_date=start_date,
            end_date=end_date,
            deadline=deadline,
            participants_count=competition_data.get("participants", 1000),
            final_ranking=competition_data.get("final_ranking"),
            final_percentile=competition_data.get("final_percentile"),
            medal_achieved=competition_data.get("medal"),
            initial_score=all_scores[0] if all_scores else 0.0,
            best_score=max(all_scores) if all_scores else 0.0,
            final_score=all_scores[-1] if all_scores else 0.0,
            score_improvement=max(all_scores) - all_scores[0] if len(all_scores) > 1 else 0.0,
            total_gpu_hours_used=total_gpu_hours,
            total_api_calls=sum(getattr(h, 'api_calls_count', 0) for h in analyzer_history),
            total_experiments=total_experiments,
            techniques_identified=techniques_identified,
            techniques_attempted=techniques_attempted,
            techniques_successful=techniques_successful
        )
        
        return summary
    
    async def _analyze_techniques(
        self,
        executor_history: List[Any],
        analysis_depth: RetrospectiveDepth
    ) -> List[TechniqueAnalysis]:
        """技術分析"""
        
        technique_stats = {}
        
        # 実行履歴から技術別統計収集
        for execution in executor_history:
            if not hasattr(execution, 'techniques_to_implement'):
                continue
                
            for technique_info in execution.techniques_to_implement:
                technique_name = technique_info.get("technique", "unknown")
                
                if technique_name not in technique_stats:
                    technique_stats[technique_name] = {
                        "attempts": 0,
                        "successes": 0,
                        "failures": 0,
                        "scores": [],
                        "execution_times": [],
                        "gpu_hours": [],
                        "failure_reasons": [],
                        "success_factors": []
                    }
                
                stats = technique_stats[technique_name]
                stats["attempts"] += 1
                
                # 成功/失敗判定
                execution_success = getattr(execution, 'success_rate', 0) > 0.5
                if execution_success:
                    stats["successes"] += 1
                    if hasattr(execution, 'best_score'):
                        stats["scores"].append(execution.best_score)
                    stats["success_factors"].append("Execution completed successfully")
                else:
                    stats["failures"] += 1
                    stats["failure_reasons"].append("Low success rate")
                
                # メトリクス収集
                if hasattr(execution, 'execution_duration'):
                    stats["execution_times"].append(execution.execution_duration)
                if hasattr(execution, 'total_gpu_hours_used'):
                    stats["gpu_hours"].append(execution.total_gpu_hours_used)
        
        # 技術分析結果生成
        analyses = []
        for technique_name, stats in technique_stats.items():
            
            success_rate = stats["successes"] / max(stats["attempts"], 1)
            avg_score = statistics.mean(stats["scores"]) if stats["scores"] else 0.0
            avg_execution_time = statistics.mean(stats["execution_times"]) if stats["execution_times"] else 0.0
            total_gpu = sum(stats["gpu_hours"])
            
            # 有効性スコア計算
            effectiveness = self._calculate_technique_effectiveness(
                success_rate, avg_score, avg_execution_time, total_gpu
            )
            
            # 難易度スコア計算
            difficulty = 1.0 - success_rate  # 成功率が低いほど難易度高
            
            # ROIスコア計算
            roi = avg_score / max(total_gpu, 0.1)  # スコア改善/GPU時間
            
            analysis = TechniqueAnalysis(
                technique_name=technique_name,
                category=self._categorize_technique(technique_name),
                attempts_count=stats["attempts"],
                success_count=stats["successes"],
                failure_count=stats["failures"],
                success_rate=success_rate,
                avg_score_improvement=avg_score,
                best_score_achieved=max(stats["scores"]) if stats["scores"] else 0.0,
                avg_execution_time_hours=avg_execution_time,
                total_gpu_hours_used=total_gpu,
                effectiveness_score=effectiveness,
                difficulty_score=difficulty,
                roi_score=roi,
                common_failures=stats["failure_reasons"][:3],
                success_factors=stats["success_factors"][:3]
            )
            
            analyses.append(analysis)
        
        # 有効性順でソート
        analyses.sort(key=lambda a: a.effectiveness_score, reverse=True)
        
        return analyses
    
    def _calculate_technique_effectiveness(
        self,
        success_rate: float,
        avg_score: float,
        avg_time: float,
        total_gpu: float
    ) -> float:
        """技術有効性スコア計算"""
        
        # 各要素を0-1に正規化
        success_component = success_rate
        score_component = min(avg_score / 0.9, 1.0)  # 0.9を最大と仮定
        time_efficiency = max(0, 1.0 - avg_time / 24.0)  # 24時間を最大と仮定
        gpu_efficiency = max(0, 1.0 - total_gpu / 10.0)  # 10時間を最大と仮定
        
        # 重み付き合計
        effectiveness = (
            success_component * 0.4 +
            score_component * 0.3 +
            time_efficiency * 0.15 +
            gpu_efficiency * 0.15
        )
        
        return min(effectiveness, 1.0)
    
    def _categorize_technique(self, technique_name: str) -> str:
        """技術カテゴライズ"""
        
        name_lower = technique_name.lower()
        
        if "ensemble" in name_lower or "boosting" in name_lower:
            return "ensemble"
        elif "stacking" in name_lower or "blending" in name_lower:
            return "stacking"
        elif "neural" in name_lower or "deep" in name_lower:
            return "deep_learning"
        elif "feature" in name_lower:
            return "feature_engineering"
        elif "optimization" in name_lower or "optuna" in name_lower:
            return "hyperparameter_optimization"
        else:
            return "other"
    
    async def _extract_learning_insights(
        self,
        competition_summary: CompetitionSummary,
        technique_analyses: List[TechniqueAnalysis],
        agent_histories: Dict[str, List[Any]],
        analysis_depth: RetrospectiveDepth
    ) -> List[LearningInsight]:
        """学習知見抽出"""
        
        insights = []
        
        # 1. 技術有効性に関する知見
        top_techniques = [t for t in technique_analyses if t.effectiveness_score > 0.7]
        if top_techniques:
            insight = LearningInsight(
                insight_id=f"insight-tech-eff-{uuid.uuid4().hex[:6]}",
                category=LearningCategory.TECHNIQUE_EFFECTIVENESS,
                title=f"高効果技術パターン発見: {competition_summary.competition_type}競技",
                description=f"以下の技術が{competition_summary.competition_type}競技で高い効果を発揮: {', '.join([t.technique_name for t in top_techniques[:3]])}",
                supporting_data={
                    "techniques": [t.technique_name for t in top_techniques],
                    "avg_effectiveness": statistics.mean([t.effectiveness_score for t in top_techniques]),
                    "avg_success_rate": statistics.mean([t.success_rate for t in top_techniques])
                },
                confidence_level=min(statistics.mean([t.success_rate for t in top_techniques]), 1.0),
                statistical_significance=0.05,  # 仮の値
                applicable_competition_types=[competition_summary.competition_type],
                applicable_contexts=["similar_dataset_size", "similar_timeline"],
                improvement_recommendations=[
                    f"次回{competition_summary.competition_type}競技では{top_techniques[0].technique_name}を優先実装",
                    "高効果技術のハイパーパラメータ最適化に重点配分"
                ],
                priority=ImprovementPriority.HIGH,
                discovered_at=datetime.utcnow()
            )
            insights.append(insight)
        
        # 2. リソース最適化に関する知見
        if competition_summary.total_gpu_hours_used > 0:
            gpu_efficiency = competition_summary.score_improvement / competition_summary.total_gpu_hours_used
            
            if gpu_efficiency > 0.01:  # 1GPU時間あたり0.01以上のスコア改善
                insight = LearningInsight(
                    insight_id=f"insight-resource-{uuid.uuid4().hex[:6]}",
                    category=LearningCategory.RESOURCE_OPTIMIZATION,
                    title="効率的GPU活用パターン確認",
                    description=f"GPU時間当たりスコア改善効率: {gpu_efficiency:.4f}",
                    supporting_data={
                        "gpu_hours_used": competition_summary.total_gpu_hours_used,
                        "score_improvement": competition_summary.score_improvement,
                        "efficiency": gpu_efficiency
                    },
                    confidence_level=0.8,
                    statistical_significance=0.05,
                    applicable_competition_types=[competition_summary.competition_type],
                    applicable_contexts=["limited_gpu_budget"],
                    improvement_recommendations=[
                        "同様の効率性を次回も維持",
                        "更なる効率化の余地を探索"
                    ],
                    priority=ImprovementPriority.MEDIUM,
                    discovered_at=datetime.utcnow()
                )
                insights.append(insight)
        
        # 3. エラーパターンに関する知見
        failed_techniques = [t for t in technique_analyses if t.success_rate < 0.5]
        if failed_techniques:
            common_failures = []
            for tech in failed_techniques:
                common_failures.extend(tech.common_failures)
            
            if common_failures:
                insight = LearningInsight(
                    insight_id=f"insight-error-{uuid.uuid4().hex[:6]}",
                    category=LearningCategory.ERROR_PATTERNS,
                    title="共通失敗パターン特定",
                    description=f"以下のエラーパターンが複数技術で発生: {', '.join(set(common_failures[:3]))}",
                    supporting_data={
                        "failed_techniques": [t.technique_name for t in failed_techniques],
                        "common_failures": list(set(common_failures)),
                        "failure_frequency": len(common_failures)
                    },
                    confidence_level=0.7,
                    statistical_significance=0.05,
                    applicable_competition_types=[competition_summary.competition_type],
                    applicable_contexts=["similar_complexity"],
                    improvement_recommendations=[
                        "事前にこれらのエラーパターンへの対策を準備",
                        "失敗しやすい技術には追加のテスト時間を確保"
                    ],
                    priority=ImprovementPriority.HIGH,
                    discovered_at=datetime.utcnow()
                )
                insights.append(insight)
        
        return insights
    
    async def _validate_existing_insights(
        self,
        competition_summary: CompetitionSummary,
        technique_analyses: List[TechniqueAnalysis]
    ) -> List[LearningInsight]:
        """既存知見検証"""
        
        validated_insights = []
        
        for insight in self.historical_insights:
            if insight.validated:
                continue
                
            # 適用可能性チェック
            if competition_summary.competition_type not in insight.applicable_competition_types:
                continue
            
            # 検証実行
            validation_result = await self._validate_insight(
                insight, competition_summary, technique_analyses
            )
            
            if validation_result["confirmed"]:
                insight.validated = True
                insight.applied_count += 1
                validated_insights.append(insight)
                
                self.logger.info(f"知見検証成功: {insight.title}")
        
        return validated_insights
    
    async def _validate_insight(
        self,
        insight: LearningInsight,
        competition_summary: CompetitionSummary,
        technique_analyses: List[TechniqueAnalysis]
    ) -> Dict[str, Any]:
        """個別知見検証"""
        
        if insight.category == LearningCategory.TECHNIQUE_EFFECTIVENESS:
            # 技術有効性知見の検証
            recommended_techniques = insight.supporting_data.get("techniques", [])
            
            for tech_analysis in technique_analyses:
                if (tech_analysis.technique_name in recommended_techniques and
                    tech_analysis.effectiveness_score >= insight.confidence_level):
                    return {
                        "confirmed": True,
                        "evidence": f"{tech_analysis.technique_name}が期待通りの効果を発揮",
                        "confidence": tech_analysis.effectiveness_score
                    }
        
        elif insight.category == LearningCategory.RESOURCE_OPTIMIZATION:
            # リソース最適化知見の検証
            expected_efficiency = insight.supporting_data.get("efficiency", 0)
            actual_efficiency = (
                competition_summary.score_improvement / 
                max(competition_summary.total_gpu_hours_used, 0.1)
            )
            
            if actual_efficiency >= expected_efficiency * 0.8:  # 80%以上の効率達成
                return {
                    "confirmed": True,
                    "evidence": f"期待効率{expected_efficiency:.4f}に対し{actual_efficiency:.4f}を達成",
                    "confidence": min(actual_efficiency / expected_efficiency, 1.0)
                }
        
        return {
            "confirmed": False,
            "evidence": "検証条件を満たさず",
            "confidence": 0.0
        }
    
    async def _analyze_resource_efficiency(
        self,
        agent_histories: Dict[str, List[Any]],
        competition_summary: CompetitionSummary
    ) -> Dict[str, Any]:
        """リソース効率性分析"""
        
        # GPU効率性
        gpu_efficiency = 0.0
        if competition_summary.total_gpu_hours_used > 0:
            gpu_efficiency = competition_summary.score_improvement / competition_summary.total_gpu_hours_used
        
        # 時間効率性
        total_competition_hours = (competition_summary.end_date - competition_summary.start_date).total_seconds() / 3600
        time_utilization = competition_summary.total_gpu_hours_used / max(total_competition_hours, 1)
        
        # API効率性
        api_efficiency = 0.0
        if competition_summary.total_api_calls > 0:
            api_efficiency = competition_summary.techniques_identified / competition_summary.total_api_calls
        
        # 実験効率性
        experiment_efficiency = 0.0
        if competition_summary.total_experiments > 0:
            experiment_efficiency = competition_summary.techniques_successful / competition_summary.total_experiments
        
        efficiency_analysis = {
            "gpu_efficiency": gpu_efficiency,
            "time_utilization": time_utilization,
            "api_efficiency": api_efficiency,
            "experiment_efficiency": experiment_efficiency
        }
        
        # 時間配分分析
        time_allocation = {
            "analysis_phase": 0.2,      # 分析20%
            "implementation_phase": 0.6, # 実装60%
            "optimization_phase": 0.2    # 最適化20%
        }
        
        return {
            "efficiency": efficiency_analysis,
            "time_allocation": time_allocation
        }
    
    async def _generate_improvement_recommendations(
        self,
        competition_summary: CompetitionSummary,
        technique_analyses: List[TechniqueAnalysis],
        new_insights: List[LearningInsight]
    ) -> Dict[str, List[str]]:
        """改善提案生成"""
        
        achievements = []
        failures = []
        immediate_improvements = []
        strategic_improvements = []
        lessons = []
        best_practices = []
        
        # 成果の特定
        if competition_summary.score_improvement > 0.05:
            achievements.append(f"大幅スコア改善達成: {competition_summary.score_improvement:.4f}")
        if competition_summary.medal_achieved:
            achievements.append(f"メダル獲得: {competition_summary.medal_achieved}")
        
        high_performing_techniques = [t for t in technique_analyses if t.effectiveness_score > 0.8]
        if high_performing_techniques:
            achievements.append(f"高効果技術の成功実装: {', '.join([t.technique_name for t in high_performing_techniques[:2]])}")
        
        # 失敗の特定
        if competition_summary.score_improvement < 0.01:
            failures.append("スコア改善不足")
        if competition_summary.techniques_successful / max(competition_summary.techniques_attempted, 1) < 0.5:
            failures.append("技術実装成功率低下")
        
        low_performing_techniques = [t for t in technique_analyses if t.effectiveness_score < 0.3]
        if low_performing_techniques:
            failures.append(f"低効果技術の選択: {', '.join([t.technique_name for t in low_performing_techniques[:2]])}")
        
        # 即座改善事項
        for technique in low_performing_techniques:
            immediate_improvements.append(f"{technique.technique_name}の実装方法見直し")
        
        if competition_summary.total_gpu_hours_used > 40:  # 40時間以上使用
            immediate_improvements.append("GPU使用量の最適化")
        
        # 戦略的改善事項
        for insight in new_insights:
            if insight.priority in [ImprovementPriority.CRITICAL, ImprovementPriority.HIGH]:
                strategic_improvements.extend(insight.improvement_recommendations)
        
        # 次回への教訓
        lessons.extend([
            f"最優先技術: {technique_analyses[0].technique_name if technique_analyses else 'なし'}",
            f"GPU予算配分: {competition_summary.total_gpu_hours_used:.1f}時間が適正"
        ])
        
        # ベストプラクティス更新
        for technique in high_performing_techniques:
            best_practices.append(f"{technique.technique_name}: {', '.join(technique.success_factors[:2])}")
        
        return {
            "achievements": achievements,
            "failures": failures,
            "immediate": immediate_improvements,
            "strategic": strategic_improvements,
            "lessons": lessons,
            "best_practices": best_practices
        }
    
    def _evaluate_overall_performance(self, competition_summary: CompetitionSummary) -> str:
        """全体パフォーマンス評価"""
        
        score = 0
        
        # スコア改善
        if competition_summary.score_improvement > 0.1:
            score += 3
        elif competition_summary.score_improvement > 0.05:
            score += 2
        elif competition_summary.score_improvement > 0.01:
            score += 1
        
        # メダル獲得
        if competition_summary.medal_achieved == "gold":
            score += 3
        elif competition_summary.medal_achieved == "silver":
            score += 2
        elif competition_summary.medal_achieved == "bronze":
            score += 1
        
        # 技術成功率
        if competition_summary.techniques_attempted > 0:
            success_rate = competition_summary.techniques_successful / competition_summary.techniques_attempted
            if success_rate > 0.8:
                score += 2
            elif success_rate > 0.6:
                score += 1
        
        # 評価変換
        if score >= 6:
            return "excellent"
        elif score >= 4:
            return "good"
        elif score >= 2:
            return "fair"
        else:
            return "poor"
    
    async def _update_knowledge_base(self, report: RetrospectiveReport):
        """知識ベース更新"""
        
        # 新しい知見を追加
        self.historical_insights.extend(report.new_insights)
        
        # 技術知識更新
        for analysis in report.technique_analyses:
            if analysis.technique_name in self.technique_knowledge:
                # 既存データとマージ
                existing = self.technique_knowledge[analysis.technique_name]
                # 簡単な平均化（実際にはより洗練された更新が必要）
                existing.effectiveness_score = (existing.effectiveness_score + analysis.effectiveness_score) / 2
                existing.attempts_count += analysis.attempts_count
            else:
                self.technique_knowledge[analysis.technique_name] = analysis
        
        # 競技パターン更新
        comp_type = report.competition_summary.competition_type
        if comp_type not in self.competition_patterns:
            self.competition_patterns[comp_type] = []
        
        self.competition_patterns[comp_type].append({
            "score_improvement": report.competition_summary.score_improvement,
            "gpu_hours_used": report.competition_summary.total_gpu_hours_used,
            "success_rate": report.competition_summary.techniques_successful / max(report.competition_summary.techniques_attempted, 1),
            "date": report.generated_at
        })
        
        self.logger.info(f"知識ベース更新完了: {len(report.new_insights)}新知見追加")
    
    async def _create_retrospective_issue(
        self,
        competition_data: Dict[str, Any],
        analysis_depth: RetrospectiveDepth
    ) -> int:
        """振り返りIssue作成"""
        
        title = f"🔍 競技振り返り分析: {competition_data.get('name', 'Unknown')}"
        
        description = f"""
## 競技振り返り分析開始

**競技名**: {competition_data.get('name', 'Unknown')}
**分析エージェントID**: `{self.agent_id}`
**分析深度**: {analysis_depth.value}
**開始時刻**: {datetime.utcnow().isoformat()}

### 分析項目
- 競技全体サマリー生成
- 技術別効果分析
- 学習知見抽出・検証
- リソース効率性評価
- 改善提案生成
- 知識ベース更新

このIssueで振り返り分析の進捗と結果をレポートします。
        """
        
        try:
            issue_data = await self.atomic_operations.create_issue(
                title=title,
                description=description,
                labels=["retrospective", "analysis", "learning"]
            )
            return issue_data["number"]
            
        except Exception as e:
            self.logger.error(f"振り返りIssue作成失敗: {e}")
            return -1
    
    async def _post_retrospective_report(self, report: RetrospectiveReport, issue_number: int):
        """振り返りレポート投稿"""
        
        performance_emoji = {
            "excellent": "🏆",
            "good": "✅", 
            "fair": "⚠️",
            "poor": "❌"
        }
        
        emoji = performance_emoji.get(report.overall_performance, "📊")
        
        content = f"""
## {emoji} 競技振り返り分析完了

### 📈 全体評価: {report.overall_performance.upper()}

### 🎯 競技サマリー
- **競技名**: {report.competition_summary.competition_name}
- **競技タイプ**: {report.competition_summary.competition_type}
- **スコア改善**: {report.competition_summary.score_improvement:.4f}
- **最高スコア**: {report.competition_summary.best_score:.4f}
- **GPU使用時間**: {report.competition_summary.total_gpu_hours_used:.1f}時間
- **実験数**: {report.competition_summary.total_experiments}
- **技術成功率**: {report.competition_summary.techniques_successful}/{report.competition_summary.techniques_attempted}

### 🏆 主な成果
{chr(10).join([f"- {achievement}" for achievement in report.key_achievements]) if report.key_achievements else "- なし"}

### ❌ 主な課題
{chr(10).join([f"- {failure}" for failure in report.major_failures]) if report.major_failures else "- なし"}

### 🔬 技術分析結果
{chr(10).join([f"- **{t.technique_name}**: 有効性{t.effectiveness_score:.2f}, 成功率{t.success_rate:.1%}" for t in report.technique_analyses[:5]]) if report.technique_analyses else "- 分析データなし"}

### 💡 新発見知見
{chr(10).join([f"- **{i.title}**: {i.description}" for i in report.new_insights]) if report.new_insights else "- 新知見なし"}

### 🎯 即座改善事項
{chr(10).join([f"- {improvement}" for improvement in report.immediate_improvements]) if report.immediate_improvements else "- なし"}

### 📚 次回への教訓
{chr(10).join([f"- {lesson}" for lesson in report.lessons_for_next_competition]) if report.lessons_for_next_competition else "- なし"}

### 📊 リソース効率性
- **GPU効率**: {report.resource_efficiency.get('gpu_efficiency', 0):.4f}
- **時間活用率**: {report.resource_efficiency.get('time_utilization', 0):.2f}
- **実験効率**: {report.resource_efficiency.get('experiment_efficiency', 0):.2f}

---
*レポートID: {report.report_id} | Retrospective Agent {self.agent_id}*
        """
        
        try:
            await self.atomic_operations.create_comment(
                issue_number=issue_number,
                comment_body=content
            )
            self.logger.info(f"振り返りレポート投稿完了: Issue #{issue_number}")
            
        except Exception as e:
            self.logger.error(f"振り返りレポート投稿失敗: {e}")
    
    async def _post_error_report(self, error: Exception, issue_number: int):
        """エラーレポート投稿"""
        
        content = f"""
## ❌ 振り返り分析エラー

**エラー発生時刻**: {datetime.utcnow().isoformat()}
**エラー内容**: {str(error)}

振り返り分析中にエラーが発生しました。システム管理者に確認を依頼してください。

---
*Retrospective Agent {self.agent_id}*
        """
        
        try:
            await self.atomic_operations.create_comment(
                issue_number=issue_number,
                comment_body=content
            )
        except Exception as e:
            self.logger.error(f"エラーレポート投稿失敗: {e}")
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """学習サマリー取得"""
        
        return {
            "agent_id": self.agent_id,
            "total_retrospectives": len(self.retrospective_history),
            "total_insights": len(self.historical_insights),
            "validated_insights": len([i for i in self.historical_insights if i.validated]),
            "technique_knowledge_count": len(self.technique_knowledge),
            "competition_patterns_tracked": len(self.competition_patterns),
            "most_effective_techniques": sorted(
                self.technique_knowledge.items(),
                key=lambda x: x[1].effectiveness_score,
                reverse=True
            )[:3],
            "recent_learning_insights": [
                {
                    "title": insight.title,
                    "category": insight.category.value,
                    "confidence": insight.confidence_level,
                    "priority": insight.priority.value
                }
                for insight in self.historical_insights[-5:]
            ]
        }