"""
深層分析エージェント

グランドマスター級技術調査・最新手法研究・実装可能性判定を統合し、
executor エージェント向けの具体的実装戦略を生成するメインエージェント。
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# 知識ベース・分析システム
from .knowledge_base.grandmaster_patterns import GrandmasterPatterns, CompetitionType
from .analyzers.technical_feasibility import TechnicalFeasibilityAnalyzer, TechnicalSpecification, TechnicalComplexity as ImplementationComplexity
from .collectors.kaggle_solutions import KaggleSolutionCollector
from .collectors.arxiv_papers import ArxivPaperCollector
from .utils.web_scraper import WebSearchIntegrator, SearchStrategy
from .utils.github_issue_reporter import GitHubIssueReporter, TechnicalAnalysisReport, ReportPriority


class AnalysisPhase(Enum):
    """分析フェーズ"""
    INITIALIZATION = "initialization"
    GRANDMASTER_ANALYSIS = "grandmaster_analysis"
    SOLUTION_COLLECTION = "solution_collection"
    PAPER_RESEARCH = "paper_research"
    WEB_INVESTIGATION = "web_investigation"
    FEASIBILITY_ANALYSIS = "feasibility_analysis"
    INTEGRATION = "integration"
    REPORTING = "reporting"
    COMPLETED = "completed"


class AnalysisScope(Enum):
    """分析スコープ"""
    QUICK = "quick"        # 30分以内
    STANDARD = "standard"  # 2時間以内
    COMPREHENSIVE = "comprehensive"  # 制限なし


@dataclass
class AnalysisRequest:
    """分析リクエスト"""
    competition_name: str
    competition_type: str  # "tabular", "computer_vision", "nlp", "time_series"
    participant_count: int
    days_remaining: int
    scope: AnalysisScope
    planner_context: Optional[Dict[str, Any]] = None
    issue_number: Optional[int] = None


@dataclass
class AnalysisResult:
    """分析結果"""
    request: AnalysisRequest
    analysis_id: str
    
    # 各分析段階の結果
    grandmaster_analysis: Dict[str, Any]
    kaggle_solutions: List[Any]
    arxiv_papers: List[Any]
    web_investigation: Dict[str, Any]
    feasibility_assessment: Dict[str, Any]
    
    # 統合結果
    recommended_techniques: List[Dict[str, Any]]
    implementation_roadmap: List[str]
    risk_assessment: List[str]
    confidence_level: float
    
    # メタデータ
    analysis_duration: float
    phases_completed: List[AnalysisPhase]
    information_sources_count: int
    created_at: datetime


class AnalyzerAgent:
    """深層分析エージェント - メインクラス"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # エージェント情報
        self.agent_id = f"analyzer-{uuid.uuid4().hex[:8]}"
        self.agent_version = "1.0.0"
        self.start_time = datetime.utcnow()
        
        # 分析コンポーネント初期化
        self.grandmaster_patterns = GrandmasterPatterns()
        self.feasibility_analyzer = TechnicalFeasibilityAnalyzer()
        self.kaggle_collector = KaggleSolutionCollector()
        self.arxiv_collector = ArxivPaperCollector()
        self.web_integrator = WebSearchIntegrator()
        self.issue_reporter = GitHubIssueReporter()
        
        # 分析履歴
        self.analysis_history: List[AnalysisResult] = []
        self.current_analysis: Optional[AnalysisResult] = None
        
        # 設定
        self.max_concurrent_analyses = 1
    
    async def execute_comprehensive_analysis(
        self,
        request: AnalysisRequest
    ) -> AnalysisResult:
        """包括的技術分析実行"""
        
        analysis_start = datetime.utcnow()
        analysis_id = f"analysis-{uuid.uuid4().hex[:8]}"
        
        self.logger.info(f"包括分析開始: {request.competition_name} (ID: {analysis_id})")
        
        try:
            # 分析結果オブジェクト初期化
            result = AnalysisResult(
                request=request,
                analysis_id=analysis_id,
                grandmaster_analysis={},
                kaggle_solutions=[],
                arxiv_papers=[],
                web_investigation={},
                feasibility_assessment={},
                recommended_techniques=[],
                implementation_roadmap=[],
                risk_assessment=[],
                confidence_level=0.0,
                analysis_duration=0.0,
                phases_completed=[],
                information_sources_count=0,
                created_at=analysis_start
            )
            
            self.current_analysis = result
            
            # 分析フェーズ順次実行
            await self._execute_analysis_phase(result, AnalysisPhase.INITIALIZATION)
            await self._execute_analysis_phase(result, AnalysisPhase.GRANDMASTER_ANALYSIS)
            
            # スコープに応じた並列分析
            if request.scope in [AnalysisScope.STANDARD, AnalysisScope.COMPREHENSIVE]:
                # 並列情報収集
                await self._execute_parallel_information_collection(result)
            else:
                # 簡易収集
                await self._execute_analysis_phase(result, AnalysisPhase.SOLUTION_COLLECTION)
            
            # 統合分析
            await self._execute_analysis_phase(result, AnalysisPhase.FEASIBILITY_ANALYSIS)
            await self._execute_analysis_phase(result, AnalysisPhase.INTEGRATION)
            await self._execute_analysis_phase(result, AnalysisPhase.REPORTING)
            
            # 完了処理
            result.analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()
            result.phases_completed.append(AnalysisPhase.COMPLETED)
            
            # 分析履歴に追加
            self.analysis_history.append(result)
            self.current_analysis = None
            
            self.logger.info(f"包括分析完了: {result.analysis_duration:.1f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"包括分析エラー: {e}")
            # エラー時も基本情報は返す
            result.analysis_duration = (datetime.utcnow() - analysis_start).total_seconds()
            result.risk_assessment.append(f"分析エラー: {str(e)}")
            return result
    
    async def _execute_analysis_phase(self, result: AnalysisResult, phase: AnalysisPhase):
        """分析フェーズ実行"""
        
        phase_start = datetime.utcnow()
        self.logger.info(f"フェーズ開始: {phase.value}")
        
        try:
            if phase == AnalysisPhase.INITIALIZATION:
                await self._phase_initialization(result)
            
            elif phase == AnalysisPhase.GRANDMASTER_ANALYSIS:
                await self._phase_grandmaster_analysis(result)
            
            elif phase == AnalysisPhase.SOLUTION_COLLECTION:
                await self._phase_solution_collection(result)
            
            elif phase == AnalysisPhase.PAPER_RESEARCH:
                await self._phase_paper_research(result)
            
            elif phase == AnalysisPhase.WEB_INVESTIGATION:
                await self._phase_web_investigation(result)
            
            elif phase == AnalysisPhase.FEASIBILITY_ANALYSIS:
                await self._phase_feasibility_analysis(result)
            
            elif phase == AnalysisPhase.INTEGRATION:
                await self._phase_integration(result)
            
            elif phase == AnalysisPhase.REPORTING:
                await self._phase_reporting(result)
            
            # フェーズ完了記録
            result.phases_completed.append(phase)
            phase_duration = (datetime.utcnow() - phase_start).total_seconds()
            self.logger.info(f"フェーズ完了: {phase.value} ({phase_duration:.1f}秒)")
            
        except Exception as e:
            self.logger.error(f"フェーズエラー: {phase.value} - {e}")
            result.risk_assessment.append(f"フェーズ{phase.value}でエラー: {str(e)}")
    
    async def _execute_parallel_information_collection(self, result: AnalysisResult):
        """並列情報収集実行"""
        
        self.logger.info("並列情報収集開始")
        
        # 並列実行タスク作成
        tasks = [
            self._execute_analysis_phase(result, AnalysisPhase.SOLUTION_COLLECTION),
            self._execute_analysis_phase(result, AnalysisPhase.PAPER_RESEARCH),
            self._execute_analysis_phase(result, AnalysisPhase.WEB_INVESTIGATION)
        ]
        
        # 並列実行（例外は各フェーズで処理）
        await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("並列情報収集完了")
    
    async def _phase_initialization(self, result: AnalysisResult):
        """初期化フェーズ"""
        
        request = result.request
        
        # コンペタイプの正規化
        competition_type_map = {
            "tabular": CompetitionType.TABULAR,
            "computer_vision": CompetitionType.COMPUTER_VISION,
            "nlp": CompetitionType.NLP,
            "time_series": CompetitionType.TIME_SERIES
        }
        
        result.competition_type_enum = competition_type_map.get(
            request.competition_type, CompetitionType.TABULAR
        )
        
        # 基本分析パラメータ設定
        result.analysis_parameters = {
            "max_techniques": 8 if request.scope == AnalysisScope.COMPREHENSIVE else 5,
            "time_limit_minutes": 120 if request.scope == AnalysisScope.COMPREHENSIVE else 30,
            "min_confidence_threshold": 0.6,
            "require_implementation_code": request.scope != AnalysisScope.QUICK
        }
        
        self.logger.info(f"初期化完了: {request.competition_name}, タイプ: {request.competition_type}")
    
    async def _phase_grandmaster_analysis(self, result: AnalysisResult):
        """グランドマスターパターン分析フェーズ"""
        
        request = result.request
        
        # グランドマスターパターン適用性分析
        gm_analysis = await self.grandmaster_patterns.analyze_pattern_applicability(
            competition_type=result.competition_type_enum,
            participant_count=request.participant_count,
            days_remaining=request.days_remaining,
            available_gpu_hours=24.0
        )
        
        result.grandmaster_analysis = gm_analysis
        result.information_sources_count += 1
        
        self.logger.info(f"グランドマスター分析完了: {gm_analysis['total_applicable_techniques']}技術")
    
    async def _phase_solution_collection(self, result: AnalysisResult):
        """Kaggle解法収集フェーズ"""
        
        request = result.request
        
        # Kaggle優勝解法収集
        solutions = await self.kaggle_collector.collect_competition_solutions(
            competition_name=request.competition_name,
            competition_type=request.competition_type,
            max_solutions=10 if request.scope == AnalysisScope.COMPREHENSIVE else 5
        )
        
        result.kaggle_solutions = solutions
        result.information_sources_count += len(solutions)
        
        self.logger.info(f"Kaggle解法収集完了: {len(solutions)}件")
    
    async def _phase_paper_research(self, result: AnalysisResult):
        """論文研究フェーズ"""
        
        request = result.request
        
        # arXiv最新論文収集
        papers = await self.arxiv_collector.collect_latest_papers(
            competition_domain=request.competition_type,
            days_back=30,
            max_papers=15 if request.scope == AnalysisScope.COMPREHENSIVE else 8
        )
        
        result.arxiv_papers = papers
        result.information_sources_count += len(papers)
        
        self.logger.info(f"論文研究完了: {len(papers)}件")
    
    async def _phase_web_investigation(self, result: AnalysisResult):
        """Web調査フェーズ"""
        
        request = result.request
        
        # WebSearch統合調査
        investigation_report = await self.web_integrator.conduct_comprehensive_investigation(
            competition_name=request.competition_name,
            competition_domain=request.competition_type,
            investigation_scope="standard" if request.scope == AnalysisScope.COMPREHENSIVE else "quick",
            time_limit_minutes=result.analysis_parameters["time_limit_minutes"] // 2
        )
        
        result.web_investigation = {
            "total_results": investigation_report.total_results,
            "high_quality_results": investigation_report.high_quality_results,
            "key_findings": investigation_report.key_findings,
            "recommended_techniques": investigation_report.recommended_techniques,
            "confidence_level": investigation_report.confidence_level
        }
        
        result.information_sources_count += investigation_report.total_results
        
        self.logger.info(f"Web調査完了: {investigation_report.total_results}件")
    
    async def _phase_feasibility_analysis(self, result: AnalysisResult):
        """実装可能性分析フェーズ"""
        
        request = result.request
        
        # 主要技術の実装可能性分析
        feasibility_results = []
        
        # グランドマスター推奨技術から上位3つ
        gm_techniques = result.grandmaster_analysis.get("top_recommendations", [])[:3]
        
        for gm_rec in gm_techniques:
            technique = gm_rec.get("technique", {})
            if isinstance(technique, dict):
                technique_name = technique.get("name", "unknown")
                
                # 技術仕様作成
                tech_spec = TechnicalSpecification(
                    name=technique_name,
                    description=f"グランドマスター推奨: {technique_name}",
                    complexity=self._map_to_technical_complexity(technique.get("implementation_difficulty", "moderate")),
                    estimated_implementation_hours=technique.get("estimated_implementation_days", 5) * 8,
                    required_libraries=technique.get("key_libraries", ["scikit-learn"]),
                    gpu_memory_gb=16.0 if technique.get("gpu_requirement", "optional") != "optional" else 0.0,
                    cpu_cores_min=2,
                    ram_gb_min=8.0,
                    disk_space_gb=5.0,
                    implementation_difficulty_factors=[],
                    common_pitfalls=[],
                    success_indicators=[]
                )
                
                # 実装可能性分析実行
                feasibility = await self.feasibility_analyzer.analyze_technique_feasibility(
                    technique_spec=tech_spec,
                    available_days=request.days_remaining,
                    current_skill_level=0.7
                )
                
                feasibility_results.append({
                    "technique": technique_name,
                    "feasibility_result": feasibility,
                    "source": "grandmaster"
                })
        
        result.feasibility_assessment = {
            "technique_assessments": feasibility_results,
            "overall_feasibility": self._calculate_overall_feasibility(feasibility_results),
            "implementation_recommendations": [
                fa["technique"] for fa in feasibility_results 
                if fa["feasibility_result"].feasibility_score > 0.6
            ]
        }
        
        self.logger.info(f"実装可能性分析完了: {len(feasibility_results)}技術")
    
    def _map_to_technical_complexity(self, difficulty_str: str):
        """難易度文字列を技術複雑度にマッピング"""
        mapping = {
            "easy": ImplementationComplexity.SIMPLE,
            "moderate": ImplementationComplexity.MODERATE,
            "hard": ImplementationComplexity.COMPLEX,
            "expert": ImplementationComplexity.RESEARCH
        }
        return mapping.get(difficulty_str, ImplementationComplexity.MODERATE)
    
    def _calculate_overall_feasibility(self, feasibility_results: List[Dict[str, Any]]) -> float:
        """全体実装可能性算出"""
        if not feasibility_results:
            return 0.0
        
        scores = [fr["feasibility_result"].feasibility_score for fr in feasibility_results]
        return sum(scores) / len(scores)
    
    async def _phase_integration(self, result: AnalysisResult):
        """統合分析フェーズ"""
        
        # 各情報源からの技術推奨を統合
        all_techniques = []
        
        # グランドマスター技術
        gm_techniques = result.grandmaster_analysis.get("top_recommendations", [])
        for gm_rec in gm_techniques:
            technique = gm_rec.get("technique", {})
            if isinstance(technique, dict):
                all_techniques.append({
                    "technique": technique.get("name", "unknown"),
                    "source": "grandmaster",
                    "score": gm_rec.get("applicability_score", 0.0),
                    "confidence": 0.9
                })
        
        # Web調査技術
        web_techniques = result.web_investigation.get("recommended_techniques", [])
        for web_rec in web_techniques:
            all_techniques.append({
                "technique": web_rec.get("technique", "unknown"),
                "source": "web_investigation", 
                "score": web_rec.get("recommendation_score", 0.0),
                "confidence": result.web_investigation.get("confidence_level", 0.5)
            })
        
        # 技術統合・重複排除・スコア正規化
        integrated_techniques = self._integrate_technique_recommendations(all_techniques)
        
        # 上位技術選択
        max_techniques = result.analysis_parameters["max_techniques"]
        result.recommended_techniques = integrated_techniques[:max_techniques]
        
        # 実装ロードマップ生成
        result.implementation_roadmap = await self._generate_implementation_roadmap(
            result.recommended_techniques, result.request.days_remaining
        )
        
        # リスク評価統合
        result.risk_assessment = await self._integrate_risk_assessment(result)
        
        # 信頼度算出
        result.confidence_level = self._calculate_overall_confidence(result)
        
        self.logger.info(f"統合分析完了: {len(result.recommended_techniques)}技術推奨")
    
    def _integrate_technique_recommendations(self, all_techniques: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """技術推奨統合"""
        
        # 技術名で集約
        technique_aggregation = {}
        
        for tech in all_techniques:
            name = tech["technique"]
            if name not in technique_aggregation:
                technique_aggregation[name] = {
                    "technique": name,
                    "sources": [],
                    "scores": [],
                    "confidences": [],
                    "mention_count": 0
                }
            
            agg = technique_aggregation[name]
            agg["sources"].append(tech["source"])
            agg["scores"].append(tech["score"])
            agg["confidences"].append(tech["confidence"])
            agg["mention_count"] += 1
        
        # 統合スコア算出
        integrated_techniques = []
        for name, agg in technique_aggregation.items():
            # 重み付き統合スコア
            avg_score = sum(agg["scores"]) / len(agg["scores"])
            avg_confidence = sum(agg["confidences"]) / len(agg["confidences"])
            mention_bonus = min(agg["mention_count"] / 3.0, 0.2)  # 複数言及ボーナス
            
            integrated_score = avg_score * avg_confidence + mention_bonus
            
            integrated_techniques.append({
                "technique": name,
                "integrated_score": integrated_score,
                "sources": list(set(agg["sources"])),
                "mention_count": agg["mention_count"],
                "avg_confidence": avg_confidence
            })
        
        # スコア順ソート
        integrated_techniques.sort(key=lambda x: x["integrated_score"], reverse=True)
        
        return integrated_techniques
    
    async def _generate_implementation_roadmap(
        self, recommended_techniques: List[Dict[str, Any]], days_remaining: int
    ) -> List[str]:
        """実装ロードマップ生成"""
        
        roadmap = []
        
        if not recommended_techniques:
            return ["技術推奨不足のため手動計画が必要"]
        
        # 3段階実装計画
        high_priority = recommended_techniques[:2]  # 上位2技術
        medium_priority = recommended_techniques[2:4] if len(recommended_techniques) > 2 else []
        
        # フェーズ1: 高優先度技術
        if high_priority:
            estimated_days_phase1 = min(days_remaining * 0.4, 7)
            roadmap.append(f"フェーズ1: 主力技術実装 ({estimated_days_phase1:.0f}日)")
            for i, tech in enumerate(high_priority, 1):
                roadmap.append(f"  {i}. {tech['technique']} (統合スコア: {tech['integrated_score']:.2f})")
        
        # フェーズ2: 中優先度技術・最適化
        if medium_priority:
            estimated_days_phase2 = min(days_remaining * 0.3, 5)
            roadmap.append(f"フェーズ2: 補完技術・最適化 ({estimated_days_phase2:.0f}日)")
            for i, tech in enumerate(medium_priority, 1):
                roadmap.append(f"  {i}. {tech['technique']} (統合スコア: {tech['integrated_score']:.2f})")
        
        # フェーズ3: 統合・調整
        remaining_days = max(days_remaining * 0.3, 2)
        roadmap.append(f"フェーズ3: 統合・最終調整 ({remaining_days:.0f}日)")
        roadmap.append("  1. 技術統合・アンサンブル構築")
        roadmap.append("  2. ハイパーパラメータ最適化")
        roadmap.append("  3. 最終検証・提出準備")
        
        return roadmap
    
    async def _integrate_risk_assessment(self, result: AnalysisResult) -> List[str]:
        """リスク評価統合"""
        
        risks = []
        
        # 情報収集品質リスク
        if result.information_sources_count < 10:
            risks.append("情報収集不足による判断精度低下リスク")
        
        # 技術複雑度リスク
        high_complexity_count = sum(
            1 for tech in result.recommended_techniques
            if "ensemble" in tech["technique"].lower() or "neural" in tech["technique"].lower()
        )
        if high_complexity_count > len(result.recommended_techniques) * 0.5:
            risks.append("高複雑度技術による実装困難リスク")
        
        # 時間制約リスク
        if result.request.days_remaining < 14:
            risks.append("時間制約による機能削減リスク")
        
        # 実装可能性リスク
        feasible_techniques = result.feasibility_assessment.get("implementation_recommendations", [])
        if len(feasible_techniques) < len(result.recommended_techniques) * 0.6:
            risks.append("実装可能性不足による計画変更リスク")
        
        # Web調査信頼度リスク
        web_confidence = result.web_investigation.get("confidence_level", 0.0)
        if web_confidence < 0.6:
            risks.append("情報信頼度不足による方向性ミスリスク")
        
        return risks[:5]  # 最大5リスク
    
    def _calculate_overall_confidence(self, result: AnalysisResult) -> float:
        """全体信頼度算出"""
        
        confidence_factors = []
        
        # 情報源数による信頼度
        source_confidence = min(result.information_sources_count / 20.0, 1.0)
        confidence_factors.append(source_confidence)
        
        # グランドマスター分析信頼度
        gm_applicable_count = result.grandmaster_analysis.get("total_applicable_techniques", 0)
        gm_confidence = min(gm_applicable_count / 5.0, 1.0)
        confidence_factors.append(gm_confidence)
        
        # Web調査信頼度
        web_confidence = result.web_investigation.get("confidence_level", 0.5)
        confidence_factors.append(web_confidence)
        
        # 実装可能性信頼度
        feasibility_confidence = result.feasibility_assessment.get("overall_feasibility", 0.5)
        confidence_factors.append(feasibility_confidence)
        
        # 統合技術数による信頼度
        technique_confidence = min(len(result.recommended_techniques) / 5.0, 1.0)
        confidence_factors.append(technique_confidence)
        
        # 重み付き平均
        weights = [0.2, 0.25, 0.2, 0.2, 0.15]  # 合計1.0
        weighted_confidence = sum(cf * w for cf, w in zip(confidence_factors, weights))
        
        return min(weighted_confidence, 0.95)  # 最大95%
    
    async def _phase_reporting(self, result: AnalysisResult):
        """レポート生成フェーズ"""
        
        # 技術分析レポート作成
        report = TechnicalAnalysisReport(
            competition_name=result.request.competition_name,
            competition_type=result.request.competition_type,
            analysis_scope=result.request.scope.value,
            
            recommended_techniques=result.recommended_techniques,
            grandmaster_pattern_analysis=result.grandmaster_analysis,
            
            implementation_feasibility=result.feasibility_assessment,
            estimated_implementation_time=f"{result.request.days_remaining}日以内",
            required_resources={
                "estimated_gpu_hours": f"{result.request.days_remaining * 2}時間",
                "memory_requirement": "16GB RAM推奨",
                "storage_requirement": "10GB以上の空き容量"
            },
            
            technical_risks=result.risk_assessment,
            implementation_constraints=[
                f"残り日数: {result.request.days_remaining}日",
                f"参加者数: {result.request.participant_count:,}名（高競合）",
                "GPU時間制限: 1日8時間"
            ],
            fallback_strategies=[
                "段階的実装（基本機能優先）",
                "既存ライブラリ活用による開発短縮",
                "保守的手法への切り替え"
            ],
            
            executor_instructions=await self._generate_executor_instructions(result),
            success_metrics=await self._generate_success_metrics(result),
            milestone_timeline=result.implementation_roadmap,
            
            confidence_level=result.confidence_level,
            information_sources=[
                f"グランドマスターパターン: {result.grandmaster_analysis.get('total_applicable_techniques', 0)}技術",
                f"Kaggle解法: {len(result.kaggle_solutions)}件",
                f"arXiv論文: {len(result.arxiv_papers)}件",
                f"Web調査: {result.web_investigation.get('total_results', 0)}件"
            ],
            analysis_duration=result.analysis_duration,
            created_at=result.created_at
        )
        
        # GitHub Issue作成
        issue_result = await self.issue_reporter.create_technical_analysis_issue(
            report=report,
            priority=ReportPriority.MEDAL_CRITICAL if result.confidence_level > 0.8 else ReportPriority.HIGH
        )
        
        result.github_issue = issue_result
        
        self.logger.info(f"レポート生成完了: Issue #{issue_result.get('issue_number', 'N/A')}")
    
    async def _generate_executor_instructions(self, result: AnalysisResult) -> List[str]:
        """executor向け指示生成"""
        
        instructions = []
        
        # 優先技術の実装指示
        for i, tech in enumerate(result.recommended_techniques[:3], 1):
            instructions.append(
                f"{i}. {tech['technique']} を優先実装 "
                f"(統合スコア: {tech['integrated_score']:.2f}, "
                f"情報源: {', '.join(tech['sources'])})"
            )
        
        # 実装戦略指示
        if result.confidence_level > 0.8:
            instructions.append("高信頼度分析結果のため積極的実装を推奨")
        else:
            instructions.append("信頼度制限のため保守的実装・段階的検証を推奨")
        
        # リソース効率指示
        if result.request.days_remaining < 7:
            instructions.append("時間制約により既存実装・ライブラリの最大活用")
        
        # 実装可能性に基づく指示
        feasible_count = len(result.feasibility_assessment.get("implementation_recommendations", []))
        if feasible_count < 3:
            instructions.append("実装可能技術が限定的のため代替手法の並行準備")
        
        return instructions
    
    async def _generate_success_metrics(self, result: AnalysisResult) -> List[str]:
        """成功指標生成"""
        
        metrics = [
            f"分析信頼度: {result.confidence_level:.1%}以上の維持",
            f"推奨技術実装: {len(result.recommended_techniques[:3])}技術の80%以上完了",
            "メダル圏内順位: 参加者上位10%以内",
            f"実装期限: {result.request.days_remaining}日以内の完了"
        ]
        
        # 競合レベルに応じた目標調整
        if result.request.participant_count > 2000:
            metrics.append("高競合対応: 上位5%以内の目標設定")
        
        return metrics
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """エージェント状態取得"""
        
        uptime = (datetime.utcnow() - self.start_time).total_seconds() / 3600
        
        return {
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "uptime_hours": uptime,
            "analysis_history_count": len(self.analysis_history),
            "current_analysis_active": self.current_analysis is not None,
            "last_analysis": {
                "competition_name": self.analysis_history[-1].request.competition_name if self.analysis_history else None,
                "confidence_level": self.analysis_history[-1].confidence_level if self.analysis_history else None,
                "duration": self.analysis_history[-1].analysis_duration if self.analysis_history else None
            } if self.analysis_history else None,
            "component_status": {
                "grandmaster_patterns": "operational",
                "feasibility_analyzer": "operational", 
                "kaggle_collector": "operational",
                "arxiv_collector": "operational",
                "web_integrator": "operational",
                "issue_reporter": "operational"
            }
        }
    
    async def get_analysis_summary(self, analysis_id: Optional[str] = None) -> Dict[str, Any]:
        """分析サマリー取得"""
        
        if analysis_id:
            # 特定分析のサマリー
            target_analysis = next(
                (a for a in self.analysis_history if a.analysis_id == analysis_id),
                None
            )
            if not target_analysis:
                return {"error": "指定された分析IDが見つかりません"}
            
            return {
                "analysis_id": target_analysis.analysis_id,
                "competition_name": target_analysis.request.competition_name,
                "confidence_level": target_analysis.confidence_level,
                "recommended_techniques_count": len(target_analysis.recommended_techniques),
                "information_sources_count": target_analysis.information_sources_count,
                "phases_completed": [p.value for p in target_analysis.phases_completed],
                "analysis_duration": target_analysis.analysis_duration
            }
        
        else:
            # 全体サマリー
            if not self.analysis_history:
                return {"message": "分析履歴がありません"}
            
            total_analyses = len(self.analysis_history)
            avg_confidence = sum(a.confidence_level for a in self.analysis_history) / total_analyses
            avg_duration = sum(a.analysis_duration for a in self.analysis_history) / total_analyses
            
            return {
                "total_analyses": total_analyses,
                "average_confidence": f"{avg_confidence:.1%}",
                "average_duration": f"{avg_duration:.1f}秒",
                "recent_competitions": [
                    a.request.competition_name for a in self.analysis_history[-5:]
                ]
            }
    
    async def execute_quick_analysis(
        self,
        competition_name: str,
        competition_type: str = "tabular"
    ) -> Dict[str, Any]:
        """クイック分析実行（テスト用）"""
        
        quick_request = AnalysisRequest(
            competition_name=competition_name,
            competition_type=competition_type,
            participant_count=1000,
            days_remaining=30,
            scope=AnalysisScope.QUICK
        )
        
        result = await self.execute_comprehensive_analysis(quick_request)
        
        return {
            "analysis_id": result.analysis_id,
            "recommended_techniques": result.recommended_techniques[:3],
            "confidence_level": result.confidence_level,
            "analysis_duration": result.analysis_duration,
            "github_issue": result.github_issue
        }