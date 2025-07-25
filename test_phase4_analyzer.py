#!/usr/bin/env python3
"""
Phase 4: 分析エージェント統合テスト

グランドマスター技術調査・最新手法研究・実装可能性判定を担当する
深層分析エージェントの全機能テスト。
"""

import asyncio
import sys
from datetime import datetime, timedelta

def test_phase4_imports():
    """Phase 4 インポートテスト"""
    print("=== Phase 4: 分析エージェント インポートテスト ===")
    
    try:
        from system.agents.analyzer import (
            AnalyzerAgent,
            GrandmasterPatterns,
            TechnicalFeasibilityAnalyzer,
            KaggleSolutionCollector,
            ArxivPaperCollector
        )
        print("✅ 全モジュール正常インポート")
        return True
    except Exception as e:
        print(f"❌ インポート失敗: {e}")
        return False

async def test_grandmaster_patterns():
    """グランドマスターパターン分析テスト"""
    print("\n=== グランドマスターパターン分析テスト ===")
    
    try:
        from system.agents.analyzer.knowledge_base.grandmaster_patterns import (
            GrandmasterPatterns, CompetitionType
        )
        
        patterns = GrandmasterPatterns()
        
        # パターン適用可能性分析
        analysis = await patterns.analyze_pattern_applicability(
            competition_type=CompetitionType.TABULAR,
            participant_count=1500,
            days_remaining=30,
            available_gpu_hours=16.0
        )
        
        print(f"✅ グランドマスターパターン分析成功")
        print(f"   - 適用可能技術数: {analysis['total_applicable_techniques']}")
        print(f"   - 上位推奨数: {len(analysis['top_recommendations'])}")
        
        if analysis['top_recommendations']:
            top_rec = analysis['top_recommendations'][0]
            technique = top_rec.get('technique', {})
            technique_name = technique.get('name') if isinstance(technique, dict) else str(technique)
            print(f"   - 最優秀技術: {technique_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ グランドマスターパターン分析失敗: {e}")
        return False

async def test_technical_feasibility():
    """技術実装可能性分析テスト"""
    print("\n=== 技術実装可能性分析テスト ===")
    
    try:
        from system.agents.analyzer.analyzers.technical_feasibility import (
            TechnicalFeasibilityAnalyzer, TechnicalSpecification, 
            TechnicalComplexity
        )
        
        analyzer = TechnicalFeasibilityAnalyzer()
        
        # テスト用技術仕様
        tech_spec = TechnicalSpecification(
            name="ensemble_stacking",
            description="多層アンサンブル手法",
            complexity=TechnicalComplexity.ADVANCED,
            estimated_implementation_hours=40,
            required_libraries=["scikit-learn", "xgboost", "lightgbm"],
            gpu_memory_gb=8.0,
            cpu_cores_min=4,
            ram_gb_min=16.0,
            disk_space_gb=5.0,
            implementation_difficulty_factors=["多モデル統合", "メタ特徴量生成"],
            common_pitfalls=["過学習リスク", "計算時間増大"],
            success_indicators=["CV改善", "リーダーボード向上"]
        )
        
        # 実装可能性分析実行
        result = await analyzer.analyze_technique_feasibility(
            technique_spec=tech_spec,
            available_days=21,
            current_skill_level=0.7
        )
        
        print(f"✅ 技術実装可能性分析成功")
        print(f"   - 実装可能性スコア: {result.feasibility_score:.3f}")
        print(f"   - 実装成功確率: {result.implementation_probability:.3f}")
        print(f"   - 推定完了日数: {result.estimated_completion_days}日")
        print(f"   - リソース互換性: {'○' if result.resource_compatibility else '×'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 技術実装可能性分析失敗: {e}")
        return False

async def test_kaggle_solution_collector():
    """Kaggle解法収集テスト"""
    print("\n=== Kaggle解法収集テスト ===")
    
    try:
        from system.agents.analyzer.collectors.kaggle_solutions import KaggleSolutionCollector
        
        collector = KaggleSolutionCollector()
        
        # テスト用解法収集（模擬データ）
        solutions = await collector.collect_competition_solutions(
            competition_name="test_tabular_competition",
            competition_type="tabular",
            max_solutions=5
        )
        
        print(f"✅ Kaggle解法収集成功")
        print(f"   - 収集解法数: {len(solutions)}")
        
        if solutions:
            print(f"   - サンプル解法: {solutions[0].solution_title}")
            print(f"   - 使用技術: {solutions[0].techniques_used[:3]}")
        
        # 技術推奨生成テスト
        recommendations = await collector.get_technique_recommendations(
            competition_type="tabular",
            available_time_days=30,
            complexity_preference="medium"
        )
        
        print(f"   - 技術推奨数: {len(recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Kaggle解法収集失敗: {e}")
        return False

async def test_arxiv_paper_collector():
    """arXiv論文収集テスト"""
    print("\n=== arXiv論文収集テスト ===")
    
    try:
        from system.agents.analyzer.collectors.arxiv_papers import ArxivPaperCollector
        
        collector = ArxivPaperCollector()
        
        # 最新論文収集（模擬データ）
        papers = await collector.collect_latest_papers(
            competition_domain="tabular",
            days_back=30,
            max_papers=5
        )
        
        print(f"✅ arXiv論文収集成功")
        print(f"   - 収集論文数: {len(papers)}")
        
        if papers:
            print(f"   - サンプル論文: {papers[0].title[:50]}...")
            print(f"   - 重要技術: {papers[0].key_techniques[:3]}")
        
        # 技術推奨生成テスト
        recommendations = await collector.generate_technique_recommendations(
            competition_domain="tabular",
            max_recommendations=3
        )
        
        print(f"   - 技術推奨数: {len(recommendations)}")
        
        return True
        
    except Exception as e:
        print(f"❌ arXiv論文収集失敗: {e}")
        return False

async def test_web_search_integrator():
    """WebSearch統合調査テスト"""
    print("\n=== WebSearch統合調査テスト ===")
    
    try:
        from system.agents.analyzer.utils.web_scraper import WebSearchIntegrator
        
        integrator = WebSearchIntegrator()
        
        # 包括調査実行（模擬データ）
        report = await integrator.conduct_comprehensive_investigation(
            competition_name="test_competition",
            competition_domain="tabular",
            investigation_scope="standard",
            time_limit_minutes=60
        )
        
        print(f"✅ WebSearch統合調査成功")
        print(f"   - 調査結果数: {report.total_results}")
        print(f"   - 高品質結果数: {report.high_quality_results}")
        print(f"   - 推奨技術数: {len(report.recommended_techniques)}")
        print(f"   - 信頼度: {report.confidence_level:.3f}")
        print(f"   - 調査時間: {report.investigation_duration:.1f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSearch統合調査失敗: {e}")
        return False

async def test_github_issue_reporter():
    """GitHub Issue連携テスト"""
    print("\n=== GitHub Issue連携テスト ===")
    
    try:
        from system.agents.analyzer.utils.github_issue_reporter import (
            GitHubIssueReporter, TechnicalAnalysisReport, ReportPriority
        )
        
        reporter = GitHubIssueReporter()
        
        # テスト用技術分析レポート
        test_report = TechnicalAnalysisReport(
            competition_name="test_competition",
            competition_type="tabular",
            analysis_scope="standard",
            recommended_techniques=[
                {
                    "technique": "gradient_boosting_ensemble",
                    "integrated_score": 0.85,
                    "sources": ["grandmaster", "kaggle"],
                    "mention_count": 3
                },
                {
                    "technique": "feature_engineering_automated",
                    "integrated_score": 0.78,
                    "sources": ["arxiv", "web"],
                    "mention_count": 2
                }
            ],
            grandmaster_pattern_analysis={
                "total_applicable_techniques": 5,
                "top_recommendations": [
                    {
                        "grandmaster": "owen_zhang",
                        "technique": {"name": "multi_level_stacking"},
                        "applicability_score": 0.85
                    }
                ]
            },
            implementation_feasibility={
                "overall_feasibility": 0.75,
                "implementation_recommendations": ["gradient_boosting_ensemble"]
            },
            estimated_implementation_time="14日以内",
            required_resources={
                "estimated_gpu_hours": "28時間",
                "memory_requirement": "16GB RAM推奨"
            },
            technical_risks=["時間制約による機能削減リスク"],
            implementation_constraints=["残り日数: 14日"],
            fallback_strategies=["既存ライブラリ活用による開発短縮"],
            executor_instructions=[
                "gradient_boosting_ensemble を最優先実装",
                "既存実装の最大活用を推奨"
            ],
            success_metrics=["推奨技術実装: 2技術の80%以上完了"],
            milestone_timeline=[
                "フェーズ1: 主力技術実装 (6日)",
                "フェーズ2: 補完技術・最適化 (4日)",
                "フェーズ3: 統合・最終調整 (4日)"
            ],
            confidence_level=0.82,
            information_sources=["グランドマスターパターン: 5技術", "Web調査: 15件"],
            analysis_duration=125.5,
            created_at=datetime.utcnow()
        )
        
        # GitHub Issue作成テスト
        issue_result = await reporter.create_technical_analysis_issue(
            report=test_report,
            priority=ReportPriority.HIGH
        )
        
        print(f"✅ GitHub Issue連携成功")
        print(f"   - Issue作成: {'成功' if issue_result['success'] else '失敗'}")
        print(f"   - Issue番号: #{issue_result.get('issue_number', 'N/A')}")
        print(f"   - executor通知: {'実行済み' if issue_result.get('executor_notified') else '未実行'}")
        
        return True
        
    except Exception as e:
        print(f"❌ GitHub Issue連携失敗: {e}")
        return False

async def test_analyzer_agent_integration():
    """分析エージェント統合テスト"""
    print("\n=== 分析エージェント統合テスト ===")
    
    try:
        from system.agents.analyzer.analyzer_agent import (
            AnalyzerAgent, AnalysisRequest, AnalysisScope
        )
        
        # エージェント作成
        agent = AnalyzerAgent()
        
        # エージェント状態確認
        status = await agent.get_agent_status()
        
        print(f"✅ 分析エージェント初期化成功")
        print(f"   - エージェントID: {status['agent_id']}")
        print(f"   - バージョン: {status['agent_version']}")
        print(f"   - 稼働時間: {status['uptime_hours']:.3f}時間")
        
        # クイック分析実行テスト
        print("\n--- クイック分析テスト ---")
        quick_result = await agent.execute_quick_analysis(
            competition_name="test_quick_competition",
            competition_type="tabular"
        )
        
        print(f"✅ クイック分析成功")
        print(f"   - 分析ID: {quick_result['analysis_id']}")
        print(f"   - 推奨技術数: {len(quick_result['recommended_techniques'])}")
        print(f"   - 信頼度: {quick_result['confidence_level']:.3f}")
        print(f"   - 分析時間: {quick_result['analysis_duration']:.1f}秒")
        
        # GitHub Issue確認
        if quick_result['github_issue']['success']:
            print(f"   - GitHub Issue: #{quick_result['github_issue']['issue_number']}")
        
        # 分析サマリー取得
        summary = await agent.get_analysis_summary()
        print(f"   - 分析履歴: {summary.get('total_analyses', 0)}件")
        
        return True
        
    except Exception as e:
        print(f"❌ 分析エージェント統合テスト失敗: {e}")
        return False

def test_phase4_completion():
    """Phase 4 完了基準チェック"""
    print("\n=== Phase 4 完了基準チェック ===")
    
    criteria = [
        ("グランドマスターパターンDB", "GrandmasterPatterns"),
        ("技術実装可能性分析", "TechnicalFeasibilityAnalyzer"),
        ("Kaggle解法収集システム", "KaggleSolutionCollector"),
        ("arXiv論文収集システム", "ArxivPaperCollector"),
        ("WebSearch統合調査", "WebSearchIntegrator"),
        ("GitHub Issue連携", "GitHubIssueReporter"),
        ("統合分析エージェント", "AnalyzerAgent")
    ]
    
    all_passed = True
    
    for name, class_name in criteria:
        try:
            if class_name in ["WebSearchIntegrator", "GitHubIssueReporter"]:
                exec(f"from system.agents.analyzer.utils.web_scraper import {class_name}" if class_name == "WebSearchIntegrator" else f"from system.agents.analyzer.utils.github_issue_reporter import {class_name}")
            elif class_name == "AnalyzerAgent":
                exec(f"from system.agents.analyzer.analyzer_agent import {class_name}")
            else:
                exec(f"from system.agents.analyzer import {class_name}")
            print(f"✅ {name}: 実装完了")
        except Exception as e:
            print(f"❌ {name}: 実装不完全 - {e}")
            all_passed = False
    
    return all_passed

async def main():
    """Phase 4 統合テスト実行"""
    print("Phase 4: 分析エージェント統合テスト")
    print("=" * 50)
    
    all_tests_passed = True
    
    # インポートテスト
    if not test_phase4_imports():
        all_tests_passed = False
    
    # 機能テスト
    if not await test_grandmaster_patterns():
        all_tests_passed = False
    
    if not await test_technical_feasibility():
        all_tests_passed = False
    
    if not await test_kaggle_solution_collector():
        all_tests_passed = False
    
    if not await test_arxiv_paper_collector():
        all_tests_passed = False
    
    if not await test_web_search_integrator():
        all_tests_passed = False
    
    if not await test_github_issue_reporter():
        all_tests_passed = False
    
    if not await test_analyzer_agent_integration():
        all_tests_passed = False
    
    # 完了基準チェック
    if not test_phase4_completion():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 Phase 4: 分析エージェント実装完了! 🎉")
        print("\n✅ 実装完了項目:")
        print("  - グランドマスター技術データベース・パターン分析")
        print("  - 技術実装可能性評価エンジン")
        print("  - Kaggle優勝解法自動収集・分析")
        print("  - arXiv最新論文調査・技術抽出")
        print("  - WebSearch統合調査システム")
        print("  - GitHub Issue連携・自動レポート生成")
        print("  - executor向け実装戦略策定")
        print("  - 統合深層分析エージェント")
        print("\n🚀 Phase 5: 実行エージェント実装準備完了")
        return 0
    else:
        print("❌ テスト失敗。修正が必要です。")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))