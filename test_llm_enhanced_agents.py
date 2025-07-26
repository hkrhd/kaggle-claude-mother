#!/usr/bin/env python3
"""
LLM強化エージェントシステム統合テスト

各エージェントでのLLMベース判断システムの動作確認・統合テスト
"""

import asyncio
import sys
import os
import logging
import subprocess
from datetime import datetime, timedelta

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

# LLMベース判断システム
from system.agents.shared.llm_decision_base import ClaudeClient, CLAUDE_MODEL
from system.agents.executor.submission_decision_agent import SubmissionDecisionAgent, SubmissionContext
from system.agents.executor.executor_agent import ExecutorAgent, ExecutionRequest, ExecutionPriority

# 競技選択エージェント（既存）
from system.agents.competition_selector.competition_selector_agent import (
    CompetitionSelectorAgent, SelectionStrategy
)

# 他のエージェント
from system.agents.analyzer.analyzer_agent import AnalyzerAgent
from system.agents.monitor.monitor_agent import MonitorAgent


def get_github_token():
    """GitHub認証トークンを動的に取得"""
    try:
        # gh auth token コマンドでトークンを取得
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # フォールバック: 環境変数から取得
        return os.environ.get("GITHUB_TOKEN", "test_token")


async def test_llm_submission_decision():
    """LLMベース提出判断テスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("🤖 LLMベース提出判断テスト開始")
    
    try:
        # Claude クライアント初期化（モデル指定付き）
        claude_client = ClaudeClient(model=CLAUDE_MODEL)
        
        # 提出判断エージェント作成
        submission_agent = SubmissionDecisionAgent(claude_client)
        
        # テスト用提出コンテキスト
        test_context = SubmissionContext(
            competition_name="Test ML Competition",
            current_best_score=0.8756,
            target_score=0.9000,
            current_rank_estimate=87,
            total_participants=3500,
            days_remaining=5,
            hours_remaining=120.0,
            
            experiments_completed=25,
            experiments_running=3,
            success_rate=0.72,
            resource_budget_remaining=0.35,
            
            score_history=[0.8234, 0.8456, 0.8621, 0.8698, 0.8756],
            score_improvement_trend=0.0058,
            plateau_duration_hours=6.5,
            
            leaderboard_top10_scores=[
                0.9234, 0.9156, 0.9089, 0.9012, 0.8967,
                0.8934, 0.8901, 0.8878, 0.8856, 0.8834
            ],
            medal_threshold_estimate=0.8900,
            current_medal_zone="none",
            
            model_stability=0.85,
            overfitting_risk=0.25,
            technical_debt_level=0.30
        )
        
        # 異なる緊急度でのテスト
        urgency_levels = ["low", "medium", "high", "critical"]
        
        for urgency in urgency_levels:
            logger.info(f"📊 提出判断テスト: 緊急度{urgency}")
            
            decision_response = await submission_agent.should_submit_competition(
                context=test_context,
                urgency=urgency
            )
            
            decision = decision_response.decision_result
            
            logger.info(f"  判断結果: {decision['decision']}")
            logger.info(f"  信頼度: {decision_response.confidence_score:.2f}")
            logger.info(f"  実行時間: {decision_response.execution_time_seconds:.1f}秒")
            logger.info(f"  推論: {decision.get('reasoning', 'N/A')[:100]}...")
            logger.info(f"  メダル確率: Gold={decision.get('medal_probability', {}).get('gold', 0):.2f}")
            
            # 決定に応じた模擬処理
            if decision['decision'] == 'SUBMIT':
                logger.info("  → ✅ 模擬提出実行")
            elif decision['decision'] == 'CONTINUE':
                logger.info("  → 🔄 実験継続")
            else:
                logger.info("  → ⏳ 待機")
            
            logger.info("")
        
        # パフォーマンス統計確認
        perf_metrics = submission_agent.get_performance_metrics()
        logger.info("📈 提出判断エージェント統計:")
        logger.info(f"  総判断回数: {perf_metrics['total_decisions']}")
        logger.info(f"  成功率: {perf_metrics['success_rate']:.1%}")
        logger.info(f"  平均応答時間: {perf_metrics['average_response_time']:.1f}秒")
        logger.info(f"  フォールバック率: {perf_metrics['fallback_rate']:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM提出判断テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integrated_executor_agent():
    """LLM統合ExecutorAgentテスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("⚡ LLM統合ExecutorAgentテスト開始")
    
    try:
        # GitHub認証トークン取得
        github_token = get_github_token()
        
        # ExecutorAgent初期化（LLMベース提出判断統合）
        executor = ExecutorAgent(
            github_token=github_token,
            repo_name="hkrhd/kaggle-claude-mother"
        )
        
        # テスト用実行リクエスト
        test_request = ExecutionRequest(
            competition_name="Test Tabular Competition",
            analyzer_issue_number=12345,
            techniques_to_implement=[
                {
                    "technique": "gradient_boosting_ensemble",
                    "integrated_score": 0.85,
                    "gpu_required": False
                },
                {
                    "technique": "feature_engineering_advanced", 
                    "integrated_score": 0.72,
                    "gpu_required": False
                }
            ],
            priority=ExecutionPriority.HIGH,
            deadline_days=7
        )
        
        # 実行実験（LLM提出判断統合版）
        logger.info("🚀 統合実行実験開始...")
        
        execution_result = await executor.execute_technical_implementation(test_request)
        
        logger.info("📊 実行結果サマリー:")
        logger.info(f"  実行ID: {execution_result.execution_id}")
        logger.info(f"  総実験数: {execution_result.total_experiments_run}")
        logger.info(f"  成功率: {execution_result.success_rate:.1%}")
        logger.info(f"  ベストスコア: {execution_result.best_score:.6f}")
        logger.info(f"  GPU時間使用: {execution_result.total_gpu_hours_used:.1f}時間")
        logger.info(f"  実行時間: {execution_result.execution_duration:.1f}秒")
        logger.info(f"  提出準備: {'完了' if execution_result.submission_ready else '未完'}")
        
        # LLM提出判断結果確認
        if hasattr(execution_result, 'llm_submission_decision'):
            llm_decision = execution_result.llm_submission_decision
            logger.info("🤖 LLM提出判断結果:")
            logger.info(f"  判断: {llm_decision.get('decision', 'N/A')}")
            logger.info(f"  信頼度: {llm_decision.get('confidence', 0):.2f}")
            logger.info(f"  推論: {llm_decision.get('reasoning', 'N/A')[:100]}...")
        
        # 提出情報確認
        if hasattr(execution_result, 'submission_info'):
            submission_info = execution_result.submission_info
            logger.info("📤 提出情報:")
            logger.info(f"  提出実行: {submission_info.get('submitted', False)}")
            if submission_info.get('reason'):
                logger.info(f"  理由: {submission_info['reason']}")
            if submission_info.get('llm_reasoning'):
                logger.info(f"  LLM推論: {submission_info['llm_reasoning'][:100]}...")
        
        # 継続実験情報確認
        if hasattr(execution_result, 'continued_experiments'):
            cont_exp = execution_result.continued_experiments
            logger.info("🔬 継続実験情報:")
            logger.info(f"  実行アクション: {cont_exp.get('executed_actions', [])}")
            logger.info(f"  トリガー: {cont_exp.get('triggered_by', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 統合ExecutorAgentテスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_agent_llm_coordination():
    """マルチエージェントLLM連携テスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("🌐 マルチエージェントLLM連携テスト開始")
    
    try:
        # GitHub認証トークン取得
        github_token = get_github_token()
        repo_name = "hkrhd/kaggle-claude-mother"
        
        # 各エージェント初期化
        competition_selector = CompetitionSelectorAgent(
            github_token=github_token,
            repo_name=repo_name
        )
        
        analyzer = AnalyzerAgent()
        
        executor = ExecutorAgent(
            github_token=github_token, 
            repo_name=repo_name
        )
        
        monitor = MonitorAgent(
            github_token=github_token,
            repo_name=repo_name
        )
        
        # 1. 競技選択（既存LLMベース）
        logger.info("🎯 段階1: LLMベース競技選択")
        
        test_competitions = [
            {
                "id": "test-tabular-competition",
                "name": "Test Tabular Competition",
                "type": "tabular",
                "deadline": (datetime.utcnow() + timedelta(days=30)).isoformat(),
                "participants": 2500,
                "prize_amount": 50000,
                "competition_category": "featured",
                "awards_medals": True
            }
        ]
        
        selection_decision = await competition_selector.evaluate_available_competitions(
            available_competitions=test_competitions,
            strategy=SelectionStrategy.BALANCED
        )
        
        logger.info(f"競技選択結果: {len(selection_decision.selected_competitions)}競技選択")
        
        if not selection_decision.selected_competitions:
            logger.warning("競技選択なし - テスト継続不可")
            return False
        
        selected_competition = selection_decision.selected_competitions[0]
        
        # 2. 分析実行（従来方式）
        logger.info("🔬 段階2: 技術分析実行")
        
        analysis_request = {
            "competition_name": selected_competition.name,
            "competition_type": selected_competition.type,
            "participant_count": selected_competition.participants,
            "days_remaining": 30
        }
        
        analysis_result = await analyzer.analyze_competition(analysis_request)
        
        logger.info(f"分析完了: {len(analysis_result.recommended_techniques)}技術推奨")
        
        # 3. 実行実行（LLMベース提出判断統合）
        logger.info("⚡ 段階3: LLM統合実行エージェント")
        
        execution_request = ExecutionRequest(
            competition_name=selected_competition.name,
            analyzer_issue_number=12345,
            techniques_to_implement=[
                {
                    "technique": tech["technique"],
                    "integrated_score": tech.get("integrated_score", 0.7),
                    "gpu_required": False
                }
                for tech in analysis_result.recommended_techniques[:2]
            ],
            priority=ExecutionPriority.HIGH,
            deadline_days=30
        )
        
        execution_result = await executor.execute_technical_implementation(execution_request)
        
        logger.info(f"実行完了: 成功率{execution_result.success_rate:.1%}")
        
        # 4. 監視エージェント（従来方式）
        logger.info("👁️ 段階4: 統合監視")
        
        # 監視対象エージェント設定
        monitored_agents = {
            "competition_selector": competition_selector,
            "analyzer": analyzer,
            "executor": executor
        }
        
        # 短時間監視テスト（5秒）
        monitoring_issue = await monitor.start_monitoring(monitored_agents)
        
        logger.info(f"監視開始: Issue #{monitoring_issue}")
        
        # 5秒間監視
        await asyncio.sleep(5)
        
        await monitor.stop_monitoring()
        
        logger.info("監視停止")
        
        # 5. 統合結果サマリー
        logger.info("📈 統合テスト結果サマリー:")
        logger.info(f"  選択競技: {selected_competition.name}")
        logger.info(f"  LLM推奨スコア: {selected_competition.llm_score:.2f}")
        logger.info(f"  分析技術数: {len(analysis_result.recommended_techniques)}")
        logger.info(f"  実行成功率: {execution_result.success_rate:.1%}")
        logger.info(f"  実行時間: {execution_result.execution_duration:.1f}秒")
        
        # LLM判断統計
        executor_perf = executor.submission_decision_agent.get_performance_metrics()
        selector_perf = competition_selector.get_selection_performance_metrics()
        
        logger.info("🤖 LLM判断統計:")
        logger.info(f"  競技選択判断: {selector_perf.get('total_decisions', 0)}回")
        logger.info(f"  提出判断: {executor_perf['total_decisions']}回")
        logger.info(f"  総合LLM成功率: {(executor_perf['success_rate'] + selector_perf.get('success_rate', 1.0)) / 2:.1%}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ マルチエージェント連携テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """メイン関数"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 LLM強化エージェントシステム統合テスト")
    print("=" * 60)
    
    test_results = []
    
    # テスト1: LLMベース提出判断
    print("\n🤖 テスト1: LLMベース提出判断")
    print("-" * 30)
    result1 = await test_llm_submission_decision()
    test_results.append(("LLM提出判断", result1))
    
    # テスト2: LLM統合ExecutorAgent
    print("\n⚡ テスト2: LLM統合ExecutorAgent")
    print("-" * 30)
    result2 = await test_integrated_executor_agent()
    test_results.append(("LLM統合ExecutorAgent", result2))
    
    # テスト3: マルチエージェントLLM連携
    print("\n🌐 テスト3: マルチエージェントLLM連携")
    print("-" * 30)
    result3 = await test_multi_agent_llm_coordination()
    test_results.append(("マルチエージェント連携", result3))
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("📊 テスト結果サマリー")
    print("-" * 30)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 テスト結果: {passed}/{len(test_results)} PASS")
    
    if passed == len(test_results):
        print("🎉 全テスト成功 - LLMベース判断システム正常動作確認")
        return 0
    else:
        print("⚠️ 一部テスト失敗 - システム確認が必要")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)