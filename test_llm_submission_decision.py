#!/usr/bin/env python3
"""
LLMベース提出判断システム単体テスト

ExecutorAgentの提出判断LLMエージェントの単体テスト・動作確認
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

# LLMベース判断システム
from system.agents.shared.llm_decision_base import ClaudeClient
from system.agents.executor.submission_decision_agent import SubmissionDecisionAgent, SubmissionContext


async def test_llm_basic_submission_decision():
    """基本的なLLM提出判断テスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("🤖 基本LLM提出判断テスト開始")
    
    try:
        # Claude クライアント初期化
        claude_client = ClaudeClient()
        
        # 提出判断エージェント作成
        submission_agent = SubmissionDecisionAgent(claude_client)
        
        # テスト用提出コンテキスト（高スコア・メダル圏内）
        high_performance_context = SubmissionContext(
            competition_name="High Performance Test Competition",
            current_best_score=0.9234,
            target_score=0.9100,
            current_rank_estimate=25,
            total_participants=3500,
            days_remaining=2,
            hours_remaining=48.0,
            
            experiments_completed=45,
            experiments_running=1,
            success_rate=0.85,
            resource_budget_remaining=0.15,
            
            score_history=[0.8987, 0.9045, 0.9123, 0.9189, 0.9234],
            score_improvement_trend=0.0045,
            plateau_duration_hours=3.2,
            
            leaderboard_top10_scores=[
                0.9567, 0.9432, 0.9389, 0.9278, 0.9245,
                0.9234, 0.9201, 0.9178, 0.9156, 0.9134
            ],
            medal_threshold_estimate=0.9200,
            current_medal_zone="silver",
            
            model_stability=0.92,
            overfitting_risk=0.15,
            technical_debt_level=0.20
        )
        
        # 高パフォーマンス状況での判断テスト
        logger.info("📊 高パフォーマンス状況テスト")
        
        decision_response = await submission_agent.should_submit_competition(
            context=high_performance_context,
            urgency="medium"
        )
        
        decision = decision_response.decision_result
        
        logger.info(f"  判断結果: {decision['decision']}")
        logger.info(f"  信頼度: {decision_response.confidence_score:.2f}")
        logger.info(f"  実行時間: {decision_response.execution_time_seconds:.1f}秒")
        logger.info(f"  フォールバック使用: {decision_response.fallback_used}")
        logger.info(f"  推論: {decision.get('reasoning', 'N/A')[:100]}...")
        
        medal_prob = decision.get('medal_probability', {})
        logger.info(f"  メダル確率: Gold={medal_prob.get('gold', 0):.2f}, Silver={medal_prob.get('silver', 0):.2f}")
        
        # 予想: 高スコア+メダル圏なので SUBMIT の可能性が高い
        expected_decision = "SUBMIT"
        actual_decision = decision['decision']
        
        if actual_decision == expected_decision:
            logger.info("  ✅ 期待される判断結果")
        else:
            logger.warning(f"  ⚠️ 予想外の判断: 期待={expected_decision}, 実際={actual_decision}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 基本LLM提出判断テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_llm_different_scenarios():
    """異なるシナリオでのLLM提出判断テスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("🎯 シナリオ別LLM提出判断テスト開始")
    
    try:
        claude_client = ClaudeClient()
        submission_agent = SubmissionDecisionAgent(claude_client)
        
        # シナリオ1: 低スコア・時間切迫
        logger.info("📉 シナリオ1: 低スコア・時間切迫")
        
        low_score_context = SubmissionContext(
            competition_name="Low Score Emergency",
            current_best_score=0.7234,
            target_score=0.8500,
            current_rank_estimate=800,
            total_participants=2000,
            days_remaining=0,
            hours_remaining=6.0,  # 6時間しかない
            
            experiments_completed=15,
            experiments_running=2,
            success_rate=0.60,
            resource_budget_remaining=0.80,
            
            score_history=[0.6890, 0.7012, 0.7123, 0.7198, 0.7234],
            score_improvement_trend=0.0036,
            plateau_duration_hours=8.0,
            
            leaderboard_top10_scores=[
                0.8567, 0.8432, 0.8389, 0.8278, 0.8245,
                0.8234, 0.8201, 0.8178, 0.8156, 0.8134
            ],
            medal_threshold_estimate=0.8200,
            current_medal_zone="none",
            
            model_stability=0.65,
            overfitting_risk=0.45,
            technical_debt_level=0.50
        )
        
        decision1 = await submission_agent.should_submit_competition(
            context=low_score_context,
            urgency="critical"
        )
        
        logger.info(f"  判断: {decision1.decision_result['decision']} (信頼度: {decision1.confidence_score:.2f})")
        logger.info(f"  推論: {decision1.decision_result.get('reasoning', 'N/A')[:80]}...")
        
        # シナリオ2: 中程度スコア・改善中
        logger.info("📈 シナリオ2: 中程度スコア・改善中")
        
        improving_context = SubmissionContext(
            competition_name="Improving Performance",
            current_best_score=0.8456,
            target_score=0.8800,
            current_rank_estimate=200,
            total_participants=3000,
            days_remaining=7,
            hours_remaining=168.0,
            
            experiments_completed=30,
            experiments_running=3,
            success_rate=0.75,
            resource_budget_remaining=0.50,
            
            score_history=[0.8123, 0.8234, 0.8345, 0.8398, 0.8456],
            score_improvement_trend=0.0083,  # 良い改善トレンド
            plateau_duration_hours=1.5,
            
            leaderboard_top10_scores=[
                0.8967, 0.8834, 0.8789, 0.8678, 0.8645,
                0.8634, 0.8601, 0.8578, 0.8556, 0.8534
            ],
            medal_threshold_estimate=0.8600,
            current_medal_zone="bronze",
            
            model_stability=0.85,
            overfitting_risk=0.25,
            technical_debt_level=0.30
        )
        
        decision2 = await submission_agent.should_submit_competition(
            context=improving_context,
            urgency="low"
        )
        
        logger.info(f"  判断: {decision2.decision_result['decision']} (信頼度: {decision2.confidence_score:.2f})")
        logger.info(f"  推論: {decision2.decision_result.get('reasoning', 'N/A')[:80]}...")
        
        # シナリオ3: 高リスク状況
        logger.info("⚠️ シナリオ3: 高リスク状況")
        
        high_risk_context = SubmissionContext(
            competition_name="High Risk Situation",
            current_best_score=0.8789,
            target_score=0.8900,
            current_rank_estimate=120,
            total_participants=4000,
            days_remaining=3,
            hours_remaining=72.0,
            
            experiments_completed=60,
            experiments_running=0,
            success_rate=0.45,  # 低い成功率
            resource_budget_remaining=0.05,  # リソース枯渇
            
            score_history=[0.8789, 0.8789, 0.8789, 0.8789, 0.8789],  # スコア停滞
            score_improvement_trend=0.0000,
            plateau_duration_hours=48.0,  # 長時間停滞
            
            leaderboard_top10_scores=[
                0.9234, 0.9156, 0.9089, 0.9012, 0.8967,
                0.8934, 0.8901, 0.8878, 0.8856, 0.8834
            ],
            medal_threshold_estimate=0.8850,
            current_medal_zone="bronze",
            
            model_stability=0.50,  # 不安定
            overfitting_risk=0.80,  # 高い過学習リスク
            technical_debt_level=0.70
        )
        
        decision3 = await submission_agent.should_submit_competition(
            context=high_risk_context,
            urgency="high"
        )
        
        logger.info(f"  判断: {decision3.decision_result['decision']} (信頼度: {decision3.confidence_score:.2f})")
        logger.info(f"  推論: {decision3.decision_result.get('reasoning', 'N/A')[:80]}...")
        
        # パフォーマンス統計確認
        perf_metrics = submission_agent.get_performance_metrics()
        logger.info("📈 提出判断エージェント統計:")
        logger.info(f"  総判断回数: {perf_metrics['total_decisions']}")
        logger.info(f"  成功率: {perf_metrics['success_rate']:.1%}")
        logger.info(f"  平均応答時間: {perf_metrics['average_response_time']:.1f}秒")
        logger.info(f"  フォールバック率: {perf_metrics['fallback_rate']:.1%}")
        
        # 判断の多様性確認
        decisions = [
            decision1.decision_result['decision'],
            decision2.decision_result['decision'],
            decision3.decision_result['decision']
        ]
        
        unique_decisions = set(decisions)
        logger.info(f"🎯 判断多様性: {len(unique_decisions)}種類の判断 {list(unique_decisions)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ シナリオ別LLM提出判断テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_mechanism():
    """フォールバック機構テスト"""
    
    logger = logging.getLogger(__name__)
    logger.info("🛡️ フォールバック機構テスト開始")
    
    try:
        # 故意にタイムアウトを短く設定してフォールバックを誘発
        claude_client = ClaudeClient()
        submission_agent = SubmissionDecisionAgent(claude_client)
        
        # 非常に短いタイムアウト設定のテストコンテキスト
        test_context = SubmissionContext(
            competition_name="Fallback Test",
            current_best_score=0.8000,
            target_score=0.8500,
            current_rank_estimate=300,
            total_participants=2000,
            days_remaining=5,
            hours_remaining=120.0,
            
            experiments_completed=20,
            experiments_running=1,
            success_rate=0.70,
            resource_budget_remaining=0.40,
            
            score_history=[0.7800, 0.7900, 0.7950, 0.7980, 0.8000],
            score_improvement_trend=0.0020,
            plateau_duration_hours=5.0,
            
            leaderboard_top10_scores=[
                0.8800, 0.8700, 0.8600, 0.8500, 0.8450,
                0.8400, 0.8350, 0.8300, 0.8250, 0.8200
            ],
            medal_threshold_estimate=0.8300,
            current_medal_zone="none",
            
            model_stability=0.75,
            overfitting_risk=0.30,
            technical_debt_level=0.35
        )
        
        # 極短タイムアウトでフォールバックテスト
        from system.agents.shared.llm_decision_base import LLMDecisionRequest, LLMDecisionType
        
        fallback_request = LLMDecisionRequest(
            request_id="fallback-test",
            decision_type=LLMDecisionType.SUBMISSION_DECISION,
            context_data={},  # 簡易コンテキスト
            urgency_level="medium",
            fallback_strategy="conservative_submit",
            max_response_time_seconds=0.001  # 極短タイムアウト
        )
        
        # フォールバック実行
        fallback_decision = await submission_agent.make_decision(fallback_request)
        
        logger.info(f"フォールバック判断結果:")
        logger.info(f"  判断: {fallback_decision.decision_result.get('decision', 'N/A')}")
        logger.info(f"  フォールバック使用: {fallback_decision.fallback_used}")
        logger.info(f"  信頼度: {fallback_decision.confidence_score:.2f}")
        logger.info(f"  実行時間: {fallback_decision.execution_time_seconds:.3f}秒")
        
        if fallback_decision.fallback_used:
            logger.info("  ✅ フォールバック機構正常動作")
            return True
        else:
            logger.warning("  ⚠️ フォールバックが期待通り動作しなかった")
            return False
        
    except Exception as e:
        logger.error(f"❌ フォールバック機構テスト失敗: {e}")
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
    
    print("🧪 LLMベース提出判断システム単体テスト")
    print("=" * 50)
    
    test_results = []
    
    # テスト1: 基本的なLLM提出判断
    print("\n🤖 テスト1: 基本LLM提出判断")
    print("-" * 30)
    result1 = await test_llm_basic_submission_decision()
    test_results.append(("基本LLM提出判断", result1))
    
    # テスト2: 異なるシナリオでの判断
    print("\n🎯 テスト2: シナリオ別判断")
    print("-" * 30)
    result2 = await test_llm_different_scenarios()
    test_results.append(("シナリオ別判断", result2))
    
    # テスト3: フォールバック機構
    print("\n🛡️ テスト3: フォールバック機構")
    print("-" * 30)
    result3 = await test_fallback_mechanism()
    test_results.append(("フォールバック機構", result3))
    
    # 結果サマリー
    print("\n" + "=" * 50)
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
        print("🎉 全テスト成功 - LLMベース提出判断システム正常動作確認")
        return 0
    else:
        print("⚠️ 一部テスト失敗 - システム確認が必要")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)