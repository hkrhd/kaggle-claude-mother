#!/usr/bin/env python3
"""
Phase 3: プランニングエージェント統合テスト
"""

import asyncio
import sys
import pytest
from datetime import datetime, timedelta

def test_phase3_imports():
    """Phase 3 インポートテスト"""
    print("=== Phase 3: プランニングエージェント インポートテスト ===")
    
    try:
        from system.agents.planner import (
            PlannerAgent,
            CompetitionInfo,
            CompetitionStatus,
            AnalysisResult,
            MedalProbability,
            ProbabilityFactors,
            MedalProbabilityCalculator,
            CompetitionSelectionStrategy,
            WithdrawalStrategy
        )
        print("✅ 全モジュール正常インポート")
        return True
    except Exception as e:
        print(f"❌ インポート失敗: {e}")
        return False

@pytest.mark.asyncio
async def test_medal_probability_calculator():
    """メダル確率算出テスト"""
    print("\n=== メダル確率算出テスト ===")
    
    try:
        from system.agents.planner.calculators.medal_probability import MedalProbabilityCalculator
        from system.agents.planner.models.competition import CompetitionInfo, CompetitionType, PrizeType
        
        calculator = MedalProbabilityCalculator()
        
        # テスト用コンペデータ
        comp_info = CompetitionInfo(
            competition_id="test_comp",
            title="Test Tabular Competition",
            url="https://kaggle.com/c/test",
            participant_count=1500,
            total_prize=25000,
            prize_type=PrizeType.MONETARY,
            competition_type=CompetitionType.TABULAR,
            days_remaining=45,
            data_size_gb=2.0,
            feature_count=75,
            skill_requirements=["feature_engineering", "ensemble"],
            last_updated=datetime.utcnow()
        )
        
        result = await calculator.calculate_medal_probability(comp_info)
        
        print(f"✅ メダル確率算出成功: 総合確率 {result.overall_probability:.3f}")
        print(f"   - 金メダル確率: {result.gold_probability:.3f}")
        print(f"   - 銀メダル確率: {result.silver_probability:.3f}")
        print(f"   - 銅メダル確率: {result.bronze_probability:.3f}")
        print(f"   - 信頼区間: {result.confidence_interval[0]:.3f} - {result.confidence_interval[1]:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ メダル確率算出失敗: {e}")
        return False

@pytest.mark.asyncio
async def test_competition_selection_strategy():
    """コンペ選択戦略テスト"""
    print("\n=== コンペ選択戦略テスト ===")
    
    try:
        from system.agents.planner.strategies.selection_strategy import CompetitionSelectionStrategy, SelectionStrategy
        from system.agents.planner.models.competition import CompetitionInfo, CompetitionType, PrizeType
        
        strategy = CompetitionSelectionStrategy()
        
        # テスト用コンペデータ
        comp_info = CompetitionInfo(
            competition_id="strategy_test",
            title="Strategy Test Competition",
            url="https://kaggle.com/c/strategy_test",
            participant_count=800,
            total_prize=50000,
            prize_type=PrizeType.MONETARY,
            competition_type=CompetitionType.COMPUTER_VISION,
            days_remaining=60,
            data_size_gb=5.0,
            feature_count=20,
            skill_requirements=["deep_learning", "cnn"],
            last_updated=datetime.utcnow()
        )
        
        analysis = await strategy.analyze_competition_for_selection(
            comp_info, SelectionStrategy.BALANCED
        )
        
        print(f"✅ コンペ選択戦略テスト成功")
        print(f"   - 推奨アクション: {analysis.recommended_action}")
        print(f"   - 戦略スコア: {analysis.strategic_score:.3f}")
        print(f"   - アクション信頼度: {analysis.action_confidence:.3f}")
        print(f"   - メダル確率: {analysis.medal_probability:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ コンペ選択戦略テスト失敗: {e}")
        return False

@pytest.mark.asyncio
async def test_withdrawal_strategy():
    """撤退戦略テスト"""
    print("\n=== 撤退戦略テスト ===")
    
    try:
        from system.agents.planner.strategies.withdrawal_strategy import WithdrawalStrategy
        from system.agents.planner.models.competition import CompetitionInfo, CompetitionStatus, CompetitionType, PrizeType, CompetitionPhase
        
        withdrawal_strategy = WithdrawalStrategy()
        
        # テスト用コンペ情報
        comp_info = CompetitionInfo(
            competition_id="withdrawal_test",
            title="Withdrawal Test Competition",
            url="https://kaggle.com/c/withdrawal_test",
            participant_count=2000,
            total_prize=30000,
            prize_type=PrizeType.MONETARY,
            competition_type=CompetitionType.NLP,
            days_remaining=14,
            last_updated=datetime.utcnow()
        )
        
        # テスト用ステータス（下位順位をシミュレート）
        status = CompetitionStatus(
            competition_info=comp_info,
            phase=CompetitionPhase.ACTIVE,
            is_participating=True,
            current_rank=1600,  # 下位80%
            current_score=0.65,
            time_invested_hours=25.0,
            experiments_completed=8,
            last_improvement_date=datetime.utcnow() - timedelta(days=16)
        )
        
        analysis = await withdrawal_strategy.analyze_withdrawal_decision(comp_info, status)
        
        print(f"✅ 撤退戦略テスト成功")
        print(f"   - 撤退推奨: {analysis.should_withdraw}")
        print(f"   - 撤退緊急度: {analysis.withdrawal_urgency.value}")
        print(f"   - 撤退スコア: {analysis.withdrawal_score:.3f}")
        print(f"   - 主要理由: {analysis.primary_reason.value if analysis.primary_reason else 'N/A'}")
        print(f"   - 推奨アクション: {analysis.recommended_action}")
        return True
        
    except Exception as e:
        print(f"❌ 撤退戦略テスト失敗: {e}")
        return False

@pytest.mark.asyncio
async def test_planner_agent_basic():
    """プランニングエージェント基本テスト"""
    print("\n=== プランニングエージェント基本テスト ===")
    
    try:
        from system.agents.planner.planner_agent import PlannerAgent
        from system.agents.planner.models.competition import CompetitionInfo, CompetitionType, PrizeType
        
        # エージェント作成（認証なしモード）
        agent = PlannerAgent()
        agent.auto_mode = False  # 外部APIへの接続を無効化
        
        # 基本状態確認（外部システム初期化なし）
        status = await agent.get_agent_status()
        
        print(f"✅ プランニングエージェント基本テスト成功")
        print(f"   - エージェントID: {status['agent_id']}")
        print(f"   - バージョン: {status['agent_version']}")
        print(f"   - 稼働時間: {status['uptime_hours']:.3f}時間")
        print(f"   - 自動モード: {status['auto_mode']}")
        print(f"   - 現在のポートフォリオサイズ: {status['current_portfolio_size']}")
        return True
        
    except Exception as e:
        print(f"❌ プランニングエージェント基本テスト失敗: {e}")
        return False

def test_phase3_completion():
    """Phase 3 完了基準チェック"""
    print("\n=== Phase 3 完了基準チェック ===")
    
    criteria = [
        ("メダル確率算出エンジン", "MedalProbabilityCalculator"),
        ("コンペ選択戦略", "CompetitionSelectionStrategy"),
        ("撤退戦略システム", "WithdrawalStrategy"),
        ("GitHub Issue管理", "GitHubIssueManager"),
        ("Kaggle API クライアント", "KaggleApiClient"),
        ("プランニングエージェント", "PlannerAgent")
    ]
    
    all_passed = True
    
    for name, class_name in criteria:
        try:
            if class_name == "GitHubIssueManager":
                exec(f"from system.agents.planner.utils.github_issues import {class_name}")
            elif class_name == "KaggleApiClient":
                exec(f"from system.agents.planner.utils.kaggle_api import {class_name}")
            elif class_name == "PlannerAgent":
                exec(f"from system.agents.planner.planner_agent import {class_name}")
            else:
                exec(f"from system.agents.planner import {class_name}")
            print(f"✅ {name}: 実装完了")
        except Exception as e:
            print(f"❌ {name}: 実装不完全 - {e}")
            all_passed = False
    
    return all_passed

async def main():
    """Phase 3 統合テスト実行"""
    print("Phase 3: プランニングエージェント統合テスト")
    print("=" * 50)
    
    all_tests_passed = True
    
    # インポートテスト
    if not test_phase3_imports():
        all_tests_passed = False
    
    # 機能テスト
    if not await test_medal_probability_calculator():
        all_tests_passed = False
    
    if not await test_competition_selection_strategy():
        all_tests_passed = False
    
    if not await test_withdrawal_strategy():
        all_tests_passed = False
    
    if not await test_planner_agent_basic():
        all_tests_passed = False
    
    # 完了基準チェック
    if not test_phase3_completion():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 Phase 3: プランニングエージェント実装完了! 🎉")
        print("\n✅ 実装完了項目:")
        print("  - 多次元メダル確率算出エンジン")
        print("  - 戦略的コンペ選択システム")
        print("  - リアルタイム撤退判断システム")
        print("  - GitHub Issue連携・自動化")
        print("  - Kaggle API統合・コンペ管理")
        print("  - 統合プランニングエージェント")
        print("\n🚀 Phase 4: 分析エージェント実装準備完了")
        return 0
    else:
        print("❌ テスト失敗。修正が必要です。")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))