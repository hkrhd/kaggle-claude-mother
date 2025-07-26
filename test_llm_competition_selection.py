#!/usr/bin/env python3
"""
LLMベース競技選択システム統合テスト

CompetitionSelectorAgent と DynamicCompetitionManager の統合動作確認
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

from system.dynamic_competition_manager.dynamic_competition_manager import DynamicCompetitionManager
from system.agents.competition_selector.competition_selector_agent import (
    CompetitionSelectorAgent, SelectionStrategy
)


async def test_llm_competition_selection():
    """LLM競技選択統合テスト"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🧪 LLM競技選択システム統合テスト開始")
    
    try:
        # DynamicCompetitionManager初期化（LLM選択統合版）
        manager = DynamicCompetitionManager(
            github_token="dummy_token",  # テスト用ダミートークン
            repo_name="test/repo"
        )
        
        logger.info("✅ DynamicCompetitionManager初期化完了")
        
        # 1. 新競技スキャン・LLM選択実行
        logger.info("🔍 新競技スキャン・LLM選択実行テスト...")
        
        selected_competitions = await manager.scan_new_competitions()
        
        logger.info(f"📊 競技選択結果: {len(selected_competitions)}競技")
        for comp in selected_competitions:
            logger.info(f"  - {comp['name']}: {comp.get('medal_probability', 0):.1%}確率")
            if comp.get('llm_selected'):
                logger.info(f"    🧠 LLM推奨: {comp.get('llm_recommendation', 'N/A')} (スコア: {comp.get('llm_score', 0):.2f})")
        
        # 2. LLM選択システム状態確認
        logger.info("📈 LLM選択システム状態確認...")
        
        llm_status = await manager.get_llm_selection_status()
        
        logger.info("🎯 LLM選択システム状態:")
        logger.info(f"  - エージェントID: {llm_status.get('llm_selection_agent_id', 'N/A')}")
        logger.info(f"  - 選択決定回数: {llm_status.get('selection_decisions_made', 0)}")
        logger.info(f"  - ポートフォリオサイズ: {llm_status.get('current_portfolio_size', 0)}")
        logger.info(f"  - メダル確率合計: {llm_status.get('portfolio_medal_probability', 0):.2f}")
        logger.info(f"  - GPU予算使用率: {llm_status.get('gpu_budget_utilization', 0):.1%}")
        
        # 3. 全体システム状態確認
        logger.info("🖥️ 全体システム状態確認...")
        
        system_status = manager.get_system_status()
        
        logger.info("⚙️ システム状態:")
        logger.info(f"  - アクティブ競技数: {system_status.get('active_competitions_count', 0)}")
        logger.info(f"  - LLM選択有効: {system_status.get('llm_selection_enabled', False)}")
        logger.info(f"  - 最終スキャン時刻: {system_status.get('last_scan_time', 'N/A')}")
        logger.info(f"  - 最終LLM選択時刻: {system_status.get('last_llm_selection_time', 'N/A')}")
        logger.info(f"  - システム健全性: {system_status.get('system_health', 'unknown')}")
        
        # 4. 単独競技選択エージェントテスト
        logger.info("🤖 単独競技選択エージェントテスト...")
        
        selector = CompetitionSelectorAgent(
            github_token="dummy_token",
            repo_name="test/repo"
        )
        
        # テスト用競技データ
        test_competitions = [
            {
                "id": "test-tabular-comp",
                "name": "Test Tabular Competition",
                "type": "tabular",
                "deadline": (datetime.utcnow() + timedelta(days=20)).isoformat(),
                "participants": 1500,
                "prize_amount": 50000,
                "competition_category": "featured",
                "awards_medals": True
            },
            {
                "id": "test-cv-comp",
                "name": "Test Computer Vision Competition",
                "type": "computer_vision",
                "deadline": (datetime.utcnow() + timedelta(days=40)).isoformat(),
                "participants": 2200,
                "prize_amount": 25000,
                "competition_category": "research",
                "awards_medals": True
            }
        ]
        
        # 各戦略でのテスト実行
        strategies = [
            SelectionStrategy.CONSERVATIVE,
            SelectionStrategy.BALANCED,
            SelectionStrategy.AGGRESSIVE
        ]
        
        for strategy in strategies:
            logger.info(f"🎯 戦略テスト: {strategy.value}")
            
            try:
                decision = await selector.evaluate_available_competitions(
                    available_competitions=test_competitions,
                    strategy=strategy
                )
                
                logger.info(f"  ✅ 選択完了: {len(decision.selected_competitions)}競技選択")
                logger.info(f"  📊 期待メダル数: {decision.expected_medal_count:.2f}")
                logger.info(f"  🎖️ 決定信頼度: {decision.decision_confidence:.1%}")
                
                for comp in decision.selected_competitions:
                    logger.info(f"    - {comp.name}: LLM推奨 {comp.llm_recommendation} (スコア: {comp.llm_score:.2f})")
                
            except Exception as e:
                logger.error(f"  ❌ 戦略{strategy.value}テスト失敗: {e}")
        
        # 5. ポートフォリオ状態確認
        logger.info("📈 ポートフォリオ状態確認...")
        
        portfolio_status = await selector.get_current_portfolio_status()
        
        logger.info("💼 ポートフォリオ状態:")
        logger.info(f"  - 選択決定回数: {portfolio_status.get('selection_decisions_made', 0)}")
        logger.info(f"  - アクティブ競技数: {portfolio_status.get('active_competitions_count', 0)}")
        logger.info(f"  - GPU予算使用率: {portfolio_status.get('gpu_budget_utilization', 0):.1%}")
        logger.info(f"  - 平均信頼度: {portfolio_status.get('avg_confidence_level', 0):.1%}")
        
        logger.info("🎉 LLM競技選択システム統合テスト完了")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 統合テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """メイン関数"""
    
    print("🧪 LLM競技選択システム統合テスト")
    print("=" * 50)
    
    success = await test_llm_competition_selection()
    
    if success:
        print("\n✅ 全テスト成功")
        return 0
    else:
        print("\n❌ テスト失敗")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)