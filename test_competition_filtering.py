#!/usr/bin/env python3
"""
コンペ選択フィルタリング機能テスト

アワードポイント・メダル獲得可能なコンペのみが選択されることを確認する。
"""

import asyncio
import logging
import os
import subprocess
from system.dynamic_competition_manager.dynamic_competition_manager import DynamicCompetitionManager

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


async def test_competition_filtering():
    """コンペフィルタリングテスト"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 コンペ選択フィルタリング機能テスト開始")
    print("=" * 60)
    
    # GitHub認証トークン取得
    github_token = get_github_token()
    
    # マネージャー初期化
    manager = DynamicCompetitionManager(
        github_token=github_token,
        repo_name="hkrhd/kaggle-claude-mother"
    )
    
    # 新競技スキャン実行
    print("📊 新競技スキャン実行中...")
    new_competitions = await manager.scan_new_competitions()
    
    print("\n🎯 スキャン結果:")
    print(f"選択されたコンペ数: {len(new_competitions)}件")
    
    if new_competitions:
        print("\n✅ 選択されたコンペ一覧:")
        for i, comp in enumerate(new_competitions, 1):
            print(f"  {i}. {comp['name']}")
            print(f"     カテゴリ: {comp.get('competition_category', 'unknown')}")
            print(f"     賞金: ${comp.get('prize_amount', 0):,}")
            print(f"     メダル対象: {comp.get('awards_medals', 'unknown')}")
            print(f"     メダル確率: {comp.get('medal_probability', 0):.1%}")
            print()
    else:
        print("❌ 選択されたコンペがありません")
    
    # 期待される結果の確認
    print("🔍 期待結果との比較:")
    
    expected_competitions = {
        "tabular-playground-series-apr-2024": "Featured競技 (賞金$25,000)",
        "plant-pathology-2024-fgvc11": "Research競技 (賞金$15,000)"
    }
    
    excluded_competitions = {
        "house-prices-advanced": "Getting Started競技 (賞金なし)",
        "nlp-getting-started": "Getting Started競技 (賞金なし)"
    }
    
    selected_ids = {comp['id'] for comp in new_competitions}
    
    # 期待される競技が選択されているか確認
    for comp_id, description in expected_competitions.items():
        if comp_id in selected_ids:
            print(f"✅ {description} - 正常に選択")
        else:
            print(f"❌ {description} - 選択されていない（エラー）")
    
    # 除外される競技が選択されていないか確認
    for comp_id, description in excluded_competitions.items():
        if comp_id in selected_ids:
            print(f"❌ {description} - 誤って選択（エラー）")
        else:
            print(f"✅ {description} - 正常に除外")
    
    print("\n" + "=" * 60)
    
    # 最終判定
    expected_count = len(expected_competitions)
    actual_count = len(new_competitions)
    
    if actual_count == expected_count:
        print(f"🎉 テスト成功: {actual_count}件のメダル獲得可能コンペのみ選択")
        return True
    else:
        print(f"❌ テスト失敗: 期待{expected_count}件、実際{actual_count}件")
        return False

async def test_medal_eligibility():
    """メダル対象判定ロジック単体テスト"""
    
    print("\n🔬 メダル対象判定ロジック単体テスト")
    print("-" * 40)
    
    manager = DynamicCompetitionManager("test", "test")
    
    test_cases = [
        # Featured競技 (賞金付き) -> True
        {
            "name": "Featured Competition",
            "competition_category": "featured",
            "prize_amount": 50000,
            "awards_medals": True,
            "expected": True
        },
        # Research競技 (賞金付き) -> True
        {
            "name": "Research Competition with Prize",
            "competition_category": "research", 
            "prize_amount": 15000,
            "awards_medals": True,
            "expected": True
        },
        # Getting Started競技 -> False
        {
            "name": "Getting Started Competition",
            "competition_category": "getting-started",
            "prize_amount": 0,
            "awards_medals": False,
            "expected": False
        },
        # Knowledge競技 -> False
        {
            "name": "Knowledge Competition",
            "competition_category": "knowledge",
            "prize_amount": 0,
            "awards_medals": False,
            "expected": False
        },
        # Playground競技 (賞金なし) -> False
        {
            "name": "Playground Competition",
            "competition_category": "playground",
            "prize_amount": 0,
            "awards_medals": False,
            "expected": False
        },
        # Playground競技 (賞金付き) -> True
        {
            "name": "Playground Competition with Prize",
            "competition_category": "playground",
            "prize_amount": 5000,
            "awards_medals": True,
            "expected": True
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        result = manager._is_medal_eligible_competition(test_case)
        expected = test_case["expected"]
        
        if result == expected:
            print(f"✅ {test_case['name']}: {result} (正解)")
        else:
            print(f"❌ {test_case['name']}: {result}, 期待値: {expected}")
            all_passed = False
    
    return all_passed

async def main():
    """メインテスト実行"""
    
    print("🚀 コンペ選択フィルタリング機能 - 総合テスト")
    print("=" * 60)
    
    # テスト1: 総合フィルタリングテスト
    filtering_test_passed = await test_competition_filtering()
    
    # テスト2: 単体ロジックテスト
    eligibility_test_passed = await test_medal_eligibility()
    
    print("\n" + "=" * 60)
    print("🏁 テスト結果まとめ")
    
    if filtering_test_passed and eligibility_test_passed:
        print("🎉 全テスト成功: Knowledge/Practice競技は正常に除外されます")
        return True
    else:
        print("❌ テスト失敗: システム改修に問題があります")
        return False

if __name__ == "__main__":
    asyncio.run(main())