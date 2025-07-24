# 戦略プランニングエージェント実装計画書

## 概要
READMEの設計に基づく最初のエージェント（`agent:planner`）の実装計画。メダル確率算出・戦略的コンペ選択・撤退判断を担当する核心エージェント。

## 実装アーキテクチャ

### 1. 技術スタック選択

#### Python + asyncio
**採用理由**: 
- GitHub Issue API操作の非同期処理が必要
- Kaggle API、Web scraping処理の並列実行効率
- uvによる軽量環境管理との親和性

#### httpx + aiofiles
**採用理由**:
- GitHub Issue APIとKaggle API同時操作
- ファイルI/O非同期化でパフォーマンス最適化
- requests比較で非同期処理完全対応

#### pydantic + dataclasses  
**採用理由**:
- コンペ情報・確率計算の型安全性保証
- Issue作成時の構造化データ検証
- JSON⇔オブジェクト変換の自動化

### 2. コアモジュール設計

```
system/agents/planner/
├── __init__.py
├── planner_agent.py        # メインエージェントクラス
├── models/
│   ├── competition.py      # コンペ情報データクラス
│   └── probability.py      # 確率計算モデル
├── calculators/
│   ├── medal_probability.py    # メダル確率算出ロジック
│   └── competition_scanner.py  # アクティブコンペスキャン
├── strategies/
│   ├── selection_strategy.py   # コンペ選択戦略
│   └── withdrawal_strategy.py  # 撤退判断戦略
└── utils/
    ├── kaggle_api.py       # Kaggle API操作
    └── github_issues.py    # Issue作成・更新
```

**設計根拠**:
- **職責分離**: 確率計算・戦略判断・API操作を完全分離
- **テスタビリティ**: 各モジュール独立テスト可能
- **拡張性**: 新しい確率算出手法・戦略を容易追加

### 3. メダル確率算出アルゴリズム

#### 多次元確率モデル
```python
medal_probability = base_score * domain_match * timing_factor * resource_efficiency

# 各要素の重み付き計算
base_score = participants_factor * prize_factor * competition_type_factor
domain_match = skill_alignment_score * past_performance_weight  
timing_factor = remaining_time / total_time * urgency_weight
resource_efficiency = estimated_compute_cost / available_resources
```

**採用根拠**:
- **多面的評価**: 単純参加者数でなく包括的リスク評価
- **動的調整**: 残り時間・リソース状況の自動反映
- **学習適応**: 過去実績による個人適応スコア

#### グランドマスター事例分析統合
**実装方針**:
- Owen Zhang/Abhishek Thakur成功パターンDB化
- 過去優勝解法の技術難易度定量化
- 個人スキルセットとのマッチング係数算出

### 4. GitHub Issue連携システム

#### 原子性操作保証
```python
async def create_strategy_issue(competition_info):
    # 重複チェック→作成→ラベル付けを原子的実行
    existing = await check_duplicate_issue(comp=competition_info.name, agent="planner")
    if existing:
        return await update_existing_issue(existing.number, new_analysis)
    
    # UUID suffix付きタイトルで重複回避
    title = f"[{competition_info.name}] planner: Medal Strategy Analysis - {uuid4().hex[:8]}"
    
    return await create_issue_with_retry(title, body, labels)
```

**採用根拠**:
- **競合防止**: 複数プランナー同時実行時の重複Issue防止
- **デッドロック回避**: 厳密順序実行（planner→analyzer→executor→monitor）
- **障害回復**: 指数バックオフによる自動リトライ

#### 厳密ラベリングシステム
```python
REQUIRED_LABELS = [
    f"agent:planner",
    f"comp:{competition_name}",
    f"status:auto-processing", 
    f"priority:medal-critical",
    f"medal-probability:{probability_tier}"
]
```

**設計意図**:
- **一意識別**: comp+agentラベルでIssue完全識別
- **状態管理**: status変更による次エージェント自動起動
- **優先制御**: medal-probabilityによる実行優先度調整

### 5. 動的コンペ管理機能

#### 週2回最適化スケジューラ
```python
@schedule.every().tuesday.at("07:00")
@schedule.every().friday.at("07:00") 
async def dynamic_optimization():
    current_competitions = await get_active_competitions()
    available_competitions = await scan_kaggle_competitions()
    
    optimal_portfolio = await calculate_optimal_3_competitions(
        current=current_competitions,
        available=available_competitions,
        threshold_score=0.7
    )
    
    if better_opportunity_found(optimal_portfolio, current_competitions):
        await execute_portfolio_rebalancing(optimal_portfolio)
```

**採用根拠**:
- **機会損失防止**: 高確率コンペの見逃し完全回避
- **リソース最適化**: 常に最高期待値3コンペ維持
- **自動判断**: 人間介入なし撤退・参戦決定

#### 撤退判断アルゴリズム
```python
def should_withdraw(competition, current_rank, remaining_time):
    medal_threshold = calculate_medal_cutoff(competition.participants)
    probability_drop = (current_rank - medal_threshold) / medal_threshold
    
    # 複数条件での撤退判断
    if probability_drop > 0.5 and remaining_time < 0.3:  # メダル圏外確定
        return True
    if better_opportunity_available() and probability_drop > 0.2:  # 機会コスト
        return True
    return False
```

**論理根拠**:
- **データ駆動**: 現在順位・残り時間の定量的判断
- **機会コスト考慮**: より良いコンペ発見時の柔軟撤退
- **金メダル優先**: 「金1個>銀2個」原則の数値化

### 6. 初期実装スコープ

#### Phase 1: コア機能（1週間）
1. **基本メダル確率算出**: 参加者数・賞金・残り時間ベース
2. **GitHub Issue作成**: 基本的なIssue連携機能
3. **Kaggle API統合**: アクティブコンペ一覧取得
4. **シンプル戦略判断**: 上位3コンペ選択ロジック

#### Phase 2: 高度機能（2週間）
1. **動的最適化**: 週2回自動スキャン・入れ替え
2. **撤退判断**: リアルタイム順位監視・撤退決定
3. **障害回復**: Issue競合・API制限への対応
4. **グランドマスター分析統合**: 過去事例パターンマッチング

#### Phase 3: 最適化（1週間）
1. **確率モデル改良**: 実績フィードバック学習
2. **パフォーマンス最適化**: API呼び出し・処理時間改善  
3. **監視・ログ**: 動作状況可視化・デバッグ機能
4. **統合テスト**: 他エージェントとの連携検証

### 7. テスト戦略

#### 単体テスト
- 確率計算アルゴリズム精度検証
- API操作（GitHub/Kaggle）モック化テスト
- 撤退判断ロジック境界値テスト

#### 統合テスト  
- Issue作成→analyzer起動の自動連携テスト
- 複数コンペ同時処理での競合状態テスト
- API制限・障害時の回復動作テスト

#### 本番環境テスト
- 実際のKaggleコンペでの確率精度検証
- 週2回スケジュール動作確認
- 長期運用でのメモリリーク・パフォーマンス監視

## 成功指標

1. **確率精度**: 算出確率とメダル実績の相関係数 > 0.8
2. **応答性能**: Issue作成から次エージェント起動まで < 5分
3. **安定性**: 1ヶ月連続運用でのクラッシュ・重複Issue発生ゼロ
4. **効率性**: 最適3コンペ維持率 > 95%（週2回チェック）

## リスク対策

### 技術リスク
- **API制限**: 指数バックオフ・キューイング機能
- **競合状態**: 原子的操作・ロック機能
- **データ不整合**: 厳密スキーマ検証・ロールバック機能

### 運用リスク  
- **無限ループ**: タイムアウト・最大試行回数制限
- **リソース枯渇**: メモリ・API制限監視・自動停止
- **誤判断**: 人間による緊急停止・設定調整機能

この実装計画により、READMEの戦略プランニングエージェント仕様を完全実現し、自律的メダル獲得システムの基盤を構築します。