---
name: 🎯 Planner - 戦略分析 (Strategy Analysis)
about: プランニングエージェントによる競技選択と戦略立案
title: "[{{COMPETITION_ID}}] Planner: {{COMPETITION_TITLE}} - Medal Strategy Analysis"
labels: agent:planner, status:auto-processing
assignees: ''
---

# 🎯 戦略プランニング分析レポート

## 📊 競技基本情報

### コンペティション詳細
- **タイトル**: {{COMPETITION_TITLE}}
- **URL**: {{COMPETITION_URL}}
- **ID**: `{{COMPETITION_ID}}`
- **種別**: {{COMPETITION_TYPE}}

### 参加状況
| 指標 | 値 |
|------|-----|
| 参加者数 | {{PARTICIPANT_COUNT}} |
| 総賞金 | {{TOTAL_PRIZE}} |
| 残り日数 | {{DAYS_REMAINING}} |
| データサイズ | {{DATA_SIZE}} |

## 🏅 メダル確率分析（LLM多段階評価）

### Stage 1: 競技深層分析
```
評価完了度: {{ANALYSIS_COMPLETENESS}}%
分析深度: {{ANALYSIS_DEPTH}}
```

**主要な発見**:
- {{KEY_FINDING_1}}
- {{KEY_FINDING_2}}
- {{KEY_FINDING_3}}

### Stage 2: 成功パターン抽出
| パターン | 適合度 | 実装難易度 |
|---------|--------|------------|
| {{PATTERN_1}} | {{MATCH_1}}% | {{DIFFICULTY_1}} |
| {{PATTERN_2}} | {{MATCH_2}}% | {{DIFFICULTY_2}} |
| {{PATTERN_3}} | {{MATCH_3}}% | {{DIFFICULTY_3}} |

### Stage 3: メダル確率計算
| メダル | 確率 | 信頼区間 |
|--------|------|----------|
| 🥇 金 | {{GOLD_PROB}}% | {{GOLD_CI}} |
| 🥈 銀 | {{SILVER_PROB}}% | {{SILVER_CI}} |
| 🥉 銅 | {{BRONZE_PROB}}% | {{BRONZE_CI}} |
| **総合** | **{{OVERALL_PROB}}%** | **{{OVERALL_CI}}** |

## 🎲 戦略的推奨事項

### 推奨アクション
**{{RECOMMENDED_ACTION}}**

**アクション信頼度**: {{ACTION_CONFIDENCE}}%

### 詳細戦略
1. **初期フェーズ** (Day 1-{{PHASE1_END}}):
   - {{PHASE1_STRATEGY}}

2. **中盤フェーズ** (Day {{PHASE2_START}}-{{PHASE2_END}}):
   - {{PHASE2_STRATEGY}}

3. **最終フェーズ** (Day {{PHASE3_START}}-終了):
   - {{PHASE3_STRATEGY}}

### リソース配分計画
- **GPU時間予算**: {{GPU_HOURS}}時間
- **優先度**: {{PRIORITY_LEVEL}}
- **並行実行可能性**: {{PARALLEL_CAPABILITY}}

## ⚠️ リスク評価

### 主要リスク
| リスク | 発生確率 | 影響度 | 対策 |
|--------|---------|--------|------|
| {{RISK_1}} | {{RISK_1_PROB}} | {{RISK_1_IMPACT}} | {{RISK_1_MITIGATION}} |
| {{RISK_2}} | {{RISK_2_PROB}} | {{RISK_2_IMPACT}} | {{RISK_2_MITIGATION}} |

### 撤退条件
- {{WITHDRAWAL_CONDITION_1}}
- {{WITHDRAWAL_CONDITION_2}}
- {{WITHDRAWAL_CONDITION_3}}

## 📈 期待成果

### 成功指標
- **目標順位**: TOP {{TARGET_PERCENTILE}}%
- **最低達成ライン**: TOP {{MINIMUM_PERCENTILE}}%
- **ROI期待値**: {{EXPECTED_ROI}}

### 学習機会
- {{LEARNING_OPPORTUNITY_1}}
- {{LEARNING_OPPORTUNITY_2}}

## 🔄 次のステップ

### Analyzer Agent への引き継ぎ事項
1. **重点分析領域**: {{FOCUS_AREA}}
2. **推奨技術調査**: {{RECOMMENDED_TECHNIQUES}}
3. **ベンチマーク目標**: {{BENCHMARK_TARGET}}

### 実行タイムライン
```mermaid
gantt
    title 実行計画
    dateFormat  YYYY-MM-DD
    section Planning
    戦略立案完了    :done, p1, {{TODAY}}, 1d
    section Analysis
    技術分析開始    :active, a1, after p1, {{ANALYSIS_DURATION}}d
    section Execution
    実装開始予定    :e1, after a1, {{EXECUTION_DURATION}}d
```

---

## 📋 メタデータ

- **分析実行時刻**: {{ANALYSIS_TIMESTAMP}}
- **処理時間**: {{PROCESSING_TIME}}秒
- **LLM呼び出し回数**: {{LLM_CALLS}}回
- **信頼度スコア**: {{CONFIDENCE_SCORE}}/100

### タグ
`#kaggle` `#{{COMPETITION_TYPE}}` `#medal-probability-{{PROBABILITY_TIER}}`

---
*このIssueは自動生成されました by Planner Agent v2.0 with LLM Enhancement*