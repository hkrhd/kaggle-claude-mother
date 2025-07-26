---
name: 📚 Retrospective - 振り返り学習 (Learning & Improvement)
about: レトロスペクティブエージェントによる学習抽出とシステム改善
title: "[{{COMPETITION_ID}}] Retrospective: {{COMPETITION_TITLE}} - Learning & System Evolution"
labels: agent:retrospective, status:learning
assignees: ''
---

# 📚 競技振り返り・学習レポート

## 🏆 競技結果サマリー
- **最終順位**: {{FINAL_RANK}}/{{TOTAL_TEAMS}} (TOP {{PERCENTILE}}%)
- **メダル獲得**: {{MEDAL_ACHIEVED}}
- **最終スコア**: {{FINAL_SCORE}}
- **目標達成度**: {{GOAL_ACHIEVEMENT}}%

## 🧠 4段階学習分析

### Stage 1: 競技包括分析
**パフォーマンス評価**: {{PERFORMANCE_RATING}}

#### 成功要因トップ3
1. **{{SUCCESS_FACTOR_1}}**
   - 貢献度: {{CONTRIBUTION_1}}%
   - 再現可能性: {{REPRODUCIBILITY_1}}/5

2. **{{SUCCESS_FACTOR_2}}**
   - 貢献度: {{CONTRIBUTION_2}}%
   - 再現可能性: {{REPRODUCIBILITY_2}}/5

3. **{{SUCCESS_FACTOR_3}}**
   - 貢献度: {{CONTRIBUTION_3}}%
   - 再現可能性: {{REPRODUCIBILITY_3}}/5

#### 改善必要領域
| 領域 | 現状 | 理想 | ギャップ | 優先度 |
|------|------|------|----------|--------|
| {{AREA_1}} | {{CURRENT_1}} | {{IDEAL_1}} | {{GAP_1}} | {{PRIORITY_1}} |
| {{AREA_2}} | {{CURRENT_2}} | {{IDEAL_2}} | {{GAP_2}} | {{PRIORITY_2}} |

### Stage 2: 学習抽出・知識体系化

#### 転移可能なパターン
| パターン名 | 説明 | 適用条件 | 期待効果 |
|-----------|------|---------|----------|
| {{PATTERN_1}} | {{DESC_1}} | {{CONDITION_1}} | {{EFFECT_1}} |
| {{PATTERN_2}} | {{DESC_2}} | {{CONDITION_2}} | {{EFFECT_2}} |

#### 新たな発見・洞察
> 💡 **{{KEY_INSIGHT_1}}**
> 
> 詳細: {{INSIGHT_DETAIL_1}}

> 💡 **{{KEY_INSIGHT_2}}**
> 
> 詳細: {{INSIGHT_DETAIL_2}}

#### アンチパターン（避けるべき事項）
- ❌ **{{ANTI_PATTERN_1}}**: {{ANTI_REASON_1}}
- ❌ **{{ANTI_PATTERN_2}}**: {{ANTI_REASON_2}}

### Stage 3: システム改善計画

#### 即時改善項目
| 改善項目 | 現在の問題 | 提案する解決策 | 期待効果 |
|---------|-----------|---------------|----------|
| {{IMPROVE_1}} | {{PROBLEM_1}} | {{SOLUTION_1}} | {{BENEFIT_1}} |
| {{IMPROVE_2}} | {{PROBLEM_2}} | {{SOLUTION_2}} | {{BENEFIT_2}} |

#### アーキテクチャ改善提案
```mermaid
graph LR
    A[現在のフロー] --> B[問題点]
    B --> C[改善案]
    C --> D[期待される効果]
    
    subgraph 具体例
    E[{{CURRENT_ARCH}}] --> F[{{ARCH_PROBLEM}}]
    F --> G[{{IMPROVED_ARCH}}]
    G --> H[{{ARCH_BENEFIT}}]
    end
```

### Stage 4: 自動実装結果

#### 実装された改善
| 改善内容 | ファイル | 変更前 | 変更後 | 効果測定 |
|---------|---------|--------|--------|----------|
| {{IMPL_1}} | {{FILE_1}} | {{BEFORE_1}} | {{AFTER_1}} | {{MEASURE_1}} |
| {{IMPL_2}} | {{FILE_2}} | {{BEFORE_2}} | {{AFTER_2}} | {{MEASURE_2}} |

## 📊 エージェント別パフォーマンス分析

### エージェント効率性
| エージェント | 精度 | 速度 | リソース効率 | 総合評価 |
|-------------|------|------|-------------|----------|
| Planner | {{PLAN_ACC}}% | {{PLAN_SPEED}} | {{PLAN_EFF}} | {{PLAN_SCORE}}/10 |
| Analyzer | {{ANAL_ACC}}% | {{ANAL_SPEED}} | {{ANAL_EFF}} | {{ANAL_SCORE}}/10 |
| Executor | {{EXEC_ACC}}% | {{EXEC_SPEED}} | {{EXEC_EFF}} | {{EXEC_SCORE}}/10 |
| Monitor | {{MON_ACC}}% | {{MON_SPEED}} | {{MON_EFF}} | {{MON_SCORE}}/10 |

### 連携効率性
```
情報フロー効率: {{INFO_FLOW_EFF}}%
意思決定一貫性: {{DECISION_CONSISTENCY}}%
全体最適化度: {{OVERALL_OPTIMIZATION}}%
```

## 🎯 将来競技への適用戦略

### 更新された選択基準
1. **{{NEW_CRITERION_1}}** (重み: {{WEIGHT_1}})
2. **{{NEW_CRITERION_2}}** (重み: {{WEIGHT_2}})
3. **{{NEW_CRITERION_3}}** (重み: {{WEIGHT_3}})

### 技術優先順位の変更
| 技術カテゴリ | 旧優先度 | 新優先度 | 変更理由 |
|-------------|---------|---------|----------|
| {{TECH_CAT_1}} | {{OLD_PRI_1}} | {{NEW_PRI_1}} | {{REASON_1}} |
| {{TECH_CAT_2}} | {{OLD_PRI_2}} | {{NEW_PRI_2}} | {{REASON_2}} |

### リスク管理の更新
- **新規識別リスク**: {{NEW_RISK}}
- **緩和戦略**: {{MITIGATION_STRATEGY}}
- **早期警告指標**: {{EARLY_WARNING}}

## 💾 知識ベース更新

### 追加されたベストプラクティス
```yaml
- name: {{BEST_PRACTICE_1}}
  context: {{CONTEXT_1}}
  implementation: {{IMPLEMENTATION_1}}
  validation: {{VALIDATION_1}}

- name: {{BEST_PRACTICE_2}}
  context: {{CONTEXT_2}}
  implementation: {{IMPLEMENTATION_2}}
  validation: {{VALIDATION_2}}
```

### 更新された意思決定テンプレート
| 決定タイプ | 評価基準 | 優先順位 |
|-----------|---------|---------|
| {{DECISION_1}} | {{CRITERIA_1}} | {{PRIORITY_1}} |
| {{DECISION_2}} | {{CRITERIA_2}} | {{PRIORITY_2}} |

## 🚀 次回への準備

### システム準備状況
- ✅ コード最適化完了: {{CODE_OPT_ITEMS}}項目
- ✅ 設定更新完了: {{CONFIG_UPDATES}}項目
- ✅ テンプレート改良: {{TEMPLATE_UPDATES}}項目
- ✅ 知識ベース拡充: {{KB_ADDITIONS}}項目

### 推奨される次の競技
1. **{{NEXT_COMP_1}}**
   - 理由: {{NEXT_REASON_1}}
   - 予想メダル確率: {{NEXT_PROB_1}}%

2. **{{NEXT_COMP_2}}**
   - 理由: {{NEXT_REASON_2}}
   - 予想メダル確率: {{NEXT_PROB_2}}%

---

## 📋 振り返りメタデータ

- **分析完了時刻**: {{ANALYSIS_TIMESTAMP}}
- **学習抽出項目数**: {{LEARNING_ITEMS}}
- **実装改善数**: {{IMPROVEMENTS_IMPLEMENTED}}
- **システム性能向上**: {{SYSTEM_IMPROVEMENT}}%
- **次回メダル確率向上**: +{{MEDAL_PROB_INCREASE}}%

### 分析深度
- **データ分析量**: {{DATA_ANALYZED}}
- **LLM分析ステージ**: {{LLM_STAGES}}
- **相関分析項目**: {{CORRELATION_ITEMS}}
- **将来予測精度**: {{PREDICTION_ACCURACY}}%

---
*このIssueは自動生成されました by Retrospective Agent v2.0 with Continuous Learning & System Evolution*