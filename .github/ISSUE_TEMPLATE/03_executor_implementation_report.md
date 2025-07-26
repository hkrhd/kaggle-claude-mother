---
name: ⚡ Executor - 実装実行 (Implementation & Execution)
about: エグゼキューターエージェントによる実装と実験管理
title: "[{{COMPETITION_ID}}] Executor: {{IMPLEMENTATION_PHASE}} - Progress Report"
labels: agent:executor, status:implementing
assignees: ''
---

# ⚡ 実装・実行レポート

## 🔗 前段階リンク
- **Analyzer Issue**: #{{ANALYZER_ISSUE_NUMBER}}
- **推奨技術**: {{RECOMMENDED_TECHNIQUES}}
- **目標スコア**: {{TARGET_SCORE}}

## 🎯 LLM提出戦略分析

### Stage 1: 実験戦略最適化
**現在の状況**
- CV スコア: {{CURRENT_CV_SCORE}}
- LB スコア: {{CURRENT_LB_SCORE}} ({{LB_RANK}}位/{{TOTAL_TEAMS}}チーム)
- 改善トレンド: {{IMPROVEMENT_TREND}}

**推奨実験優先順位**
| 実験 | 期待改善 | 所要時間 | 優先度 |
|------|---------|---------|--------|
| {{EXP_1}} | +{{GAIN_1}}% | {{TIME_1}}h | {{PRIORITY_1}} |
| {{EXP_2}} | +{{GAIN_2}}% | {{TIME_2}}h | {{PRIORITY_2}} |
| {{EXP_3}} | +{{GAIN_3}}% | {{TIME_3}}h | {{PRIORITY_3}} |

### Stage 2: 競合動向分析
**リーダーボード動態**
```
トップとの差: {{GAP_TO_TOP}}
メダル圏との差: {{GAP_TO_MEDAL}}
追い上げ速度: {{CATCH_UP_RATE}}
```

**競合の推定戦略**
- 🥇 上位チーム: {{TOP_TEAM_STRATEGY}}
- 🏃 急上昇チーム: {{RISING_TEAM_STRATEGY}}
- 🎯 要注意: {{WATCH_OUT_TEAMS}}

### Stage 3: 提出判断
**提出推奨度**: {{SUBMISSION_CONFIDENCE}}%

**判断根拠**:
- {{SUBMISSION_REASON_1}}
- {{SUBMISSION_REASON_2}}
- {{SUBMISSION_REASON_3}}

## 💻 実装進捗

### 完了タスク ✅
- [x] {{COMPLETED_1}} - スコア改善: +{{IMPROVEMENT_1}}%
- [x] {{COMPLETED_2}} - スコア改善: +{{IMPROVEMENT_2}}%
- [x] {{COMPLETED_3}} - スコア改善: +{{IMPROVEMENT_3}}%

### 実行中タスク 🔄
- [ ] {{IN_PROGRESS_1}} - 進捗: {{PROGRESS_1}}%
  - 推定完了: {{ETA_1}}
  - 現在の課題: {{ISSUE_1}}

### 待機中タスク ⏳
- [ ] {{PENDING_1}}
- [ ] {{PENDING_2}}

## 🔬 実験結果サマリー

### 成功した実験
| 実験名 | ベースライン | 結果 | 改善率 | 実装時間 |
|--------|------------|------|--------|----------|
| {{SUCCESS_1}} | {{BASE_1}} | {{RESULT_1}} | +{{IMPROVE_1}}% | {{TIME_1}}h |
| {{SUCCESS_2}} | {{BASE_2}} | {{RESULT_2}} | +{{IMPROVE_2}}% | {{TIME_2}}h |

### 失敗した実験
| 実験名 | 期待値 | 結果 | 原因分析 |
|--------|--------|------|----------|
| {{FAILED_1}} | {{EXPECT_1}} | {{ACTUAL_1}} | {{REASON_1}} |

## 🏗️ 実装詳細

### アーキテクチャ
```python
# 主要なパイプライン構造
{{ARCHITECTURE_SNIPPET}}
```

### 主要な工夫点
1. **{{INNOVATION_1}}**
   ```python
   {{CODE_SNIPPET_1}}
   ```

2. **{{INNOVATION_2}}**
   ```python
   {{CODE_SNIPPET_2}}
   ```

## 📈 パフォーマンス指標

### モデル性能
| モデル | CV Score | LB Score (Public) | 訓練時間 | 推論時間 |
|--------|----------|------------------|----------|----------|
| {{MODEL_1}} | {{CV_1}} | {{LB_1}} | {{TRAIN_1}} | {{INFER_1}} |
| {{MODEL_2}} | {{CV_2}} | {{LB_2}} | {{TRAIN_2}} | {{INFER_2}} |
| **アンサンブル** | **{{CV_ENS}}** | **{{LB_ENS}}** | - | {{INFER_ENS}} |

### リソース使用状況
- **GPU時間消費**: {{GPU_USED}}/{{GPU_BUDGET}}時間 ({{GPU_PERCENT}}%)
- **メモリ最大使用**: {{MAX_MEMORY}}GB
- **ストレージ使用**: {{STORAGE_USED}}GB

## 🎲 次の戦略

### 短期計画（24時間以内）
1. {{SHORT_TERM_1}}
2. {{SHORT_TERM_2}}

### 中期計画（2-3日）
1. {{MID_TERM_1}}
2. {{MID_TERM_2}}

### 最終日戦略
- **提出タイミング**: {{SUBMISSION_TIMING}}
- **アンサンブル戦略**: {{ENSEMBLE_STRATEGY}}
- **リスクヘッジ**: {{RISK_HEDGE}}

## ⚠️ 課題と対策

### 現在の課題
| 課題 | 影響度 | 対策 | 期限 |
|------|--------|------|------|
| {{ISSUE_1}} | {{IMPACT_1}} | {{SOLUTION_1}} | {{DEADLINE_1}} |
| {{ISSUE_2}} | {{IMPACT_2}} | {{SOLUTION_2}} | {{DEADLINE_2}} |

## 🔄 Monitor Agent への引き継ぎ

### 監視重点項目
- **メモリリーク監視**: {{MEMORY_WATCH}}
- **実行時間監視**: {{TIME_WATCH}}
- **スコア異常検知**: {{SCORE_WATCH}}

### アラート設定
- CV/LBスコア乖離 > {{SCORE_GAP_THRESHOLD}}%
- GPU使用率 > {{GPU_THRESHOLD}}%
- エラー率 > {{ERROR_THRESHOLD}}%

---

## 📋 実行メタデータ

- **報告生成時刻**: {{REPORT_TIMESTAMP}}
- **総実験数**: {{TOTAL_EXPERIMENTS}}
- **成功率**: {{SUCCESS_RATE}}%
- **LLM提出分析**: {{LLM_CALLS}}回実行
- **現在のメダル確率**: {{MEDAL_PROBABILITY}}%

### 実行環境
- **使用環境**: {{EXECUTION_ENV}}
- **Pythonバージョン**: {{PYTHON_VERSION}}
- **主要ライブラリ**: {{KEY_LIBRARIES}}

---
*このIssueは自動生成されました by Executor Agent v2.0 with Multi-Stage Submission Strategy*