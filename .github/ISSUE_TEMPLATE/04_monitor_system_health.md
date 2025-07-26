---
name: 🔍 Monitor - システム監視 (System Monitoring)
about: モニターエージェントによる予防的監視と自動修復
title: "[{{COMPETITION_ID}}] Monitor: {{MONITORING_TYPE}} - System Health Report"
labels: agent:monitor, status:monitoring
assignees: ''
---

# 🔍 システム監視・健全性レポート

## 🔗 実行コンテキスト
- **Executor Issue**: #{{EXECUTOR_ISSUE_NUMBER}}
- **監視期間**: {{MONITORING_START}} 〜 {{MONITORING_END}}
- **監視モード**: {{MONITORING_MODE}}

## 🛡️ 予防的監視分析（3段階LLM分析）

### Stage 1: パターン認識・異常予測
**システム健全性スコア**: {{HEALTH_SCORE}}/100

**検出されたパターン**
| パターン | 信頼度 | 予測される問題 | 推定発生時刻 |
|---------|--------|---------------|--------------|
| {{PATTERN_1}} | {{CONF_1}}% | {{PROBLEM_1}} | {{TIME_1}} |
| {{PATTERN_2}} | {{CONF_2}}% | {{PROBLEM_2}} | {{TIME_2}} |

**早期警告サイン** ⚠️
- {{WARNING_SIGN_1}}
- {{WARNING_SIGN_2}}
- {{WARNING_SIGN_3}}

### Stage 2: 異常診断・影響分析

#### 検出された異常
| 異常ID | 種別 | 深刻度 | 影響範囲 | メダルへの影響 |
|--------|------|--------|----------|----------------|
| {{ANOMALY_1}} | {{TYPE_1}} | {{SEV_1}} | {{SCOPE_1}} | {{IMPACT_1}} |
| {{ANOMALY_2}} | {{TYPE_2}} | {{SEV_2}} | {{SCOPE_2}} | {{IMPACT_2}} |

#### 根本原因分析
```mermaid
graph TD
    A[{{ROOT_CAUSE}}] --> B[{{SYMPTOM_1}}]
    A --> C[{{SYMPTOM_2}}]
    B --> D[{{EFFECT_1}}]
    C --> E[{{EFFECT_2}}]
    D --> F[{{FINAL_IMPACT}}]
    E --> F
```

### Stage 3: 自動修復戦略

**実行された自動修復** ✅
| 修復アクション | 実行時刻 | 結果 | 効果 |
|---------------|---------|------|------|
| {{REPAIR_1}} | {{TIME_1}} | {{RESULT_1}} | {{EFFECT_1}} |
| {{REPAIR_2}} | {{TIME_2}} | {{RESULT_2}} | {{EFFECT_2}} |

**保留中の修復提案** ⏳
- {{PENDING_REPAIR_1}} - 推奨度: {{REC_1}}%
- {{PENDING_REPAIR_2}} - 推奨度: {{REC_2}}%

## 📊 システムメトリクス

### パフォーマンス指標
```
CPU使用率: {{CPU_USAGE}}% ({{CPU_TREND}})
メモリ使用率: {{MEMORY_USAGE}}% ({{MEMORY_TREND}})
GPU使用率: {{GPU_USAGE}}% ({{GPU_TREND}})
ディスクI/O: {{DISK_IO}} MB/s
```

### 実験進行状況
| 実験ID | ステータス | 進捗 | 推定完了時刻 | 異常フラグ |
|--------|----------|------|-------------|-----------|
| {{EXP_1}} | {{STATUS_1}} | {{PROG_1}}% | {{ETA_1}} | {{FLAG_1}} |
| {{EXP_2}} | {{STATUS_2}} | {{PROG_2}}% | {{ETA_2}} | {{FLAG_2}} |

### エラー統計
- **総エラー数**: {{TOTAL_ERRORS}}
- **回復済み**: {{RECOVERED_ERRORS}}
- **未解決**: {{UNRESOLVED_ERRORS}}
- **エラー率**: {{ERROR_RATE}}% ({{ERROR_TREND}})

## 🔮 予測分析

### 今後24時間の予測
**リスクレベル**: {{RISK_LEVEL}}

| 予測項目 | 確率 | 影響度 | 推奨対策 |
|---------|------|--------|----------|
| {{PRED_1}} | {{PROB_1}}% | {{IMP_1}} | {{ACTION_1}} |
| {{PRED_2}} | {{PROB_2}}% | {{IMP_2}} | {{ACTION_2}} |

### リソース枯渇予測
- **GPU時間残**: {{GPU_REMAINING}}時間 (枯渇予測: {{GPU_DEPLETION}})
- **メモリ圧迫**: {{MEMORY_PRESSURE}} (限界到達: {{MEMORY_LIMIT}})
- **ストレージ**: {{STORAGE_FREE}}GB (枯渇予測: {{STORAGE_DEPLETION}})

## 🚨 アラート・対応履歴

### 発生したアラート
| 時刻 | レベル | 内容 | 対応 | 結果 |
|------|--------|------|------|------|
| {{ALERT_TIME_1}} | {{LEVEL_1}} | {{CONTENT_1}} | {{RESPONSE_1}} | {{OUTCOME_1}} |
| {{ALERT_TIME_2}} | {{LEVEL_2}} | {{CONTENT_2}} | {{RESPONSE_2}} | {{OUTCOME_2}} |

### カスケード障害の防止
**検出されたカスケードリスク**: {{CASCADE_RISK}}%
- 一次障害点: {{PRIMARY_FAILURE}}
- 予想される連鎖: {{CASCADE_CHAIN}}
- 防止措置: {{PREVENTION_MEASURES}}

## 💊 改善提案

### 即時対応推奨
1. **{{IMMEDIATE_1}}**
   - 理由: {{REASON_1}}
   - 期待効果: {{EFFECT_1}}

2. **{{IMMEDIATE_2}}**
   - 理由: {{REASON_2}}
   - 期待効果: {{EFFECT_2}}

### 中期的改善
- {{MEDIUM_TERM_1}}
- {{MEDIUM_TERM_2}}

## 📈 学習と最適化

### 監視精度の改善
- **誤検知率**: {{FALSE_POSITIVE}}% → {{NEW_FALSE_POSITIVE}}%
- **見逃し率**: {{FALSE_NEGATIVE}}% → {{NEW_FALSE_NEGATIVE}}%
- **予測精度**: {{PREDICTION_ACC}}% → {{NEW_PREDICTION_ACC}}%

### 適用された学習
1. {{LEARNING_1}}
2. {{LEARNING_2}}

## 🔄 Retrospective Agent への引き継ぎ

### 重要な発見
- **成功パターン**: {{SUCCESS_PATTERN}}
- **失敗パターン**: {{FAILURE_PATTERN}}
- **改善機会**: {{IMPROVEMENT_OPPORTUNITY}}

### システム改善提案
1. {{SYSTEM_IMPROVEMENT_1}}
2. {{SYSTEM_IMPROVEMENT_2}}

---

## 📋 監視メタデータ

- **レポート生成時刻**: {{REPORT_TIMESTAMP}}
- **監視サイクル数**: {{MONITORING_CYCLES}}
- **自動修復実行数**: {{AUTO_REPAIRS}}
- **ダウンタイム**: {{DOWNTIME}}分 ({{UPTIME}}%)
- **LLM分析実行**: {{LLM_ANALYSES}}回

### 監視設定
- **監視間隔**: {{MONITORING_INTERVAL}}
- **アラート閾値**: {{ALERT_THRESHOLDS}}
- **自動修復モード**: {{AUTO_REPAIR_MODE}}

---
*このIssueは自動生成されました by Monitor Agent v2.0 with Predictive Monitoring & Auto-Remediation*