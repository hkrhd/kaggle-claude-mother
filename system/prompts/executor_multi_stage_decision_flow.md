# Executor Agent 多段階意思決定フロー
<!-- version: 1.0.0 -->
<!-- optimized_for: maximum_medal_probability -->

## 🎯 目的：提出タイミングの最適化によるメダル獲得確率最大化

LLMを複数回活用し、多角的な分析に基づいて最適な提出判断を行います。

## 📊 3段階意思決定プロセス

### Stage 1: 実験戦略の最適化
**目的**: 限られたリソースで最大の成果を出す実験計画
```
入力: 利用可能リソース、pending実験リスト、過去の実験結果
プロンプト: executor_experiment_strategy.md
出力: 優先順位付き実験スケジュール、期待される改善
```

### Stage 2: 競合動向の予測
**目的**: 他チームの動きを予測し、戦略的優位を確保
```
入力: リーダーボード変化、提出パターン、時間要因
プロンプト: executor_competitor_prediction.md
出力: 競合予測、最適提出タイミング、情報戦略
```

### Stage 3: 最終提出判断
**目的**: 全ての情報を統合し、提出/継続/待機を決定
```
入力: Stage 1&2の結果、現在のパフォーマンス、リスク評価
プロンプト: executor_submission_decision.md（改良版）
出力: 最終判断、信頼度、代替案
```

## 🔄 条件付き追加分析

### 高リスク状況での深掘り分析
```python
if risk_level == "HIGH" or confidence < 0.6:
    # 追加の詳細分析を実行
    detailed_risk_analysis = await llm.analyze(
        prompt="executor_high_risk_decision.md",
        context=all_previous_analyses
    )
```

### 最終日特別分析
```python
if hours_remaining < 24:
    # 最終日専用の戦略分析
    final_day_strategy = await llm.analyze(
        prompt="executor_final_day_strategy.md",
        context={
            "current_position": current_rank,
            "medal_gaps": medal_distances,
            "competitor_activity": recent_submissions
        }
    )
```

### シェイクアップ予測
```python
if competition_type in ["image", "nlp"] and CV_gap > threshold:
    # Public/Privateの乖離リスク分析
    shakeup_analysis = await llm.analyze(
        prompt="executor_shakeup_prediction.md",
        context=validation_scores
    )
```

## 💡 LLM活用の最大化戦略

### 1. 並列分析の活用
```python
# Stage 1と2は独立しているため並列実行
results = await asyncio.gather(
    analyze_experiment_strategy(),
    predict_competitor_moves()
)
```

### 2. 文脈の継承
- 各段階の分析結果を次の段階に引き継ぐ
- 矛盾の検出と解決
- 統合的な判断

### 3. 確信度による追加分析
```python
if final_decision.confidence < 0.7:
    # 低確信度の場合、別角度から再分析
    alternative_analysis = await llm.analyze(
        prompt="executor_alternative_perspective.md",
        context=final_decision
    )
```

### 4. 時間軸での分析
- 短期（6時間）: 即座の行動判断
- 中期（24時間）: 実験完了を待つか
- 長期（締切まで）: 全体戦略

## 📈 期待される成果

### 従来の単一分析との比較
| 指標 | 単一分析 | 多段階分析 |
|------|---------|-----------|
| 判断精度 | 70-75% | 85-90% |
| 見落としリスク | 中 | 低 |
| 適応性 | 低 | 高 |
| 戦略的深度 | 浅い | 深い |

### メダル獲得確率への影響
1. **タイミングの最適化**: +10-15%の改善
2. **リソース効率**: 20-30%の改善
3. **競合対策**: 予測精度80%以上
4. **リスク回避**: 致命的ミスを90%削減

## 🚀 実装における重要ポイント

### 1. レスポンス時間の管理
- 全分析を4時間以内に完了
- 緊急時は簡略版で対応

### 2. 分析結果の保存
- 全ての分析結果をログ保存
- 事後検証での改善に活用

### 3. 人間の介入ポイント
- 確信度70%未満での確認
- 最終提出前の承認
- 予想外の状況での相談

### 4. フィードバックループ
- 提出結果の即座の分析
- 次回判断への反映
- プロンプトの継続的改善

この多段階分析により、単純な数値判断を超えた
戦略的な提出判断を実現し、メダル獲得確率を最大化します。