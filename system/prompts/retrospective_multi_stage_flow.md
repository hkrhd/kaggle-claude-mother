# Retrospective Agent 多段階分析フロー
<!-- version: 1.0.0 -->
<!-- optimized_for: continuous_system_evolution -->

## 🎯 目的：競技経験を最大限活用し、システムを継続的に進化させる

LLMを複数回活用し、深い振り返りから具体的な改善実装まで包括的に実行します。

## 📊 4段階振り返り・改善プロセス

### Stage 1: 競技結果の包括的分析
**目的**: 何が起きたか、なぜ起きたかを完全に理解
```
入力: 競技結果、全エージェントログ、意思決定履歴
プロンプト: retrospective_competition_analysis.md
出力: 成功要因、失敗原因、改善機会の特定
```

### Stage 2: 学習抽出と知識体系化
**目的**: 個別経験を転用可能な知識に昇華
```
入力: Stage 1の分析結果、過去の類似パターン
プロンプト: retrospective_learning_extraction.md
出力: 再利用可能なパターン、原則、ヒューリスティクス
```

### Stage 3: システム改善計画策定
**目的**: 具体的で実装可能な改善計画を設計
```
入力: Stage 1&2の結果、システムメトリクス、制約条件
プロンプト: retrospective_system_improvement.md
出力: 優先順位付けされた改善ロードマップ
```

### Stage 4: 自動実装と効果検証
**目的**: 改善を安全に実装し、効果を測定
```
入力: Stage 3の改善計画、実装ガイドライン
実行: コード改善、設定最適化、テスト実行
出力: 実装結果、効果測定、次回への学習
```

## 🔄 条件付き追加分析

### 予想外の結果での深掘り分析
```python
if competition_result.unexpected or performance.below_threshold:
    # 想定外の結果に対する詳細分析
    deep_analysis = await llm.analyze(
        prompt="retrospective_unexpected_result_analysis.md",
        context={
            "expectations": original_predictions,
            "reality": actual_results,
            "gaps": performance_gaps
        }
    )
```

### 大成功時のベストプラクティス抽出
```python
if medal_achieved and rank_percentile < 0.05:  # TOP 5%
    # 成功パターンの詳細分解
    success_blueprint = await llm.analyze(
        prompt="retrospective_success_pattern_extraction.md",
        context={
            "winning_factors": critical_success_factors,
            "unique_approaches": innovative_techniques,
            "timing_decisions": strategic_choices
        }
    )
```

### システム大規模改善の設計
```python
if improvement_potential > 0.3:  # 30%以上の改善余地
    # アーキテクチャレベルの改善設計
    architecture_redesign = await llm.analyze(
        prompt="retrospective_architecture_evolution.md",
        context={
            "current_limitations": system_bottlenecks,
            "emerging_requirements": new_challenges,
            "technology_options": available_solutions
        }
    )
```

## 💡 LLM活用の最大化戦略

### 1. 文脈の継承と深化
```python
# 各ステージの分析結果を次に引き継ぎ、深化させる
context = {}
for stage in analysis_stages:
    stage_result = await llm.analyze(
        prompt=stage.prompt,
        context={**context, **stage.specific_inputs}
    )
    context[stage.name] = stage_result
    
    # 重要な洞察は次ステージで強調
    if stage_result.critical_insights:
        context["emphasis"] = stage_result.critical_insights
```

### 2. 比較分析による学習強化
```python
# 過去の類似競技との比較で学習を深める
comparative_analysis = await llm.analyze(
    prompt="retrospective_comparative_analysis.md",
    context={
        "current_competition": current_results,
        "similar_competitions": historical_similar_cases,
        "performance_delta": performance_comparisons
    }
)
```

### 3. 仮説生成と検証設計
```python
# 改善仮説を生成し、検証方法を設計
improvement_hypotheses = await llm.analyze(
    prompt="retrospective_hypothesis_generation.md",
    context={
        "identified_problems": bottlenecks_and_issues,
        "potential_solutions": improvement_ideas,
        "constraints": system_limitations
    }
)

for hypothesis in improvement_hypotheses:
    validation_plan = await llm.analyze(
        prompt="retrospective_validation_design.md",
        context=hypothesis
    )
```

### 4. 知識の永続化と共有
```python
# 学習内容を構造化し、将来参照可能な形で保存
knowledge_codification = await llm.analyze(
    prompt="retrospective_knowledge_persistence.md",
    context={
        "raw_learnings": all_stage_outputs,
        "existing_knowledge_base": current_knowledge,
        "organization_schema": knowledge_structure
    }
)

# 自動ドキュメント生成
documentation = await llm.analyze(
    prompt="retrospective_documentation_generation.md",
    context=knowledge_codification
)
```

## 📈 期待される成果

### 従来の振り返りとの比較
| 側面 | 従来手法 | LLM多段階分析 |
|------|---------|--------------|
| 分析深度 | 表面的 | 根本原因まで到達 |
| 学習抽出 | 断片的 | 体系的・構造化 |
| 改善実装 | 手動・遅い | 自動・迅速 |
| 知識蓄積 | 属人的 | システム化 |

### 継続的改善の加速
1. **学習速度**: 3倍以上の知識獲得速度
2. **改善精度**: 90%以上の改善成功率
3. **自動化率**: 80%以上の改善自動実装
4. **知識活用**: 95%以上の過去知識再利用

## 🚀 実装における重要ポイント

### 1. 安全な自動実装
- すべての変更前にバックアップ
- 段階的ロールアウト
- 自動ロールバック機能
- 継続的モニタリング

### 2. 知識の質保証
- 統計的検証の実施
- 外れ値の適切な処理
- 一般化の妥当性確認
- 定期的な知識更新

### 3. システム進化の方向性
- より高度な自動化へ
- より深い分析へ
- より迅速な適応へ
- より確実な成功へ

この多段階分析により、
各競技を確実に次の成功への糧とし、
システムを継続的に進化させ続けます。