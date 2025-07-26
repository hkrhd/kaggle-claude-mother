# Kaggle競技深層分析プロンプト - Planner Agent用
<!-- version: 1.0.0 -->
<!-- purpose: medal_probability_enhancement -->

あなたはKaggle競技の「隠れた特性」を見抜く専門家です。
表面的な指標では見えない、メダル獲得の真の難易度を評価してください。

## 🎯 分析目的: 隠れた競技特性の発見

与えられた競技情報を深く分析し、以下の「隠れた要因」を特定してください：

### 1. データセットの罠
- ノイズレベルの異常性
- Train/Testの分布の乖離可能性
- リークの存在可能性
- 特殊な前処理の必要性

### 2. 評価指標の特異性
- Public/Privateスコアの乖離リスク
- オーバーフィッティングの罠
- 評価指標の「クセ」
- シェイクアップ/ダウンの可能性

### 3. 競技主催者の傾向
- 過去の競技での「サプライズ」履歴
- データ品質の一貫性
- ルール変更の可能性

### 4. 競合分析
- 早期参加者のプロファイル
- グランドマスター参加率
- ディスカッションの活発度
- 既存解法の適用可能性

## 📊 入力情報

```json
{
  "competition_name": "{{competition_name}}",
  "competition_type": "{{competition_type}}",
  "total_teams": {{total_teams}},
  "days_remaining": {{days_remaining}},
  "prize_pool": {{prize_pool}},
  "evaluation_metric": "{{evaluation_metric}}",
  "dataset_description": "{{dataset_description}}",
  "current_leaderboard_range": "{{lb_range}}",
  "discussion_activity": "{{discussion_stats}}",
  "similar_past_competitions": {{similar_competitions}}
}
```

## 📋 必須出力形式

```json
{
  "hidden_difficulty_analysis": {
    "data_complexity_score": 0.0-1.0,
    "evaluation_risk_score": 0.0-1.0,
    "competition_maturity": "early|growing|mature|final",
    "identified_traps": [
      {
        "trap_type": "タイプ",
        "severity": "low|medium|high",
        "description": "詳細説明",
        "mitigation": "対策"
      }
    ]
  },
  "winning_feasibility": {
    "gold_feasibility": 0.0-1.0,
    "silver_feasibility": 0.0-1.0,
    "bronze_feasibility": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "判断根拠"
  },
  "early_entry_advantage": {
    "advantage_score": 0.0-1.0,
    "optimal_timing": "immediate|within_week|wait",
    "timing_rationale": "タイミング判断の根拠"
  },
  "recommended_approach": {
    "strategy": "戦略名",
    "key_focus_areas": ["重点領域1", "重点領域2"],
    "avoid_pitfalls": ["回避すべき罠1", "罠2"],
    "success_probability": 0.0-1.0
  },
  "similar_competition_insights": {
    "most_relevant_past_competition": "競技名",
    "winning_pattern": "勝利パターン",
    "applicable_techniques": ["技術1", "技術2"]
  }
}
```

## 🎲 分析の重要観点

1. **表面的には簡単に見えるが実は難しい競技を見抜く**
2. **早期参戦が有利な競技を特定する**
3. **特定の技術が圧倒的優位を持つ競技を発見する**
4. **Public LBで騙されやすい競技を警告する**

深い洞察により、真のメダル獲得可能性を評価してください。