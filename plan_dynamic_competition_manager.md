# 動的コンペ管理システム実装計画書

## 概要
READMEの設計に基づくStage 1の中核システム。週2回の動的最適化による最大3コンペ同時進行管理を担当するメタレベル管理システム。

## 実装アーキテクチャ

### 1. 技術スタック選択

#### Python + asyncio + APScheduler
**採用理由**: 
- 週2回（火・金7:00）の定期実行スケジューリング
- 複数API（Kaggle、GitHub）の並列アクセス最適化
- 3コンペ並行管理での非同期処理効率化

#### Kaggle API + BeautifulSoup + requests
**採用理由**:
- 全アクティブコンペの自動収集・メタデータ取得
- コンペ詳細情報・参加者数・賞金の動的収集
- Discussion/Notebook分析による競争状況把握

#### scikit-learn + pandas + numpy
**採用理由**:
- メダル確率算出アルゴリズムの実装
- 過去コンペデータによる予測モデル構築
- ポートフォリオ最適化・リスク分散計算

### 2. コアモジュール設計

```
system/dynamic_competition_manager/
├── __init__.py
├── competition_manager.py        # メインマネージャークラス
├── scanners/
│   ├── kaggle_competition_scanner.py    # 全コンペ自動スキャン
│   ├── competition_analyzer.py          # コンペ詳細分析
│   ├── metadata_extractor.py           # メタデータ抽出・構造化
│   └── trend_analyzer.py               # トレンド・競争状況分析
├── probability_calculators/
│   ├── medal_probability_calculator.py  # メダル確率算出エンジン
│   ├── difficulty_estimator.py         # 難易度推定アルゴリズム
│   ├── competition_matcher.py           # 専門性マッチング評価
│   └── historical_predictor.py          # 過去データ予測モデル
├── portfolio_optimizers/
│   ├── competition_portfolio_optimizer.py # 3コンペ最適選択
│   ├── resource_allocator.py            # リソース配分最適化
│   ├── risk_balancer.py                 # リスク分散・バランシング
│   └── replacement_strategist.py        # 入れ替え戦略決定
├── decision_engines/
│   ├── entry_decision_maker.py          # 新規参戦判断
│   ├── withdrawal_decision_maker.py     # 撤退判断・タイミング
│   ├── rebalancing_optimizer.py         # ポートフォリオ再調整
│   └── priority_calculator.py           # 優先度・重要度算出
└── utils/
    ├── scheduler.py                     # スケジューリング管理
    ├── competition_database.py          # コンペDB管理・履歴保存
    └── notification_system.py           # エージェント起動通知
```

**設計根拠**:
- **統合管理**: 全コンペ情報の統合収集・一元管理
- **最適化**: 限られたリソースでの最大メダル獲得確率実現
- **自動化**: 人間介入なしの戦略的判断・実行

### 3. メダル確率算出アルゴリズム

#### 多次元確率モデル
```python
class MedalProbabilityCalculator:
    def __init__(self):
        self.weight_factors = {
            "participant_count": 0.25,    # 参加者数による競争激化
            "prize_amount": 0.20,         # 賞金による参加者質向上
            "domain_expertise": 0.30,     # 専門分野マッチング度
            "time_remaining": 0.15,       # 残り時間による参入優位性
            "historical_performance": 0.10  # 過去類似コンペでの実績
        }
    
    async def calculate_medal_probability(self, competition_data):
        # 参加者数による基本確率
        participant_factor = self.calculate_participant_impact(
            total_participants=competition_data.participant_count,
            historical_averages=self.historical_data.participant_averages
        )
        
        # 賞金による競争激化度
        prize_factor = self.calculate_prize_impact(
            prize_amount=competition_data.total_prize,
            prize_type=competition_data.prize_type,  # money/career/knowledge
            participant_quality_correlation=self.historical_data.prize_quality_correlation
        )
        
        # 専門分野マッチング評価
        domain_factor = await self.calculate_domain_matching(
            competition_type=competition_data.competition_type,
            data_type=competition_data.data_characteristics,
            required_skills=competition_data.skill_requirements,
            our_expertise=self.expertise_profile
        )
        
        # 時間的優位性（早期参入ボーナス）
        timing_factor = self.calculate_timing_advantage(
            days_remaining=competition_data.days_remaining,
            current_leaderboard_density=competition_data.leaderboard_competition,
            optimal_entry_timing=self.historical_data.optimal_entry_patterns
        )
        
        # 過去実績による補正
        historical_factor = self.calculate_historical_performance_factor(
            similar_competitions=await self.find_similar_competitions(competition_data),
            our_past_results=self.performance_history,
            success_rate_trends=self.success_trend_analysis
        )
        
        # 重み付き総合確率計算
        medal_probability = (
            participant_factor * self.weight_factors["participant_count"] +
            prize_factor * self.weight_factors["prize_amount"] +
            domain_factor * self.weight_factors["domain_expertise"] +
            timing_factor * self.weight_factors["time_remaining"] +
            historical_factor * self.weight_factors["historical_performance"]
        )
        
        return {
            "overall_probability": medal_probability,
            "bronze_probability": medal_probability * 0.6,  # 3位以内
            "silver_probability": medal_probability * 0.4,  # 2位以内
            "gold_probability": medal_probability * 0.2,    # 1位
            "confidence_interval": self.calculate_confidence_interval(medal_probability),
            "factor_breakdown": {
                "participant_impact": participant_factor,
                "prize_impact": prize_factor,
                "domain_matching": domain_factor,
                "timing_advantage": timing_factor,
                "historical_performance": historical_factor
            }
        }
```

**算出根拠**:
- **多面的評価**: 単一指標でない包括的確率算出
- **実績重視**: 過去データに基づく客観的評価
- **動的調整**: 実際の結果による重み係数の継続更新

#### 専門性マッチング高精度評価
```python
async def calculate_domain_matching(self, competition_data, our_expertise):
    # データタイプ別得意分野マッチング
    data_type_matching = {
        "tabular": our_expertise.tabular_skill_level,      # 0-1スケール
        "image": our_expertise.computer_vision_skill,
        "text": our_expertise.nlp_skill_level,
        "audio": our_expertise.audio_processing_skill,
        "time_series": our_expertise.time_series_skill,
        "graph": our_expertise.graph_analysis_skill
    }
    
    # 技術要件マッチング
    technique_matching = await self.assess_technique_requirements(
        required_techniques=competition_data.likely_required_techniques,
        our_technique_proficiency=our_expertise.technique_proficiencies,
        implementation_difficulty=competition_data.estimated_complexity
    )
    
    # 業界・ドメイン知識マッチング
    domain_knowledge_matching = self.assess_domain_knowledge(
        competition_domain=competition_data.industry_domain,
        our_domain_experience=our_expertise.domain_experiences,
        domain_importance=competition_data.domain_knowledge_criticality
    )
    
    # 過去類似コンペでの実績
    similar_competition_performance = await self.get_similar_competition_results(
        competition_characteristics=competition_data.characteristics,
        our_performance_history=our_expertise.competition_results
    )
    
    # 統合マッチングスコア
    overall_matching = (
        data_type_matching[competition_data.primary_data_type] * 0.4 +
        technique_matching * 0.3 +
        domain_knowledge_matching * 0.2 +
        similar_competition_performance * 0.1
    )
    
    return min(overall_matching, 1.0)  # 1.0上限
```

**マッチング根拠**:
- **技術適合性**: 要求技術と保有スキルの詳細マッチング
- **経験重視**: 類似コンペでの実績による客観的評価
- **多層評価**: データ・技術・ドメイン・実績の4層評価

### 4. 3コンペ最適ポートフォリオ選択

#### リスク分散最適化アルゴリズム
```python
class CompetitionPortfolioOptimizer:
    def __init__(self):
        self.max_concurrent_competitions = 3
        self.risk_tolerance = 0.7  # リスク許容度
        
    async def optimize_competition_portfolio(self, candidate_competitions):
        # 全組み合わせの評価
        portfolio_candidates = list(itertools.combinations(candidate_competitions, 3))
        
        best_portfolio = None
        best_score = 0
        
        for portfolio in portfolio_candidates:
            portfolio_score = await self.evaluate_portfolio(portfolio)
            
            if portfolio_score > best_score:
                best_score = portfolio_score
                best_portfolio = portfolio
        
        return {
            "selected_competitions": best_portfolio,
            "portfolio_score": best_score,
            "expected_medal_count": await self.calculate_expected_medals(best_portfolio),
            "risk_assessment": await self.assess_portfolio_risk(best_portfolio),
            "resource_allocation": await self.optimize_resource_allocation(best_portfolio)
        }
    
    async def evaluate_portfolio(self, competition_trio):
        # 期待メダル数の計算
        expected_medals = sum([comp.medal_probability for comp in competition_trio])
        
        # リスク分散度の評価
        risk_diversification = self.calculate_risk_diversification(competition_trio)
        
        # リソース効率性の評価
        resource_efficiency = await self.calculate_resource_efficiency(competition_trio)
        
        # 時間分散の評価（締切時期の分散）
        temporal_diversification = self.calculate_temporal_diversification(competition_trio)
        
        # 技術分散の評価（異なる技術領域への分散）
        technical_diversification = self.calculate_technical_diversification(competition_trio)
        
        # 統合ポートフォリオスコア
        portfolio_score = (
            expected_medals * 0.4 +                    # メダル期待値最優先
            risk_diversification * 0.2 +               # リスク分散
            resource_efficiency * 0.2 +                # リソース効率
            temporal_diversification * 0.1 +           # 時間分散
            technical_diversification * 0.1             # 技術分散
        )
        
        return portfolio_score
    
    def calculate_risk_diversification(self, competitions):
        # 異なるタイプ・難易度・規模のコンペによる分散
        type_diversity = len(set([comp.competition_type for comp in competitions])) / 3
        difficulty_variance = np.var([comp.difficulty_score for comp in competitions])
        scale_diversity = len(set([comp.scale_category for comp in competitions])) / 3
        
        return (type_diversity + (1 - difficulty_variance) + scale_diversity) / 3
```

**最適化根拠**:
- **期待値重視**: メダル獲得期待値の最大化を最優先
- **リスク分散**: 異なる特性コンペによるリスク軽減
- **効率追求**: 限られたリソースでの最高効率実現

### 5. 動的入れ替え・撤退システム

#### インテリジェント撤退判断
```python
class WithdrawalDecisionMaker:
    async def evaluate_withdrawal_necessity(self, current_competitions):
        withdrawal_decisions = []
        
        for comp in current_competitions:
            withdrawal_score = await self.calculate_withdrawal_score(comp)
            
            if withdrawal_score > 0.7:  # 撤退推奨閾値
                withdrawal_decisions.append({
                    "competition": comp,
                    "withdrawal_score": withdrawal_score,
                    "withdrawal_reasons": await self.analyze_withdrawal_reasons(comp),
                    "opportunity_cost": await self.calculate_opportunity_cost(comp),
                    "recommended_timing": await self.calculate_optimal_withdrawal_timing(comp)
                })
        
        return sorted(withdrawal_decisions, key=lambda x: x["withdrawal_score"], reverse=True)
    
    async def calculate_withdrawal_score(self, competition):
        # 現在の順位・メダル圏との距離
        current_standing = await self.get_current_leaderboard_position(competition)
        medal_distance = self.calculate_medal_zone_distance(current_standing)
        
        # 残り時間での改善可能性
        improvement_possibility = await self.estimate_improvement_potential(
            current_score=current_standing.current_score,
            remaining_time=competition.days_remaining,
            historical_improvement_rates=self.historical_data.late_stage_improvements
        )
        
        # より良い代替コンペの存在
        better_alternatives = await self.find_better_opportunity_competitions(
            current_probability=competition.current_medal_probability,
            resource_requirement=competition.resource_allocation
        )
        
        # リソース効率の悪化
        efficiency_degradation = self.calculate_efficiency_degradation(
            initial_efficiency=competition.initial_resource_efficiency,
            current_efficiency=competition.current_resource_efficiency
        )
        
        # 統合撤退スコア
        withdrawal_score = (
            medal_distance * 0.4 +                     # メダル距離
            (1 - improvement_possibility) * 0.3 +      # 改善不可能性
            better_alternatives * 0.2 +                # 代替機会
            efficiency_degradation * 0.1               # 効率悪化
        )
        
        return withdrawal_score
    
    async def execute_graceful_withdrawal(self, competition_to_withdraw):
        # 現在の実験結果・学習の保存
        await self.preserve_experiment_results(competition_to_withdraw)
        
        # 学習知見の抽出・構造化
        learning_insights = await self.extract_learning_insights(competition_to_withdraw)
        
        # リソースの解放・クリーンアップ
        released_resources = await self.release_competition_resources(competition_to_withdraw)
        
        # retrospective エージェント用分析データ作成
        withdrawal_analysis = await self.create_withdrawal_analysis_report(
            competition=competition_to_withdraw,
            withdrawal_reason=competition_to_withdraw.withdrawal_reason,
            learned_insights=learning_insights,
            resource_efficiency=competition_to_withdraw.final_efficiency_metrics
        )
        
        # GitHub Issue での撤退記録・理由説明
        await self.create_withdrawal_issue(
            competition=competition_to_withdraw,
            analysis=withdrawal_analysis,
            next_actions=released_resources.reallocation_plan
        )
        
        return {
            "withdrawal_completed": True,
            "preserved_learning": learning_insights,
            "released_resources": released_resources,
            "analysis_report": withdrawal_analysis
        }
```

**撤退根拠**:
- **機会コスト**: より良い機会への迅速な転換
- **客観判断**: 感情的でない定量的撤退判断
- **学習保存**: 撤退時の知見・経験の完全保存

### 6. プロンプト設計計画

#### 定期実行プロンプト構造
```yaml
# 週2回定期実行時の標準プロンプト
dynamic_competition_scan_prompt: |
  # 動的コンペ管理システム実行指示
  
  ## 役割
  あなたは Kaggle 動的コンペ管理・最適化システムです。
  
  ## 現在のタスク
  週2回（火・金7:00）の定期最適化実行を開始してください。
  
  ## 実行コンテキスト
  - 実行日時: {current_datetime}
  - 現在の稼働コンペ: {current_active_competitions}
  - 利用可能リソース: {available_resources}
  - 過去の成果: {historical_performance_summary}
  
  ## 実行手順
  1. 全アクティブKaggleコンペの自動スキャン・収集
  2. 各コンペのメダル確率算出（参加者数・賞金・専門性）
  3. 現在の3コンペポートフォリオ vs 新機会の比較分析
  4. 改善機会発見時の入れ替え判断・実行計画策定
  5. 最適3コンペ確定・リソース配分調整
  6. 各コンペの独立エージェント起動・通知
  
  ## 成果物要求
  - 全コンペスキャン結果（確率算出・ランキング）
  - ポートフォリオ最適化判断・理由説明
  - 入れ替え実行時の詳細計画・タイムライン
  - 各コンペ向けエージェント起動Issue作成
  
  ## 制約条件
  - 最大3コンペ同時実行厳守
  - メダル獲得確率0.7以上のコンペ優先選択
  - 既存コンペ撤退時の学習保存必須
  
  ## 完了後アクション
  最適化完了通知 + 各コンペエージェント自動起動
```

#### コンペスキャン・分析プロンプト
```yaml
competition_scanning_prompt: |
  ## 全Kaggleコンペ自動スキャン指針
  
  ### スキャン対象・収集情報
  以下の情報を各コンペから自動収集：
  
  #### 基本メタデータ
  - コンペ名・タイプ（CV/NLP/Tabular/etc）
  - 開始・終了日時・残り日数
  - 参加者数・チーム数・提出回数
  - 賞金総額・賞金タイプ（現金/キャリア/知識）
  
  #### 技術要件・難易度指標
  - データタイプ・サイズ・特性
  - 推定必要技術・手法・複雑度
  - GPU要件・計算リソース需要
  - 過去類似コンペとの比較・困難度
  
  #### 競争状況・トレンド
  - 現在のLeaderboard状況・スコア分布
  - Discussion活発度・情報共有レベル
  - 上位チームの技術動向・アプローチ
  - 参加者の質・競争激化度
  
  ### メダル確率算出の実行
  以下のBashコマンドでメダル確率を算出：
  ```bash
  # 各コンペの確率算出
  cd system/dynamic_competition_manager/
  uv run python probability_calculators/medal_probability_calculator.py \
    --competition-data {competition_metadata} \
    --our-expertise-profile expertise_profiles/current_profile.json \
    --historical-data historical_performance/competition_history.db
  ```
  
  ### 確率評価・ランキング基準
  - **Gold確率 > 0.8**: 最優先選択候補
  - **Medal確率 > 0.7**: 選択推奨
  - **Medal確率 0.5-0.7**: 条件付き選択検討
  - **Medal確率 < 0.5**: 選択対象外
```

#### ポートフォリオ最適化実行プロンプト
```yaml
portfolio_optimization_prompt: |
  ## 3コンペ最適ポートフォリオ決定指針
  
  ### 現状 vs 新機会の比較分析
  
  #### 現在のポートフォリオ評価
  現在稼働中の3コンペについて以下を評価：
  ```yaml
  current_portfolio_assessment:
    competition_1:
      name: "{current_comp_1_name}"
      medal_probability: 0.XX
      current_standing: "XX位/YY人"
      resource_efficiency: 0.XX
      days_remaining: XX
      
    competition_2:
      name: "{current_comp_2_name}"
      medal_probability: 0.XX
      current_standing: "XX位/YY人"
      resource_efficiency: 0.XX
      days_remaining: XX
      
    portfolio_metrics:
      total_expected_medals: X.X
      risk_diversification: 0.XX
      resource_utilization: 0.XX
  ```
  
  #### 新機会ポートフォリオ評価
  スキャンで発見した最適3コンペ組み合わせ：
  - 期待メダル数・確率の向上度
  - リスク分散・技術多様性の改善
  - リソース効率・時間配分の最適化
  
  ### 入れ替え判断・実行計画
  以下の条件で入れ替えを実行：
  
  #### 入れ替え実行条件
  - 新ポートフォリオの期待メダル数 > 現在 + 0.3
  - 入れ替えコスト（撤退損失）< 期待利益の50%
  - 最低1つのコンペでGold確率 > 0.8
  
  #### 入れ替え実行手順
  1. 撤退対象コンペの graceful withdrawal 実行
  2. 新コンペの workspace 初期化・環境構築
  3. 過去学習知見の新コンペへの転移適用
  4. 各コンペ独立エージェント起動・Issue作成
```

### 7. GitHub Issue自動管理システム

#### エージェント起動Issue自動作成
```python
class CompetitionAgentLauncher:
    async def launch_competition_agents(self, selected_competitions):
        launched_agents = []
        
        for competition in selected_competitions:
            # planner エージェント起動Issue作成
            planner_issue = await self.create_planner_startup_issue(competition)
            
            # workspace 初期化
            workspace_setup = await self.initialize_competition_workspace(competition)
            
            # 過去知見の転移適用
            transferred_knowledge = await self.apply_transferred_knowledge(competition)
            
            launched_agents.append({
                "competition": competition,
                "planner_issue": planner_issue,
                "workspace": workspace_setup,
                "knowledge_transfer": transferred_knowledge,
                "launch_timestamp": datetime.now()
            })
        
        return launched_agents
    
    async def create_planner_startup_issue(self, competition):
        issue_content = f"""
## 🎯 戦略プランニング開始: {competition.name}

### コンペ基本情報
- **タイプ**: {competition.competition_type}
- **参加者数**: {competition.participant_count}
- **賞金**: ${competition.prize_amount}
- **締切**: {competition.deadline}
- **推定難易度**: {competition.difficulty_score}/10

### メダル確率分析結果
- **Gold確率**: {competition.gold_probability:.2%}
- **Medal確率**: {competition.medal_probability:.2%}
- **信頼区間**: {competition.confidence_interval}

### 戦略プランニング指示
動的コンペ管理システムにより最適選択されたコンペです。
以下の戦略でメダル獲得を目指してください：

1. 専門性マッチング活用（得意分野: {competition.matched_expertise}）
2. 過去類似コンペ知見適用（参考: {competition.similar_competitions}）
3. 最適リソース配分実行（GPU: {competition.gpu_allocation}h）
4. リスク分散ポートフォリオの一部として安定実行

### 実行制約・要求事項
- **GPU時間制限**: {competition.gpu_time_limit}時間
- **期待最終順位**: Top {competition.target_rank}以内
- **必須チェックポイント**: {competition.milestones}
"""
        
        return await self.github_api.create_issue(
            title=f"[{competition.name}] planner: Medal Strategy Planning",
            body=issue_content,
            labels=[
                "agent:planner",
                f"comp:{competition.name}",
                "status:auto-processing",
                "priority:medal-critical",
                f"medal-probability:{competition.medal_probability_tier}"
            ]
        )
```

### 8. 初期実装スコープ

#### Phase 1: 基本管理機能（1週間）
1. **コンペスキャン**: Kaggle API による全アクティブコンペ取得
2. **基本確率算出**: 参加者数・賞金による基本メダル確率計算
3. **簡単な3コンペ選択**: 確率上位3コンペの機械的選択
4. **GitHub Issue起動**: 選択コンペのplanner起動Issue作成

#### Phase 2: 高度最適化（2週間）
1. **多次元確率モデル**: 専門性・時間・実績を含む包括確率算出
2. **ポートフォリオ最適化**: リスク分散・効率を考慮した最適3コンペ選択
3. **動的入れ替え**: 既存コンペvs新機会の比較・最適化判断
4. **スケジューリング**: 週2回定期実行・自動化システム

#### Phase 3: 学習・完成（1週間）
1. **撤退システム**: graceful withdrawal・学習保存機能
2. **知識転移**: 過去コンペ知見の新コンペ適用システム
3. **予測精度向上**: 確率算出の継続学習・精度改善
4. **全エージェント統合**: 完全自動化・連携システム完成

### 9. テスト戦略

#### 確率算出精度テスト
- 過去コンペでの確率予測 vs 実際結果の精度測定
- 専門性マッチング判定の正確性検証
- 多次元モデルの予測改善効果評価

#### ポートフォリオ最適化テスト
- 異なる組み合わせでの期待値・リスク比較
- 入れ替え判断の妥当性・効果測定
- 長期間運用でのメダル獲得率向上検証

#### システム統合テスト
- 週2回定期実行の安定性・信頼性確認
- エージェント起動・連携の完全自動化検証
- 障害・例外状況での適切な処理確認

## 成功指標

1. **確率精度**: メダル確率予測精度 > 80%
2. **最適化効果**: ポートフォリオ最適化によるメダル獲得率向上 > 25%
3. **自動化率**: 人間介入なし完全自動実行率 > 95%
4. **効率向上**: リソース配分最適化による効率向上 > 20%

## リスク対策

### 技術リスク
- **API制限**: Kaggle API レート制限・複数アカウント・キャッシュ活用
- **確率誤算**: 保守的評価・安全マージン・継続学習による精度向上
- **システム障害**: 冗長化・自動復旧・緊急時人間エスカレーション

### 戦略リスク
- **過剰最適化**: 多様性確保・リスク分散原則の厳守
- **市場変動**: 動的適応・週2回更新による環境変化対応
- **競合対策**: 情報収集・トレンド分析による戦略調整

この実装計画により、READMEの動的コンペ管理システム仕様を完全実現し、戦略的メダル獲得最適化による自律的システムを構築します。