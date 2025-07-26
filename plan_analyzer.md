# 深層分析エージェント実装計画書

## 概要
READMEの設計に基づく第2エージェント（`agent:analyzer`）の実装計画。グランドマスター級技術調査・最新手法研究・実装可能性判定を担当する技術特化エージェント。

## 実装アーキテクチャ

### 1. 技術スタック選択

#### Python + asyncio + aiohttp
**採用理由**: 
- 複数API（Kaggle、arXiv、GitHub）の並列アクセス最適化
- Discussion・Code分析の非同期処理効率向上
- 大量技術文献スクレイピングの高速化

#### Beautiful Soup 4 + selenium
**採用理由**:
- Kaggle Discussion/Code動的コンテンツ取得
- arXiv論文PDF自動ダウンロード・メタデータ抽出
- JavaScript必須サイトでの技術情報収集対応

#### scikit-learn + transformers + sentence-transformers
**採用理由**:
- 技術文書・解法の意味的類似性分析
- グランドマスター解法パターンの自動分類
- 実装難易度・技術要件の定量評価

### 2. コアモジュール設計

```
system/agents/analyzer/
├── __init__.py
├── analyzer_agent.py          # メインエージェントクラス
├── collectors/
│   ├── kaggle_solutions.py    # Kaggle優勝解法収集
│   ├── arxiv_papers.py        # arXiv最新論文収集
│   ├── github_repos.py        # 関連リポジトリ分析
│   └── discussion_crawler.py  # Discussion分析・要約
├── analyzers/
│   ├── technical_feasibility.py    # 実装可能性判定
│   ├── performance_estimator.py    # 性能・計算コスト推定
│   ├── difficulty_scorer.py        # 技術難易度評価
│   ├── gpu_requirement_analyzer.py # GPU最適化要件分析
│   └── llm_multi_stage_analyzer.py # 多段階LLM分析オーケストレータ
├── knowledge_base/
│   ├── grandmaster_patterns.py     # グランドマスター解法DB
│   ├── technique_catalog.py        # 手法カタログ管理
│   └── implementation_templates.py # 実装テンプレート生成
└── utils/
    ├── web_scraper.py         # 高度スクレイピング
    ├── pdf_extractor.py       # PDF解析・要約
    ├── semantic_matcher.py    # 意味的マッチング
    └── llm_client.py          # LLM APIクライアント
```

**設計根拠**:
- **情報源多様化**: Kaggle/arXiv/GitHub統合で漏れなき技術調査
- **自動評価**: 人間判断不要の定量的実装可能性評価
- **知識蓄積**: 過去調査結果の再利用・パターン学習

### 3. 多段階LLM分析システム

#### 5段階深層分析プロセス
```python
class MultiStageLLMAnalyzer:
    """API制限を考慮せず、必要なだけLLMを活用して最高精度の分析を実現"""
    
    async def execute_full_analysis(self, competition_info):
        # Stage 1: 競技全体像の把握
        initial_analysis = await self.stage1_initial_analysis(competition_info)
        
        # Stage 2: グランドマスター解法の個別深層分析（TOP10を個別に）
        gm_analyses = []
        for solution in self.get_top_solutions(10):
            analysis = await self.stage2_gm_solution_analysis(solution)
            gm_analyses.append(analysis)
        
        # Stage 3: 解法パターンの統合と洞察抽出
        pattern_synthesis = await self.stage3_pattern_synthesis(gm_analyses)
        
        # Stage 4: 実装戦略の詳細設計
        implementation_strategy = await self.stage4_implementation_design(
            pattern_synthesis, self.resource_constraints
        )
        
        # Stage 5: 競合動向を考慮した最終戦略
        final_strategy = await self.stage5_competitive_strategy(
            all_previous_analyses=[
                initial_analysis, gm_analyses, 
                pattern_synthesis, implementation_strategy
            ],
            current_leaderboard=self.get_current_leaderboard()
        )
        
        # 追加の適応的分析（必要に応じて）
        if self.needs_data_anomaly_investigation(initial_analysis):
            await self.additional_data_anomaly_analysis()
        
        if self.found_novel_techniques(pattern_synthesis):
            await self.additional_novel_approach_evaluation()
        
        return self.compile_final_report(all_analyses)
```

**設計根拠**:
- **精度最優先**: API料金を考慮せず、必要なだけ深い分析
- **段階的深化**: 広く浅くから始め、重要部分を深掘り
- **適応的分析**: 発見に応じて追加分析を実施

### 4. グランドマスター解法分析システム

#### Owen Zhang/Abhishek Thakur解法パターンDB
```python
GRANDMASTER_PATTERNS = {
    "owen_zhang": {
        "signature_techniques": [
            "multi_level_stacking",
            "feature_interaction_mining", 
            "ensemble_diversity_optimization",
            "validation_strategy_innovation"
        ],
        "implementation_difficulty": 0.8,  # 0-1スケール
        "gpu_requirement": "optional",
        "success_rate": 0.85
    },
    "abhishek_thakur": {
        "signature_techniques": [
            "automated_feature_engineering",
            "hyperparameter_optimization_bayesian",
            "cross_validation_advanced",
            "model_selection_meta_learning"
        ],
        "implementation_difficulty": 0.7,
        "gpu_requirement": "recommended", 
        "success_rate": 0.82
    }
}
```

**採用根拠**:
- **パターン認識**: 成功手法の構造化・定量化
- **再現性評価**: 個人スキル・環境での実装可能性判定
- **効率優先**: 高成功率手法への集中リソース配分

#### LLMを活用した深層実装可能性評価
```python
async def calculate_deep_feasibility_score(self, technique_info):
    # ルールベースの初期評価
    basic_factors = {
        "implementation_complexity": technique_info.complexity_score,
        "available_libraries": check_library_support(technique_info.requirements),
        "gpu_compatibility": assess_gpu_requirements(technique_info.compute_needs),
        "time_constraint": estimate_implementation_time(technique_info.scope),
        "domain_expertise": match_skill_requirements(technique_info.expertise_level)
    }
    
    # LLMによる深層評価（複数回呼び出し）
    # 1. 技術の本質的理解
    essence_understanding = await self.llm_client.analyze(
        prompt="analyzer_grandmaster_solution_extraction.md",
        data=technique_info
    )
    
    # 2. 実装上の落とし穴予測
    pitfall_prediction = await self.llm_client.analyze(
        prompt="analyzer_implementation_pitfalls.md",
        data={
            "technique": technique_info,
            "essence": essence_understanding
        }
    )
    
    # 3. ROIの詳細評価
    roi_evaluation = await self.llm_client.analyze(
        prompt="analyzer_implementation_roi_evaluation.md",
        data={
            "technique": technique_info,
            "basic_factors": basic_factors,
            "pitfalls": pitfall_prediction
        }
    )
    
    # 統合スコア算出
    return {
        "feasibility_score": roi_evaluation.adjusted_roi,
        "confidence": roi_evaluation.confidence,
        "critical_factors": roi_evaluation.critical_factors,
        "implementation_plan": roi_evaluation.optimal_plan
    }
```

**設計意図**:
- **多面的評価**: 技術・環境・時間・スキルの包括判定
- **リスク最小化**: 実装失敗確率の事前算出・回避
- **優先順位付け**: 確実性高い手法から順次実装

### 4. 最新技術自動収集システム

#### arXiv論文監視・フィルタリング
```python
async def monitor_arxiv_papers(competition_domain):
    # 関連分野の最新論文自動取得
    domains = get_relevant_domains(competition_domain)
    
    for domain in domains:
        papers = await arxiv_api.search({
            "category": domain,
            "submitted": "recent:7d",  # 週1回の新規論文
            "relevance_filter": "machine_learning AND (kaggle OR competition)"
        })
        
        for paper in papers:
            # 実装可能性の事前スクリーニング
            feasibility = await quick_feasibility_check(paper.abstract, paper.methods)
            if feasibility > 0.6:
                await deep_analysis_queue.add(paper)
```

**採用根拠**:
- **最新技術活用**: 競合他者より先進的手法導入
- **効率スクリーニング**: 実装困難技術の早期除外
- **継続監視**: 手動調査不要の自動最新情報取得

#### Kaggle優勝解法自動分析
```python
async def analyze_winning_solutions(competition_type):
    # 同タイプコンペの過去優勝解法収集
    similar_comps = await find_similar_competitions(competition_type)
    
    solutions = []
    for comp in similar_comps:
        # 上位解法のコード・Discussion自動収集
        top_solutions = await kaggle_api.get_competition_solutions(
            comp.id, rank_range=(1, 10)
        )
        
        for solution in top_solutions:
            analysis = await analyze_solution_components(
                code=solution.notebooks,
                discussion=solution.posts,
                performance=solution.scores
            )
            solutions.append(analysis)
    
    # 共通パターン・成功要因の抽出
    return extract_success_patterns(solutions)
```

**技術根拠**:
- **実績重視**: 理論より実証済み手法の優先適用
- **パターン学習**: 成功解法の共通要素自動抽出
- **競合分析**: 他チーム戦略の定量的理解・対策

### 5. 技術難易度・GPU要件分析

#### 計算複雑度自動評価
```python
def estimate_computational_requirements(algorithm_spec):
    complexity_factors = {
        "data_size_dependency": analyze_big_o_complexity(algorithm_spec.operations),
        "model_complexity": count_parameters(algorithm_spec.model_architecture),  
        "training_iterations": estimate_convergence_time(algorithm_spec.hyperparams),
        "memory_footprint": calculate_memory_usage(algorithm_spec.data_pipeline)
    }
    
    # GPU要件の自動判定
    gpu_necessity = {
        "required": complexity_factors["model_complexity"] > 1e6,  # 100万パラメータ超
        "recommended": complexity_factors["training_iterations"] > 1000,
        "optional": complexity_factors["memory_footprint"] < 8  # 8GB未満
    }
    
    return {
        "complexity_score": normalize_complexity(complexity_factors),
        "gpu_requirement": determine_gpu_level(gpu_necessity),
        "estimated_runtime": project_execution_time(complexity_factors)
    }
```

**設計根拠**:
- **リソース最適化**: 限られたGPU時間の効率的配分
- **実装優先度**: 計算コスト vs 性能向上の定量判断
- **障害予防**: メモリ不足・時間超過の事前回避

### 6. GitHub Issue連携・技術提案システム

#### 構造化技術分析レポート作成
```python
async def create_technical_analysis_issue(competition_name, analysis_results):
    # 分析結果の構造化・優先順位付け
    prioritized_techniques = rank_by_medal_impact(analysis_results.techniques)
    
    issue_body = f"""
## 🔬 技術分析結果: {competition_name}

### 推奨実装手法（メダル確率順）
{format_technique_recommendations(prioritized_techniques[:5])}

### グランドマスター解法適用性
{format_grandmaster_analysis(analysis_results.grandmaster_patterns)}

### 実装可能性評価
{format_feasibility_matrix(analysis_results.feasibility_scores)}

### GPU最適化要件
{format_gpu_requirements(analysis_results.compute_needs)}

### 技術リスク評価
{format_risk_assessment(analysis_results.risk_factors)}
"""

    await github_api.create_issue(
        title=f"[{competition_name}] analyzer: Technical Implementation Strategy",
        body=issue_body,
        labels=[
            f"agent:analyzer",
            f"comp:{competition_name}",
            "status:completed",
            "priority:medal-critical"
        ]
    )
```

**採用根拠**:
- **意思決定支援**: 技術選択の定量的根拠提供
- **リスク明示**: 実装困難性・失敗要因の事前警告
- **次段階連携**: executor向け具体的実装指針提供

### 7. 知識ベース学習・蓄積システム

#### 技術成功・失敗パターンDB
```python
class TechnicalKnowledgeBase:
    def __init__(self):
        self.success_patterns = {}  # 成功技術・条件の蓄積
        self.failure_patterns = {}  # 失敗要因・回避策の学習
        self.technique_effectiveness = {}  # 手法別効果測定
    
    async def update_from_competition_result(self, comp_result):
        # 実装技術と結果の相関分析
        for technique in comp_result.applied_techniques:
            effectiveness = calculate_technique_contribution(
                technique=technique,
                final_score=comp_result.medal_result,
                baseline_score=comp_result.baseline_performance
            )
            
            self.technique_effectiveness[technique.name] = {
                "success_rate": update_success_rate(technique.name, comp_result.success),
                "average_improvement": update_score_improvement(technique.name, effectiveness),
                "implementation_difficulty": technique.actual_difficulty,
                "failure_reasons": extract_failure_patterns(comp_result.errors)
            }
    
    def get_recommendations(self, competition_context):
        # 過去実績基づく最適技術推奨
        return rank_techniques_by_historical_success(
            context=competition_context,
            knowledge_base=self.technique_effectiveness
        )
```

**学習根拠**:
- **継続改善**: 実績フィードバックによる推奨精度向上
- **失敗学習**: 同じ技術選択ミスの反復防止
- **個別最適化**: 個人実装能力・環境への適応学習

### 8. プロンプト設計計画

#### エージェント起動プロンプト構造
```yaml
# analyzer エージェント起動時の標準プロンプト
analyzer_activation_prompt: |
  # 深層分析エージェント実行指示
  
  ## 役割
  あなたは Kaggle グランドマスター級技術調査・分析エージェントです。
  
  ## 現在のタスク
  GitHub Issue: "{issue_title}" の技術分析を実行してください。
  
  ## 実行コンテキスト
  - 作業ディレクトリ: competitions/{competition_name}/
  - 対象コンペ: {competition_name}
  - 前段階(planner)の戦略: {planner_strategy_summary}
  
  ## 実行手順
  1. WebSearch: "{competition_name} kaggle winner solution" で優勝解法調査
  2. WebSearch: "arXiv machine learning {competition_domain}" で最新論文調査  
  3. 実装可能性評価（複雑度・GPU要件・時間制約）
  4. グランドマスター解法パターンとのマッチング分析
  5. 技術推奨レポート作成（構造化markdown）
  6. executor用実装指針生成
  
  ## 成果物要求
  - 推奨技術Top5（実装可能性スコア付き）
  - GPU最適化要件分析
  - 実装リスク評価・回避策
  - executor向け具体的実装指針
  
  ## 制約条件
  - 実行時間制限: 120分
  - メダル獲得確率重視（実験的手法は除外）
  - 実装困難技術は明確な警告付与
  
  ## 完了後アクション
  GitHub Issue更新 + executor エージェント起動通知
```

#### Issue読み取り・解析プロンプト
```yaml
issue_analysis_prompt: |
  以下のGitHub Issueの内容を分析し、analyzer としての具体的実行計画を策定してください：
  
  Issue Title: {issue_title}
  Issue Body: {issue_body}
  Issue Labels: {issue_labels}
  
  ## 抽出すべき情報
  1. 対象コンペ名・タイプ（CV/NLP/Tabular/etc）
  2. プランナーが算出したメダル確率・戦略方針
  3. 技術調査の優先領域・制約条件
  4. 前回類似コンペでの失敗・成功パターン
  
  ## 調査計画の具体化
  上記情報に基づき、以下を決定：
  - WebSearch クエリの最適化（検索効率向上）
  - 調査技術の優先順位（メダル確率重視）
  - 実装可能性判定の基準調整
  - executor への引き継ぎ情報の詳細化
```

#### 技術調査実行プロンプト
```yaml
research_execution_prompt: |
  ## WebSearch実行指針
  
  ### 段階1: 優勝解法パターン調査
  クエリ例:
  - "{competition_name} kaggle winner solution code"
  - "{competition_type} kaggle grandmaster techniques 2024" 
  - "Owen Zhang {competition_domain} approach"
  - "Abhishek Thakur {competition_type} strategy"
  
  ### 段階2: 最新技術論文調査  
  クエリ例:
  - "arXiv {competition_domain} machine learning 2024"
  - "{specific_technique} implementation GPU optimization"
  - "{competition_type} ensemble methods recent advances"
  
  ### 段階3: 実装事例・困難度調査
  クエリ例:
  - "{technique_name} implementation difficulty time"
  - "{framework} {technique} GPU memory requirements"
  - "kaggle kernel {technique} implementation examples"
  
  ## 情報評価・フィルタリング基準
  各調査結果に対し以下を評価：
  1. メダル獲得への直接寄与度（1-5点）
  2. 実装可能性・技術難易度（1-5点）  
  3. GPU/時間リソース要件（1-5点）
  4. 過去成功実績・信頼性（1-5点）
  
  総合スコア = (寄与度 * 0.4) + (実装可能性 * 0.3) + (リソース効率 * 0.2) + (信頼性 * 0.1)
```

#### 分析結果構造化プロンプト
```yaml
analysis_structuring_prompt: |
  収集した技術情報を以下の構造でmarkdown形式に整理してください：
  
  ## 🔬 技術分析結果: {competition_name}
  
  ### 📊 推奨実装手法（優先順位順）
  各手法について：
  ```yaml
  technique_1:
    name: "手法名"
    feasibility_score: 0.85  # 0-1スケール
    medal_contribution: 0.90  # メダル寄与期待値
    implementation_time: "3-5日"
    gpu_requirement: "optional/recommended/required"
    risk_factors: ["リスク1", "リスク2"]
    reference_sources: ["URL1", "URL2"]
  ```
  
  ### 🏆 グランドマスター解法適用性
  - Owen Zhang パターンマッチング結果
  - Abhishek Thakur 手法適用可能性  
  - 成功確率・実装困難度評価
  
  ### ⚡ GPU最適化要件
  - 必須GPU仕様・メモリ要件
  - 並列処理最適化ポイント
  - クラウド実行環境推奨設定
  
  ### 🚨 技術リスク評価
  - 実装失敗の主要リスク要因
  - 緊急時の代替手法・フォールバック
  - 時間制約下での最小限実装案
  
  ### 🎯 executor向け実装指針
  具体的な実装ステップ・コード方針・テンプレート案
```

#### Issue更新・次エージェント連携プロンプト
```yaml
handoff_prompt: |
  分析完了後、以下の手順で executor エージェント起動を実行：
  
  ## GitHub Issue更新
  1. 現Issue（analyzer担当）にコメント追加：
     ```markdown
     ## ✅ 深層分析完了
     
     {分析結果サマリー}
     
     **次段階**: executor による実装フェーズ開始
     **推奨手法**: {top_technique_name} (確率: {success_probability})
     **実装期限**: {estimated_completion_date}
     ```
  
  2. Issue ラベル更新：
     - 削除: "status:auto-processing"  
     - 追加: "status:completed"
  
  ## executor起動Issue作成
  新しいGitHub Issue作成：
  ```yaml
  title: "[{competition_name}] executor: High-Performance Implementation"
  body: |
    ## 🏗️ 実装フェーズ開始
    
    **技術分析結果**: {analyzer_issue_url}
    **推奨手法**: {prioritized_techniques}
    **GPU要件**: {gpu_specifications}
    **実装指針**: {implementation_guidelines}
    
    ## 実行計画
    {executor_specific_instructions}
  
  labels:
    - "agent:executor"
    - "comp:{competition_name}"
    - "status:auto-processing"
    - "priority:medal-critical"
  ```
  
  ## 実行確認
  作成後、executor エージェントの自動起動を確認。5分以内に反応がない場合は手動エスカレーション。
```

#### エラーハンドリング・フォールバック戦略
```yaml
error_handling_prompts:
  web_search_failure: |
    WebSearch API制限・障害時の代替戦略：
    1. Read ツールで過去分析結果ファイル確認（competitions/{comp}/cache/）
    2. ローカル知識ベースからの類似コンペ手法抽出
    3. 保守的実装手法（scikit-learn基本アルゴリズム）での暫定提案
    4. リスク明記での executor 引き継ぎ実行
  
  information_overload: |
    大量情報取得時の優先化戦略：
    1. メダル寄与度スコア上位5件に限定
    2. 実装可能性 > 0.7 の手法のみ採用
    3. 複雑な手法は段階的実装計画で分割提案
    4. 時間制約考慮での最小限・最大限実装案の併記
  
  technical_analysis_failure: |
    技術評価・判定困難時の対応：
    1. "実装リスク高" 警告付きでの暫定提案
    2. 代替手法3案の並記（保守・標準・挑戦）
    3. executor での段階的検証・採用判断への委託
    4. 人間エスカレーション条件の明記
```

### 9. 初期実装スコープ

#### Phase 1: 多段階LLM分析基盤構築（1週間）
1. **5段階LLM分析フレームワーク**: 
   - Stage 1-5の分析パイプライン実装
   - 並列LLM呼び出し最適化
   - 分析結果キャッシュシステム
2. **グランドマスター解法深層分析**: 
   - TOP10解法の個別LLM分析
   - 解法の「なぜ」を理解するプロンプト
   - 成功パターンの本質抽出
3. **実装ROI詳細評価システム**: 
   - 技術ごとのLLM評価
   - メダル獲得への寄与度定量化
   - 優先順位付き実装計画
4. **改善されたGitHub Issueテンプレート**: 
   - 多段階分析結果の構造化表示
   - 実装推奨の明確な優先順位
   - 次エージェントへの明確な指示

#### Phase 2: 競合分析と差別化戦略（2週間）
1. **競合動向LLM分析**: 
   - リーダーボード変化の深層分析
   - 他チーム戦略の推測
   - 差別化機会の特定
2. **適応的追加分析システム**: 
   - データ異常検出時の深掘り
   - 新技術発見時の詳細評価
   - 競合予想外動向への対応
3. **パターン統合と勝利方程式**: 
   - 複数解法からの共通成功要因
   - メダル獲得への最短経路
   - リスクヘッジ戦略
4. **arXiv最新論文監視**: 
   - LLMによる関連性評価
   - 実装可能性の即座判断
   - 競争優位性の評価

#### Phase 3: フィードバックループと最適化（1週間）
1. **分析結果の検証と改善**: 
   - LLM分析精度の追跡
   - 予測と実績の乖離分析
   - プロンプトの継続的改善
2. **知識ベース自動更新**: 
   - 成功/失敗パターンの蓄積
   - 競技タイプ別攻略法DB
   - 次競技への転移学習
3. **Executorエージェント連携強化**: 
   - 詳細実装指針の自動生成
   - コードテンプレート提供
   - リアルタイム進捗監視
4. **メダル獲得確率最大化調整**: 
   - 分析結果の重み付け最適化
   - リソース配分の動的調整
   - 最終盤戦略の精緻化

### 10. テスト戦略

#### 技術調査精度テスト
- 過去コンペでの技術推奨 vs 実際優勝解法の一致率測定
- グランドマスター解法パターン認識の精度検証
- 実装可能性判定の正確性（実装成功率との相関）

#### 情報収集効率テスト
- arXiv/Kaggle自動収集の網羅性・重複排除性能
- Discussion/Code分析の要約精度・重要情報抽出率
- 複数API並列アクセスの安定性・レート制限対応

#### 連携動作テスト
- planner起動からanalyzer実行開始までの応答時間
- analyzer完了からexecutor通知までの自動連携検証
- Issue作成・ラベリングの正確性・重複防止

## 成功指標

1. **技術推奨精度**: 推奨技術のメダル寄与率 > 85%（LLM多段階分析により向上）
2. **分析深度**: 各解法の「なぜ」の理解度 > 90%
3. **情報収集効率**: 5段階分析完了時間 < 4時間（並列化により）
4. **実装成功率**: 推奨技術の実装完了率 > 90%
5. **知識蓄積効果**: 同類コンペでの推奨精度向上率 > 25%
6. **差別化成功率**: 他チームが見落とした技術の発見率 > 30%

## リスク対策

### 技術リスク
- **情報過多**: 優先度スコアリング・上位N件制限
- **API制限**: 複数アカウント・指数バックオフ・キャッシュ活用
- **スクレイピング障害**: User-Agent回転・プロキシ・セッション管理

### 判断リスク
- **推奨技術偏向**: 多様性確保・リスク分散原則
- **実装困難技術推奨**: 保守的評価・段階的実装検証
- **時間超過**: タイムアウト・最小限実装での暫定提案

この実装計画により、READMEの深層分析エージェント仕様を完全実現し、グランドマスター級技術の体系的活用システムを構築します。