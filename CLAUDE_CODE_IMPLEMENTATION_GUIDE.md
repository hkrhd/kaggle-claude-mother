# Kaggle Claude Mother - ClaudeCode実装指示プロンプト

## 🎯 実装戦略概要

既存の詳細planファイル群を最大活用し、**依存関係順序に基づく段階的実装**で複雑なマルチエージェントシステムを確実に構築します。

### 実装依存関係・優先順序
```
1. 基盤システム (1-2週間)
   ├── Issue安全連携システム (plan_issue_safety_system.md)
   └── 動的コンペ管理システム (plan_dynamic_competition_manager.md)

2. エージェント順次実装 (5週間)
   ├── プランニングエージェント (plan_planner.md)
   ├── 分析エージェント (plan_analyzer.md)  
   ├── 実行エージェント (plan_executor.md)
   ├── モニタリングエージェント (plan_monitor.md)
   └── 反省エージェント (plan_retrospective.md)

3. 統合・最適化 (1週間)
   └── システム全体統合・本番稼働
```

---

## 📋 実装指示プロンプト集

### Phase 1: Issue安全連携システム実装

```
# Issue安全連携システム実装指示

## 実装目標
plan_issue_safety_system.md の完全実装。GitHub Issue APIによる安全なエージェント間連携、競合状態・デッドロック防止、原子性操作保証を実現する。

## 実装手順

### Step 1: 基盤構築 (1-2日)
1. system/issue_safety_system/ ディレクトリ構造作成
2. pyproject.toml に必要依存関係追加:
   ```toml
   dependencies = [
     "asyncio",
     "aiohttp", 
     "PyGithub",
     "github3.py",
     "redis",
     "filelock",
     "networkx"
   ]
   ```

### Step 2: 原子性操作システム (2-3日)
plan_issue_safety_system.md の「3. 原子性操作・競合回避システム」に従い実装:

- `concurrency_control/atomic_operations.py`
- `concurrency_control/deadlock_prevention.py`
- `concurrency_control/lock_manager.py`

**重要**: plan内の具体的コード例を参考に、楽観的ロック・ETAGシステムを実装

### Step 3: エージェント依存関係管理 (2-3日)
- `dependency_trackers/agent_dependency_graph.py`
- `state_machines/agent_state_tracker.py`
- `state_machines/workflow_orchestrator.py`

### Step 4: GitHub API安全ラッパー (1-2日)
- `utils/github_api_wrapper.py`
- `utils/retry_mechanism.py`
- エラーハンドリング・指数バックオフ実装

## 完了基準
- 原子的Issue作成・更新が動作
- デッドロック防止機能が機能
- コンペ分離が完全に保証
- plan_issue_safety_system.md の全テスト項目がPASS

## 次ステップ
動的コンペ管理システム実装に移行
```

### Phase 2: 動的コンペ管理システム実装

```
# 動的コンペ管理システム実装指示

## 実装目標
plan_dynamic_competition_manager.md の完全実装。週2回の動的最適化による最大3コンペ同時進行管理を実現する。

## 前提条件
Issue安全連携システムが完成していること

## 実装手順

### Step 1: 基盤構築 (1日)
system/dynamic_competition_manager/ ディレクトリ構造作成

### Step 2: メダル確率算出エンジン (3-4日)
plan_dynamic_competition_manager.md の「3. メダル確率算出アルゴリズム」を実装:

```python
# 重要: plan内の具体的アルゴリズムを参考に実装
class MedalProbabilityCalculator:
    async def calculate_medal_probability(self, competition_data):
        # plan_dynamic_competition_manager.md:69-137 のアルゴリズム実装
```

### Step 3: ポートフォリオ最適化 (2-3日)  
- `portfolio_optimizers/competition_portfolio_optimizer.py`
- 3コンペ最適選択・リスク分散アルゴリズム

### Step 4: 撤退・入れ替えシステム (2-3日)
- `decision_engines/withdrawal_decision_maker.py`
- plan内の撤退スコア算出ロジック実装

### Step 5: スケジューラー・自動実行 (2日)
- `utils/scheduler.py`
- 週2回定期実行・エージェント起動通知

## 完了基準
- メダル確率算出精度 > 80%
- 3コンペポートフォリオ最適化が動作  
- 週2回自動スケジュール実行
- plan_dynamic_competition_manager.md の全成功指標達成

## 次ステップ
プランニングエージェント実装開始
```

### Phase 3: プランニングエージェント実装

```
# プランニングエージェント実装指示

## 実装目標
plan_planner.md の完全実装。メダル確率算出・戦略的コンペ選択・撤退判断を担当する核心エージェント。

## 前提条件
- Issue安全連携システム完成
- 動的コンペ管理システム完成

## エージェント起動プロンプト
plan_planner.md の「6. プロンプト設計計画」(L410-449) のプロンプトを使用:

```yaml
planner_activation_prompt: |
  # 戦略プランニングエージェント実行指示
  
  ## 役割
  あなたは Kaggle メダル確率算出・戦略策定エージェントです。
  
  ## 現在のタスク
  GitHub Issue: "{issue_title}" の戦略分析を実行してください。
  
  ## 実行コンテキスト
  - 作業ディレクトリ: competitions/{competition_name}/
  - 対象コンペ: {competition_name}
  - 動的管理システムからの選択理由: {selection_rationale}
  
  ## 実行手順 (plan_planner.md準拠)
  1. コンペ基本情報の詳細収集・分析
  2. メダル確率の多次元算出・検証
  3. 専門性マッチング評価・強み活用戦略
  4. 撤退条件・リスク管理設定
  5. analyzer エージェント向け戦略Issue作成
  
  ## 完了後アクション
  GitHub Issue更新 + analyzer エージェント起動通知
```

## 実装手順

### Step 1: エージェント基盤 (1-2日)
system/agents/planner/ 構造作成・基本クラス実装

### Step 2: 確率算出ロジック (2-3日)
plan_planner.md の「3. メダル確率算出アルゴリズム」実装:
- `calculators/medal_probability.py`
- 多次元確率モデル (L56-64)

### Step 3: 戦略策定システム (2-3日)
- `strategies/selection_strategy.py`
- `strategies/withdrawal_strategy.py`
- 撤退判断アルゴリズム (L139-151)

### Step 4: GitHub Issue連携 (1-2日)
- Issue安全連携システムを活用
- analyzer起動Issue自動作成

## 完了基準
- 確率算出精度 > 80% (plan_planner.md:196)
- Issue作成→analyzer起動 < 5分 (plan_planner.md:197)
- 安定性: 1ヶ月連続運用成功 (plan_planner.md:199)

## 次ステップ
分析エージェント実装開始
```

### Phase 4: 分析エージェント実装

```
# 分析エージェント実装指示

## 実装目標
plan_analyzer.md の完全実装。グランドマスター級技術調査・最新手法研究・実装可能性判定。

## エージェント起動プロンプト
plan_analyzer.md の「8. プロンプト設計計画」(L296-333) を使用

## 実装手順

### Step 1: 技術調査システム (3-4日)
plan_analyzer.md の「3. グランドマスター解法分析システム」実装:

```python
# Owen Zhang/Abhishek Thakur パターンDB (L62-87)
GRANDMASTER_PATTERNS = {
    "owen_zhang": {
        "signature_techniques": ["multi_level_stacking", ...],
        "implementation_difficulty": 0.8,
        "success_rate": 0.85
    }
}
```

### Step 2: 実装可能性判定 (2-3日)
- `analyzers/technical_feasibility.py`
- plan内の実装可能性スコアリング (L95-115)

### Step 3: 最新技術収集 (2-3日)
- `collectors/arxiv_papers.py`
- `collectors/kaggle_solutions.py`
- WebSearch活用の自動調査システム

### Step 4: 構造化レポート生成 (1-2日)
- plan_analyzer.md の技術分析レポート形式 (L216-247)
- executor向け実装指針生成

## 完了基準
- 技術推奨精度 > 70% (plan_analyzer.md:542)
- 情報収集完了時間 < 2時間 (plan_analyzer.md:543)
- 実装成功率 > 80% (plan_analyzer.md:544)

## 次ステップ
実行エージェント実装開始
```

### Phase 5: 実行エージェント実装

```
# 実行エージェント実装指示

## 実装目標
plan_executor.md の完全実装。クラウド実行・無料リソース最大活用・自動実験。

## エージェント起動プロンプト
plan_executor.md の「6. プロンプト設計計画」(L283-320) を使用

## 実装手順

### Step 1: クラウド統合基盤 (3-4日)
plan_executor.md の「3. クラウド実行環境統合システム」実装:
- `cloud_managers/kaggle_kernel_manager.py`
- `cloud_managers/colab_execution_manager.py`
- `cloud_managers/paperspace_manager.py`

### Step 2: リソース最適化 (2-3日)
```python
# plan_executor.md のリソース配分アルゴリズム (L68-103)
class CloudResourceOptimizer:
    async def optimize_experiment_allocation(self, experiments, competition_priority):
        # 具体実装はplan参照
```

### Step 3: 自動ノートブック生成 (3-4日)
- `code_generators/notebook_generator.py`
- analyzerの技術指針 → 実行可能コード変換
- 環境別最適化 (Kaggle/Colab/Paperspace)

### Step 4: 並列実験・最適化 (2-3日)  
- `optimization_engines/hyperparameter_tuner.py`
- ベイズ最適化・Optuna統合

## 完了基準
- クラウド実行成功率 > 95% (plan_executor.md:510)
- スコア改善率 > 10% (plan_executor.md:511)
- GPU活用率 > 85% (plan_executor.md:512)

## 次ステップ
モニタリングエージェント実装開始
```

### Phase 6: モニタリングエージェント実装

```
# モニタリングエージェント実装指示

## 実装目標
plan_monitor.md の完全実装。継続学習・失敗分析・動的最適化。

## エージェント起動プロンプト
plan_monitor.md の「6. プロンプト設計計画」(L410-449) を使用

## 実装手順

### Step 1: リアルタイム監視 (2-3日)
plan_monitor.md の「3. リアルタイム実行監視システム」実装:
- `continuous_monitors/execution_monitor.py`
- executor実行の継続監視・異常検出

### Step 2: 失敗分析エンジン (3-4日)
```python
# plan_monitor.md の失敗分析 (L214-304)
class FailureAnalysisEngine:
    async def analyze_experiment_failure(self, failed_experiment):
        # 多次元失敗要因分析実装
```

### Step 3: 転移学習システム (2-3日)
- `knowledge_managers/transfer_learner.py`
- コンペ横断知識転移 (L314-400)

### Step 4: 予測・最適化提案 (2-3日)
- `prediction_engines/success_predictor.py`
- 動的最適化提案・戦略調整

## 完了基準
- 監視精度 > 90% (plan_monitor.md:667)
- 予測精度 > 80% (plan_monitor.md:668)
- 改善効果 > 15% (plan_monitor.md:669)

## 次ステップ
反省エージェント実装開始
```

### Phase 7: 反省エージェント実装

```
# 反省エージェント実装指示

## 実装目標
plan_retrospective.md の完全実装。システム自己改善・マザーリポジトリ自動更新。

## エージェント起動プロンプト
plan_retrospective.md の「6. プロンプト設計計画」(L443-481) を使用

## 実装手順

### Step 1: システム分析エンジン (3-4日)
plan_retrospective.md の「3. システム総合分析・改善識別システム」実装:
```python
# システム性能分析 (L68-140)
class SystemPerformanceAnalyzer:
    async def analyze_comprehensive_performance(self, historical_data):
        # 包括的性能分析実装
```

### Step 2: 自動改善システム (3-4日)
- `improvement_engines/code_optimizer.py`
- `improvement_engines/template_enhancer.py`
- plan内の自動コード改善アルゴリズム (L224-318)

### Step 3: テンプレート最適化 (2-3日)
- `auto_implementers/template_generator.py`
- 使用データ基づくテンプレート改良 (L328-432)

### Step 4: 自動Issue管理 (1-2日)
- `utils/git_operations.py`
- 改善Issue自動作成・更新・クローズ (L615-710)

## 完了基準
- 改善効果 > 20% (plan_retrospective.md:756)
- 自動化率 > 90% (plan_retrospective.md:757)
- 安全性: 失敗率 < 5% (plan_retrospective.md:758)

## 次ステップ
システム統合・最適化
```

### Phase 8: システム統合・本番稼働

```
# システム統合・最適化実装指示

## 統合目標
全エージェント・基盤システムの完全統合、本番環境での自律稼働実現。

## 統合手順

### Step 1: E2Eテスト実装 (2-3日)
全planファイルのテスト戦略を統合:
- 単一コンペライフサイクル完全テスト
- 3コンペ並行実行テスト
- システム障害復旧テスト

### Step 2: 本番環境セットアップ (1-2日)
- GitHub Actions ワークフロー作成
- 環境変数・認証設定自動化
- 監視・ログシステム構築

### Step 3: 初回3コンペ選択・起動 (1日)
- 動的コンペ管理システムの初回実行
- 最適3コンペ選択・エージェント起動
- 全フロー動作確認

### Step 4: 継続監視・最適化 (継続)
- システム性能監視
- 自動改善サイクル稼働確認
- メダル獲得実績追跡

## 成功基準
- 完全自動化率 > 95%
- メダル獲得確率精度 > 80%
- システム稼働率 > 99%
- 人間介入頻度 < 週1回

## 本番稼働完了
週2回動的最適化・3コンペ並行管理・完全自動メダル獲得システム稼働開始
```

---

## 🎯 重要な実装ポイント

### 1. planファイル活用の徹底
- **具体的アルゴリズム**: 各planファイルのコード例を必ず参照・実装
- **プロンプト再利用**: 既存のエージェント起動プロンプトをそのまま活用
- **成功指標厳守**: 各planの成功基準を必ず満たす実装

### 2. 依存関係順序の厳守
```
基盤システム → プランニング → 分析 → 実行 → モニタリング → 反省
```
**絶対に順序を飛ばさない**: 各段階の完了確認後に次段階着手

### 3. 段階的検証・テスト
- **各Phase完了時**: 該当planファイルの全テスト項目実行
- **統合時**: 複数エージェント連携の動作確認
- **本番前**: E2Eテスト・長期稼働テスト必須

### 4. GitHub Issue安全連携の徹底活用
- **全エージェント**: Issue安全連携システム経由でのみ通信
- **コンペ分離**: `comp:{competition_name}` ラベルでの厳密分離
- **原子性保証**: 全Issue操作で原子性・競合回避機能使用

---

## 📋 実装チェックリスト

### Phase 1-2: 基盤システム ✓
- [ ] Issue安全連携システム完成
- [ ] 動的コンペ管理システム完成
- [ ] 基盤統合テストPASS

### Phase 3-7: エージェント実装 ✓
- [ ] プランニングエージェント完成
- [ ] 分析エージェント完成  
- [ ] 実行エージェント完成
- [ ] モニタリングエージェント完成
- [ ] 反省エージェント完成

### Phase 8: システム統合 ✓
- [ ] 全エージェント統合完了
- [ ] E2Eテスト全PASS
- [ ] 本番環境稼働開始

### 最終目標達成 🏆
- [ ] 週2回動的最適化自動実行
- [ ] 最大3コンペ並行管理
- [ ] 完全自動メダル獲得システム稼働
- [ ] 人間介入なしの自律運用実現

---

この実装指示に従うことで、7つの詳細planファイルを最大活用し、複雑なマルチエージェントシステムを確実に構築できます。各段階で必ずplanファイルの具体的アルゴリズム・プロンプト・成功指標を参照して実装してください。