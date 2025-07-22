# Kaggle コンペ用リポジトリ初期化

あなたは経験豊富なKaggleデータサイエンティストです。指定されたコンペ用の最適化されたリポジトリを作成し、優勝に向けた完璧な作業環境を初期化してください。

## 使用方法
```
/k_init [competition-name]
```

## 実行内容

### 1. 新規コンペディレクトリ作成
- `competitions/[competition-name]/` ディレクトリを作成
- 以下の基本構造を生成:
```
competitions/[competition-name]/
├── notebooks/
│   └── submission.ipynb       # 提出用の単一ノートブック
├── data/                      # データセット格納
├── insights/
│   └── discussion_summary.md  # Discussion分析結果
├── cache/
│   └── discussions/           # Discussion分析キャッシュ
├── .claude/
│   └── commands/              # カスタムスラッシュコマンド
├── pyproject.toml             # uv設定（コンペ専用依存関係）
├── README.md                  # コンペ固有のREADME（優勝戦略含む）
└── .python-version            # Python バージョン指定
```

### 2. uv環境構築
```bash
cd competitions/[competition-name]
uv init --python 3.11
uv add pandas scikit-learn matplotlib seaborn jupyter kaggle
```

### 3. Kaggle APIでデータダウンロード
```bash
uv run kaggle competitions download -c [competition-name] -p ./data/
unzip ./data/[competition-name].zip -d ./data/
```

### 4. テンプレートファイルのコピー
以下の具体的なコマンドを順番に実行してテンプレートを安全にコピー:

```bash
# 4.1 必要なディレクトリの作成（エラーハンドリング付き）
mkdir -p "competitions/[competition-name]/.claude/commands" || {
    echo "Error: Failed to create .claude/commands directory"
    exit 1
}

# 4.2 カスタムスラッシュコマンドのコピー（存在確認付き）
if [ -d "templates/custom-slash-commands" ]; then
    cp -r templates/custom-slash-commands/* "competitions/[competition-name]/.claude/commands/" 2>/dev/null && \
    echo "✓ Custom slash commands copied successfully" || \
    echo "Warning: Some custom slash commands may not have been copied"
else
    echo "Warning: templates/custom-slash-commands directory not found"
fi

# 4.3 pyproject.toml設定ファイルのコピーと調整
if [ -f "templates/pyproject.toml.template" ]; then
    cp "templates/pyproject.toml.template" "competitions/[competition-name]/pyproject.toml" && \
    echo "✓ pyproject.toml template copied successfully" || {
        echo "Error: Failed to copy pyproject.toml template"
        exit 1
    }
    
    # プロジェクト名をコンペ名に置換
    sed -i 's/name = "kaggle-competition"/name = "[competition-name]"/' \
        "competitions/[competition-name]/pyproject.toml" 2>/dev/null || \
        echo "Warning: Failed to update project name in pyproject.toml"
    
    # 説明をコンペ固有に更新
    sed -i 's/description = "Kaggle competition workspace"/description = "[competition-name] Kaggle competition workspace"/' \
        "competitions/[competition-name]/pyproject.toml" 2>/dev/null || \
        echo "Warning: Failed to update project description"
else
    echo "Error: templates/pyproject.toml.template not found"
    exit 1
fi

# 4.4 追加テンプレートファイルの処理（将来の拡張対応）
if [ -d "templates/notebooks" ]; then
    mkdir -p "competitions/[competition-name]/notebooks"
    cp -r templates/notebooks/* "competitions/[competition-name]/notebooks/" 2>/dev/null && \
    echo "✓ Notebook templates copied" || \
    echo "Warning: Notebook templates copy failed"
fi

# 4.5 コピー作業の検証
echo "=== Template Copy Verification ==="
echo "Checking copied files:"
ls -la "competitions/[competition-name]/.claude/commands/" 2>/dev/null | grep -v "total" || echo "No custom commands copied"
echo "pyproject.toml status:"
[ -f "competitions/[competition-name]/pyproject.toml" ] && echo "✓ pyproject.toml exists" || echo "✗ pyproject.toml missing"
echo "==============================="
```

**コピー作業のポイント:**
- **エラーハンドリング**: 各ステップで失敗時の適切な処理を実装
- **存在確認**: ファイル・ディレクトリの存在を事前確認
- **プレースホルダー置換**: `[competition-name]` を実際のコンペ名に自動置換
- **検証ステップ**: コピー完了後に結果を確認・報告
- **段階的処理**: 部分的な失敗でも可能な限り処理を継続
- **拡張性**: 将来のテンプレート追加に対応した構造

### 5. コンペリポジトリ内での追加セットアップ
```bash
cd competitions/[competition-name]
uv sync
uv run pre-commit install
```
- **依存関係同期**: `uv sync` でプロジェクト依存関係を確実にインストール
- **Pre-commit設定**: コード品質管理のためのpre-commitフックを有効化
- **開発環境準備**: リンター、フォーマッターなどの開発ツール環境を整備

### 6. コンペ優勝戦略README生成
- 以下の項目を含む包括的なREADMEを `README.md` として生成:

#### 6.1 コンペ概要・情報収集
- **コンペ詳細**: WebSearchでコンペの目的、評価指標、データセット概要を調査
- **ルール確認**: 外部データ使用可否、提出ルール、期限の詳細
- **賞金・参加者数**: モチベーション維持のための基本情報
- **過去の類似コンペ**: 同種コンペの勝利解法パターンをWebSearchで調査
- **ドメイン知識**: 問題領域の専門知識をWebSearchで収集

#### 6.2 2024年最新勝利戦略（WebSearch結果を基に記載）
- **特徴量エンジニアリング戦略**: 
  - GPU加速cuDF-pandasを使用した10,000+特徴量の高速生成・検証
  - Target Encoding、PCA結合、1D CNN特徴抽出の活用
  - COL1, COL2, STAT組み合わせの系統的探索
- **アンサンブル・スタッキング戦略**:
  - 多層スタッキング：GBDT、NN、SVR、KNNの多様性確保
  - cuMLを使用したGPU加速でのモデル大量訓練
  - 異なる前処理・アーキテクチャによるモデル多様性戦略
- **クロスバリデーション戦略**:
  - コンペ開始時からの堅牢なCV戦略確立
  - 時系列性・グループ性を考慮した適切な分割設計
  - Private/Public LB乖離対策のためのCV重視

#### 6.3 マザーリポジトリ活用情報
- **環境管理方針**: uvを使用した独立環境構築（マザーリポジトリとの分離）
- **テンプレート活用**: `templates/`からの再利用可能コードブロック活用
- **提出方式**: 単一ノートブック提出の標準的パターン
- **カスタムコマンド**: Discussion更新チェック（/k_update-insights）の活用
- **リポジトリ独立性**: コンペ完了後はマザーから独立した作業継続

#### 6.4 実行ロードマップ
- **初期設定（1-2日）**: 環境構築、データ理解、CV戦略設計
- **Week 1-2**: EDA完了、ベースライン確立、特徴量生成パイプライン構築
- **Week 3-4**: GPU加速特徴量エンジニアリング、単体モデル最適化
- **Week 5-6**: 多層スタッキング構築、アンサンブル最適化
- **Week 7**: 最終提出準備、2つの提出枠戦略

#### 6.5 最新技術活用計画
- **GPU加速**: NVIDIA cuDF-pandas、cuMLの活用
- **ハイブリッドアーキテクチャ**: CNN+Transformer組み合わせ
- **医療画像分野**: 高度なデータ拡張技術
- **1D CNN**: タブラーデータでの1D CNN→2D CNN段階的アプローチ

#### 6.6 Discussion・コミュニティ戦略
- **毎日のDiscussion確認**: 新手法・データリーク・バグ情報収集
- **過去勝利解法研究**: Kaggle Solutionsからのブループリント学習
- **コミュニティ貢献**: 知見共有によるネットワーク構築とフィードバック獲得
- **Grandmaster知見活用**: 2024年最新のGrandmaster戦略適用

#### 6.7 計算資源・時間管理
- **GPU効率活用**: 特徴量生成・モデル訓練の並列化
- **高影響・低労力タスク**: 優先度マトリクス設計
- **リスク管理**: 提出直前の安全策・バックアップ戦略

## 実行指示

1. まず引数で指定されたコンペ名が有効かKaggle APIで確認
2. 上記の初期化手順を順番に実行（分析は実行しない）
3. 各ステップの進捗状況を報告
4. エラーが発生した場合は代替手段を提案
5. **重要**: 初期化完了後、ユーザーに以下を明確に伝達:
   ```
   初期化が完了しました。今後の作業は以下のディレクトリで実行してください：
   cd competitions/[competition-name]
   
   このディレクトリ内で Claude Code を使用することで、
   マザーリポジトリから独立した完全な作業環境が利用できます。
   ```

## 初期化後の独立作業について

**完全な環境独立性**: この初期化により、`competitions/[competition-name]/`ディレクトリは以下を含む完全に独立した作業環境となります：

- **独立したuv環境**: コンペ専用の依存関係管理
- **カスタムコマンド**: Discussion更新チェックなどの専用ワークフロー  
- **優勝戦略README**: 2024年最新の勝利手法を含む包括的な戦略ガイド
- **マザーリポジトリとの分離**: 今後マザーリポジトリへの依存なし

**注意**: 
- Kaggle API の認証設定（`~/.kaggle/kaggle.json`）が必要です
- 初期化は設定とREADME生成のみに特化し、実際の分析は行いません
- 全ての分析・モデリング作業は初期化後にコンペディレクトリ内で実行してください