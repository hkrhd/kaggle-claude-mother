# Kaggle Claude Mother

Claude Codeを活用してKaggleコンペを効率的に攻略するためのマザーリポジトリ。

コンペごとのリポジトリ作成時に使用するカスタムコマンドとテンプレートを提供します。各コンペは`competitions/`ディレクトリ以下に独立したリポジトリとして作成され、個別にuvで環境管理を行います。

## 概要

- **カスタムコマンド提供**: Claude Codeで使用するコンペ管理コマンド（/find-comp、/start等）
- **テンプレート管理**: コンペ用リポジトリのひな形
- **ディレクトリ管理**: 各コンペの作業領域を整理

## ワークフロー

```bash
# 1. おすすめコンペを検索
/find-comp

# 2. 選択したコンペ用リポジトリを作成（competitions/以下に作成）
/start titanic

# 3. コンペディレクトリに移動して作業開始
cd competitions/titanic

# 4. 個別に環境構築（初回のみ）
uv init
uv add pandas scikit-learn matplotlib seaborn jupyter

# 5. 以降の作業はコンペディレクトリ内で実行
# ノートブック実行、データ分析、モデル訓練など

# 6. 定期的にDiscussion更新をチェック
/update-insights
```

## ディレクトリ構造

```
kaggle-claude-mother/                 # マザーリポジトリ（コマンド・テンプレート管理）
├── .claude/
│   └── commands/                     # Claude Codeカスタムコマンド
│       ├── find-comp.md              # コンペ検索・提案
│       ├── start.md                  # リポジトリ作成＋初期分析
│       └── update-insights.md        # Discussion更新チェック
├── templates/                        # コンペ用テンプレート
│   ├── notebooks/                    # 分析ノートブック雛形
│   └── pyproject.toml.template       # uv設定テンプレート
└── competitions/                     # 各コンペの作業ディレクトリ
    ├── titanic/                      # 例：タイタニックコンペ
    │   ├── pyproject.toml            # 個別のuv環境
    │   ├── .venv/                    # コンペ専用仮想環境
    │   ├── notebooks/
    │   ├── data/
    │   └── insights/
    └── house-prices/                 # 例：住宅価格予測コンペ
        ├── pyproject.toml
        ├── .venv/
        └── ...
```

## セットアップ

マザーリポジトリは指示とテンプレートの管理のみ行います。実際の開発環境は各コンペディレクトリで個別に構築します。

```bash
# 1. マザーリポジトリのクローン
ghq get your-username/kaggle-claude-mother
cd $(ghq root)/github.com/your-username/kaggle-claude-mother

# 2. Kaggle API設定
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Claude Code読み込み
claude load

# 4. 新しいコンペを始める場合
/start competition-name  # competitions/competition-name/ が作成される

# 5. コンペディレクトリに移動
cd competitions/competition-name

# 6. 環境構築と作業実行
uv init
uv add pandas scikit-learn matplotlib seaborn jupyter kaggle
# 以降の分析・モデリング作業はこのディレクトリ内で実行
```

## 提出方式

**単一ノートブック提出**が標準的な方式です。Kaggleコンペでは1つのノートブックで全ての処理（EDA、前処理、モデリング、予測）を完結させます。

```bash
# 各コンペディレクトリに移動して実行
cd competitions/competition-name
uv run kaggle kernels push -p ./notebooks/submission.ipynb
```

**重要**: 全ての作業（データ分析、モデル訓練、提出）は対応するコンペディレクトリ内で実行してください。

## 共通コードブロック

再利用可能なコードブロックは`templates/`ディレクトリに配置され、新しいコンペ作成時に各コンペディレクトリにコピーされます。

## エージェント間連携システム（GitHub Issue API）

ファイルベースではなく**GitHub Issueベース**でエージェント間のやり取りを行います：

### エージェント構成と役割

#### 1. プランニングエージェント (`agent:planner`)
- **主要責務**: 全体戦略・タイムライン策定
- **作業内容**:
  - コンペ概要の分析とアプローチ選定
  - 段階別実行計画の立案
  - リスク評価とマイルストーン設定
- **Issue作成例**: "Strategy Planning: [コンペ名] 攻略戦略"

#### 2. 分析エージェント (`agent:analyzer`) 
- **主要責務**: データ理解・外部情報収集
- **作業内容**:
  - EDA実行とデータ特性把握
  - Discussion分析と有用情報抽出
  - 競合解法の調査・ベンチマーク分析
- **Issue作成例**: "Analysis: データ分析とDiscussion調査"

#### 3. 実行エージェント (`agent:executor`)
- **主要責務**: 実装・実験・モデル構築
- **作業内容**:
  - 特徴量エンジニアリング実装
  - モデル訓練・バリデーション
  - アンサンブル構築・提出ファイル作成
- **コメント報告**: 実験結果、スコア、課題

#### 4. モニタリングエージェント (`agent:monitor`)
- **主要責務**: 進捗管理・情報更新監視  
- **作業内容**:
  - Issue進捗の追跡とボトルネック特定
  - Discussion新規投稿の監視と重要度評価
  - リーダーボード変動とデッドライン管理
- **自動化機能**: `/update-insights`による定期チェック

### ラベル体系（最小構成）
- `agent:planner` - プランニングエージェント担当
- `agent:analyzer` - 分析エージェント担当
- `agent:executor` - 実行エージェント担当
- `agent:monitor` - モニタリングエージェント担当
- `status:todo` - 未着手
- `status:in-progress` - 作業中
- `priority:high/medium/low` - 優先度

### 連携フロー
1. **プランニングエージェント** → 戦略Issue作成（全体計画をbodyに記載）
2. **分析エージェント** → データ分析・Discussion調査Issue作成、findings報告
3. **実行エージェント** → 実装Issue作成、実験結果をコメントで逐次報告
4. **モニタリングエージェント** → 各Issueの進捗を監視、課題があれば新Issue作成
5. **完了時** → 各エージェントがIssueクローズ、最終結果をコメントで記録
6. **コンペ共通記憶** → MDファイルで管理継続（Issue履歴とは別に保持）

## カスタムコマンド詳細

> **注意**: 
> - `k_`で始まる単語（例：`k_init`、`k_start`等）は、このリポジトリ専用のカスタムスラッシュコマンドを指します。
> - `k_init`作業中はweb検索は必要ありません。

### `/find-comp`

- アクティブなコンペを取得
- メダル獲得可能性、参加者数、賞金を分析
- おすすめ度をスコアリングして提示

### `/start [competition-name]`

- `competitions/[competition-name]/`ディレクトリ作成
- テンプレートからの初期ファイル配置
- データダウンロード
- Discussion分析（上位投稿、有用な手法を抽出）
- 初期EDA自動実行

### `/update-insights`

- 前回チェック以降の新規Discussionを分析
- 各コンペの`insights/`ディレクトリに知見を追記
- 重要な更新があれば通知

## Discussion分析機能

各コンペのDiscussionから以下を自動抽出：

- 高評価の解法アプローチ
- 有効な特徴量エンジニアリング
- バグ情報・注意点

キャッシュ戦略：

- 初回実行時：全Discussion取得
- 2回目以降：差分のみ取得（各コンペの`cache/discussions/last_check.json`）

## リポジトリ管理方針

**マルチリポジトリ管理**: 各コンペは独立したGitリポジトリとして管理されます。マザーリポジトリとは別の個別リポジトリです。

各コンペは`competitions/`ディレクトリ以下で管理され、以下の構造に従います：

```
competitions/[competition-name]/
├── pyproject.toml           # uv設定（コンペ専用依存関係）
├── .venv/                   # 仮想環境（個別管理）
├── notebooks/
│   └── submission.ipynb     # 提出用の単一ノートブック
├── data/                    # ダウンロードしたデータセット
├── insights/
│   └── discussion_summary.md # 抽出された知見
└── cache/
    └── discussions/         # Discussion分析キャッシュ
```

**管理方針:**
- 各コンペは`competitions/`以下で独立したGitリポジトリとして管理
- コンペごとに独立したuv環境を構築
- マザーリポジトリ、各コンペリポジトリは完全に分離
- **重要**: マザーリポジトリ直下にはコンペデータやモデルファイルは置かない
- テンプレートとカスタムコマンドのみをマザーで管理