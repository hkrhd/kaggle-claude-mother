# Kaggle Claude Mother

Claude Codeを活用してKaggleコンペを効率的に攻略するためのマザーリポジトリ。

## 概要

- **自動コンペ検索・分析**: 参加価値の高いコンペを自動で提案
- **Discussion自動分析**: コンペのDiscussionを定期的に分析し、知見を蓄積
- **リポジトリ自動生成**: コンペごとに最適化されたリポジトリを自動作成

## ワークフロー

```bash
# 1. おすすめコンペを検索
/find-comp

# 2. 選択したコンペ用リポジトリを作成（Discussion分析込み）
/start titanic

# 3. 定期的にDiscussion更新をチェック
/update-insights
```

## ディレクトリ構造

```
kaggle-claude-mother/
├── .claude/
│   └── commands/         # カスタムコマンド
│       ├── find-comp.md  # コンペ検索・提案
│       ├── start.md      # リポジトリ作成＋初期分析
│       └── update-insights.md # Discussion更新チェック
├── templates/
│   ├── notebooks/        # 分析ノートブック
│   └── src/             # 共通モジュール
├── cache/
│   └── discussions/      # Discussion分析キャッシュ
└── insights/            # コンペごとの知見DB
```

## セットアップ

```bash
# 1. クローン
ghq get your-username/kaggle-claude-mother
cd $(ghq root)/github.com/your-username/kaggle-claude-mother

# 2. 環境構築
uv sync

# 3. Kaggle API設定
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 4. Claude Code読み込み
claude load
```

## カスタムコマンド詳細

### `/find-comp`

- アクティブなコンペを取得
- メダル獲得可能性、参加者数、賞金を分析
- おすすめ度をスコアリングして提示

### `/start [competition-name]`

- 新規リポジトリ作成
- データダウンロード
- Discussion分析（上位投稿、有用な手法を抽出）
- 初期EDA自動実行

### `/update-insights`

- 前回チェック以降の新規Discussionを分析
- `insights/[competition]/`に知見を追記
- 重要な更新があれば通知

## Discussion分析機能

各コンペのDiscussionから以下を自動抽出：

- 高評価の解法アプローチ
- 有効な特徴量エンジニアリング
- バグ情報・注意点

キャッシュ戦略：

- 初回実行時：全Discussion取得
- 2回目以降：差分のみ取得（`cache/discussions/[comp-name]/last_check.json`）

## 生成されるリポジトリ構造

```
kaggle-[competition-name]/
├── notebooks/
│   ├── 01_eda_auto.ipynb    # 自動生成されたEDA
│   └── 02_baseline.ipynb    # Discussion分析に基づくベースライン
├── insights/
│   └── discussion_summary.md # 抽出された知見
└── src/                     # マザーリポジトリから継承
```

