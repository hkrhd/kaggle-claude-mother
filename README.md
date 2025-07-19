# Kaggle Claude Mother

Claude Codeを活用したKaggleコンペティション攻略のためのマザーリポジトリです。このリポジトリをベースに、新しいコンペごとに専用リポジトリを自動生成し、効率的にメダル獲得を目指します。

## 特徴

- **Claude Code最適化**: カスタムスラッシュコマンドとテンプレートでClaude Codeを最大活用
- **自動化ワークフロー**: データダウンロード、EDA、モデリング、提出までの一連の流れを自動化
- **再利用可能な構造**: 過去のコンペで得られた知見をテンプレート化して蓄積
- **リポジトリ自動生成**: 新しいコンペ用リポジトリを1コマンドで作成

## ディレクトリ構造

```
kaggle-claude-mother/
├── README.md                    # このファイル
├── .claude/                     # Claude Code設定
│   ├── commands/               # カスタムスラッシュコマンド
│   │   ├── eda.md             # EDA実行コマンド
│   │   ├── model.md           # モデリングコマンド
│   │   ├── submit.md          # 提出コマンド
│   │   └── setup-comp.md      # 新コンペセットアップ
│   └── settings.json          # Claude Code設定
├── templates/                   # テンプレートファイル
│   ├── notebooks/             # Jupyterノートブックテンプレート
│   │   ├── 01_eda.ipynb       # EDAテンプレート
│   │   ├── 02_modeling.ipynb  # モデリングテンプレート
│   │   └── 03_ensemble.ipynb  # アンサンブルテンプレート
│   ├── src/                   # Pythonスクリプトテンプレート
│   │   ├── config.py          # 設定管理
│   │   ├── features.py        # 特徴量エンジニアリング
│   │   ├── models.py          # モデル定義
│   │   └── utils.py           # ユーティリティ関数
│   └── project_template/      # 新プロジェクト用テンプレート
├── scripts/                    # 自動化スクリプト
│   ├── setup_competition.py   # 新コンペ用リポジトリ作成
│   ├── kaggle_api.py          # Kaggle API操作
│   └── submission.py          # 自動提出
├── configs/                    # 設定ファイル
│   ├── model_configs.yaml     # モデル設定
│   └── competition_types.yaml # コンペタイプ別設定
└── pyproject.toml             # Python依存関係
```

## セットアップ

### 1. 環境準備

```bash
# リポジトリクローン
ghq get hkrhd/kaggle-claude-mother
cd $(ghq root)/github.com/hkrhd/kaggle-claude-mother

# Python環境セットアップ（uvを使用）
uv sync
```

### 2. Kaggle API設定

```bash
# Kaggle API tokenをダウンロードして配置
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Claude Code設定

```bash
# Claude Code設定を読み込み
claude load
```

## 使用方法

### 新しいコンペティション開始

```bash
# Claude Codeで新しいコンペ用リポジトリを作成
/setup-comp titanic-advanced
```

これにより以下が自動実行されます：
1. 新しいリポジトリ `kaggle-titanic-advanced` を作成
2. テンプレートファイルをコピー
3. コンペデータをダウンロード
4. 初期EDAノートブックを準備

### 分析ワークフロー

1. **EDA実行**: `/eda` - 探索的データ分析を自動実行
2. **モデリング**: `/model` - ベースラインモデルから高度なモデルまで段階的に実装
3. **提出**: `/submit` - 予測結果を生成してKaggleに自動提出

## 成功事例と学習

このリポジトリは以下の方針で継続的に改善されます：

1. **コンペ終了後の振り返り**: 有効だった手法をテンプレートに反映
2. **Claude Code最適化**: より効率的なプロンプトとワークフローを蓄積
3. **自動化レベル向上**: 手動作業を段階的に自動化

## 注意事項

- Claude Codeを使用した実装例は研究目的であり、Kaggleの利用規約を遵守してください
- APIキーや個人情報は適切に管理し、リポジトリにコミットしないでください
- 各コンペの特性に応じてテンプレートをカスタマイズして使用してください

## ライセンス

MIT License

## 貢献

プルリクエストやIssueでの改善提案を歓迎します。
