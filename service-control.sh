#!/bin/bash
# Kaggle Claude Mother サービス制御スクリプト

SERVICE_NAME="kaggle-claude-mother.service"

case "$1" in
    start)
        echo "🚀 Kaggle Claude Mother サービス開始中..."
        systemctl --user start $SERVICE_NAME
        echo "✅ サービス開始完了"
        systemctl --user status $SERVICE_NAME --no-pager
        ;;
    stop)
        echo "🛑 Kaggle Claude Mother サービス停止中..."
        systemctl --user stop $SERVICE_NAME
        echo "✅ サービス停止完了"
        ;;
    restart)
        echo "🔄 Kaggle Claude Mother サービス再起動中..."
        systemctl --user restart $SERVICE_NAME
        echo "✅ サービス再起動完了"
        systemctl --user status $SERVICE_NAME --no-pager
        ;;
    status)
        echo "📊 Kaggle Claude Mother サービス状態:"
        systemctl --user status $SERVICE_NAME --no-pager
        ;;
    logs)
        echo "📋 Kaggle Claude Mother サービスログ:"
        journalctl --user -u $SERVICE_NAME -f
        ;;
    enable)
        echo "⚙️ Kaggle Claude Mother サービス自動起動有効化..."
        systemctl --user enable $SERVICE_NAME
        echo "✅ 自動起動有効化完了"
        ;;
    disable)
        echo "⚙️ Kaggle Claude Mother サービス自動起動無効化..."
        systemctl --user disable $SERVICE_NAME
        echo "✅ 自動起動無効化完了"
        ;;
    clean)
        echo "🧹 まっさらな状態に初期化中..."
        systemctl --user stop $SERVICE_NAME 2>/dev/null || true
        systemctl --user disable $SERVICE_NAME 2>/dev/null || true
        rm -rf /home/hh/ws/github.com/hkrhd/kaggle-claude-mother/logs/*
        echo "✅ システム初期化完了"
        ;;
    *)
        echo "使用方法: $0 {start|stop|restart|status|logs|enable|disable|clean}"
        echo ""
        echo "コマンド説明:"
        echo "  start   - サービス開始"
        echo "  stop    - サービス停止"
        echo "  restart - サービス再起動"
        echo "  status  - サービス状態確認"
        echo "  logs    - サービスログ表示（リアルタイム）"
        echo "  enable  - 自動起動有効化"
        echo "  disable - 自動起動無効化"
        echo "  clean   - まっさら状態に初期化"
        exit 1
        ;;
esac