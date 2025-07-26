#!/usr/bin/env python3
"""
監視サービス - systemdタイマーから定期実行される監視スクリプト
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timezone

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

from system.agents.monitor.monitor_agent import MonitorAgent
from system.config.system_config import ConfigManager


async def run_monitoring_check():
    """監視チェック実行"""
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 設定読み込み
        config_manager = ConfigManager("config/system.yaml")
        config = config_manager.load_config()
        
        # GitHub認証
        github_token = config.github_token
        if not github_token:
            import subprocess
            try:
                result = subprocess.run(
                    ['gh', 'auth', 'token'], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                github_token = result.stdout.strip()
            except subprocess.CalledProcessError:
                logger.error("GitHub認証トークン取得失敗")
                return 1
        
        # 監視エージェント初期化
        monitor = MonitorAgent(
            github_token=github_token,
            repo_name=config.repo_name or "hkrhd/kaggle-claude-mother"
        )
        
        logger.info(f"🔍 定期監視チェック開始: {datetime.now(timezone.utc).isoformat()}")
        
        # サービス健全性チェック実行
        health_check = await monitor._perform_service_health_check()
        
        # リソース監視
        await monitor._check_system_resources(health_check)
        
        # 自動修復（必要な場合）
        if monitor.auto_repair_enabled and (
            not health_check.service_active or 
            health_check.error_patterns_detected or
            health_check.memory_usage_percent > monitor.alert_thresholds["memory_usage_threshold_percent"]
        ):
            await monitor._perform_auto_repair(health_check)
        
        # 健全性履歴に追加
        monitor.service_health_history.append(health_check)
        
        # 定期レポート投稿（30分毎）
        if len(monitor.service_health_history) % 30 == 0:
            await monitor._post_service_status_report(health_check)
        
        logger.info(f"✅ 定期監視チェック完了: サービス{'アクティブ' if health_check.service_active else '非アクティブ'}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 監視チェックエラー: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_monitoring_check())
    sys.exit(exit_code)