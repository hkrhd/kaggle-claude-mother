"""
参加確認監視システム

Issue状態を監視し、クローズされたらオーケストレーションを開始するシステム
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from system.competition_participation_manager import CompetitionParticipationManager, ParticipationStatus
from system.orchestrator.master_orchestrator import MasterOrchestrator


class ParticipationMonitor:
    """参加確認監視システム"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        
        # 参加管理
        self.participation_manager = CompetitionParticipationManager(github_token, repo_name)
        
        # オーケストレーター（参加確認後に起動）
        self.orchestrator = MasterOrchestrator()
        
        # 監視設定
        self.monitoring_interval_seconds = 60  # 1分間隔でチェック
        self.is_monitoring = False
        
    async def start_monitoring(self):
        """監視開始"""
        
        self.logger.info("🔍 参加確認監視開始")
        self.is_monitoring = True
        
        while self.is_monitoring:
            try:
                # 参加リクエスト状況確認
                monitoring_result = await self.participation_manager.monitor_participation_requests()
                
                # 新規参加確認があった場合
                for competition_id in monitoring_result["newly_registered"]:
                    await self._start_competition_orchestration(competition_id)
                
                # 監視サマリーログ
                if monitoring_result["checked_requests"] > 0:
                    self.logger.info(f"監視結果: {monitoring_result}")
                
                # 次回チェックまで待機
                await asyncio.sleep(self.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"監視エラー: {e}")
                await asyncio.sleep(self.monitoring_interval_seconds)
    
    async def _start_competition_orchestration(self, competition_id: str):
        """コンペオーケストレーション開始"""
        
        self.logger.info(f"🚀 オーケストレーション開始: {competition_id}")
        
        try:
            # 参加確認済みコンペに対してオーケストレーション実行
            result = await self.orchestrator.orchestrate_competition(
                competition_name=competition_id,
                orchestration_mode="sequential"
            )
            
            self.logger.info(f"✅ オーケストレーション完了: {competition_id}")
            
        except Exception as e:
            self.logger.error(f"オーケストレーションエラー ({competition_id}): {e}")
    
    def stop_monitoring(self):
        """監視停止"""
        
        self.logger.info("🛑 参加確認監視停止")
        self.is_monitoring = False
    
    async def check_single_competition(self, competition_id: str) -> Optional[ParticipationStatus]:
        """単一コンペの参加状態確認"""
        
        return await self.participation_manager.check_participation_status(competition_id)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """監視状態取得"""
        
        participation_summary = self.participation_manager.get_participation_summary()
        
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "participation_summary": participation_summary,
            "current_time": datetime.now(timezone.utc).isoformat()
        }


async def main():
    """メイン実行（テスト用）"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    github_token = os.getenv("GITHUB_TOKEN") or open(os.path.expanduser("~/.github_token")).read().strip()
    repo_name = "hkrhd/kaggle-claude-mother"
    
    monitor = ParticipationMonitor(github_token, repo_name)
    
    print("🔍 参加確認監視システム開始")
    print("   - Issue #16 (House Prices) がクローズされるとオーケストレーション開始")
    print("   - Ctrl+C で停止")
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 監視停止")
        monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())