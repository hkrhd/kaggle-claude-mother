#!/usr/bin/env python3
"""
Claude Mother System - メインアプリケーション

Kaggle競技完全自動化システムのメインエントリーポイント。
全エージェント統合・自律実行・監視・学習機能を提供。
"""

import asyncio
import sys
import os
import logging
import argparse
import signal
from datetime import datetime
from typing import Dict, List, Any, Optional

# パス設定
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

# システムコンポーネント
from system.orchestrator.master_orchestrator import MasterOrchestrator, OrchestrationMode
from system.config.system_config import ConfigManager, get_config
from system.competition_manager.dynamic_competition_manager import DynamicCompetitionManager


class ClaudeMotherSystem:
    """Claude Mother System メインアプリケーション"""
    
    def __init__(self, config_path: str = "config/system.yaml"):
        # 設定読み込み
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        # ログ設定
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # システムコンポーネント
        self.orchestrator: Optional[MasterOrchestrator] = None
        self.competition_manager: Optional[DynamicCompetitionManager] = None
        
        # 状態管理
        self.is_running = False
        self.autonomous_mode = False
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"🚀 Claude Mother System 初期化完了 (v{self.config.version})")
    
    def setup_logging(self):
        """ログ設定"""
        
        log_level = getattr(logging, self.config.log_level.value)
        
        # ログディレクトリ作成
        log_dir = os.path.dirname(self.config.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # ログフォーマット
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # ファイルハンドラー
        file_handler = logging.FileHandler(
            self.config.log_file_path, 
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # ルートロガー設定
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    async def initialize_system(self):
        """システム初期化"""
        
        self.logger.info("システム初期化開始...")
        
        try:
            # 設定検証
            validation_errors = self.config_manager.validate_config()
            if validation_errors:
                for error in validation_errors:
                    self.logger.error(f"設定エラー: {error}")
                raise ValueError(f"設定に問題があります: {validation_errors}")
            
            # マスターオーケストレーター初期化
            self.orchestrator = MasterOrchestrator(
                github_token=self.config.github_token,
                repo_name=self.config.repo_name,
                config=self.config.__dict__
            )
            
            # 競技管理システム初期化
            self.competition_manager = DynamicCompetitionManager(
                github_token=self.config.github_token,
                repo_name=self.config.repo_name
            )
            
            self.is_running = True
            self.logger.info("✅ システム初期化完了")
            
        except Exception as e:
            self.logger.error(f"❌ システム初期化失敗: {e}")
            raise
    
    async def run_competition(
        self,
        competition_name: str,
        competition_type: str = "tabular",
        orchestration_mode: str = "adaptive",
        max_gpu_hours: float = None
    ) -> Dict[str, Any]:
        """単一競技実行"""
        
        if not self.is_running:
            await self.initialize_system()
        
        self.logger.info(f"🎯 競技実行開始: {competition_name}")
        
        # 競技データ作成
        competition_data = {
            "id": f"manual-{competition_name.lower().replace(' ', '-')}",
            "name": competition_name,
            "type": competition_type,
            "deadline": (datetime.utcnow().add(days=30)).isoformat(),
            "priority": "high",
            "resource_budget": {
                "max_gpu_hours": max_gpu_hours or self.config.default_gpu_budget_hours,
                "max_api_calls": self.config.default_api_calls_limit,
                "max_execution_time_hours": self.config.max_execution_time_hours
            },
            "target_performance": {
                "min_score_improvement": 0.05,
                "target_ranking_percentile": 0.1
            }
        }
        
        # オーケストレーションモード変換
        mode_mapping = {
            "sequential": OrchestrationMode.SEQUENTIAL,
            "parallel": OrchestrationMode.PARALLEL,
            "adaptive": OrchestrationMode.ADAPTIVE,
            "emergency": OrchestrationMode.EMERGENCY
        }
        mode = mode_mapping.get(orchestration_mode, OrchestrationMode.ADAPTIVE)
        
        # 競技実行
        try:
            result = await self.orchestrator.orchestrate_competition(
                competition_data=competition_data,
                orchestration_mode=mode
            )
            
            self.logger.info(f"🏁 競技実行完了: {competition_name} (成功: {result.success})")
            
            return {
                "success": result.success,
                "orchestration_id": result.orchestration_id,
                "final_phase": result.final_phase.value,
                "overall_score": result.overall_score,
                "total_duration_hours": result.total_duration_hours,
                "resource_consumption": result.resource_consumption,
                "phase_results": result.phase_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ 競技実行エラー: {competition_name} - {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def start_autonomous_mode(self):
        """自律実行モード開始"""
        
        if not self.is_running:
            await self.initialize_system()
        
        self.logger.info("🤖 自律実行モード開始")
        self.autonomous_mode = True
        
        try:
            await self.orchestrator.start_autonomous_mode()
        except Exception as e:
            self.logger.error(f"❌ 自律実行モードエラー: {e}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        
        if not self.orchestrator:
            return {
                "status": "not_initialized",
                "message": "System not initialized"
            }
        
        orchestrator_status = await self.orchestrator.get_system_status()
        
        return {
            "status": "running" if self.is_running else "stopped",
            "autonomous_mode": self.autonomous_mode,
            "environment": self.config.environment.value,
            "system_version": self.config.version,
            "orchestrator_status": orchestrator_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def stop_system(self):
        """システム停止"""
        
        self.logger.info("🛑 システム停止中...")
        
        self.is_running = False
        self.autonomous_mode = False
        
        # 各コンポーネントの停止処理
        if self.orchestrator:
            # アクティブな実行の完了待機
            if hasattr(self.orchestrator, 'active_contexts'):
                active_count = len(self.orchestrator.active_contexts)
                if active_count > 0:
                    self.logger.info(f"⏳ アクティブな実行完了待機中: {active_count}件")
                    # 実際の実装では適切な停止処理を行う
        
        self.logger.info("✅ システム停止完了")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー"""
        
        self.logger.info(f"📶 シグナル受信: {signum}")
        
        # 非同期停止処理
        asyncio.create_task(self.stop_system())


async def main():
    """メイン関数"""
    
    # コマンドライン引数パース
    parser = argparse.ArgumentParser(
        description="Claude Mother System - Kaggle競技完全自動化システム"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["single", "autonomous", "status"],
        default="single",
        help="実行モード"
    )
    
    parser.add_argument(
        "--competition",
        help="競技名（singleモード用）"
    )
    
    parser.add_argument(
        "--type",
        choices=["tabular", "computer-vision", "nlp"],
        default="tabular",
        help="競技タイプ"
    )
    
    parser.add_argument(
        "--orchestration",
        choices=["sequential", "parallel", "adaptive", "emergency"],
        default="adaptive",
        help="オーケストレーションモード"
    )
    
    parser.add_argument(
        "--gpu-hours",
        type=float,
        help="最大GPU時間"
    )
    
    parser.add_argument(
        "--config",
        default="config/system.yaml",
        help="設定ファイルパス"
    )
    
    args = parser.parse_args()
    
    # システム初期化
    system = ClaudeMotherSystem(args.config)
    
    try:
        if args.mode == "single":
            # 単一競技実行
            if not args.competition:
                print("❌ エラー: --competition が必要です")
                return 1
            
            print(f"🎯 競技実行: {args.competition}")
            
            result = await system.run_competition(
                competition_name=args.competition,
                competition_type=args.type,
                orchestration_mode=args.orchestration,
                max_gpu_hours=args.gpu_hours
            )
            
            print("\n📊 実行結果:")
            print(f"成功: {'✅' if result['success'] else '❌'}")
            if result['success']:
                print(f"オーケストレーションID: {result['orchestration_id']}")
                print(f"最終フェーズ: {result['final_phase']}")
                print(f"総合スコア: {result['overall_score']:.2f}")
                print(f"実行時間: {result['total_duration_hours']:.1f}時間")
                print(f"GPU使用時間: {result['resource_consumption'].get('total_gpu_hours', 0):.1f}時間")
            else:
                print(f"エラー: {result.get('error', 'Unknown error')}")
            
            return 0 if result['success'] else 1
        
        elif args.mode == "autonomous":
            # 自律実行モード
            print("🤖 自律実行モード開始")
            print("Ctrl+C で停止できます")
            
            await system.start_autonomous_mode()
            
            return 0
        
        elif args.mode == "status":
            # システム状態確認
            await system.initialize_system()
            status = await system.get_system_status()
            
            print("\n📊 システム状態:")
            print(f"ステータス: {status['status']}")
            print(f"環境: {status['environment']}")
            print(f"バージョン: {status['system_version']}")
            print(f"自律モード: {'有効' if status['autonomous_mode'] else '無効'}")
            
            if 'orchestrator_status' in status:
                orch_status = status['orchestrator_status']
                print(f"処理済み競技数: {orch_status['total_competitions_handled']}")
                print(f"成功率: {orch_status['success_rate']:.1%}")
                print(f"稼働時間: {orch_status['uptime_hours']:.1f}時間")
            
            return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  ユーザー中断")
        await system.stop_system()
        return 0
    
    except Exception as e:
        print(f"\n❌ システムエラー: {e}")
        await system.stop_system()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)