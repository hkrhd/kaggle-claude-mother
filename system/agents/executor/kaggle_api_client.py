"""
Kaggle API クライアント

実際のKaggle APIを使用したデータ取得・提出・スコア監視システム。
Disaster Tweetsコンペでのメダル級スコア達成を目指す。
"""

import os
import json
import logging
import asyncio
import zipfile
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("Kaggle API不利用: pip install kaggle が必要")


@dataclass
class SubmissionResult:
    """提出結果"""
    submission_id: str
    status: str
    public_score: Optional[float]
    private_score: Optional[float]
    submission_time: datetime
    file_name: str
    description: str


@dataclass
class CompetitionDataset:
    """コンペデータセット"""
    competition_name: str
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    sample_submission: pd.DataFrame
    data_description: str
    downloaded_at: datetime


class KaggleAPIClient:
    """Kaggle API クライアント"""
    
    def __init__(self, kaggle_config_path: str = "kaggle.json"):
        self.logger = logging.getLogger(__name__)
        self.kaggle_config_path = kaggle_config_path
        self.api = None
        self.authenticated = False
        
        # データ保存パス
        self.data_dir = Path("data")
        self.submissions_dir = Path("submissions")
        self.data_dir.mkdir(exist_ok=True)
        self.submissions_dir.mkdir(exist_ok=True)
        
        self._setup_kaggle_api()
    
    def _setup_kaggle_api(self):
        """Kaggle API 設定"""
        
        try:
            if not KAGGLE_AVAILABLE:
                self.logger.error("Kaggle APIライブラリが利用不可")
                return
            
            # Kaggle認証設定
            config_path = Path(self.kaggle_config_path)
            if config_path.exists():
                # 環境変数にKaggle認証情報を設定
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                os.environ['KAGGLE_USERNAME'] = config['username']
                os.environ['KAGGLE_KEY'] = config['key']
                
                self.logger.info(f"Kaggle認証設定完了: {config['username']}")
            else:
                self.logger.error(f"Kaggle設定ファイルが見つかりません: {config_path}")
                return
            
            # API初期化
            self.api = KaggleApi()
            self.api.authenticate()
            self.authenticated = True
            
            self.logger.info("✅ Kaggle API認証成功")
            
        except Exception as e:
            self.logger.error(f"Kaggle API設定失敗: {e}")
            self.authenticated = False
    
    async def download_competition_data(self, competition_name: str) -> Optional[CompetitionDataset]:
        """コンペデータダウンロード"""
        
        if not self.authenticated:
            self.logger.error("Kaggle API未認証")
            return None
        
        try:
            self.logger.info(f"データダウンロード開始: {competition_name}")
            
            # コンペデータダウンロード
            comp_data_dir = self.data_dir / competition_name
            comp_data_dir.mkdir(exist_ok=True)
            
            # ダウンロード実行（同期処理）
            self.api.competition_download_files(
                competition=competition_name,
                path=str(comp_data_dir),
                quiet=False
            )
            
            # ZIPファイル展開
            zip_files = list(comp_data_dir.glob("*.zip"))
            for zip_file in zip_files:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(comp_data_dir)
                zip_file.unlink()  # ZIPファイル削除
            
            # データファイル読み込み
            train_file = comp_data_dir / "train.csv"
            test_file = comp_data_dir / "test.csv"
            sample_file = comp_data_dir / "sample_submission.csv"
            
            if not all([train_file.exists(), test_file.exists(), sample_file.exists()]):
                self.logger.error(f"必要なデータファイルが見つかりません: {comp_data_dir}")
                return None
            
            # DataFrame読み込み
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            sample_submission = pd.read_csv(sample_file)
            
            # データ特性分析
            data_description = f"""
データセット概要:
- 訓練データ: {train_data.shape[0]:,}行, {train_data.shape[1]}列
- テストデータ: {test_data.shape[0]:,}行, {test_data.shape[1]}列
- 提出形式: {sample_submission.shape[0]:,}行, {sample_submission.shape[1]}列

訓練データ列:
{', '.join(train_data.columns.tolist())}

テストデータ列:
{', '.join(test_data.columns.tolist())}

提出データ列:
{', '.join(sample_submission.columns.tolist())}
"""
            
            dataset = CompetitionDataset(
                competition_name=competition_name,
                train_data=train_data,
                test_data=test_data,
                sample_submission=sample_submission,
                data_description=data_description,
                downloaded_at=datetime.now(timezone.utc)
            )
            
            self.logger.info(f"✅ データダウンロード完了: {competition_name}")
            self.logger.info(f"  - 訓練: {train_data.shape}, テスト: {test_data.shape}")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"データダウンロードエラー: {e}")
            return None
    
    async def submit_predictions(
        self,
        competition_name: str,
        predictions_df: pd.DataFrame,
        description: str = "Claude Code Submission"
    ) -> Optional[SubmissionResult]:
        """予測結果提出"""
        
        if not self.authenticated:
            self.logger.error("Kaggle API未認証")
            return None
        
        try:
            self.logger.info(f"提出開始: {competition_name}")
            
            # 提出ファイル準備
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            submission_file = self.submissions_dir / f"{competition_name}_{timestamp}.csv"
            
            # CSV保存
            predictions_df.to_csv(submission_file, index=False)
            
            # 提出実行
            self.api.competition_submit(
                file_name=str(submission_file),
                message=f"{description} - {timestamp}",
                competition=competition_name
            )
            
            # 提出履歴取得（最新提出のIDを取得）
            await asyncio.sleep(2)  # API制限対応
            submissions = self.api.competition_submissions(competition_name)
            latest_submission = submissions[0] if submissions else None
            
            if latest_submission:
                result = SubmissionResult(
                    submission_id=str(latest_submission.get('ref', 'unknown')),
                    status=latest_submission.get('status', 'unknown'),
                    public_score=latest_submission.get('publicScore'),
                    private_score=latest_submission.get('privateScore'),
                    submission_time=datetime.now(timezone.utc),
                    file_name=submission_file.name,
                    description=description
                )
                
                self.logger.info(f"✅ 提出完了: {result.submission_id}")
                if result.public_score:
                    self.logger.info(f"  - パブリックスコア: {result.public_score}")
                
                return result
            else:
                self.logger.warning("提出ID取得失敗")
                return None
                
        except Exception as e:
            self.logger.error(f"提出エラー: {e}")
            return None
    
    async def get_leaderboard_position(self, competition_name: str) -> Optional[Dict[str, Any]]:
        """リーダーボード順位取得"""
        
        if not self.authenticated:
            return None
        
        try:
            # 自分の提出履歴取得
            submissions = self.api.competition_submissions(competition_name)
            if not submissions:
                return None
            
            # 最高スコア提出を特定
            best_submission = max(
                [s for s in submissions if s.get('publicScore') is not None],
                key=lambda x: x.get('publicScore', 0),
                default=None
            )
            
            if not best_submission:
                return None
            
            # リーダーボード取得（概算）
            leaderboard = self.api.competition_leaderboard_view(competition_name)
            total_teams = leaderboard.get('totalTeams', 1000)
            
            # 自分のスコアに基づく推定順位
            best_score = best_submission.get('publicScore', 0)
            estimated_rank = int(total_teams * (1 - best_score))  # 簡易推定
            
            return {
                "competition_name": competition_name,
                "current_rank": estimated_rank,
                "total_teams": total_teams,
                "best_score": best_score,
                "percentile": (1 - estimated_rank / total_teams) * 100,
                "medal_zone": self._calculate_medal_zone(estimated_rank, total_teams),
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"リーダーボード取得エラー: {e}")
            return None
    
    def _calculate_medal_zone(self, rank: int, total_teams: int) -> str:
        """メダルゾーン判定"""
        
        if rank <= max(1, int(total_teams * 0.01)):  # 上位1%
            return "gold"
        elif rank <= max(1, int(total_teams * 0.05)):  # 上位5%
            return "silver"
        elif rank <= max(1, int(total_teams * 0.10)):  # 上位10%
            return "bronze"
        else:
            return "no_medal"
    
    async def monitor_submission_scoring(
        self,
        competition_name: str,
        submission_id: str,
        max_wait_minutes: int = 30
    ) -> Optional[SubmissionResult]:
        """提出スコアリング監視"""
        
        if not self.authenticated:
            return None
        
        try:
            self.logger.info(f"スコアリング監視開始: {submission_id}")
            
            start_time = datetime.now()
            max_wait = timedelta(minutes=max_wait_minutes)
            
            while datetime.now() - start_time < max_wait:
                # 提出履歴取得
                submissions = self.api.competition_submissions(competition_name)
                target_submission = None
                
                for sub in submissions:
                    if str(sub.get('ref', '')) == submission_id:
                        target_submission = sub
                        break
                
                if target_submission:
                    status = target_submission.get('status', 'unknown')
                    public_score = target_submission.get('publicScore')
                    
                    self.logger.info(f"提出状態: {status}, スコア: {public_score}")
                    
                    if status == 'complete' and public_score is not None:
                        result = SubmissionResult(
                            submission_id=submission_id,
                            status=status,
                            public_score=public_score,
                            private_score=target_submission.get('privateScore'),
                            submission_time=datetime.now(timezone.utc),
                            file_name="monitored_submission",
                            description="Score monitoring"
                        )
                        
                        self.logger.info(f"✅ スコアリング完了: {public_score}")
                        return result
                    
                    elif status in ['error', 'cancelled']:
                        self.logger.error(f"提出エラー: {status}")
                        return None
                
                # 30秒待機
                await asyncio.sleep(30)
            
            self.logger.warning("スコアリング監視タイムアウト")
            return None
            
        except Exception as e:
            self.logger.error(f"スコアリング監視エラー: {e}")
            return None
    
    async def get_competition_info(self, competition_name: str) -> Optional[Dict[str, Any]]:
        """コンペ情報取得"""
        
        if not self.authenticated:
            return None
        
        try:
            # コンペ情報取得
            competition = self.api.competition_view(competition_name)
            
            return {
                "id": competition.get('id'),
                "title": competition.get('title'),
                "description": competition.get('description', '')[:500],
                "url": competition.get('url'),
                "deadline": competition.get('deadline'),
                "category": competition.get('category'),
                "reward": competition.get('reward'),
                "teamCount": competition.get('teamCount', 0),
                "userHasEntered": competition.get('userHasEntered', False),
                "evaluationMetric": competition.get('evaluationMetric'),
                "submissionsPerDay": competition.get('submissionsPerDay', 5)
            }
            
        except Exception as e:
            self.logger.error(f"コンペ情報取得エラー: {e}")
            return None
    
    async def get_submission_history(self, competition_name: str) -> List[Dict[str, Any]]:
        """提出履歴取得"""
        
        if not self.authenticated:
            return []
        
        try:
            submissions = self.api.competition_submissions(competition_name)
            
            return [
                {
                    "id": sub.get('ref'),
                    "fileName": sub.get('fileName'),
                    "date": sub.get('date'),
                    "status": sub.get('status'),
                    "publicScore": sub.get('publicScore'),
                    "privateScore": sub.get('privateScore'),
                    "description": sub.get('description', '')
                }
                for sub in submissions[:10]  # 最新10件
            ]
            
        except Exception as e:
            self.logger.error(f"提出履歴取得エラー: {e}")
            return []
    
    def is_available(self) -> bool:
        """API利用可能性確認"""
        return KAGGLE_AVAILABLE and self.authenticated
    
    def get_api_status(self) -> Dict[str, Any]:
        """API状態取得"""
        
        return {
            "kaggle_library_available": KAGGLE_AVAILABLE,
            "authenticated": self.authenticated,
            "config_file_exists": Path(self.kaggle_config_path).exists(),
            "data_directory": str(self.data_dir),
            "submissions_directory": str(self.submissions_dir)
        }