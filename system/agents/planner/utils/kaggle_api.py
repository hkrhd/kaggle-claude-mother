"""
Kaggle API クライアント

プランニングエージェント用のKaggle API操作・データ収集機能。
コンペ情報取得・参加状況管理・リーダーボード監視の統合クライアント。
"""

import asyncio
import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
try:
    import aiohttp
    import requests
except ImportError:
    # テスト環境用モック実装
    class MockAiohttp:
        class ClientSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, url, **kwargs):
                return MockResponse()
            async def post(self, url, **kwargs):
                return MockResponse()
    
    class MockResponse:
        async def json(self):
            return {"competitions": [], "datasets": []}
        async def text(self):
            return "{}"
        @property
        def status(self):
            return 200
    
    class MockRequests:
        @staticmethod
        def get(url, **kwargs):
            return MockResponse()
        @staticmethod
        def post(url, **kwargs):
            return MockResponse()
    
    aiohttp = MockAiohttp()
    requests = MockRequests()
from pathlib import Path

from ..models.competition import CompetitionInfo, CompetitionType, PrizeType


class KaggleApiClient:
    """Kaggle API クライアント"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # 認証設定
        self.credentials_path = credentials_path or os.path.expanduser("~/.kaggle/kaggle.json")
        self.username = None
        self.api_key = None
        
        # API設定
        self.base_url = "https://www.kaggle.com/api/v1"
        self.timeout = 30
        self.max_retries = 3
        
        # キャッシュ設定
        self.cache_duration = 3600  # 1時間
        self.competition_cache = {}
        
        # レート制限
        self.rate_limit_delay = 1.0  # 1秒間隔
        self.last_request_time = 0.0
    
    async def initialize(self) -> bool:
        """クライアント初期化"""
        
        try:
            # 認証情報読み込み
            if not await self._load_credentials():
                return False
            
            # 接続テスト
            if not await self._test_connection():
                return False
            
            self.logger.info("Kaggle API クライアント初期化完了")
            return True
            
        except Exception as e:
            self.logger.error(f"Kaggle API クライアント初期化失敗: {e}")
            return False
    
    async def get_active_competitions(
        self,
        category: Optional[str] = None,
        sort_by: str = "deadline",
        search: Optional[str] = None
    ) -> List[CompetitionInfo]:
        """アクティブコンペリスト取得"""
        
        self.logger.info("アクティブコンペリスト取得開始")
        
        try:
            # パラメータ設定
            params = {
                "group": "active",
                "sortBy": sort_by,
                "page": 1
            }
            
            if category:
                params["category"] = category
            if search:
                params["search"] = search
            
            # API リクエスト
            response_data = await self._make_request("GET", "/competitions/list", params=params)
            
            if not response_data:
                return []
            
            # CompetitionInfo オブジェクトに変換
            competitions = []
            for comp_data in response_data:
                competition_info = await self._convert_to_competition_info(comp_data)
                if competition_info:
                    competitions.append(competition_info)
            
            self.logger.info(f"アクティブコンペリスト取得完了: {len(competitions)}件")
            return competitions
            
        except Exception as e:
            self.logger.error(f"アクティブコンペリスト取得失敗: {e}")
            return []
    
    async def get_competition_details(
        self,
        competition_id: str,
        include_leaderboard: bool = False
    ) -> Optional[CompetitionInfo]:
        """コンペ詳細情報取得"""
        
        self.logger.info(f"コンペ詳細情報取得開始: {competition_id}")
        
        try:
            # キャッシュチェック
            cache_key = f"{competition_id}_details"
            if cache_key in self.competition_cache:
                cached_data, cache_time = self.competition_cache[cache_key]
                if datetime.utcnow().timestamp() - cache_time < self.cache_duration:
                    self.logger.info(f"キャッシュからコンペ詳細取得: {competition_id}")
                    return cached_data
            
            # コンペ基本情報取得
            comp_data = await self._make_request("GET", f"/competitions/{competition_id}")
            if not comp_data:
                return None
            
            # 詳細情報統合
            competition_info = await self._convert_to_competition_info(comp_data, detailed=True)
            
            # リーダーボード情報追加
            if include_leaderboard and competition_info:
                leaderboard_data = await self._get_competition_leaderboard(competition_id)
                if leaderboard_data:
                    competition_info = await self._merge_leaderboard_data(
                        competition_info, leaderboard_data
                    )
            
            # キャッシュ保存
            if competition_info:
                self.competition_cache[cache_key] = (competition_info, datetime.utcnow().timestamp())
            
            self.logger.info(f"コンペ詳細情報取得完了: {competition_id}")
            return competition_info
            
        except Exception as e:
            self.logger.error(f"コンペ詳細情報取得失敗 ({competition_id}): {e}")
            return None
    
    async def get_user_competitions(
        self,
        username: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """ユーザー参加コンペ一覧取得"""
        
        target_username = username or self.username
        self.logger.info(f"ユーザー参加コンペ取得開始: {target_username}")
        
        try:
            response_data = await self._make_request(
                "GET", 
                f"/users/{target_username}/competitions"
            )
            
            if not response_data:
                return []
            
            self.logger.info(f"ユーザー参加コンペ取得完了: {len(response_data)}件")
            return response_data
            
        except Exception as e:
            self.logger.error(f"ユーザー参加コンペ取得失敗: {e}")
            return []
    
    async def join_competition(self, competition_id: str) -> Dict[str, Any]:
        """コンペ参加"""
        
        self.logger.info(f"コンペ参加開始: {competition_id}")
        
        try:
            # 参加API呼び出し
            response_data = await self._make_request(
                "POST",
                f"/competitions/{competition_id}/join"
            )
            
            if response_data:
                self.logger.info(f"コンペ参加成功: {competition_id}")
                return {"success": True, "data": response_data}
            else:
                return {"success": False, "error": "参加リクエスト失敗"}
                
        except Exception as e:
            self.logger.error(f"コンペ参加失敗 ({competition_id}): {e}")
            return {"success": False, "error": str(e)}
    
    async def get_competition_data_info(
        self,
        competition_id: str
    ) -> Dict[str, Any]:
        """コンペデータ情報取得"""
        
        try:
            response_data = await self._make_request(
                "GET",
                f"/competitions/{competition_id}/data/list"
            )
            
            if not response_data:
                return {}
            
            # データファイル情報統合
            data_info = {
                "total_size_bytes": sum(file.get("totalBytes", 0) for file in response_data),
                "file_count": len(response_data),
                "files": [
                    {
                        "name": file.get("name", ""),
                        "size_bytes": file.get("totalBytes", 0),
                        "description": file.get("description", "")
                    }
                    for file in response_data
                ]
            }
            
            return data_info
            
        except Exception as e:
            self.logger.error(f"コンペデータ情報取得失敗 ({competition_id}): {e}")
            return {}
    
    async def search_competitions(
        self,
        query: str,
        category: Optional[str] = None,
        sort_by: str = "relevance"
    ) -> List[CompetitionInfo]:
        """コンペ検索"""
        
        self.logger.info(f"コンペ検索開始: {query}")
        
        try:
            params = {
                "search": query,
                "sortBy": sort_by,
                "page": 1
            }
            
            if category:
                params["category"] = category
            
            response_data = await self._make_request("GET", "/competitions/list", params=params)
            
            if not response_data:
                return []
            
            # 検索結果変換
            competitions = []
            for comp_data in response_data:
                competition_info = await self._convert_to_competition_info(comp_data)
                if competition_info:
                    competitions.append(competition_info)
            
            self.logger.info(f"コンペ検索完了: {len(competitions)}件")
            return competitions
            
        except Exception as e:
            self.logger.error(f"コンペ検索失敗: {e}")
            return []
    
    async def _load_credentials(self) -> bool:
        """認証情報読み込み"""
        
        try:
            if not os.path.exists(self.credentials_path):
                self.logger.error(f"Kaggle認証ファイル未発見: {self.credentials_path}")
                return False
            
            with open(self.credentials_path, 'r') as f:
                credentials = json.load(f)
            
            self.username = credentials.get("username")
            self.api_key = credentials.get("key")
            
            if not self.username or not self.api_key:
                self.logger.error("Kaggle認証情報不完全")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"認証情報読み込み失敗: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """接続テスト"""
        
        try:
            response_data = await self._make_request("GET", "/competitions/list", params={"page": 1})
            return response_data is not None
            
        except Exception as e:
            self.logger.error(f"接続テスト失敗: {e}")
            return False
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """API リクエスト実行"""
        
        # レート制限チェック
        await self._enforce_rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        auth = (self.username, self.api_key)
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, auth=auth, params=params, timeout=self.timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, auth=auth, params=params, json=data, timeout=self.timeout)
                else:
                    self.logger.error(f"未対応HTTPメソッド: {method}")
                    return None
                
                # レスポンス処理
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 401:
                    self.logger.error("Kaggle API認証失敗")
                    return None
                elif response.status_code == 403:
                    self.logger.error("Kaggle API アクセス権限なし")
                    return None
                elif response.status_code == 429:
                    # レート制限
                    wait_time = 2 ** attempt
                    self.logger.warning(f"レート制限エラー、{wait_time}秒待機")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    self.logger.error(f"API エラー: {response.status_code} - {response.text}")
                    return None
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.logger.error(f"API リクエスト失敗 (最終試行): {e}")
                    return None
                else:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"API リクエスト失敗 (試行{attempt + 1}), {wait_time}秒後リトライ: {e}")
                    await asyncio.sleep(wait_time)
        
        return None
    
    async def _enforce_rate_limit(self):
        """レート制限実行"""
        
        current_time = datetime.utcnow().timestamp()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = datetime.utcnow().timestamp()
    
    async def _convert_to_competition_info(
        self,
        comp_data: Dict[str, Any],
        detailed: bool = False
    ) -> Optional[CompetitionInfo]:
        """API データを CompetitionInfo に変換"""
        
        try:
            # 基本情報
            competition_id = comp_data.get("ref", "")
            title = comp_data.get("title", "")
            url = f"https://www.kaggle.com/c/{competition_id}"
            
            # 参加・賞金情報
            total_teams = comp_data.get("totalTeams", 0)
            participant_count = total_teams  # 同値として扱う
            
            # 賞金情報
            total_prize = float(comp_data.get("totalPrize", 0))
            reward_type = comp_data.get("rewardType", "")
            prize_type = self._parse_prize_type(reward_type)
            
            # 時間情報
            deadline_str = comp_data.get("deadline")
            deadline = None
            days_remaining = 0
            
            if deadline_str:
                try:
                    deadline = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
                    days_remaining = max(0, (deadline - datetime.now(deadline.tzinfo)).days)
                except:
                    pass
            
            # コンペ種別・評価指標
            categories = comp_data.get("categories", [])
            tags = comp_data.get("tags", [])
            competition_type = self._parse_competition_type(categories, tags, title)
            evaluation_metric = comp_data.get("evaluationMetric", "")
            
            # 詳細情報（詳細モード時）
            data_size_gb = 1.0  # デフォルト値
            feature_count = 50  # デフォルト値
            
            if detailed:
                # データ情報取得試行
                data_info = await self.get_competition_data_info(competition_id)
                if data_info:
                    data_size_gb = data_info.get("total_size_bytes", 0) / (1024**3)  # GB変換
            
            # 外部データ・GPU設定
            enabled_data_types = comp_data.get("enabledDataTypes", {})
            external_data_allowed = enabled_data_types.get("externalData", True)
            internet_access_allowed = enabled_data_types.get("internetAccess", False)
            gpu_quota_enabled = enabled_data_types.get("gpuQuotaTime", True)
            
            # CompetitionInfo 作成
            competition_info = CompetitionInfo(
                competition_id=competition_id,
                title=title,
                url=url,
                total_teams=total_teams,
                participant_count=participant_count,
                total_prize=total_prize,
                prize_type=prize_type,
                deadline=deadline,
                days_remaining=days_remaining,
                competition_type=competition_type,
                evaluation_metric=evaluation_metric,
                data_size_gb=data_size_gb,
                feature_count=feature_count,
                tags=tags,
                external_data_allowed=external_data_allowed,
                internet_access_allowed=internet_access_allowed,
                gpu_quota_enabled=gpu_quota_enabled,
                last_updated=datetime.utcnow(),
                data_collection_status="completed"
            )
            
            return competition_info
            
        except Exception as e:
            self.logger.error(f"CompetitionInfo 変換失敗: {e}")
            return None
    
    def _parse_prize_type(self, reward_type: str) -> PrizeType:
        """賞金種別解析"""
        
        reward_lower = reward_type.lower()
        
        if any(keyword in reward_lower for keyword in ["money", "monetary", "cash", "$"]):
            return PrizeType.MONETARY
        elif any(keyword in reward_lower for keyword in ["point", "ranking"]):
            return PrizeType.POINTS
        elif any(keyword in reward_lower for keyword in ["knowledge", "learning", "educational"]):
            return PrizeType.KNOWLEDGE
        elif any(keyword in reward_lower for keyword in ["swag", "merchandise"]):
            return PrizeType.SWAG
        else:
            return PrizeType.MONETARY  # デフォルト
    
    def _parse_competition_type(
        self,
        categories: List[str],
        tags: List[str],
        title: str
    ) -> CompetitionType:
        """コンペ種別解析"""
        
        # カテゴリ優先判定
        category_mapping = {
            "tabular": CompetitionType.TABULAR,
            "computer-vision": CompetitionType.COMPUTER_VISION,
            "nlp": CompetitionType.NLP,
            "time-series": CompetitionType.TIME_SERIES,
            "audio": CompetitionType.AUDIO,
            "graph": CompetitionType.GRAPH,
            "multi-modal": CompetitionType.MULTI_MODAL,
            "reinforcement-learning": CompetitionType.REINFORCEMENT_LEARNING,
            "code": CompetitionType.CODE_COMPETITION,
            "getting-started": CompetitionType.GETTING_STARTED
        }
        
        for category in categories:
            if category.lower() in category_mapping:
                return category_mapping[category.lower()]
        
        # タグとタイトルによる判定
        combined_text = " ".join(tags + [title]).lower()
        
        type_keywords = {
            CompetitionType.COMPUTER_VISION: ["image", "vision", "cnn", "computer vision", "cv"],
            CompetitionType.NLP: ["nlp", "text", "language", "bert", "transformer", "natural language"],
            CompetitionType.TIME_SERIES: ["time series", "forecasting", "temporal", "time-series"],
            CompetitionType.AUDIO: ["audio", "sound", "speech", "music"],
            CompetitionType.GRAPH: ["graph", "network", "node", "edge"],
            CompetitionType.MULTI_MODAL: ["multi-modal", "multimodal", "multi modal"],
            CompetitionType.REINFORCEMENT_LEARNING: ["reinforcement", "rl", "agent", "policy"],
            CompetitionType.CODE_COMPETITION: ["code", "algorithm", "programming", "coding"]
        }
        
        for comp_type, keywords in type_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return comp_type
        
        return CompetitionType.TABULAR  # デフォルト
    
    async def _get_competition_leaderboard(
        self,
        competition_id: str
    ) -> Optional[Dict[str, Any]]:
        """コンペリーダーボード取得"""
        
        try:
            response_data = await self._make_request(
                "GET",
                f"/competitions/{competition_id}/leaderboard",
                params={"page": 1}
            )
            
            return response_data
            
        except Exception as e:
            self.logger.warning(f"リーダーボード取得失敗 ({competition_id}): {e}")
            return None
    
    async def _merge_leaderboard_data(
        self,
        competition_info: CompetitionInfo,
        leaderboard_data: Dict[str, Any]
    ) -> CompetitionInfo:
        """リーダーボード情報統合"""
        
        try:
            # リーダーボード統計
            submissions = leaderboard_data.get("submissions", [])
            
            if submissions:
                # 参加者数更新（より正確な値）
                competition_info.participant_count = len(submissions)
                competition_info.total_teams = len(submissions)
            
            return competition_info
            
        except Exception as e:
            self.logger.warning(f"リーダーボード統合失敗: {e}")
            return competition_info