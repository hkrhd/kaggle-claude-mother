"""
コンペティションスキャナー

Kaggle APIを使用したアクティブコンペの自動検出・情報収集。
動的コンペ管理システムとの連携による最適化対象の抽出。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import os

import requests
try:
    import pandas as pd
except ImportError:
    # テスト環境用モック実装
    class MockPandas:
        class DataFrame:
            def __init__(self, data=None):
                self.data = data or []
            def to_dict(self, orient='records'):
                return []
            def iterrows(self):
                return iter([])
            def __len__(self):
                return 0
            def empty(self):
                return True
        @staticmethod
        def read_csv(*args, **kwargs):
            return MockPandas.DataFrame()
    pd = MockPandas()

from ..models.competition import (
    CompetitionInfo, CompetitionType, PrizeType, CompetitionPhase,
    calculate_days_remaining, calculate_competition_phase,
    extract_skill_requirements
)


class CompetitionScanner:
    """コンペティションスキャナー"""
    
    def __init__(self, kaggle_credentials_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Kaggle API設定
        self.kaggle_credentials_path = kaggle_credentials_path or os.path.expanduser("~/.kaggle/kaggle.json")
        self.api_base_url = "https://www.kaggle.com/api/v1"
        
        # スキャン設定
        self.scan_categories = [
            "tabular", "computer-vision", "nlp", "time-series",
            "feature-engineering", "getting-started"
        ]
        
        # フィルタリング条件
        self.min_days_remaining = 7      # 最小残り日数
        self.max_participants = 10000    # 最大参加者数
        self.min_participants = 10       # 最小参加者数
        self.max_competitions_scan = 50  # 一度にスキャンする最大数
        
        # キャッシュ設定
        self.cache_duration_hours = 6
        self.competition_cache = {}
        self.last_scan_time = None
    
    async def scan_active_competitions(
        self,
        force_refresh: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[CompetitionInfo]:
        """アクティブコンペスキャン"""
        
        self.logger.info("アクティブコンペスキャン開始")
        
        try:
            # キャッシュチェック
            if not force_refresh and self._is_cache_valid():
                self.logger.info("キャッシュからコンペ情報取得")
                return list(self.competition_cache.values())
            
            # Kaggle API認証
            if not await self._authenticate_kaggle_api():
                self.logger.error("Kaggle API認証失敗")
                return []
            
            # コンペリスト取得
            raw_competitions = await self._fetch_competitions_from_api()
            
            # 情報の詳細化・フィルタリング
            filtered_competitions = await self._process_and_filter_competitions(
                raw_competitions, filters
            )
            
            # キャッシュ更新
            self.competition_cache = {
                comp.competition_id: comp for comp in filtered_competitions
            }
            self.last_scan_time = datetime.utcnow()
            
            self.logger.info(f"コンペスキャン完了: {len(filtered_competitions)}件取得")
            return filtered_competitions
            
        except Exception as e:
            self.logger.error(f"コンペスキャン失敗: {e}")
            return []
    
    async def get_competition_details(
        self,
        competition_id: str,
        force_refresh: bool = False
    ) -> Optional[CompetitionInfo]:
        """特定コンペの詳細情報取得"""
        
        try:
            # キャッシュチェック
            if not force_refresh and competition_id in self.competition_cache:
                return self.competition_cache[competition_id]
            
            # API認証
            if not await self._authenticate_kaggle_api():
                return None
            
            # 詳細情報取得
            competition_data = await self._fetch_competition_details(competition_id)
            if not competition_data:
                return None
            
            # CompetitionInfo作成
            competition_info = await self._create_competition_info(competition_data)
            
            # キャッシュ更新
            self.competition_cache[competition_id] = competition_info
            
            return competition_info
            
        except Exception as e:
            self.logger.error(f"コンペ詳細取得失敗 ({competition_id}): {e}")
            return None
    
    async def scan_competition_statistics(
        self,
        competition_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """コンペ統計情報収集"""
        
        statistics = {}
        
        for comp_id in competition_ids:
            try:
                stats = await self._collect_competition_statistics(comp_id)
                statistics[comp_id] = stats
            except Exception as e:
                self.logger.warning(f"統計情報収集失敗 ({comp_id}): {e}")
                statistics[comp_id] = {}
        
        return statistics
    
    async def find_similar_competitions(
        self,
        reference_competition: CompetitionInfo,
        min_similarity: float = 0.7
    ) -> List[CompetitionInfo]:
        """類似コンペ検索"""
        
        try:
            # 全アクティブコンペ取得
            all_competitions = await self.scan_active_competitions()
            
            similar_competitions = []
            
            for comp in all_competitions:
                if comp.competition_id == reference_competition.competition_id:
                    continue
                
                # 類似度計算
                similarity = await self._calculate_competition_similarity(
                    reference_competition, comp
                )
                
                if similarity >= min_similarity:
                    similar_competitions.append((comp, similarity))
            
            # 類似度順でソート
            similar_competitions.sort(key=lambda x: x[1], reverse=True)
            
            return [comp for comp, _ in similar_competitions]
            
        except Exception as e:
            self.logger.error(f"類似コンペ検索失敗: {e}")
            return []
    
    def _is_cache_valid(self) -> bool:
        """キャッシュ有効性チェック"""
        if not self.last_scan_time or not self.competition_cache:
            return False
        
        cache_age = datetime.utcnow() - self.last_scan_time
        return cache_age.total_seconds() < self.cache_duration_hours * 3600
    
    async def _authenticate_kaggle_api(self) -> bool:
        """Kaggle API認証"""
        try:
            # 認証ファイル存在チェック
            if not os.path.exists(self.kaggle_credentials_path):
                self.logger.error(f"Kaggle認証ファイルが見つかりません: {self.kaggle_credentials_path}")
                return False
            
            # 認証情報読み込み
            with open(self.kaggle_credentials_path, 'r') as f:
                credentials = json.load(f)
            
            self.kaggle_username = credentials.get("username")
            self.kaggle_key = credentials.get("key")
            
            if not self.kaggle_username or not self.kaggle_key:
                self.logger.error("Kaggle認証情報が不完全です")
                return False
            
            # 簡単な認証テスト
            test_url = f"{self.api_base_url}/competitions/list"
            response = requests.get(
                test_url,
                auth=(self.kaggle_username, self.kaggle_key),
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Kaggle API認証成功")
                return True
            else:
                self.logger.error(f"Kaggle API認証失敗: {response.status_code}")
                return False
            
        except Exception as e:
            self.logger.error(f"Kaggle API認証エラー: {e}")
            return False
    
    async def _fetch_competitions_from_api(self) -> List[Dict[str, Any]]:
        """API からコンペリスト取得"""
        
        try:
            # アクティブコンペ取得
            competitions_url = f"{self.api_base_url}/competitions/list"
            response = requests.get(
                competitions_url,
                auth=(self.kaggle_username, self.kaggle_key),
                params={
                    "group": "active",
                    "sortBy": "deadline",
                    "page": 1
                },
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error(f"コンペリスト取得失敗: {response.status_code}")
                return []
            
            competitions_data = response.json()
            
            # 制限数まで取得
            limited_competitions = competitions_data[:self.max_competitions_scan]
            
            self.logger.info(f"APIから{len(limited_competitions)}件のコンペ取得")
            return limited_competitions
            
        except Exception as e:
            self.logger.error(f"コンペAPI取得エラー: {e}")
            return []
    
    async def _fetch_competition_details(self, competition_id: str) -> Optional[Dict[str, Any]]:
        """特定コンペの詳細情報取得"""
        
        try:
            details_url = f"{self.api_base_url}/competitions/{competition_id}"
            response = requests.get(
                details_url,
                auth=(self.kaggle_username, self.kaggle_key),
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.warning(f"コンペ詳細取得失敗 ({competition_id}): {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"コンペ詳細取得エラー ({competition_id}): {e}")
            return None
    
    async def _process_and_filter_competitions(
        self,
        raw_competitions: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]]
    ) -> List[CompetitionInfo]:
        """コンペ情報処理・フィルタリング"""
        
        processed_competitions = []
        
        for comp_data in raw_competitions:
            try:
                # CompetitionInfo作成
                competition_info = await self._create_competition_info(comp_data)
                
                # 基本フィルタリング
                if not await self._passes_basic_filters(competition_info):
                    continue
                
                # 追加フィルタリング
                if filters and not await self._passes_custom_filters(competition_info, filters):
                    continue
                
                processed_competitions.append(competition_info)
                
            except Exception as e:
                self.logger.warning(f"コンペ処理失敗: {e}")
                continue
        
        return processed_competitions
    
    async def _create_competition_info(self, comp_data: Dict[str, Any]) -> CompetitionInfo:
        """API データから CompetitionInfo 作成"""
        
        # 基本情報
        competition_id = comp_data.get("ref", "")
        title = comp_data.get("title", "")
        url = f"https://www.kaggle.com/c/{competition_id}"
        
        # 参加・賞金情報
        total_teams = comp_data.get("totalTeams", 0)
        total_prize = float(comp_data.get("totalPrize", 0))
        reward_type = comp_data.get("rewardType", "")
        
        # 賞金種別判定
        if reward_type.lower() in ["money", "monetary", "cash"]:
            prize_type = PrizeType.MONETARY
        elif reward_type.lower() in ["points", "ranking"]:
            prize_type = PrizeType.POINTS
        elif reward_type.lower() in ["knowledge", "learning"]:
            prize_type = PrizeType.KNOWLEDGE
        else:
            prize_type = PrizeType.MONETARY  # デフォルト
        
        # 時間情報
        deadline_str = comp_data.get("deadline")
        deadline = None
        days_remaining = 0
        if deadline_str:
            try:
                deadline = datetime.fromisoformat(deadline_str.replace('Z', '+00:00'))
                days_remaining = calculate_days_remaining(deadline)
            except:
                pass
        
        # コンペ種別判定
        categories = comp_data.get("categories", [])
        tags = comp_data.get("tags", [])
        competition_type = await self._determine_competition_type(categories, tags, title)
        
        # データ特性推定
        data_characteristics = await self._estimate_data_characteristics(
            competition_type, comp_data
        )
        
        # スキル要件抽出
        skill_requirements = extract_skill_requirements(
            competition_type, comp_data.get("evaluationMetric", ""), tags
        )
        
        return CompetitionInfo(
            competition_id=competition_id,
            title=title,
            url=url,
            total_teams=total_teams,
            participant_count=total_teams,  # 同値として扱う
            total_prize=total_prize,
            prize_type=prize_type,
            deadline=deadline,
            days_remaining=days_remaining,
            competition_type=competition_type,
            evaluation_metric=comp_data.get("evaluationMetric", ""),
            tags=tags,
            skill_requirements=skill_requirements,
            data_size_gb=data_characteristics.get("size_gb", 1.0),
            feature_count=data_characteristics.get("feature_count", 50),
            external_data_allowed=comp_data.get("enabledDataTypes", {}).get("externalData", True),
            internet_access_allowed=comp_data.get("enabledDataTypes", {}).get("internetAccess", False),
            gpu_quota_enabled=comp_data.get("enabledDataTypes", {}).get("gpuQuotaTime", True),
            last_updated=datetime.utcnow(),
            data_collection_status="completed"
        )
    
    async def _determine_competition_type(
        self,
        categories: List[str],
        tags: List[str],
        title: str
    ) -> CompetitionType:
        """コンペ種別判定"""
        
        # カテゴリベースの判定
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
        
        # タグベースの判定
        tag_keywords = {
            CompetitionType.COMPUTER_VISION: ["image", "vision", "cnn", "computer vision"],
            CompetitionType.NLP: ["nlp", "text", "language", "bert", "transformer"],
            CompetitionType.TIME_SERIES: ["time series", "forecasting", "temporal"],
            CompetitionType.AUDIO: ["audio", "sound", "speech"],
            CompetitionType.GRAPH: ["graph", "network", "node"],
            CompetitionType.REINFORCEMENT_LEARNING: ["reinforcement", "rl", "agent"],
            CompetitionType.CODE_COMPETITION: ["code", "algorithm", "programming"]
        }
        
        all_text = " ".join(tags + [title]).lower()
        
        for comp_type, keywords in tag_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                return comp_type
        
        # デフォルト: テーブルデータ
        return CompetitionType.TABULAR
    
    async def _estimate_data_characteristics(
        self,
        competition_type: CompetitionType,
        comp_data: Dict[str, Any]
    ) -> Dict[str, Union[float, int]]:
        """データ特性推定"""
        
        # コンペ種別による推定
        type_estimates = {
            CompetitionType.TABULAR: {"size_gb": 0.5, "feature_count": 50},
            CompetitionType.COMPUTER_VISION: {"size_gb": 5.0, "feature_count": 20},
            CompetitionType.NLP: {"size_gb": 2.0, "feature_count": 30},
            CompetitionType.TIME_SERIES: {"size_gb": 1.0, "feature_count": 25},
            CompetitionType.AUDIO: {"size_gb": 8.0, "feature_count": 15},
            CompetitionType.GRAPH: {"size_gb": 1.5, "feature_count": 35},
            CompetitionType.MULTI_MODAL: {"size_gb": 10.0, "feature_count": 40},
            CompetitionType.REINFORCEMENT_LEARNING: {"size_gb": 0.1, "feature_count": 10}
        }
        
        return type_estimates.get(competition_type, {"size_gb": 1.0, "feature_count": 30})
    
    async def _passes_basic_filters(self, competition_info: CompetitionInfo) -> bool:
        """基本フィルタリング"""
        
        # 残り日数チェック
        if competition_info.days_remaining < self.min_days_remaining:
            return False
        
        # 参加者数チェック
        if (competition_info.participant_count < self.min_participants or
            competition_info.participant_count > self.max_participants):
            return False
        
        # 基本的な情報完全性チェック
        if not competition_info.competition_id or not competition_info.title:
            return False
        
        return True
    
    async def _passes_custom_filters(
        self,
        competition_info: CompetitionInfo,
        filters: Dict[str, Any]
    ) -> bool:
        """カスタムフィルタリング"""
        
        # 種別フィルタ
        if "competition_types" in filters:
            allowed_types = filters["competition_types"]
            if competition_info.competition_type not in allowed_types:
                return False
        
        # 賞金フィルタ
        if "min_prize" in filters:
            if competition_info.total_prize < filters["min_prize"]:
                return False
        
        if "max_prize" in filters:
            if competition_info.total_prize > filters["max_prize"]:
                return False
        
        # 参加者数フィルタ
        if "min_participants" in filters:
            if competition_info.participant_count < filters["min_participants"]:
                return False
        
        if "max_participants" in filters:
            if competition_info.participant_count > filters["max_participants"]:
                return False
        
        return True
    
    async def _collect_competition_statistics(self, competition_id: str) -> Dict[str, Any]:
        """コンペ統計情報収集"""
        
        try:
            # リーダーボード情報取得
            leaderboard_url = f"{self.api_base_url}/competitions/{competition_id}/leaderboard"
            response = requests.get(
                leaderboard_url,
                auth=(self.kaggle_username, self.kaggle_key),
                timeout=30
            )
            
            stats = {
                "leaderboard_entries": 0,
                "score_range": {"min": 0.0, "max": 1.0},
                "recent_activity": "unknown"
            }
            
            if response.status_code == 200:
                leaderboard_data = response.json()
                stats["leaderboard_entries"] = len(leaderboard_data.get("submissions", []))
                
                # スコア範囲計算
                scores = [s.get("score", 0) for s in leaderboard_data.get("submissions", [])]
                if scores:
                    stats["score_range"] = {"min": min(scores), "max": max(scores)}
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"統計情報収集エラー ({competition_id}): {e}")
            return {}
    
    async def _calculate_competition_similarity(
        self,
        comp1: CompetitionInfo,
        comp2: CompetitionInfo
    ) -> float:
        """コンペ類似度計算"""
        
        similarity_score = 0.0
        
        # 種別類似度 (40%)
        if comp1.competition_type == comp2.competition_type:
            similarity_score += 0.4
        
        # 参加者数類似度 (20%)
        if comp1.participant_count > 0 and comp2.participant_count > 0:
            participant_ratio = min(comp1.participant_count, comp2.participant_count) / max(comp1.participant_count, comp2.participant_count)
            similarity_score += 0.2 * participant_ratio
        
        # 賞金類似度 (20%)
        if comp1.total_prize > 0 and comp2.total_prize > 0:
            prize_ratio = min(comp1.total_prize, comp2.total_prize) / max(comp1.total_prize, comp2.total_prize)
            similarity_score += 0.2 * prize_ratio
        
        # タグ類似度 (20%)
        tags1 = set(comp1.tags)
        tags2 = set(comp2.tags)
        if tags1 or tags2:
            tag_intersection = len(tags1 & tags2)
            tag_union = len(tags1 | tags2)
            if tag_union > 0:
                tag_similarity = tag_intersection / tag_union
                similarity_score += 0.2 * tag_similarity
        
        return similarity_score