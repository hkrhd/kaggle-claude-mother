"""
競技選択エージェント

メダル獲得可能性・戦略的価値・リソース効率性を総合評価し、
LLMベースの意思決定により最適な競技ポートフォリオを選択する専門エージェント。
"""

import asyncio
import logging
import uuid
import json
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# LLM推論システム
from openai import AsyncOpenAI
import anthropic

# GitHub Issue連携
from ...issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from ...issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations

# 競技管理システム
from ...dynamic_competition_manager.medal_probability_calculators.medal_probability_calculator import (
    MedalProbabilityCalculator, CompetitionData, CompetitionType, PrizeType
)


class SelectionStrategy(Enum):
    """選択戦略"""
    CONSERVATIVE = "conservative"      # 保守的: 高確率・低リスク重視
    BALANCED = "balanced"             # バランス型: 確率・価値・多様性バランス
    AGGRESSIVE = "aggressive"         # 積極的: 高価値・高リターン追求
    PORTFOLIO_OPTIMAL = "portfolio_optimal"  # ポートフォリオ最適化


class CompetitionCategory(Enum):
    """競技カテゴリ分類"""
    FEATURED_HIGH_PRIZE = "featured_high"      # Featured・高賞金
    FEATURED_STANDARD = "featured_standard"    # Featured・標準賞金
    RESEARCH_INNOVATION = "research_innovation"  # Research・革新的
    RESEARCH_STANDARD = "research_standard"    # Research・標準
    TABULAR_MASTERY = "tabular_mastery"       # テーブル特化・習熟
    COMPUTER_VISION = "computer_vision"       # CV専門
    NLP_LANGUAGE = "nlp_language"            # NLP専門
    TIME_SERIES = "time_series"              # 時系列専門


@dataclass
class CompetitionProfile:
    """競技プロファイル"""
    # 基本情報
    id: str
    name: str
    type: str
    category: CompetitionCategory
    
    # 競技属性
    deadline: datetime
    participants: int
    prize_amount: float
    has_medals: bool
    
    # メダル分析
    medal_probability: float
    bronze_probability: float
    silver_probability: float
    gold_probability: float
    
    # 戦略評価
    strategic_value: float        # 戦略的価値 (0.0-1.0)
    skill_alignment: float        # スキル適合度 (0.0-1.0)
    resource_efficiency: float    # リソース効率 (0.0-1.0)
    portfolio_synergy: float      # ポートフォリオシナジー (0.0-1.0)
    
    # リスク評価
    technical_risk: float         # 技術実装リスク (0.0-1.0)
    competition_risk: float       # 競合レベルリスク (0.0-1.0)
    timeline_risk: float          # 時間制約リスク (0.0-1.0)
    
    # 推定コスト
    estimated_gpu_hours: float
    estimated_development_days: float
    confidence_level: float
    
    # LLM判断結果
    llm_recommendation: str       # "strongly_recommend" | "recommend" | "neutral" | "avoid"
    llm_reasoning: List[str]
    llm_score: float             # LLM総合スコア (0.0-1.0)


@dataclass
class SelectionDecision:
    """選択決定"""
    decision_id: str
    timestamp: datetime
    strategy: SelectionStrategy
    
    # 選択結果
    selected_competitions: List[CompetitionProfile]
    rejected_competitions: List[CompetitionProfile]
    deferred_competitions: List[CompetitionProfile]
    
    # 意思決定根拠
    selection_reasoning: List[str]
    portfolio_optimization: Dict[str, Any]
    resource_allocation: Dict[str, float]
    
    # 評価指標
    expected_medal_count: float
    total_expected_value: float
    portfolio_risk_score: float
    
    # 信頼性
    decision_confidence: float
    llm_consensus_score: float


class CompetitionSelectorAgent:
    """競技選択エージェント - メインクラス"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        
        # エージェント情報
        self.agent_id = f"selector-{uuid.uuid4().hex[:8]}"
        self.agent_version = "1.0.0"
        self.start_time = datetime.utcnow()
        
        # LLM クライアント初期化
        self.openai_client = AsyncOpenAI()  # APIキーは環境変数から自動取得
        self.anthropic_client = anthropic.AsyncAnthropic()  # APIキーは環境変数から自動取得
        
        # GitHub Issue連携
        self.github_wrapper = GitHubApiWrapper(
            access_token=github_token,
            repo_name=repo_name
        )
        self.atomic_operations = AtomicIssueOperations(
            github_client=self.github_wrapper.github,
            repo_name=repo_name
        )
        
        # 分析コンポーネント
        self.medal_calculator = MedalProbabilityCalculator()
        
        # 選択履歴
        self.selection_history: List[SelectionDecision] = []
        self.current_portfolio: List[CompetitionProfile] = []
        
        # 設定
        self.max_concurrent_competitions = 3
        self.min_medal_probability_threshold = 0.15
        self.max_total_gpu_budget = 120.0  # 時間
        self.selection_interval_hours = 12  # 選択実行間隔
        
        # パフォーマンス履歴（学習用）
        self.historical_performance: Dict[str, Any] = {}
        
        self.logger.info(f"CompetitionSelectorAgent初期化完了: {self.agent_id}")
    
    async def evaluate_available_competitions(
        self,
        available_competitions: List[Dict[str, Any]],
        strategy: SelectionStrategy = SelectionStrategy.BALANCED
    ) -> SelectionDecision:
        """利用可能競技評価・選択実行"""
        
        self.logger.info(f"競技選択評価開始: {len(available_competitions)}競技, 戦略: {strategy.value}")
        
        decision_id = f"decision-{uuid.uuid4().hex[:8]}"
        
        try:
            # 1. 競技プロファイル生成
            competition_profiles = []
            for comp_data in available_competitions:
                profile = await self._create_competition_profile(comp_data)
                competition_profiles.append(profile)
            
            # 2. LLM総合評価実行
            llm_evaluated_profiles = await self._perform_llm_evaluation(
                competition_profiles, strategy
            )
            
            # 3. ポートフォリオ最適化
            optimization_result = await self._optimize_competition_portfolio(
                llm_evaluated_profiles, strategy
            )
            
            # 4. 最終選択決定
            decision = await self._make_final_selection_decision(
                decision_id, llm_evaluated_profiles, optimization_result, strategy
            )
            
            # 5. 選択レポート生成・GitHub Issue作成
            await self._create_selection_report(decision)
            
            # 6. 履歴更新
            self.selection_history.append(decision)
            self.current_portfolio = decision.selected_competitions
            
            self.logger.info(f"競技選択完了: {len(decision.selected_competitions)}競技選択")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"競技選択評価エラー: {e}")
            
            # エラー時のフォールバック決定
            fallback_decision = SelectionDecision(
                decision_id=decision_id,
                timestamp=datetime.utcnow(),
                strategy=strategy,
                selected_competitions=[],
                rejected_competitions=[],
                deferred_competitions=[],
                selection_reasoning=[f"評価エラーによりフォールバック: {str(e)}"],
                portfolio_optimization={},
                resource_allocation={},
                expected_medal_count=0.0,
                total_expected_value=0.0,
                portfolio_risk_score=1.0,
                decision_confidence=0.0,
                llm_consensus_score=0.0
            )
            
            return fallback_decision
    
    async def _create_competition_profile(self, comp_data: Dict[str, Any]) -> CompetitionProfile:
        """競技プロファイル作成"""
        
        try:
            # 基本情報抽出
            comp_id = comp_data.get("id", "unknown")
            comp_name = comp_data.get("name", "Unknown Competition")
            comp_type = comp_data.get("type", "tabular")
            
            # カテゴリ分類
            category = await self._classify_competition_category(comp_data)
            
            # メダル確率算出
            medal_probs = await self._calculate_detailed_medal_probabilities(comp_data)
            
            # 戦略評価指標算出
            strategic_metrics = await self._calculate_strategic_metrics(comp_data, comp_type)
            
            # リスク評価
            risk_metrics = await self._assess_competition_risks(comp_data, comp_type)
            
            # リソース推定
            resource_estimates = await self._estimate_resource_requirements(comp_data, comp_type)
            
            profile = CompetitionProfile(
                id=comp_id,
                name=comp_name,
                type=comp_type,
                category=category,
                deadline=datetime.fromisoformat(comp_data.get("deadline", datetime.utcnow().isoformat())),
                participants=comp_data.get("participants", 1000),
                prize_amount=comp_data.get("prize_amount", 0.0),
                has_medals=comp_data.get("awards_medals", True),
                
                medal_probability=medal_probs["overall"],
                bronze_probability=medal_probs["bronze"],
                silver_probability=medal_probs["silver"],
                gold_probability=medal_probs["gold"],
                
                strategic_value=strategic_metrics["strategic_value"],
                skill_alignment=strategic_metrics["skill_alignment"],
                resource_efficiency=strategic_metrics["resource_efficiency"],
                portfolio_synergy=strategic_metrics["portfolio_synergy"],
                
                technical_risk=risk_metrics["technical_risk"],
                competition_risk=risk_metrics["competition_risk"],
                timeline_risk=risk_metrics["timeline_risk"],
                
                estimated_gpu_hours=resource_estimates["gpu_hours"],
                estimated_development_days=resource_estimates["development_days"],
                confidence_level=strategic_metrics["confidence_level"],
                
                # LLM評価は後で実行
                llm_recommendation="neutral",
                llm_reasoning=[],
                llm_score=0.5
            )
            
            return profile
            
        except Exception as e:
            self.logger.error(f"競技プロファイル作成エラー {comp_data.get('name', 'Unknown')}: {e}")
            
            # エラー時のデフォルトプロファイル
            return CompetitionProfile(
                id=comp_data.get("id", "error"),
                name=comp_data.get("name", "Error Competition"),
                type="tabular",
                category=CompetitionCategory.FEATURED_STANDARD,
                deadline=datetime.utcnow() + timedelta(days=30),
                participants=1000,
                prize_amount=0.0,
                has_medals=False,
                medal_probability=0.0,
                bronze_probability=0.0,
                silver_probability=0.0,
                gold_probability=0.0,
                strategic_value=0.0,
                skill_alignment=0.0,
                resource_efficiency=0.0,
                portfolio_synergy=0.0,
                technical_risk=1.0,
                competition_risk=1.0,
                timeline_risk=1.0,
                estimated_gpu_hours=0.0,
                estimated_development_days=0.0,
                confidence_level=0.0,
                llm_recommendation="avoid",
                llm_reasoning=["プロファイル作成エラー"],
                llm_score=0.0
            )
    
    async def _classify_competition_category(self, comp_data: Dict[str, Any]) -> CompetitionCategory:
        """競技カテゴリ分類"""
        
        comp_type = comp_data.get("type", "tabular")
        prize_amount = comp_data.get("prize_amount", 0.0)
        category = comp_data.get("competition_category", "").lower()
        
        # Featured competitions
        if category == "featured":
            if prize_amount >= 50000:
                return CompetitionCategory.FEATURED_HIGH_PRIZE
            else:
                return CompetitionCategory.FEATURED_STANDARD
        
        # Research competitions
        elif category == "research":
            # 革新性判定（簡易版 - 実際にはより詳細な分析が必要）
            name_lower = comp_data.get("name", "").lower()
            if any(keyword in name_lower for keyword in ["novel", "new", "innovative", "breakthrough"]):
                return CompetitionCategory.RESEARCH_INNOVATION
            else:
                return CompetitionCategory.RESEARCH_STANDARD
        
        # Type-based classification
        elif comp_type == "tabular":
            return CompetitionCategory.TABULAR_MASTERY
        elif comp_type == "computer_vision":
            return CompetitionCategory.COMPUTER_VISION
        elif comp_type in ["nlp", "text"]:
            return CompetitionCategory.NLP_LANGUAGE
        elif comp_type == "time_series":
            return CompetitionCategory.TIME_SERIES
        
        else:
            return CompetitionCategory.FEATURED_STANDARD
    
    async def _calculate_detailed_medal_probabilities(self, comp_data: Dict[str, Any]) -> Dict[str, float]:
        """詳細メダル確率算出"""
        
        try:
            # CompetitionDataオブジェクト作成（既存のロジック活用）
            comp_type_mapping = {
                "tabular": CompetitionType.TABULAR,
                "computer_vision": CompetitionType.COMPUTER_VISION,
                "nlp": CompetitionType.NLP,
                "time_series": CompetitionType.TIME_SERIES
            }
            
            comp_type = comp_type_mapping.get(comp_data.get("type", "tabular"), CompetitionType.TABULAR)
            prize_amount = comp_data.get("prize_amount", 0.0)
            
            competition_data = CompetitionData(
                competition_id=comp_data.get("id", "unknown"),
                title=comp_data.get("name", "Unknown"),
                competition_type=comp_type,
                participant_count=comp_data.get("participants", 1000),
                total_prize=prize_amount,
                prize_type=PrizeType.MONETARY if prize_amount > 0 else PrizeType.KNOWLEDGE,
                days_remaining=(datetime.fromisoformat(comp_data.get("deadline", datetime.utcnow().isoformat())).replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days,
                data_characteristics={
                    "dataset_size": "medium",
                    "feature_count": 50 if comp_type == CompetitionType.TABULAR else 1000,
                    "data_quality": "high"
                },
                skill_requirements=["feature_engineering", "ensemble_methods"] if comp_type == CompetitionType.TABULAR else ["deep_learning"],
                leaderboard_competition=min(1.0, comp_data.get("participants", 1000) / 2000)
            )
            
            # 全体メダル確率算出
            result = await self.medal_calculator.calculate_medal_probability(competition_data)
            overall_prob = result.overall_medal_probability
            
            # メダル別確率推定（簡易モデル）
            # 通常、Bronze > Silver > Gold の順で確率が下がる
            bronze_prob = overall_prob * 0.6  # 全体確率の60%
            silver_prob = overall_prob * 0.3   # 全体確率の30%
            gold_prob = overall_prob * 0.1     # 全体確率の10%
            
            return {
                "overall": overall_prob,
                "bronze": bronze_prob,
                "silver": silver_prob,
                "gold": gold_prob
            }
            
        except Exception as e:
            self.logger.error(f"メダル確率算出エラー: {e}")
            return {
                "overall": 0.0,
                "bronze": 0.0,
                "silver": 0.0,
                "gold": 0.0
            }
    
    async def _calculate_strategic_metrics(self, comp_data: Dict[str, Any], comp_type: str) -> Dict[str, float]:
        """戦略評価指標算出"""
        
        try:
            # 戦略的価値算出
            prize_amount = comp_data.get("prize_amount", 0.0)
            participants = comp_data.get("participants", 1000)
            
            # 賞金価値正規化 (0-1)
            prize_value = min(1.0, prize_amount / 100000)  # $100k を基準とした正規化
            
            # 競争密度調整 (参加者が多すぎると価値低下)
            competition_density = max(0.1, 1.0 - (participants - 1000) / 10000)
            
            strategic_value = (prize_value * 0.7 + competition_density * 0.3)
            
            # スキル適合度 (競技タイプ別)
            skill_alignment_map = {
                "tabular": 0.9,      # 最も得意
                "computer_vision": 0.7,
                "nlp": 0.6,
                "time_series": 0.8,
                "audio": 0.5,
                "graph": 0.4
            }
            skill_alignment = skill_alignment_map.get(comp_type, 0.5)
            
            # リソース効率性 (賞金/予想工数比)
            estimated_hours = max(40, participants / 50)  # 参加者数ベースの工数推定
            resource_efficiency = min(1.0, (prize_amount / max(1, estimated_hours)) / 100)
            
            # ポートフォリオシナジー (現在のポートフォリオとの組み合わせ効果)
            current_types = [comp.type for comp in self.current_portfolio]
            type_diversity_bonus = 0.2 if comp_type not in current_types else 0.0
            portfolio_synergy = 0.5 + type_diversity_bonus
            
            # 信頼度 (データの完全性に基づく)
            data_completeness = 1.0 if all(key in comp_data for key in ["name", "type", "deadline", "participants"]) else 0.7
            confidence_level = data_completeness
            
            return {
                "strategic_value": strategic_value,
                "skill_alignment": skill_alignment,
                "resource_efficiency": resource_efficiency,
                "portfolio_synergy": portfolio_synergy,
                "confidence_level": confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"戦略指標算出エラー: {e}")
            return {
                "strategic_value": 0.0,
                "skill_alignment": 0.0,
                "resource_efficiency": 0.0,
                "portfolio_synergy": 0.0,
                "confidence_level": 0.0
            }
    
    async def _assess_competition_risks(self, comp_data: Dict[str, Any], comp_type: str) -> Dict[str, float]:
        """競技リスク評価"""
        
        try:
            participants = comp_data.get("participants", 1000)
            deadline = datetime.fromisoformat(comp_data.get("deadline", datetime.utcnow().isoformat()))
            days_remaining = (deadline - datetime.utcnow()).days
            
            # 技術実装リスク
            technical_risk_map = {
                "tabular": 0.2,      # 低リスク
                "computer_vision": 0.5,  # 中リスク
                "nlp": 0.6,          # 高リスク
                "time_series": 0.4,
                "audio": 0.8,        # 非常に高リスク
                "graph": 0.7
            }
            technical_risk = technical_risk_map.get(comp_type, 0.5)
            
            # 競合レベルリスク (参加者数ベース)
            if participants < 500:
                competition_risk = 0.2    # 低競合
            elif participants < 2000:
                competition_risk = 0.4    # 中競合
            elif participants < 5000:
                competition_risk = 0.7    # 高競合
            else:
                competition_risk = 0.9    # 超高競合
            
            # 時間制約リスク
            if days_remaining > 60:
                timeline_risk = 0.1      # 十分な時間
            elif days_remaining > 30:
                timeline_risk = 0.3      # 適度な時間
            elif days_remaining > 14:
                timeline_risk = 0.6      # 時間制約あり
            else:
                timeline_risk = 0.9      # 厳重な時間制約
            
            return {
                "technical_risk": technical_risk,
                "competition_risk": competition_risk,
                "timeline_risk": timeline_risk
            }
            
        except Exception as e:
            self.logger.error(f"リスク評価エラー: {e}")
            return {
                "technical_risk": 0.5,
                "competition_risk": 0.5,
                "timeline_risk": 0.5
            }
    
    async def _estimate_resource_requirements(self, comp_data: Dict[str, Any], comp_type: str) -> Dict[str, float]:
        """リソース要件推定"""
        
        try:
            participants = comp_data.get("participants", 1000)
            
            # GPU時間推定 (競技タイプ・参加者数ベース)
            base_gpu_hours = {
                "tabular": 8,
                "computer_vision": 24,
                "nlp": 20,
                "time_series": 12,
                "audio": 30,
                "graph": 16
            }
            
            base_hours = base_gpu_hours.get(comp_type, 8)
            
            # 参加者数による調整 (競争が激しいほどより多くのリソースが必要)
            competition_multiplier = 1.0 + (participants - 1000) / 5000
            
            estimated_gpu_hours = base_hours * max(1.0, competition_multiplier)
            
            # 開発日数推定
            estimated_development_days = max(7, estimated_gpu_hours / 3)  # 1日3時間の作業想定
            
            return {
                "gpu_hours": estimated_gpu_hours,
                "development_days": estimated_development_days
            }
            
        except Exception as e:
            self.logger.error(f"リソース推定エラー: {e}")
            return {
                "gpu_hours": 10.0,
                "development_days": 7.0
            }
    
    async def _perform_llm_evaluation(
        self,
        profiles: List[CompetitionProfile],
        strategy: SelectionStrategy
    ) -> List[CompetitionProfile]:
        """LLM総合評価実行"""
        
        self.logger.info(f"LLM評価開始: {len(profiles)}競技")
        
        # 並列LLM評価実行
        llm_tasks = []
        for profile in profiles:
            task = self._evaluate_single_competition_with_llm(profile, strategy)
            llm_tasks.append(task)
        
        # 並列実行
        llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
        
        # 結果統合
        evaluated_profiles = []
        for i, (profile, result) in enumerate(zip(profiles, llm_results)):
            if isinstance(result, Exception):
                self.logger.error(f"LLM評価失敗 {profile.name}: {result}")
                # エラー時は元のプロファイルを使用
                evaluated_profiles.append(profile)
            else:
                evaluated_profiles.append(result)
        
        self.logger.info(f"LLM評価完了: {len(evaluated_profiles)}競技")
        return evaluated_profiles
    
    async def _evaluate_single_competition_with_llm(
        self,
        profile: CompetitionProfile,
        strategy: SelectionStrategy
    ) -> CompetitionProfile:
        """単一競技LLM評価"""
        
        try:
            # LLM評価プロンプト生成
            evaluation_prompt = self._create_competition_evaluation_prompt(profile, strategy)
            
            # Claude (Anthropic) で評価実行
            response = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt
                    }
                ]
            )
            
            # 回答解析
            llm_evaluation = self._parse_llm_evaluation_response(response.content[0].text)
            
            # プロファイル更新
            profile.llm_recommendation = llm_evaluation["recommendation"]
            profile.llm_reasoning = llm_evaluation["reasoning"]
            profile.llm_score = llm_evaluation["score"]
            
            return profile
            
        except Exception as e:
            self.logger.error(f"LLM評価エラー {profile.name}: {e}")
            
            # エラー時のデフォルト評価
            profile.llm_recommendation = "neutral"
            profile.llm_reasoning = [f"LLM評価エラー: {str(e)}"]
            profile.llm_score = 0.5
            
            return profile
    
    def _create_competition_evaluation_prompt(
        self,
        profile: CompetitionProfile,
        strategy: SelectionStrategy
    ) -> str:
        """競技評価プロンプト生成"""
        
        # 現在のポートフォリオ情報
        current_portfolio_info = ""
        if self.current_portfolio:
            current_portfolio_info = f"""
現在の競技ポートフォリオ:
{chr(10).join([f"- {comp.name} ({comp.type}, メダル確率: {comp.medal_probability:.1%})" for comp in self.current_portfolio])}
"""
        
        # 戦略別の重視ポイント
        strategy_focus = {
            SelectionStrategy.CONSERVATIVE: "メダル確率・安全性を最重視。リスクを避けて確実な成果を狙う。",
            SelectionStrategy.BALANCED: "メダル確率・価値・多様性のバランスを重視。リスクと収益の最適化。",
            SelectionStrategy.AGGRESSIVE: "高価値・高リターンを最重視。リスクを許容して大きな成果を狙う。",
            SelectionStrategy.PORTFOLIO_OPTIMAL: "ポートフォリオ全体の最適化を重視。分散効果・シナジーを考慮。"
        }
        
        current_strategy_focus = strategy_focus.get(strategy, "バランス重視")
        
        prompt = f"""あなたはKaggle競技選択の専門家です。以下の競技を評価し、参加すべきかを判断してください。

## 選択戦略
{current_strategy_focus}

## 評価対象競技
**競技名**: {profile.name}
**種類**: {profile.type}
**カテゴリ**: {profile.category.value}
**締切**: {profile.deadline.strftime('%Y-%m-%d')} ({(profile.deadline - datetime.utcnow()).days}日後)
**参加者数**: {profile.participants:,}名
**賞金**: ${profile.prize_amount:,.0f}
**メダル対象**: {'有' if profile.has_medals else '無'}

## 分析結果
**メダル確率**: 全体{profile.medal_probability:.1%} (金{profile.gold_probability:.1%}/銀{profile.silver_probability:.1%}/銅{profile.bronze_probability:.1%})
**戦略的価値**: {profile.strategic_value:.2f}
**スキル適合度**: {profile.skill_alignment:.2f}
**リソース効率**: {profile.resource_efficiency:.2f}
**ポートフォリオシナジー**: {profile.portfolio_synergy:.2f}

## リスク評価
**技術リスク**: {profile.technical_risk:.2f}
**競合リスク**: {profile.competition_risk:.2f}
**時間リスク**: {profile.timeline_risk:.2f}

## リソース推定
**GPU時間**: {profile.estimated_gpu_hours:.1f}時間
**開発期間**: {profile.estimated_development_days:.1f}日
**分析信頼度**: {profile.confidence_level:.1%}

{current_portfolio_info}

## システム制約
- 最大同時競技数: {self.max_concurrent_competitions}競技
- 総GPU予算: {self.max_total_gpu_budget}時間
- 最低メダル確率: {self.min_medal_probability_threshold:.1%}

以下のフォーマットで回答してください:

RECOMMENDATION: [strongly_recommend|recommend|neutral|avoid]
SCORE: [0.0-1.0の数値]
REASONING:
- 理由1
- 理由2
- 理由3

戦略的観点から、この競技への参加を推奨しますか？具体的な理由と共に評価してください。"""

        return prompt
    
    def _parse_llm_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """LLM評価回答解析"""
        
        try:
            lines = response_text.strip().split('\n')
            
            recommendation = "neutral"
            score = 0.5
            reasoning = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("RECOMMENDATION:"):
                    recommendation = line.split(":", 1)[1].strip().lower()
                elif line.startswith("SCORE:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(1.0, score))  # 0.0-1.0にクランプ
                    except ValueError:
                        score = 0.5
                elif line.startswith("REASONING:"):
                    current_section = "reasoning"
                elif current_section == "reasoning" and line.startswith("- "):
                    reasoning.append(line[2:])  # "- "を除去
            
            # 推奨レベル正規化
            valid_recommendations = ["strongly_recommend", "recommend", "neutral", "avoid"]
            if recommendation not in valid_recommendations:
                recommendation = "neutral"
            
            return {
                "recommendation": recommendation,
                "score": score,
                "reasoning": reasoning if reasoning else ["LLM評価解析エラー"]
            }
            
        except Exception as e:
            self.logger.error(f"LLM評価解析エラー: {e}")
            return {
                "recommendation": "neutral",
                "score": 0.5,
                "reasoning": [f"評価解析エラー: {str(e)}"]
            }
    
    async def _optimize_competition_portfolio(
        self,
        profiles: List[CompetitionProfile],
        strategy: SelectionStrategy
    ) -> Dict[str, Any]:
        """競技ポートフォリオ最適化"""
        
        self.logger.info("ポートフォリオ最適化実行")
        
        try:
            # 制約条件
            max_competitions = self.max_concurrent_competitions
            max_gpu_budget = self.max_total_gpu_budget
            min_medal_prob = self.min_medal_probability_threshold
            
            # フィルタリング
            eligible_profiles = [
                p for p in profiles 
                if p.medal_probability >= min_medal_prob and p.llm_recommendation in ["strongly_recommend", "recommend"]
            ]
            
            if not eligible_profiles:
                # 基準を満たす競技がない場合、基準を緩和
                eligible_profiles = [p for p in profiles if p.medal_probability >= min_medal_prob * 0.5]
            
            # 戦略別最適化
            if strategy == SelectionStrategy.CONSERVATIVE:
                # 高確率・低リスク優先
                eligible_profiles.sort(
                    key=lambda p: (p.medal_probability * (1 - p.technical_risk) * (1 - p.timeline_risk)), 
                    reverse=True
                )
            
            elif strategy == SelectionStrategy.AGGRESSIVE:
                # 高価値・高リターン優先
                eligible_profiles.sort(
                    key=lambda p: (p.strategic_value * p.llm_score * (p.gold_probability + p.silver_probability * 0.5)),
                    reverse=True
                )
            
            elif strategy == SelectionStrategy.PORTFOLIO_OPTIMAL:
                # ポートフォリオシナジー・分散効果優先
                eligible_profiles.sort(
                    key=lambda p: (p.portfolio_synergy * p.medal_probability * p.skill_alignment),
                    reverse=True
                )
            
            else:  # BALANCED
                # バランススコア算出
                for profile in eligible_profiles:
                    balance_score = (
                        profile.medal_probability * 0.3 +
                        profile.strategic_value * 0.25 +
                        profile.llm_score * 0.2 +
                        profile.skill_alignment * 0.15 +
                        (1 - profile.technical_risk) * 0.1
                    )
                    profile.balance_score = balance_score
                
                eligible_profiles.sort(key=lambda p: p.balance_score, reverse=True)
            
            # リソース制約による選択
            selected_profiles = []
            total_gpu_hours = 0.0
            
            for profile in eligible_profiles:
                if (len(selected_profiles) < max_competitions and 
                    total_gpu_hours + profile.estimated_gpu_hours <= max_gpu_budget):
                    
                    selected_profiles.append(profile)
                    total_gpu_hours += profile.estimated_gpu_hours
            
            # 最適化結果
            optimization_result = {
                "strategy": strategy.value,
                "eligible_count": len(eligible_profiles),
                "selected_count": len(selected_profiles),
                "total_gpu_hours": total_gpu_hours,
                "gpu_utilization": total_gpu_hours / max_gpu_budget,
                "expected_medals": sum(p.medal_probability for p in selected_profiles),
                "total_strategic_value": sum(p.strategic_value for p in selected_profiles),
                "portfolio_diversity": len(set(p.type for p in selected_profiles)),
                "avg_confidence": sum(p.confidence_level for p in selected_profiles) / max(1, len(selected_profiles))
            }
            
            self.logger.info(f"ポートフォリオ最適化完了: {len(selected_profiles)}競技選択")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"ポートフォリオ最適化エラー: {e}")
            return {
                "strategy": strategy.value,
                "eligible_count": 0,
                "selected_count": 0,
                "error": str(e)
            }
    
    async def _make_final_selection_decision(
        self,
        decision_id: str,
        profiles: List[CompetitionProfile],
        optimization_result: Dict[str, Any],
        strategy: SelectionStrategy
    ) -> SelectionDecision:
        """最終選択決定"""
        
        try:
            # 最適化結果から選択済み競技を抽出
            selected_profiles = []
            rejected_profiles = []
            deferred_profiles = []
            
            # 選択ロジック
            min_medal_prob = self.min_medal_probability_threshold
            
            for profile in profiles:
                if (profile.llm_recommendation == "strongly_recommend" or
                    (profile.llm_recommendation == "recommend" and profile.medal_probability >= min_medal_prob)):
                    
                    if len(selected_profiles) < self.max_concurrent_competitions:
                        selected_profiles.append(profile)
                    else:
                        deferred_profiles.append(profile)
                
                elif profile.llm_recommendation == "avoid" or profile.medal_probability < min_medal_prob * 0.5:
                    rejected_profiles.append(profile)
                
                else:
                    deferred_profiles.append(profile)
            
            # 選択理由生成
            selection_reasoning = [
                f"{strategy.value}戦略に基づく選択実行",
                f"{len(selected_profiles)}競技を選択 (最大{self.max_concurrent_competitions})",
                f"総予想メダル数: {sum(p.medal_probability for p in selected_profiles):.2f}",
                f"LLM推奨競技を優先選択"
            ]
            
            if optimization_result.get("error"):
                selection_reasoning.append(f"最適化エラー: {optimization_result['error']}")
            
            # リソース配分
            total_gpu_hours = sum(p.estimated_gpu_hours for p in selected_profiles)
            resource_allocation = {
                "total_gpu_hours": total_gpu_hours,
                "gpu_utilization": total_gpu_hours / self.max_total_gpu_budget,
                "competitions_count": len(selected_profiles),
                "avg_development_days": sum(p.estimated_development_days for p in selected_profiles) / max(1, len(selected_profiles))
            }
            
            # 評価指標算出
            expected_medal_count = sum(p.medal_probability for p in selected_profiles)
            total_expected_value = sum(p.strategic_value * p.medal_probability for p in selected_profiles)
            portfolio_risk_score = sum(p.technical_risk * p.competition_risk for p in selected_profiles) / max(1, len(selected_profiles))
            
            # 信頼性評価
            decision_confidence = sum(p.confidence_level for p in selected_profiles) / max(1, len(selected_profiles))
            llm_consensus_score = sum(1 for p in selected_profiles if p.llm_recommendation in ["strongly_recommend", "recommend"]) / max(1, len(profiles))
            
            decision = SelectionDecision(
                decision_id=decision_id,
                timestamp=datetime.utcnow(),
                strategy=strategy,
                selected_competitions=selected_profiles,
                rejected_competitions=rejected_profiles,
                deferred_competitions=deferred_profiles,
                selection_reasoning=selection_reasoning,
                portfolio_optimization=optimization_result,
                resource_allocation=resource_allocation,
                expected_medal_count=expected_medal_count,
                total_expected_value=total_expected_value,
                portfolio_risk_score=portfolio_risk_score,
                decision_confidence=decision_confidence,
                llm_consensus_score=llm_consensus_score
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"最終選択決定エラー: {e}")
            
            # エラー時の空決定
            return SelectionDecision(
                decision_id=decision_id,
                timestamp=datetime.utcnow(),
                strategy=strategy,
                selected_competitions=[],
                rejected_competitions=[],
                deferred_competitions=profiles,
                selection_reasoning=[f"決定エラー: {str(e)}"],
                portfolio_optimization={},
                resource_allocation={},
                expected_medal_count=0.0,
                total_expected_value=0.0,
                portfolio_risk_score=1.0,
                decision_confidence=0.0,
                llm_consensus_score=0.0
            )
    
    async def _create_selection_report(self, decision: SelectionDecision):
        """選択レポート作成・GitHub Issue投稿"""
        
        try:
            # レポート内容生成
            report_content = self._generate_selection_report_content(decision)
            
            # GitHub Issue作成
            issue_data = await self.atomic_operations.create_issue(
                title=f"🎯 競技選択決定 - {decision.strategy.value} ({len(decision.selected_competitions)}競技)",
                description=report_content,
                labels=["competition-selection", "decision", f"strategy-{decision.strategy.value}"]
            )
            
            self.logger.info(f"選択レポート作成完了: Issue #{issue_data['number']}")
            
        except Exception as e:
            self.logger.error(f"選択レポート作成エラー: {e}")
    
    def _generate_selection_report_content(self, decision: SelectionDecision) -> str:
        """選択レポート内容生成"""
        
        # 戦略アイコン
        strategy_icons = {
            SelectionStrategy.CONSERVATIVE: "🛡️",
            SelectionStrategy.BALANCED: "⚖️",
            SelectionStrategy.AGGRESSIVE: "🚀",
            SelectionStrategy.PORTFOLIO_OPTIMAL: "📊"
        }
        
        strategy_icon = strategy_icons.get(decision.strategy, "🎯")
        
        content = f"""# {strategy_icon} Kaggle競技選択決定レポート

## 📋 決定サマリー
- **決定ID**: `{decision.decision_id}`
- **決定時刻**: {decision.timestamp.isoformat()}
- **選択戦略**: {strategy_icon} {decision.strategy.value}
- **決定信頼度**: {decision.decision_confidence:.1%}
- **LLMコンセンサス**: {decision.llm_consensus_score:.1%}

## 🏆 選択結果
### ✅ 選択競技 ({len(decision.selected_competitions)}競技)
{chr(10).join([
    f"**{i+1}. {comp.name}**"
    f"  - 種類: {comp.type} | メダル確率: {comp.medal_probability:.1%}"
    f"  - LLM推奨: {comp.llm_recommendation} (スコア: {comp.llm_score:.2f})"
    f"  - GPU時間: {comp.estimated_gpu_hours:.1f}h | 開発期間: {comp.estimated_development_days:.1f}日"
    f"  - 締切: {comp.deadline.strftime('%Y-%m-%d')} ({(comp.deadline - datetime.utcnow()).days}日後)"
    for i, comp in enumerate(decision.selected_competitions)
]) if decision.selected_competitions else "なし"}

### ❌ 却下競技 ({len(decision.rejected_competitions)}競技)
{chr(10).join([
    f"- **{comp.name}**: {comp.llm_recommendation} | メダル確率{comp.medal_probability:.1%}"
    for comp in decision.rejected_competitions[:5]  # 上位5個のみ表示
]) if decision.rejected_competitions else "なし"}

### ⏸️ 保留競技 ({len(decision.deferred_competitions)}競技)
{chr(10).join([
    f"- **{comp.name}**: {comp.llm_recommendation} | メダル確率{comp.medal_probability:.1%}"
    for comp in decision.deferred_competitions[:5]  # 上位5個のみ表示
]) if decision.deferred_competitions else "なし"}

## 📊 ポートフォリオ分析
### 期待成果
- **予想メダル数**: {decision.expected_medal_count:.2f}個
- **総戦略的価値**: {decision.total_expected_value:.2f}
- **ポートフォリオリスク**: {decision.portfolio_risk_score:.2f}

### リソース配分
- **総GPU時間**: {decision.resource_allocation.get('total_gpu_hours', 0):.1f}時間 / {self.max_total_gpu_budget}時間
- **GPU利用率**: {decision.resource_allocation.get('gpu_utilization', 0):.1%}
- **平均開発期間**: {decision.resource_allocation.get('avg_development_days', 0):.1f}日

## 🧠 LLM評価詳細
{chr(10).join([
    f"### {comp.name}"
    f"- **推奨**: {comp.llm_recommendation} (スコア: {comp.llm_score:.2f})"
    f"- **理由**: {', '.join(comp.llm_reasoning[:3])}"  # 上位3つの理由
    for comp in decision.selected_competitions
]) if decision.selected_competitions else "選択競技なし"}

## 🎯 選択理由
{chr(10).join([f"- {reason}" for reason in decision.selection_reasoning])}

## 📈 ポートフォリオ最適化結果
```json
{json.dumps(decision.portfolio_optimization, indent=2, default=str)}
```

---
*自動生成レポート | Competition Selector Agent `{self.agent_id}` | {datetime.utcnow().isoformat()}*"""

        return content
    
    async def get_current_portfolio_status(self) -> Dict[str, Any]:
        """現在のポートフォリオ状態取得"""
        
        current_time = datetime.utcnow()
        
        # アクティブ競技統計
        active_competitions = [comp for comp in self.current_portfolio if comp.deadline > current_time]
        expired_competitions = [comp for comp in self.current_portfolio if comp.deadline <= current_time]
        
        if active_competitions:
            total_medal_probability = sum(comp.medal_probability for comp in active_competitions)
            total_gpu_hours = sum(comp.estimated_gpu_hours for comp in active_competitions)
            avg_confidence = sum(comp.confidence_level for comp in active_competitions) / len(active_competitions)
            type_distribution = {}
            for comp in active_competitions:
                type_distribution[comp.type] = type_distribution.get(comp.type, 0) + 1
        else:
            total_medal_probability = total_gpu_hours = avg_confidence = 0.0
            type_distribution = {}
        
        return {
            "agent_id": self.agent_id,
            "portfolio_last_updated": self.selection_history[-1].timestamp.isoformat() if self.selection_history else None,
            "active_competitions_count": len(active_competitions),
            "expired_competitions_count": len(expired_competitions),
            "total_medal_probability": total_medal_probability,
            "total_gpu_hours_allocated": total_gpu_hours,
            "gpu_budget_utilization": total_gpu_hours / self.max_total_gpu_budget,
            "avg_confidence_level": avg_confidence,
            "type_distribution": type_distribution,
            "selection_decisions_made": len(self.selection_history),
            "last_selection_strategy": self.selection_history[-1].strategy.value if self.selection_history else None,
            "active_competitions": [
                {
                    "name": comp.name,
                    "type": comp.type,
                    "medal_probability": comp.medal_probability,
                    "days_remaining": (comp.deadline - current_time).days,
                    "llm_recommendation": comp.llm_recommendation
                }
                for comp in active_competitions
            ]
        }
    
    async def update_competition_performance(self, competition_id: str, actual_result: Dict[str, Any]):
        """競技パフォーマンス更新（学習用）"""
        
        try:
            # 実績データを履歴に記録
            if competition_id not in self.historical_performance:
                self.historical_performance[competition_id] = []
            
            performance_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "competition_id": competition_id,
                "actual_result": actual_result,
                "predicted_medal_probability": None,
                "predicted_strategic_value": None
            }
            
            # 予測値も記録（対応する選択決定から取得）
            for decision in self.selection_history:
                for comp in decision.selected_competitions:
                    if comp.id == competition_id:
                        performance_record["predicted_medal_probability"] = comp.medal_probability
                        performance_record["predicted_strategic_value"] = comp.strategic_value
                        break
            
            self.historical_performance[competition_id].append(performance_record)
            
            self.logger.info(f"競技パフォーマンス更新: {competition_id}")
            
        except Exception as e:
            self.logger.error(f"パフォーマンス更新エラー: {e}")
    
    def get_selection_performance_metrics(self) -> Dict[str, Any]:
        """選択パフォーマンス指標取得"""
        
        if not self.historical_performance:
            return {"message": "パフォーマンス履歴なし"}
        
        # 精度分析
        total_predictions = 0
        correct_medal_predictions = 0
        
        for comp_id, records in self.historical_performance.items():
            for record in records:
                if record.get("predicted_medal_probability") is not None:
                    total_predictions += 1
                    
                    # 実際にメダルを獲得したかチェック
                    actual_medal = record["actual_result"].get("medal_achieved", False)
                    predicted_prob = record["predicted_medal_probability"]
                    
                    # 確率0.5以上を「メダル予測」として扱う
                    predicted_medal = predicted_prob >= 0.5
                    
                    if actual_medal == predicted_medal:
                        correct_medal_predictions += 1
        
        accuracy = correct_medal_predictions / max(1, total_predictions)
        
        return {
            "total_competitions_tracked": len(self.historical_performance),
            "total_predictions": total_predictions,
            "prediction_accuracy": accuracy,
            "selection_decisions_made": len(self.selection_history),
            "avg_competitions_per_decision": len(self.current_portfolio) / max(1, len(self.selection_history))
        }