"""
LLMベース判断システム共通基盤

全エージェントで共通利用するLLM判断インターフェース・プロンプトエンジン・
フォールバック戦略を提供する基盤クラス。
"""

import asyncio
import logging
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from enum import Enum

# Claude Code実行時のモデル定数
# 実際のClaude Code実行時は --model sonnet オプションを使用
# 例: claude code --model sonnet "プロンプト"
CLAUDE_MODEL = "sonnet"  # 常に最新のSonnetを使用

# デフォルトモデル設定（全判断タイプで統一）
DEFAULT_MODEL_SUBMISSION = CLAUDE_MODEL     # 提出判断用
DEFAULT_MODEL_TECHNICAL = CLAUDE_MODEL      # 技術統合用
DEFAULT_MODEL_DIAGNOSTIC = CLAUDE_MODEL     # 異常診断用
DEFAULT_MODEL_RESOURCE = CLAUDE_MODEL       # リソース配分用
DEFAULT_MODEL_RISK = CLAUDE_MODEL           # リスク評価用

# Claude APIクライアント（仮想実装）
class ClaudeClient:
    """Claude API統合クライアント（仮想実装）"""
    
    def __init__(self, api_key: str = "demo_key", model: str = None):
        self.api_key = api_key
        # モデルが指定されていない場合はデフォルトを使用
        self.model = model if model else CLAUDE_MODEL
    
    async def complete(
        self,
        prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.3
    ) -> str:
        """Claude API呼び出し（模擬実装）"""
        
        # 実際の実装では anthropic ライブラリを使用
        # 現在は決定論的模擬応答を返す
        
        if "提出判断" in prompt:
            return self._generate_submission_decision_response(prompt)
        elif "技術統合" in prompt:
            return self._generate_technical_integration_response(prompt)
        elif "異常診断" in prompt:
            return self._generate_diagnostic_response(prompt)
        else:
            return self._generate_generic_response(prompt)
    
    def _generate_submission_decision_response(self, prompt: str) -> str:
        """提出判断応答生成"""
        return json.dumps({
            "decision": "SUBMIT",
            "confidence": 0.85,
            "reasoning": "スコア改善トレンドが安定、競合分析で上位10%確率が高い",
            "risk_assessment": "中リスク：更なる改善余地はあるが確実性を重視",
            "alternative_actions": ["継続実験", "パラメータ微調整"],
            "estimated_final_rank_range": [50, 150]
        }, ensure_ascii=False)
    
    def _generate_technical_integration_response(self, prompt: str) -> str:
        """技術統合応答生成"""
        return json.dumps({
            "integrated_recommendations": [
                {
                    "technique": "gradient_boosting_ensemble", 
                    "priority": 1,
                    "integration_confidence": 0.92,
                    "synergy_score": 0.85
                },
                {
                    "technique": "feature_engineering_advanced",
                    "priority": 2, 
                    "integration_confidence": 0.78,
                    "synergy_score": 0.70
                }
            ],
            "integration_strategy": "段階的実装：主力技術確立後に補完技術追加",
            "risk_mitigation": ["基本実装の確実性確保", "段階的検証"],
            "estimated_improvement": 0.15
        }, ensure_ascii=False)
    
    def _generate_diagnostic_response(self, prompt: str) -> str:
        """異常診断応答生成"""
        return json.dumps({
            "diagnosis": "メモリリーク＋API制限の複合問題",
            "severity": "MEDIUM",
            "root_cause": "長時間実行による累積メモリ使用＋API呼び出し頻度過多",
            "recommended_actions": [
                {"action": "プロセス再起動", "priority": 1, "estimated_effect": 0.8},
                {"action": "API呼び出し間隔調整", "priority": 2, "estimated_effect": 0.6}
            ],
            "prevention_measures": ["定期メモリクリア", "API制限監視強化"],
            "estimated_resolution_time": "15分"
        }, ensure_ascii=False)
    
    def _generate_generic_response(self, prompt: str) -> str:
        """汎用応答生成"""
        return json.dumps({
            "analysis": "高品質な判断が可能",
            "confidence": 0.75,
            "recommendations": ["最適化継続", "監視強化"],
            "next_actions": ["データ分析", "戦略見直し"]
        }, ensure_ascii=False)


class LLMDecisionType(Enum):
    """LLM判断タイプ"""
    SUBMISSION_DECISION = "submission_decision"
    TECHNICAL_INTEGRATION = "technical_integration"
    DIAGNOSTIC_ANALYSIS = "diagnostic_analysis"
    RESOURCE_ALLOCATION = "resource_allocation"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class LLMDecisionRequest:
    """LLM判断リクエスト"""
    request_id: str
    decision_type: LLMDecisionType
    context_data: Dict[str, Any]
    urgency_level: str  # "low", "medium", "high", "critical"
    fallback_strategy: str
    max_response_time_seconds: int = 30
    
    def to_prompt_context(self) -> str:
        """プロンプト文脈データ生成"""
        return json.dumps(self.context_data, indent=2, ensure_ascii=False)


@dataclass
class LLMDecisionResponse:
    """LLM判断応答"""
    request_id: str
    decision_type: LLMDecisionType
    decision_result: Dict[str, Any]
    confidence_score: float
    reasoning: str
    alternative_options: List[str]
    execution_time_seconds: float
    fallback_used: bool
    created_at: datetime
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """高信頼度判定"""
        return self.confidence_score >= threshold


class LLMDecisionEngine(ABC):
    """LLM判断エンジン基底クラス"""
    
    def __init__(self, agent_name: str, claude_client: ClaudeClient):
        self.agent_name = agent_name
        self.claude_client = claude_client
        self.logger = logging.getLogger(f"{__name__}.{agent_name}")
        
        # 判断履歴
        self.decision_history: List[LLMDecisionResponse] = []
        
        # パフォーマンス統計
        self.total_decisions = 0
        self.successful_decisions = 0
        self.fallback_usage_count = 0
        self.average_response_time = 0.0
    
    async def make_decision(
        self, 
        request: LLMDecisionRequest
    ) -> LLMDecisionResponse:
        """LLM判断実行（メインエントリポイント）"""
        
        decision_start = datetime.utcnow()
        
        try:
            self.logger.info(f"LLM判断開始: {request.decision_type.value} (ID: {request.request_id})")
            
            # プロンプト生成
            prompt = await self._generate_prompt(request)
            
            # LLM呼び出し（タイムアウト制御付き）
            try:
                llm_response = await asyncio.wait_for(
                    self.claude_client.complete(
                        prompt=prompt,
                        max_tokens=self._get_max_tokens(request.decision_type),
                        temperature=self._get_temperature(request.decision_type)
                    ),
                    timeout=request.max_response_time_seconds
                )
                
                # 応答解析・構造化
                decision_result = await self._parse_llm_response(llm_response, request)
                fallback_used = False
                
            except (asyncio.TimeoutError, Exception) as e:
                self.logger.warning(f"LLM呼び出し失敗: {e}")
                # フォールバック実行
                decision_result = await self._execute_fallback_strategy(request)
                fallback_used = True
                self.fallback_usage_count += 1
            
            # 応答オブジェクト作成
            execution_time = (datetime.utcnow() - decision_start).total_seconds()
            
            response = LLMDecisionResponse(
                request_id=request.request_id,
                decision_type=request.decision_type,
                decision_result=decision_result,
                confidence_score=decision_result.get("confidence", 0.5),
                reasoning=decision_result.get("reasoning", "推論データなし"),
                alternative_options=decision_result.get("alternative_actions", []),
                execution_time_seconds=execution_time,
                fallback_used=fallback_used,
                created_at=decision_start
            )
            
            # 統計更新
            self._update_performance_stats(response)
            
            # 履歴記録
            self.decision_history.append(response)
            
            self.logger.info(f"LLM判断完了: 信頼度{response.confidence_score:.2f}, {execution_time:.1f}秒")
            return response
            
        except Exception as e:
            self.logger.error(f"LLM判断エラー: {e}")
            
            # 緊急フォールバック
            emergency_result = await self._emergency_fallback(request)
            execution_time = (datetime.utcnow() - decision_start).total_seconds()
            
            return LLMDecisionResponse(
                request_id=request.request_id,
                decision_type=request.decision_type,
                decision_result=emergency_result,
                confidence_score=0.3,
                reasoning="緊急フォールバック実行",
                alternative_options=[],
                execution_time_seconds=execution_time,
                fallback_used=True,
                created_at=decision_start
            )
    
    @abstractmethod
    async def _generate_prompt(self, request: LLMDecisionRequest) -> str:
        """判断タイプ固有のプロンプト生成"""
        pass
    
    @abstractmethod
    async def _parse_llm_response(
        self, 
        llm_response: str, 
        request: LLMDecisionRequest
    ) -> Dict[str, Any]:
        """LLM応答解析・構造化"""
        pass
    
    @abstractmethod
    async def _execute_fallback_strategy(
        self, 
        request: LLMDecisionRequest
    ) -> Dict[str, Any]:
        """フォールバック戦略実行"""
        pass
    
    async def _emergency_fallback(
        self, 
        request: LLMDecisionRequest
    ) -> Dict[str, Any]:
        """緊急フォールバック（全判断タイプ共通）"""
        
        return {
            "decision": "CONSERVATIVE",
            "confidence": 0.3,
            "reasoning": "システム異常のため保守的判断を選択",
            "fallback_reason": "emergency_fallback_activated"
        }
    
    def _get_max_tokens(self, decision_type: LLMDecisionType) -> int:
        """判断タイプ別最大トークン数"""
        token_map = {
            LLMDecisionType.SUBMISSION_DECISION: 2000,
            LLMDecisionType.TECHNICAL_INTEGRATION: 3000,
            LLMDecisionType.DIAGNOSTIC_ANALYSIS: 2500,
            LLMDecisionType.RESOURCE_ALLOCATION: 2000,
            LLMDecisionType.RISK_ASSESSMENT: 2500
        }
        return token_map.get(decision_type, 2000)
    
    def _get_temperature(self, decision_type: LLMDecisionType) -> float:
        """判断タイプ別temperature設定"""
        temp_map = {
            LLMDecisionType.SUBMISSION_DECISION: 0.2,  # 保守的
            LLMDecisionType.TECHNICAL_INTEGRATION: 0.4,  # バランス
            LLMDecisionType.DIAGNOSTIC_ANALYSIS: 0.1,   # 非常に保守的
            LLMDecisionType.RESOURCE_ALLOCATION: 0.3,   # やや保守的
            LLMDecisionType.RISK_ASSESSMENT: 0.2       # 保守的
        }
        return temp_map.get(decision_type, 0.3)
    
    def _get_model(self, decision_type: LLMDecisionType) -> str:
        """判断タイプ別モデル選択
        
        プロンプトごとに最適なモデルを選択。
        ファイル上部の定数を変更することで簡単にモデル変更可能。
        """
        model_map = {
            LLMDecisionType.SUBMISSION_DECISION: DEFAULT_MODEL_SUBMISSION,
            LLMDecisionType.TECHNICAL_INTEGRATION: DEFAULT_MODEL_TECHNICAL,
            LLMDecisionType.DIAGNOSTIC_ANALYSIS: DEFAULT_MODEL_DIAGNOSTIC,
            LLMDecisionType.RESOURCE_ALLOCATION: DEFAULT_MODEL_RESOURCE,
            LLMDecisionType.RISK_ASSESSMENT: DEFAULT_MODEL_RISK
        }
        return model_map.get(decision_type, CLAUDE_MODEL)
    
    def _update_performance_stats(self, response: LLMDecisionResponse):
        """パフォーマンス統計更新"""
        
        self.total_decisions += 1
        
        if response.confidence_score >= 0.6:
            self.successful_decisions += 1
        
        # 移動平均での応答時間更新
        if self.average_response_time == 0.0:
            self.average_response_time = response.execution_time_seconds
        else:
            alpha = 0.1  # 移動平均係数
            self.average_response_time = (
                alpha * response.execution_time_seconds + 
                (1 - alpha) * self.average_response_time
            )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """パフォーマンス指標取得"""
        
        success_rate = self.successful_decisions / max(1, self.total_decisions)
        fallback_rate = self.fallback_usage_count / max(1, self.total_decisions)
        
        return {
            "agent_name": self.agent_name,
            "total_decisions": self.total_decisions,
            "success_rate": success_rate,
            "fallback_rate": fallback_rate,
            "average_response_time": self.average_response_time,
            "last_decision_time": self.decision_history[-1].created_at.isoformat() if self.decision_history else None
        }