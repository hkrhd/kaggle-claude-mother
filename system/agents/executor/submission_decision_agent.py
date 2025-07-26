"""
提出判断LLMエージェント

ExecutorAgentの提出判断を高度化するLLMベース判断システム。
スコア・競合分析・リスクを総合的に評価して最適な提出タイミングを決定。
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..shared.llm_decision_base import (
    LLMDecisionEngine, LLMDecisionRequest, LLMDecisionResponse, 
    LLMDecisionType, ClaudeClient
)


@dataclass
class SubmissionContext:
    """提出判断コンテキスト"""
    competition_name: str
    current_best_score: float
    target_score: float
    current_rank_estimate: Optional[int]
    total_participants: int
    days_remaining: int
    hours_remaining: float
    
    # 実験実行状況
    experiments_completed: int
    experiments_running: int
    success_rate: float
    resource_budget_remaining: float
    
    # スコア改善履歴
    score_history: List[float]
    score_improvement_trend: float  # 直近の改善率
    plateau_duration_hours: float   # スコア停滞時間
    
    # 競合・リーダーボード情報
    leaderboard_top10_scores: List[float]
    medal_threshold_estimate: float
    current_medal_zone: str  # "gold", "silver", "bronze", "none"
    
    # リスク要因
    model_stability: float  # 0.0-1.0
    overfitting_risk: float # 0.0-1.0
    technical_debt_level: float # 0.0-1.0


class SubmissionDecisionAgent(LLMDecisionEngine):
    """提出判断LLMエージェント"""
    
    def __init__(self, claude_client: ClaudeClient):
        super().__init__("submission_decision", claude_client)
        
        # 提出判断固有設定
        self.min_confidence_threshold = 0.7
        self.conservative_mode = False  # True時はより慎重な判断
        
        # 提出履歴（学習用）
        self.submission_history: List[Dict[str, Any]] = []
    
    async def should_submit_competition(
        self,
        context: SubmissionContext,
        urgency: str = "medium"
    ) -> LLMDecisionResponse:
        """競技提出判断メイン"""
        
        request = LLMDecisionRequest(
            request_id=f"submit-{context.competition_name}-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            decision_type=LLMDecisionType.SUBMISSION_DECISION,
            context_data={
                "competition_name": context.competition_name,
                "performance_metrics": {
                    "current_best_score": context.current_best_score,
                    "target_score": context.target_score,
                    "score_achievement_ratio": context.current_best_score / max(0.001, context.target_score),
                    "rank_estimate": context.current_rank_estimate,
                    "medal_zone": context.current_medal_zone
                },
                "time_constraints": {
                    "days_remaining": context.days_remaining,
                    "hours_remaining": context.hours_remaining,
                    "time_pressure_level": "high" if context.hours_remaining < 24 else "medium" if context.hours_remaining < 72 else "low"
                },
                "execution_status": {
                    "experiments_completed": context.experiments_completed,
                    "experiments_running": context.experiments_running,
                    "success_rate": context.success_rate,
                    "resource_budget_remaining": context.resource_budget_remaining
                },
                "performance_trends": {
                    "score_history": context.score_history[-10:],  # 最新10件
                    "improvement_trend": context.score_improvement_trend,
                    "plateau_duration_hours": context.plateau_duration_hours,
                    "is_improving": context.score_improvement_trend > 0.001
                },
                "competitive_landscape": {
                    "total_participants": context.total_participants,
                    "leaderboard_top10": context.leaderboard_top10_scores,
                    "medal_threshold": context.medal_threshold_estimate,
                    "competitive_pressure": len([s for s in context.leaderboard_top10_scores if s > context.current_best_score])
                },
                "risk_assessment": {
                    "model_stability": context.model_stability,
                    "overfitting_risk": context.overfitting_risk,
                    "technical_debt": context.technical_debt_level,
                    "overall_risk_level": (context.overfitting_risk + context.technical_debt_level + (1 - context.model_stability)) / 3
                }
            },
            urgency_level=urgency,
            fallback_strategy="conservative_submit",
            max_response_time_seconds=20  # 提出判断は迅速性重視
        )
        
        return await self.make_decision(request)
    
    async def _generate_prompt(self, request: LLMDecisionRequest) -> str:
        """提出判断用プロンプト生成"""
        
        context = request.context_data
        
        return f"""# Kaggle競技提出判断タスク

あなたはKaggle競技において最適な提出タイミングを判断する専門エージェントです。
与えられた情報を総合的に分析し、メダル獲得確率を最大化する判断を行ってください。

## 競技情報
**競技名**: {context['competition_name']}
**参加者数**: {context['competitive_landscape']['total_participants']:,}名
**残り時間**: {context['time_constraints']['days_remaining']}日 ({context['time_constraints']['hours_remaining']:.1f}時間)

## 現在のパフォーマンス
**現在のベストスコア**: {context['performance_metrics']['current_best_score']:.6f}
**目標スコア**: {context['performance_metrics']['target_score']:.6f}
**達成率**: {context['performance_metrics']['score_achievement_ratio']:.1%}
**推定順位**: {context['performance_metrics']['rank_estimate'] or 'N/A'}
**現在のメダル圏**: {context['performance_metrics']['medal_zone']}

## 実験実行状況
**完了実験数**: {context['execution_status']['experiments_completed']}
**実行中実験数**: {context['execution_status']['experiments_running']}
**成功率**: {context['execution_status']['success_rate']:.1%}
**残りリソース予算**: {context['execution_status']['resource_budget_remaining']:.1%}

## パフォーマンス推移
**最近のスコア履歴**: {context['performance_trends']['score_history']}
**改善トレンド**: {context['performance_trends']['improvement_trend']:.6f}
**停滞時間**: {context['performance_trends']['plateau_duration_hours']:.1f}時間
**改善中**: {'Yes' if context['performance_trends']['is_improving'] else 'No'}

## 競合状況
**TOP10スコア**: {context['competitive_landscape']['leaderboard_top10']}
**メダル閾値推定**: {context['competitive_landscape']['medal_threshold']:.6f}
**上位競合者数**: {context['competitive_landscape']['competitive_pressure']}名

## リスク評価
**モデル安定性**: {context['risk_assessment']['model_stability']:.2f}
**過学習リスク**: {context['risk_assessment']['overfitting_risk']:.2f}
**技術的負債**: {context['risk_assessment']['technical_debt']:.2f}
**総合リスクレベル**: {context['risk_assessment']['overall_risk_level']:.2f}

## 判断要請
上記の情報を総合的に分析し、以下の形式でJSON応答してください：

```json
{{
  "decision": "SUBMIT" | "CONTINUE" | "WAIT",
  "confidence": 0.0-1.0,
  "reasoning": "判断根拠の詳細説明",
  "risk_assessment": "提出/継続に伴うリスク評価",
  "alternative_actions": ["代替案1", "代替案2", "代替案3"],
  "estimated_final_rank_range": [最小順位, 最大順位],
  "medal_probability": {{
    "gold": 0.0-1.0,
    "silver": 0.0-1.0, 
    "bronze": 0.0-1.0,
    "none": 0.0-1.0
  }},
  "timeline_recommendation": "推奨タイムライン",
  "key_factors": ["判断に影響した主要要因1", "要因2", "要因3"]
}}
```

## 判断基準
1. **メダル獲得確率最大化**を最優先
2. **リスクとリターンのバランス**を考慮
3. **時間制約とリソース効率**を評価
4. **競合状況と市場動向**を分析
5. **技術的安定性と信頼性**を重視

現在の緊急度: {request.urgency_level}
        """
    
    async def _parse_llm_response(
        self, 
        llm_response: str, 
        request: LLMDecisionRequest
    ) -> Dict[str, Any]:
        """LLM応答解析・構造化"""
        
        try:
            # JSON抽出・パース
            response_data = json.loads(llm_response)
            
            # 必須フィールド検証
            required_fields = ["decision", "confidence", "reasoning"]
            for field in required_fields:
                if field not in response_data:
                    raise ValueError(f"必須フィールド不足: {field}")
            
            # 判断の正規化
            decision = response_data["decision"].upper()
            if decision not in ["SUBMIT", "CONTINUE", "WAIT"]:
                decision = "WAIT"  # デフォルト
            
            # 信頼度正規化
            confidence = max(0.0, min(1.0, float(response_data["confidence"])))
            
            # 構造化結果
            structured_result = {
                "decision": decision,
                "confidence": confidence,
                "reasoning": response_data.get("reasoning", ""),
                "risk_assessment": response_data.get("risk_assessment", ""),
                "alternative_actions": response_data.get("alternative_actions", []),
                "estimated_final_rank_range": response_data.get("estimated_final_rank_range", [100, 500]),
                "medal_probability": response_data.get("medal_probability", {
                    "gold": 0.0, "silver": 0.0, "bronze": 0.1, "none": 0.9
                }),
                "timeline_recommendation": response_data.get("timeline_recommendation", ""),
                "key_factors": response_data.get("key_factors", [])
            }
            
            self.logger.info(f"LLM提出判断: {decision} (信頼度: {confidence:.2f})")
            return structured_result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"LLM応答解析失敗: {e}")
            # 解析失敗時はフォールバックへ
            raise e
    
    async def _execute_fallback_strategy(
        self, 
        request: LLMDecisionRequest
    ) -> Dict[str, Any]:
        """フォールバック戦略実行（ルールベース判断）"""
        
        context = request.context_data
        
        # ルールベース判断ロジック
        decision = "WAIT"
        confidence = 0.6
        reasoning = "ルールベース判断: "
        
        performance = context["performance_metrics"]
        time_info = context["time_constraints"] 
        trends = context["performance_trends"]
        risks = context["risk_assessment"]
        
        # 判断ルール適用
        score_ratio = performance["score_achievement_ratio"]
        hours_left = time_info["hours_remaining"]
        is_improving = trends["is_improving"]
        overall_risk = risks["overall_risk_level"]
        
        if score_ratio >= 1.0 and performance["medal_zone"] in ["gold", "silver"]:
            # 目標達成+メダル圏 → 提出
            decision = "SUBMIT"
            confidence = 0.85
            reasoning += "目標スコア達成かつメダル圏内"
            
        elif hours_left < 12 and performance["medal_zone"] != "none":
            # 時間切迫+メダル圏 → 提出  
            decision = "SUBMIT"
            confidence = 0.75
            reasoning += "締切間近でメダル圏内確保"
            
        elif hours_left < 6:
            # 緊急提出
            decision = "SUBMIT"
            confidence = 0.65
            reasoning += "締切直前のため緊急提出"
            
        elif is_improving and overall_risk < 0.5:
            # 改善中+低リスク → 継続
            decision = "CONTINUE"
            confidence = 0.7
            reasoning += "スコア改善中かつリスク低"
            
        else:
            # デフォルト: 待機
            decision = "WAIT"
            confidence = 0.6
            reasoning += "状況観察が適切"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "risk_assessment": f"総合リスク: {overall_risk:.2f}",
            "alternative_actions": ["手動判断", "追加実験", "保守的提出"],
            "estimated_final_rank_range": [200, 800],
            "medal_probability": {
                "gold": 0.05, "silver": 0.15, "bronze": 0.25, "none": 0.55
            },
            "timeline_recommendation": f"残り{hours_left:.1f}時間での戦略調整",
            "key_factors": ["時間制約", "スコア達成率", "リスクレベル"],
            "fallback_reason": "LLM応答失敗によるルールベース実行"
        }
    
    def enable_conservative_mode(self, enabled: bool = True):
        """保守的モード切り替え"""
        self.conservative_mode = enabled
        self.logger.info(f"保守的モード: {'有効' if enabled else '無効'}")
    
    def add_submission_result(
        self,
        decision_id: str,
        actual_rank: int,
        actual_score: float,
        medal_achieved: Optional[str] = None
    ):
        """提出結果記録（学習データ蓄積）"""
        
        result_record = {
            "decision_id": decision_id,
            "timestamp": datetime.utcnow().isoformat(),
            "actual_rank": actual_rank,
            "actual_score": actual_score,
            "medal_achieved": medal_achieved,
            "prediction_accuracy": None  # 後で計算
        }
        
        # 過去の判断と照合
        for decision in self.decision_history:
            if decision.request_id == decision_id:
                estimated_range = decision.decision_result.get("estimated_final_rank_range", [0, 1000])
                predicted_min, predicted_max = estimated_range
                
                # 予測精度算出
                if predicted_min <= actual_rank <= predicted_max:
                    accuracy = 1.0
                else:
                    # 範囲外れ度合いに基づく精度
                    if actual_rank < predicted_min:
                        accuracy = max(0.0, 1.0 - (predicted_min - actual_rank) / predicted_min)
                    else:
                        accuracy = max(0.0, 1.0 - (actual_rank - predicted_max) / predicted_max)
                
                result_record["prediction_accuracy"] = accuracy
                break
        
        self.submission_history.append(result_record)
        self.logger.info(f"提出結果記録: {decision_id} → 順位{actual_rank}, スコア{actual_score:.6f}")
    
    def get_decision_performance_summary(self) -> Dict[str, Any]:
        """判断パフォーマンスサマリー"""
        
        if not self.submission_history:
            return {"message": "提出結果データなし"}
        
        # 統計計算
        ranks = [r["actual_rank"] for r in self.submission_history]
        scores = [r["actual_score"] for r in self.submission_history]
        accuracies = [r["prediction_accuracy"] for r in self.submission_history if r["prediction_accuracy"] is not None]
        
        medals = [r["medal_achieved"] for r in self.submission_history if r["medal_achieved"]]
        medal_rate = len(medals) / len(self.submission_history)
        
        return {
            "total_submissions": len(self.submission_history),
            "average_rank": sum(ranks) / len(ranks),
            "best_rank": min(ranks),
            "average_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "medal_achievement_rate": medal_rate,
            "medal_distribution": {
                "gold": medals.count("gold"),
                "silver": medals.count("silver"), 
                "bronze": medals.count("bronze")
            },
            "prediction_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "conservative_mode_enabled": self.conservative_mode
        }