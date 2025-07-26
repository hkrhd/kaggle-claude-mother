"""
コンペ参加管理システム

エージェントがコンペを選択した際に、ユーザーに手動参加登録を依頼するIssueを作成し、
参加確認後にオーケストレーションを開始する管理システム。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from system.issue_safety_system.utils.github_api_wrapper import GitHubApiWrapper
from system.issue_safety_system.concurrency_control.atomic_operations import AtomicIssueOperations


class ParticipationStatus(Enum):
    """参加状態"""
    PENDING_USER_REGISTRATION = "pending_user_registration"
    REGISTERED = "registered" 
    DECLINED = "declined"
    TIMEOUT = "timeout"


@dataclass
class CompetitionParticipationRequest:
    """コンペ参加リクエスト"""
    competition_id: str
    competition_name: str
    competition_url: str
    medal_probability: float
    recommended_by_agent: str
    selection_reason: str
    deadline: str
    prize_info: str
    participants_count: int
    status: ParticipationStatus = ParticipationStatus.PENDING_USER_REGISTRATION
    issue_number: Optional[int] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class CompetitionParticipationManager:
    """コンペ参加管理マネージャー"""
    
    def __init__(self, github_token: str, repo_name: str):
        self.logger = logging.getLogger(__name__)
        
        # GitHub連携
        self.github_wrapper = GitHubApiWrapper(
            access_token=github_token,
            repo_name=repo_name
        )
        self.atomic_operations = AtomicIssueOperations(
            github_client=self.github_wrapper.github,
            repo_name=repo_name
        )
        
        # 参加リクエスト管理
        self.pending_requests: Dict[str, CompetitionParticipationRequest] = {}
        self.completed_requests: List[CompetitionParticipationRequest] = []
        
        # 設定
        self.participation_timeout_hours = 48  # 48時間以内に参加確認
        
    async def request_user_participation(
        self,
        competition_id: str,
        competition_name: str,
        competition_url: str,
        medal_probability: float,
        selection_reason: str,
        deadline: str = "不明",
        prize_info: str = "不明",
        participants_count: int = 0,
        recommended_by_agent: str = "dynamic_competition_manager"
    ) -> CompetitionParticipationRequest:
        """ユーザーにコンペ参加登録を依頼"""
        
        self.logger.info(f"🎯 コンペ参加依頼開始: {competition_name}")
        
        # 参加リクエスト作成
        request = CompetitionParticipationRequest(
            competition_id=competition_id,
            competition_name=competition_name,
            competition_url=competition_url,
            medal_probability=medal_probability,
            recommended_by_agent=recommended_by_agent,
            selection_reason=selection_reason,
            deadline=deadline,
            prize_info=prize_info,
            participants_count=participants_count
        )
        
        # GitHub Issue作成
        issue_number = await self._create_participation_issue(request)
        request.issue_number = issue_number
        
        # 管理リストに追加
        self.pending_requests[competition_id] = request
        
        self.logger.info(f"✅ 参加依頼Issue作成: #{issue_number}")
        
        return request
    
    async def _create_participation_issue(self, request: CompetitionParticipationRequest) -> int:
        """参加依頼Issue作成"""
        
        title = f"🏆 コンペ参加登録依頼: {request.competition_name}"
        
        # メダル確率に基づく推奨度判定
        if request.medal_probability >= 0.7:
            recommendation_level = "🥇 **強く推奨** (高確率)"
            priority_label = "priority:medal-critical"
        elif request.medal_probability >= 0.5:
            recommendation_level = "🥈 **推奨** (中確率)"
            priority_label = "priority:high"
        else:
            recommendation_level = "🥉 **検討推奨** (低確率)"
            priority_label = "priority:medium"
        
        body = f"""
## 🎯 コンペ参加登録のお願い

システムが以下のコンペへの参加を推奨しています。**手動でKaggleサイトにて参加登録をお願いします。**

### 📊 コンペ詳細情報

**コンペ名**: [{request.competition_name}]({request.competition_url})
**メダル獲得確率**: {request.medal_probability:.1%}
**推奨レベル**: {recommendation_level}
**参加者数**: {request.participants_count:,}名
**締切**: {request.deadline}
**賞金**: {request.prize_info}

### 🤖 選択理由

{request.selection_reason}

### 📝 参加手順

1. **Kaggleサイトアクセス**: [{request.competition_name}]({request.competition_url})
2. **Join Competitionボタンクリック**: 参加登録を完了
3. **Rules Acceptanceの確認**: 規約に同意
4. **このIssueをクローズ**: 参加完了後に手動でクローズしてください

### ⚡ 自動開始について

**このIssueがクローズされると、システムが自動的にこのコンペのオーケストレーションを開始します。**

- 🎯 戦略プランニング開始
- 🔬 グランドマスター解法分析
- 🏗️ GPU実装・並列実験
- 🧠 継続学習・最適化

### ⏰ タイムアウト

**48時間以内**に参加確認がない場合、このリクエストは自動的にタイムアウトします。

---

**推奨エージェント**: `{request.recommended_by_agent}`
**作成時刻**: {request.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

> 💡 **ヒント**: 参加後すぐにクローズしていただければ、システムが即座に分析・実装を開始します。
"""
        
        # Issue作成
        result = await self.atomic_operations.create_issue(
            title=title,
            description=body,
            labels=[
                "participation-request",
                priority_label,
                f"comp:{request.competition_id}",
                "status:pending-user-action",
                "agent:participation-manager"
            ]
        )
        
        if result["success"]:
            return result["number"]
        else:
            raise Exception(f"Issue作成失敗: {result}")
    
    async def check_participation_status(self, competition_id: str) -> Optional[ParticipationStatus]:
        """参加状態確認"""
        
        if competition_id not in self.pending_requests:
            return None
        
        request = self.pending_requests[competition_id]
        
        try:
            # Issue状態確認
            issue = await self.github_wrapper.get_issue_safely(request.issue_number)
            
            if not issue.success:
                self.logger.error(f"Issue取得失敗: #{request.issue_number}")
                return request.status
            
            # クローズ確認
            if issue.issue.state == "closed":
                request.status = ParticipationStatus.REGISTERED
                
                # 完了リストに移動
                self.completed_requests.append(request)
                del self.pending_requests[competition_id]
                
                self.logger.info(f"✅ 参加確認完了: {request.competition_name}")
                
                return ParticipationStatus.REGISTERED
            
            # タイムアウト確認
            hours_elapsed = (datetime.now(timezone.utc) - request.created_at).total_seconds() / 3600
            if hours_elapsed > self.participation_timeout_hours:
                request.status = ParticipationStatus.TIMEOUT
                
                # タイムアウトコメント追加
                await self._add_timeout_comment(request)
                
                # 完了リストに移動
                self.completed_requests.append(request)
                del self.pending_requests[competition_id]
                
                self.logger.warning(f"⏰ 参加確認タイムアウト: {request.competition_name}")
                
                return ParticipationStatus.TIMEOUT
            
            return request.status
            
        except Exception as e:
            self.logger.error(f"参加状態確認エラー: {e}")
            return request.status
    
    async def _add_timeout_comment(self, request: CompetitionParticipationRequest):
        """タイムアウトコメント追加"""
        
        try:
            comment = f"""
## ⏰ 参加確認タイムアウト

{self.participation_timeout_hours}時間以内に参加確認がなかったため、このリクエストをタイムアウトとしました。

**次回参加希望の場合**:
- 新しいコンペスキャンで再度推奨される可能性があります
- 手動でシステムに参加リクエストを送ることも可能です

このIssueは自動的にクローズされます。

---
*自動タイムアウト処理 - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""
            
            await self.atomic_operations.create_comment(
                issue_number=request.issue_number,
                comment=comment
            )
            
            # Issueクローズ
            await self.github_wrapper.close_issue_safely(request.issue_number)
            
        except Exception as e:
            self.logger.error(f"タイムアウトコメント追加エラー: {e}")
    
    async def get_ready_competitions(self) -> List[str]:
        """参加確認済みでオーケストレーション開始可能なコンペリスト取得"""
        
        ready_competitions = []
        
        # 全ての pending requests を確認
        for competition_id in list(self.pending_requests.keys()):
            status = await self.check_participation_status(competition_id)
            
            if status == ParticipationStatus.REGISTERED:
                ready_competitions.append(competition_id)
        
        return ready_competitions
    
    async def monitor_participation_requests(self) -> Dict[str, Any]:
        """参加リクエスト監視（定期実行用）"""
        
        monitoring_result = {
            "checked_requests": 0,
            "newly_registered": [],
            "timed_out": [],
            "still_pending": []
        }
        
        for competition_id in list(self.pending_requests.keys()):
            monitoring_result["checked_requests"] += 1
            
            status = await self.check_participation_status(competition_id) 
            
            if status == ParticipationStatus.REGISTERED:
                monitoring_result["newly_registered"].append(competition_id)
            elif status == ParticipationStatus.TIMEOUT:
                monitoring_result["timed_out"].append(competition_id)
            elif status == ParticipationStatus.PENDING_USER_REGISTRATION:
                monitoring_result["still_pending"].append(competition_id)
        
        if monitoring_result["newly_registered"]:
            self.logger.info(f"🎉 新規参加確認: {monitoring_result['newly_registered']}")
        
        if monitoring_result["timed_out"]:
            self.logger.warning(f"⏰ タイムアウト: {monitoring_result['timed_out']}")
        
        return monitoring_result
    
    def get_participation_summary(self) -> Dict[str, Any]:
        """参加状況サマリー取得"""
        
        return {
            "pending_requests_count": len(self.pending_requests),
            "completed_requests_count": len(self.completed_requests),
            "pending_competitions": list(self.pending_requests.keys()),
            "recent_completions": [
                {
                    "competition_id": req.competition_id,
                    "competition_name": req.competition_name,
                    "status": req.status.value,
                    "completed_at": req.created_at.isoformat()
                }
                for req in self.completed_requests[-5:]  # 最新5件
            ]
        }


# テスト関数削除済み（house prices関連）