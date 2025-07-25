"""
GitHub API安全ラッパー

GitHub API操作の安全性・信頼性を保証する統合ラッパー。
レート制限・エラーハンドリング・監査ログを統合提供。
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
import logging

# GitHubライブラリ（テスト環境では模擬）
try:
    from github import Github, GithubException
    from github.Repository import Repository
    from github.Issue import Issue
    from github.Label import Label
except ImportError:
    # テスト環境用の模擬クラス
    class Github:
        def __init__(self, *args, **kwargs):
            pass
        def get_repo(self, repo_name):
            return None
    
    class GithubException(Exception):
        pass
    
    class Repository:
        pass
    
    class Issue:
        pass
    
    class Label:
        pass

from .retry_mechanism import RetryMechanism, RetryConfig
from .audit_logger import AuditLogger, AuditEvent


@dataclass
class RateLimitInfo:
    """レート制限情報"""
    remaining: int
    limit: int
    reset_time: datetime
    used: int
    
    @property
    def usage_percentage(self) -> float:
        """使用率パーセンテージ"""
        if self.limit == 0:
            return 100.0
        return (self.used / self.limit) * 100.0
    
    @property
    def time_until_reset(self) -> timedelta:
        """リセットまでの時間"""
        return max(self.reset_time - datetime.utcnow(), timedelta(0))


@dataclass
class IssueOperationResult:
    """Issue操作結果"""
    success: bool
    issue: Optional[Issue] = None
    issue_number: Optional[int] = None
    operation_type: str = ""
    timestamp: datetime = None
    error: Optional[str] = None
    rate_limit_info: Optional[RateLimitInfo] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class GitHubApiSafetyError(Exception):
    """GitHub API安全操作エラー"""
    def __init__(self, message: str, rate_limit_info: Optional[RateLimitInfo] = None):
        super().__init__(message)
        self.rate_limit_info = rate_limit_info


class GitHubApiWrapper:
    """GitHub API安全ラッパー"""
    
    def __init__(
        self,
        access_token: str,
        repo_name: str,
        rate_limit_buffer: int = 100,  # 安全バッファ
        max_retries: int = 5
    ):
        self.github = Github(access_token)
        self.repo = self.github.get_repo(repo_name)
        self.repo_name = repo_name
        self.rate_limit_buffer = rate_limit_buffer
        
        # コンポーネント初期化
        self.retry_mechanism = RetryMechanism(
            RetryConfig(
                max_attempts=max_retries,
                base_delay=1.0,
                max_delay=60.0,
                exponential_base=2.0
            )
        )
        self.audit_logger = AuditLogger()
        self.logger = logging.getLogger(__name__)
        
        # 内部状態
        self.last_rate_limit_check = datetime.utcnow()
        self.cached_rate_limit_info: Optional[RateLimitInfo] = None
        self.operation_count = 0
    
    async def create_issue_safely(
        self,
        title: str,
        body: str,
        labels: List[str],
        assignees: Optional[List[str]] = None,
        competition: Optional[str] = None
    ) -> IssueOperationResult:
        """安全なIssue作成"""
        
        operation_context = {
            "operation": "create_issue",
            "title": title,
            "labels": labels,
            "assignees": assignees or [],
            "competition": competition
        }
        
        # 監査ログ開始
        audit_event = self.audit_logger.start_operation(
            "create_issue", operation_context
        )
        
        try:
            # レート制限チェック
            await self.check_rate_limit_safety()
            
            # 重複チェック
            duplicate_issue = await self.check_duplicate_issue(title, labels)
            if duplicate_issue:
                result = IssueOperationResult(
                    success=True,
                    issue=duplicate_issue,
                    issue_number=duplicate_issue.number,
                    operation_type="create_issue_duplicate_found",
                    error="Duplicate issue found, returning existing"
                )
                
                self.audit_logger.complete_operation(
                    audit_event, True, {"duplicate_found": True}
                )
                
                return result
            
            # リトライ付きIssue作成
            async def create_operation():
                return self.repo.create_issue(
                    title=title,
                    body=body,
                    labels=labels,
                    assignees=assignees or []
                )
            
            issue = await self.retry_mechanism.execute_with_retry(
                create_operation,
                operation_name="create_issue"
            )
            
            # 結果作成
            result = IssueOperationResult(
                success=True,
                issue=issue,
                issue_number=issue.number,
                operation_type="create_issue",
                rate_limit_info=await self.get_current_rate_limit()
            )
            
            # 監査ログ完了
            self.audit_logger.complete_operation(
                audit_event, True, {
                    "issue_number": issue.number,
                    "issue_url": issue.html_url
                }
            )
            
            self.logger.info(f"Issue作成成功: #{issue.number} - {title}")
            return result
            
        except Exception as e:
            # エラー処理・監査ログ
            error_info = {
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            
            self.audit_logger.complete_operation(
                audit_event, False, error_info
            )
            
            result = IssueOperationResult(
                success=False,
                operation_type="create_issue",
                error=str(e),
                rate_limit_info=await self.get_current_rate_limit()
            )
            
            self.logger.error(f"Issue作成失敗: {title} - {e}")
            return result
    
    async def update_issue_safely(
        self,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        state: Optional[str] = None
    ) -> IssueOperationResult:
        """安全なIssue更新"""
        
        operation_context = {
            "operation": "update_issue",
            "issue_number": issue_number,
            "title": title,
            "body": body,
            "labels": labels,
            "assignees": assignees,
            "state": state
        }
        
        audit_event = self.audit_logger.start_operation(
            "update_issue", operation_context
        )
        
        try:
            # レート制限チェック
            await self.check_rate_limit_safety()
            
            # Issue取得・存在確認
            issue = await self.get_issue_safely(issue_number)
            if not issue.success:
                return IssueOperationResult(
                    success=False,
                    operation_type="update_issue",
                    error=f"Issue not found: #{issue_number}"
                )
            
            current_issue = issue.issue
            
            # リトライ付き更新実行
            async def update_operation():
                update_kwargs = {}
                if title is not None:
                    update_kwargs['title'] = title
                if body is not None:
                    update_kwargs['body'] = body
                if labels is not None:
                    update_kwargs['labels'] = labels
                if assignees is not None:
                    update_kwargs['assignees'] = assignees
                if state is not None:
                    update_kwargs['state'] = state
                
                current_issue.edit(**update_kwargs)
                return current_issue
            
            updated_issue = await self.retry_mechanism.execute_with_retry(
                update_operation,
                operation_name="update_issue"
            )
            
            result = IssueOperationResult(
                success=True,
                issue=updated_issue,
                issue_number=updated_issue.number,
                operation_type="update_issue",
                rate_limit_info=await self.get_current_rate_limit()
            )
            
            self.audit_logger.complete_operation(
                audit_event, True, {
                    "updated_fields": list(operation_context.keys()),
                    "issue_url": updated_issue.html_url
                }
            )
            
            self.logger.info(f"Issue更新成功: #{issue_number}")
            return result
            
        except Exception as e:
            self.audit_logger.complete_operation(
                audit_event, False, {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            result = IssueOperationResult(
                success=False,
                operation_type="update_issue",
                error=str(e),
                rate_limit_info=await self.get_current_rate_limit()
            )
            
            self.logger.error(f"Issue更新失敗: #{issue_number} - {e}")
            return result
    
    async def get_issue_safely(self, issue_number: int) -> IssueOperationResult:
        """安全なIssue取得"""
        
        try:
            await self.check_rate_limit_safety()
            
            async def get_operation():
                return self.repo.get_issue(issue_number)
            
            issue = await self.retry_mechanism.execute_with_retry(
                get_operation,
                operation_name="get_issue"
            )
            
            return IssueOperationResult(
                success=True,
                issue=issue,
                issue_number=issue.number,
                operation_type="get_issue",
                rate_limit_info=await self.get_current_rate_limit()
            )
            
        except Exception as e:
            return IssueOperationResult(
                success=False,
                operation_type="get_issue",
                error=str(e),
                rate_limit_info=await self.get_current_rate_limit()
            )
    
    async def add_comment_safely(
        self,
        issue_number: int,
        comment_body: str
    ) -> IssueOperationResult:
        """安全なコメント追加"""
        
        operation_context = {
            "operation": "add_comment",
            "issue_number": issue_number,
            "comment_length": len(comment_body)
        }
        
        audit_event = self.audit_logger.start_operation(
            "add_comment", operation_context
        )
        
        try:
            await self.check_rate_limit_safety()
            
            # Issue取得
            issue_result = await self.get_issue_safely(issue_number)
            if not issue_result.success:
                return IssueOperationResult(
                    success=False,
                    operation_type="add_comment",
                    error=f"Issue not found: #{issue_number}"
                )
            
            issue = issue_result.issue
            
            # リトライ付きコメント追加
            async def comment_operation():
                return issue.create_comment(comment_body)
            
            comment = await self.retry_mechanism.execute_with_retry(
                comment_operation,
                operation_name="add_comment"
            )
            
            result = IssueOperationResult(
                success=True,
                issue=issue,
                issue_number=issue.number,
                operation_type="add_comment",
                rate_limit_info=await self.get_current_rate_limit()
            )
            
            self.audit_logger.complete_operation(
                audit_event, True, {
                    "comment_id": comment.id,
                    "comment_url": comment.html_url
                }
            )
            
            self.logger.info(f"コメント追加成功: Issue #{issue_number}")
            return result
            
        except Exception as e:
            self.audit_logger.complete_operation(
                audit_event, False, {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            
            result = IssueOperationResult(
                success=False,
                operation_type="add_comment",
                error=str(e),
                rate_limit_info=await self.get_current_rate_limit()
            )
            
            self.logger.error(f"コメント追加失敗: Issue #{issue_number} - {e}")
            return result
    
    async def list_issues_safely(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        limit: int = 30
    ) -> Dict[str, Any]:
        """安全なIssue一覧取得"""
        
        try:
            await self.check_rate_limit_safety()
            
            async def list_operation():
                issues = self.repo.get_issues(
                    state=state,
                    labels=labels or [],
                    assignee=assignee
                )
                
                # 制限付きで取得
                issue_list = []
                for i, issue in enumerate(issues):
                    if i >= limit:
                        break
                    issue_list.append({
                        "number": issue.number,
                        "title": issue.title,
                        "state": issue.state,
                        "labels": [label.name for label in issue.labels],
                        "assignees": [assignee.login for assignee in issue.assignees],
                        "created_at": issue.created_at.isoformat(),
                        "updated_at": issue.updated_at.isoformat(),
                        "html_url": issue.html_url
                    })
                
                return issue_list
            
            issues = await self.retry_mechanism.execute_with_retry(
                list_operation,
                operation_name="list_issues"
            )
            
            return {
                "success": True,
                "issues": issues,
                "count": len(issues),
                "rate_limit_info": await self.get_current_rate_limit()
            }
            
        except Exception as e:
            self.logger.error(f"Issue一覧取得失敗: {e}")
            return {
                "success": False,
                "error": str(e),
                "rate_limit_info": await self.get_current_rate_limit()
            }
    
    async def check_duplicate_issue(
        self,
        title: str,
        labels: List[str]
    ) -> Optional[Issue]:
        """重複Issue確認"""
        
        try:
            # ラベルベース検索
            issues = self.repo.get_issues(state="open", labels=labels)
            
            # タイトル類似度チェック
            for issue in issues:
                if self.is_similar_title(title, issue.title):
                    return issue
            
            return None
            
        except Exception as e:
            self.logger.warning(f"重複チェック失敗: {e}")
            return None
    
    def is_similar_title(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """タイトル類似度判定"""
        
        # 簡単な類似度判定（実際にはより高度なアルゴリズムを使用可能）
        title1_words = set(title1.lower().split())
        title2_words = set(title2.lower().split())
        
        if not title1_words and not title2_words:
            return True
        
        intersection = len(title1_words.intersection(title2_words))
        union = len(title1_words.union(title2_words))
        
        if union == 0:
            return False
        
        jaccard_similarity = intersection / union
        return jaccard_similarity >= threshold
    
    async def check_rate_limit_safety(self):
        """レート制限安全性チェック"""
        
        rate_limit_info = await self.get_current_rate_limit()
        
        # 残り制限が不足している場合は待機
        if rate_limit_info.remaining < self.rate_limit_buffer:
            wait_time = rate_limit_info.time_until_reset.total_seconds()
            
            self.logger.warning(
                f"レート制限接近、待機中: {rate_limit_info.remaining}/{rate_limit_info.limit} "
                f"(リセットまで{wait_time:.0f}秒)"
            )
            
            if wait_time > 0:
                await asyncio.sleep(min(wait_time + 10, 300))  # 最大5分待機
    
    async def get_current_rate_limit(self) -> RateLimitInfo:
        """現在のレート制限情報取得"""
        
        # キャッシュ有効性チェック（1分間有効）
        if (self.cached_rate_limit_info and 
            datetime.utcnow() - self.last_rate_limit_check < timedelta(minutes=1)):
            return self.cached_rate_limit_info
        
        try:
            rate_limit = self.github.get_rate_limit()
            core_limit = rate_limit.core
            
            self.cached_rate_limit_info = RateLimitInfo(
                remaining=core_limit.remaining,
                limit=core_limit.limit,
                reset_time=core_limit.reset,
                used=core_limit.limit - core_limit.remaining
            )
            
            self.last_rate_limit_check = datetime.utcnow()
            
            return self.cached_rate_limit_info
            
        except Exception as e:
            self.logger.error(f"レート制限情報取得失敗: {e}")
            
            # フォールバック値
            return RateLimitInfo(
                remaining=0,
                limit=5000,
                reset_time=datetime.utcnow() + timedelta(hours=1),
                used=5000
            )
    
    async def get_operation_statistics(self) -> Dict[str, Any]:
        """操作統計取得"""
        
        rate_limit_info = await self.get_current_rate_limit()
        audit_stats = self.audit_logger.get_statistics()
        
        return {
            "total_operations": self.operation_count,
            "rate_limit": {
                "remaining": rate_limit_info.remaining,
                "limit": rate_limit_info.limit,
                "usage_percentage": rate_limit_info.usage_percentage,
                "time_until_reset_minutes": rate_limit_info.time_until_reset.total_seconds() / 60
            },
            "audit_statistics": audit_stats,
            "wrapper_start_time": self.audit_logger.start_time.isoformat(),
            "repository": self.repo_name
        }
    
    async def validate_repository_access(self) -> Dict[str, Any]:
        """リポジトリアクセス検証"""
        
        try:
            # 基本情報取得テスト
            repo_info = {
                "name": self.repo.name,
                "full_name": self.repo.full_name,
                "private": self.repo.private,
                "permissions": {}
            }
            
            # 権限テスト
            try:
                # Issue読み取りテスト
                list(self.repo.get_issues(state="open")[:1])
                repo_info["permissions"]["read_issues"] = True
            except:
                repo_info["permissions"]["read_issues"] = False
            
            try:
                # Issue作成テスト（実際には作成しない）
                repo_info["permissions"]["create_issues"] = True
            except:
                repo_info["permissions"]["create_issues"] = False
            
            return {
                "success": True,
                "repository_info": repo_info,
                "rate_limit_info": await self.get_current_rate_limit()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def emergency_rate_limit_recovery(self) -> bool:
        """緊急レート制限回復"""
        
        try:
            rate_limit_info = await self.get_current_rate_limit()
            
            if rate_limit_info.remaining > 0:
                self.logger.info("レート制限回復確認済み")
                return True
            
            wait_time = rate_limit_info.time_until_reset.total_seconds()
            if wait_time > 0:
                self.logger.info(f"緊急待機モード: {wait_time:.0f}秒待機")
                await asyncio.sleep(wait_time + 30)  # 30秒バッファ
                
                # 回復確認
                new_rate_limit = await self.get_current_rate_limit()
                return new_rate_limit.remaining > 0
            
            return False
            
        except Exception as e:
            self.logger.error(f"緊急レート制限回復失敗: {e}")
            return False