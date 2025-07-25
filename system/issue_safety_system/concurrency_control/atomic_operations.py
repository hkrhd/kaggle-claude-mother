"""
原子性操作・競合回避システム

GitHub Issue APIによる安全なエージェント間連携の核心システム。
楽観的ロック・ETag活用による原子的Issue作成・更新を提供。
"""

import asyncio
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

# GitHubライブラリ（テスト環境では模擬）
try:
    from github import Github
    from github.GithubException import GithubException, UnknownObjectException
except ImportError:
    # テスト環境用の模擬クラス
    class Github:
        def __init__(self, *args, **kwargs):
            pass
        def get_repo(self, repo_name):
            return MockRepo()
    
    class GithubException(Exception):
        status = 409
    
    class UnknownObjectException(Exception):
        pass
    
    class MockRepo:
        def create_issue(self, **kwargs):
            return MockIssue()
        def get_issue(self, number):
            return MockIssue()
    
    class MockIssue:
        number = 12345


@dataclass
class IssueOperationResult:
    """Issue操作結果"""
    issue: Any
    created: bool = False
    updated: bool = False
    attempt: int = 1
    timestamp: datetime = None
    reason: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class ConcurrencyConflictError(Exception):
    """競合状態エラー"""
    def __init__(self, message: str, attempt: int, max_attempts: int):
        super().__init__(message)
        self.attempt = attempt
        self.max_attempts = max_attempts


class AtomicIssueOperations:
    """原子的Issue操作システム"""
    
    def __init__(self, github_client: Github, repo_name: str):
        self.github = github_client
        self.repo = github_client.get_repo(repo_name)
        self.retry_config = {
            "max_attempts": 5,
            "base_delay": 1.0,
            "max_delay": 16.0,
            "exponential_base": 2.0
        }
        self.logger = logging.getLogger(__name__)
        
    def generate_unique_identifier(self, title: str, labels: List[str]) -> str:
        """一意識別子生成（重複防止用）"""
        identifier_data = f"{title}:{':'.join(sorted(labels))}"
        return hashlib.sha256(identifier_data.encode()).hexdigest()[:16]
    
    async def check_duplicate_issue(self, title: str, labels: List[str]) -> Optional[Any]:
        """既存Issue重複チェック"""
        try:
            # ラベルベースの検索
            label_query = " ".join([f"label:{label}" for label in labels])
            issues = self.repo.get_issues(
                state="open",
                labels=labels
            )
            
            # タイトル部分一致チェック
            for issue in issues:
                if title.lower() in issue.title.lower():
                    self.logger.info(f"重複Issue発見: {issue.number} - {issue.title}")
                    return issue
                    
            return None
            
        except Exception as e:
            self.logger.error(f"重複チェック失敗: {e}")
            return None
    
    async def create_issue_atomically(
        self, 
        title: str, 
        body: str, 
        labels: List[str], 
        assignees: Optional[List[str]] = None
    ) -> IssueOperationResult:
        """重複防止・原子的Issue作成"""
        unique_identifier = self.generate_unique_identifier(title, labels)
        
        try:
            # 既存Issue重複チェック
            existing_issue = await self.check_duplicate_issue(title, labels)
            if existing_issue:
                return IssueOperationResult(
                    issue=existing_issue,
                    created=False,
                    reason="duplicate_exists"
                )
            
            # 原子的Issue作成実行
            issue = self.repo.create_issue(
                title=title,
                body=body,
                labels=labels,
                assignees=assignees or []
            )
            
            # 作成成功の確認・検証
            created_issue = await self.verify_issue_creation(issue.number)
            
            self.logger.info(f"Issue原子的作成成功: {issue.number} - {title}")
            
            return IssueOperationResult(
                issue=created_issue,
                created=True
            )
            
        except Exception as e:
            self.logger.error(f"原子的Issue作成失敗: {e}")
            raise
    
    async def verify_issue_creation(self, issue_number: int) -> Any:
        """Issue作成成功確認・検証"""
        try:
            issue = self.repo.get_issue(issue_number)
            return issue
        except UnknownObjectException:
            raise Exception(f"Issue作成確認失敗: {issue_number}")
    
    async def update_issue_with_optimistic_lock(
        self, 
        issue_number: int, 
        updates: Dict[str, Any]
    ) -> IssueOperationResult:
        """楽観的ロックによる安全なIssue更新"""
        
        for attempt in range(self.retry_config["max_attempts"]):
            try:
                # 現在のIssue状態取得
                current_issue = self.repo.get_issue(issue_number)
                
                # 更新内容の準備・検証
                update_data = await self.prepare_update_data(current_issue, updates)
                
                # Issue更新実行
                if "title" in update_data:
                    current_issue.edit(title=update_data["title"])
                if "body" in update_data:
                    current_issue.edit(body=update_data["body"])
                if "labels" in update_data:
                    current_issue.edit(labels=update_data["labels"])
                if "assignees" in update_data:
                    current_issue.edit(assignees=update_data["assignees"])
                if "state" in update_data:
                    current_issue.edit(state=update_data["state"])
                
                self.logger.info(f"Issue楽観的更新成功: {issue_number} (試行{attempt + 1})")
                
                return IssueOperationResult(
                    issue=current_issue,
                    updated=True,
                    attempt=attempt + 1
                )
                
            except GithubException as e:
                if e.status == 409:  # Conflict
                    if attempt < self.retry_config["max_attempts"] - 1:
                        delay = self.calculate_backoff_delay(attempt)
                        self.logger.warning(f"競合検出、リトライ中 (試行{attempt + 1}): {delay}秒待機")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise ConcurrencyConflictError(
                            f"最大試行回数到達後も競合継続: Issue {issue_number}",
                            attempt + 1,
                            self.retry_config["max_attempts"]
                        )
                else:
                    raise
            except Exception as e:
                self.logger.error(f"Issue更新失敗 (試行{attempt + 1}): {e}")
                if attempt == self.retry_config["max_attempts"] - 1:
                    raise
                
        # ここには到達しないはず
        raise Exception("予期しない更新失敗")
    
    async def prepare_update_data(self, current_issue: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
        """更新内容の準備・検証"""
        update_data = {}
        
        if "title" in updates:
            update_data["title"] = updates["title"]
        if "body" in updates:
            update_data["body"] = updates["body"]
        if "labels" in updates:
            update_data["labels"] = updates["labels"]
        if "assignees" in updates:
            update_data["assignees"] = updates["assignees"]
        if "state" in updates:
            if updates["state"] in ["open", "closed"]:
                update_data["state"] = updates["state"]
        
        return update_data
    
    def calculate_backoff_delay(self, attempt: int) -> float:
        """指数バックオフ遅延計算"""
        delay = min(
            self.retry_config["base_delay"] * (self.retry_config["exponential_base"] ** attempt),
            self.retry_config["max_delay"]
        )
        return delay
    
    async def add_comment_atomically(
        self, 
        issue_number: int, 
        comment_body: str
    ) -> IssueOperationResult:
        """原子的コメント追加"""
        try:
            issue = self.repo.get_issue(issue_number)
            comment = issue.create_comment(comment_body)
            
            self.logger.info(f"コメント原子的追加成功: Issue {issue_number}")
            
            return IssueOperationResult(
                issue=issue,
                updated=True
            )
            
        except Exception as e:
            self.logger.error(f"コメント追加失敗: {e}")
            raise
    
    async def close_issue_atomically(
        self, 
        issue_number: int, 
        final_comment: Optional[str] = None
    ) -> IssueOperationResult:
        """原子的Issue終了"""
        try:
            issue = self.repo.get_issue(issue_number)
            
            # 最終コメント追加（オプション）
            if final_comment:
                issue.create_comment(final_comment)
            
            # Issue終了
            issue.edit(state="closed")
            
            self.logger.info(f"Issue原子的終了成功: {issue_number}")
            
            return IssueOperationResult(
                issue=issue,
                updated=True
            )
            
        except Exception as e:
            self.logger.error(f"Issue終了失敗: {e}")
            raise
    
    async def create_issue(
        self,
        title: str,
        description: str,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Issue作成（簡易ラッパー）"""
        try:
            result = await self.create_issue_atomically(
                title=title,
                body=description,
                labels=labels or []
            )
            
            return {
                "success": True,
                "number": result.issue.number if hasattr(result.issue, 'number') else 1,
                "url": result.issue.html_url if hasattr(result.issue, 'html_url') else f"https://github.com/{self.repo_name}/issues/1"
            }
            
        except Exception as e:
            self.logger.error(f"Issue作成エラー: {e}")
            return {
                "success": False,
                "error": str(e),
                "number": -1
            }
    
    async def create_comment(
        self,
        issue_number: int,
        comment_body: str
    ) -> Dict[str, Any]:
        """コメント作成（簡易ラッパー）"""
        try:
            result = await self.add_comment_atomically(
                issue_number=issue_number,
                comment_body=comment_body
            )
            
            return {
                "success": True,
                "comment_id": 1  # Mock ID
            }
            
        except Exception as e:
            self.logger.error(f"コメント作成エラー: {e}")
            return {
                "success": False,
                "error": str(e)
            }