"""
分散ロック管理システム

Redis・ファイルロック・GitHub Issue APIを活用した分散ロック機構。
複数エージェントの同時実行における排他制御を提供。
"""

import asyncio
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import json

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# FileLock（テスト環境では模擬）
try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError:
    # テスト環境用の模擬filelock
    class FileLockTimeout(Exception):
        pass
    
    class FileLock:
        def __init__(self, lock_file, timeout=-1):
            self.lock_file = lock_file
            self.timeout = timeout
            self.is_locked = False
        
        def acquire(self, timeout=None, poll_interval=0.05):
            self.is_locked = True
            return self
        
        def release(self, force=False):
            self.is_locked = False
        
        def __enter__(self):
            return self.acquire()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()


@dataclass
class LockInfo:
    """ロック情報"""
    lock_id: str
    owner: str
    resource: str
    acquired_at: datetime
    expires_at: datetime
    lock_type: str
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lock_id": self.lock_id,
            "owner": self.owner,
            "resource": self.resource,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "lock_type": self.lock_type,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LockInfo':
        return cls(
            lock_id=data["lock_id"],
            owner=data["owner"],
            resource=data["resource"],
            acquired_at=datetime.fromisoformat(data["acquired_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            lock_type=data["lock_type"],
            metadata=data.get("metadata", {})
        )


class LockAcquisitionError(Exception):
    """ロック取得失敗エラー"""
    def __init__(self, message: str, lock_info: Optional[LockInfo] = None):
        super().__init__(message)
        self.lock_info = lock_info


class LockManager:
    """分散ロック管理システム"""
    
    def __init__(
        self, 
        redis_url: Optional[str] = None,
        lock_dir: str = "/tmp/kaggle_claude_locks",
        github_client: Optional[Any] = None,
        repo_name: Optional[str] = None
    ):
        self.redis_client = None
        self.lock_dir = lock_dir
        self.github_client = github_client
        self.repo_name = repo_name
        self.logger = logging.getLogger(__name__)
        
        # ロックディレクトリ作成
        os.makedirs(lock_dir, exist_ok=True)
        
        # Redis接続（利用可能な場合）
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.logger.info("Redis分散ロック有効化")
            except Exception as e:
                self.logger.warning(f"Redis接続失敗、ファイルロックにフォールバック: {e}")
        
        # アクティブロック追跡
        self.active_locks: Dict[str, LockInfo] = {}
    
    def generate_lock_id(self, resource: str, owner: str) -> str:
        """ロックID生成"""
        lock_data = f"{resource}:{owner}:{time.time()}"
        return hashlib.sha256(lock_data.encode()).hexdigest()[:16]
    
    @asynccontextmanager
    async def acquire_lock(
        self,
        resource: str,
        owner: str,
        timeout: int = 30,
        ttl: int = 300,
        lock_type: str = "exclusive"
    ) -> AsyncContextManager[LockInfo]:
        """ロック取得・自動解放コンテキストマネージャー"""
        
        lock_id = self.generate_lock_id(resource, owner)
        lock_info = None
        
        try:
            # ロック取得
            lock_info = await self.acquire_lock_internal(
                resource, owner, lock_id, timeout, ttl, lock_type
            )
            
            self.logger.info(f"ロック取得成功: {lock_id} ({resource})")
            yield lock_info
            
        except Exception as e:
            self.logger.error(f"ロック取得失敗: {resource} - {e}")
            raise
        
        finally:
            # ロック自動解放
            if lock_info:
                await self.release_lock(lock_info.lock_id)
    
    async def acquire_lock_internal(
        self,
        resource: str,
        owner: str,
        lock_id: str,
        timeout: int,
        ttl: int,
        lock_type: str
    ) -> LockInfo:
        """内部ロック取得実装"""
        
        start_time = time.time()
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        while time.time() - start_time < timeout:
            try:
                # 複数バックエンドでのロック試行
                success = False
                
                # 1. Redis分散ロック（優先）
                if self.redis_client:
                    success = await self.acquire_redis_lock(
                        resource, owner, lock_id, ttl
                    )
                
                # 2. ファイルロック（フォールバック）
                if not success:
                    success = await self.acquire_file_lock(
                        resource, owner, lock_id, ttl
                    )
                
                # 3. GitHub Issue ロック（最終手段）
                if not success and self.github_client:
                    success = await self.acquire_github_lock(
                        resource, owner, lock_id, ttl
                    )
                
                if success:
                    lock_info = LockInfo(
                        lock_id=lock_id,
                        owner=owner,
                        resource=resource,
                        acquired_at=datetime.utcnow(),
                        expires_at=expires_at,
                        lock_type=lock_type
                    )
                    
                    self.active_locks[lock_id] = lock_info
                    return lock_info
                
            except Exception as e:
                self.logger.warning(f"ロック取得試行失敗: {e}")
            
            # 短時間待機後リトライ
            await asyncio.sleep(0.1)
        
        # タイムアウト
        raise LockAcquisitionError(f"ロック取得タイムアウト: {resource}")
    
    async def acquire_redis_lock(
        self, 
        resource: str, 
        owner: str, 
        lock_id: str, 
        ttl: int
    ) -> bool:
        """Redis分散ロック取得"""
        if not self.redis_client:
            return False
        
        try:
            lock_key = f"kaggle_lock:{resource}"
            lock_value = f"{owner}:{lock_id}"
            
            # Redis SET NX EX でロック取得
            result = await self.redis_client.set(
                lock_key, 
                lock_value, 
                nx=True,  # キーが存在しない場合のみ設定
                ex=ttl    # TTL設定
            )
            
            return result is True
            
        except Exception as e:
            self.logger.error(f"Redis ロック取得失敗: {e}")
            return False
    
    async def acquire_file_lock(
        self, 
        resource: str, 
        owner: str, 
        lock_id: str, 
        ttl: int
    ) -> bool:
        """ファイルロック取得"""
        try:
            lock_file = os.path.join(
                self.lock_dir, 
                f"{hashlib.md5(resource.encode()).hexdigest()}.lock"
            )
            
            file_lock = FileLock(lock_file, timeout=0.1)
            
            # ノンブロッキングロック試行
            file_lock.acquire(blocking=False)
            
            # ロック情報をファイルに記録
            lock_info_file = lock_file + ".info"
            lock_data = {
                "owner": owner,
                "lock_id": lock_id,
                "acquired_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
                "resource": resource
            }
            
            with open(lock_info_file, 'w') as f:
                json.dump(lock_data, f)
            
            # ロック解放処理を遅延実行
            asyncio.create_task(self.schedule_file_lock_release(lock_file, ttl))
            
            return True
            
        except FileLockTimeout:
            return False
        except Exception as e:
            self.logger.error(f"ファイルロック取得失敗: {e}")
            return False
    
    async def schedule_file_lock_release(self, lock_file: str, ttl: int):
        """ファイルロック自動解放スケジュール"""
        await asyncio.sleep(ttl)
        try:
            file_lock = FileLock(lock_file)
            if file_lock.is_locked:
                file_lock.release()
            
            # 情報ファイル削除
            lock_info_file = lock_file + ".info"
            if os.path.exists(lock_info_file):
                os.remove(lock_info_file)
                
        except Exception as e:
            self.logger.warning(f"ファイルロック自動解放失敗: {e}")
    
    async def acquire_github_lock(
        self, 
        resource: str, 
        owner: str, 
        lock_id: str, 
        ttl: int
    ) -> bool:
        """GitHub Issue ロック取得"""
        if not self.github_client or not self.repo_name:
            return False
        
        try:
            repo = self.github_client.get_repo(self.repo_name)
            
            # ロック用Issue検索
            lock_title = f"[LOCK] {resource}"
            existing_issues = repo.get_issues(
                state="open",
                labels=["system:lock"]
            )
            
            # 既存ロックチェック
            for issue in existing_issues:
                if lock_title in issue.title:
                    # 期限切れロックの確認
                    if await self.is_github_lock_expired(issue):
                        # 期限切れロック削除
                        issue.edit(state="closed")
                        continue
                    else:
                        # アクティブロック存在
                        return False
            
            # 新しいロック Issue 作成
            lock_body = f"""
**Lock Information**
- Resource: `{resource}`
- Owner: `{owner}`
- Lock ID: `{lock_id}`
- Acquired At: `{datetime.utcnow().isoformat()}`
- Expires At: `{(datetime.utcnow() + timedelta(seconds=ttl)).isoformat()}`

**Auto-Release**: This lock will be automatically released after {ttl} seconds.
"""
            
            issue = repo.create_issue(
                title=lock_title,
                body=lock_body,
                labels=["system:lock", f"owner:{owner}"]
            )
            
            # 自動解放スケジュール
            asyncio.create_task(
                self.schedule_github_lock_release(issue.number, ttl)
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"GitHub ロック取得失敗: {e}")
            return False
    
    async def is_github_lock_expired(self, issue: Any) -> bool:
        """GitHub ロック期限切れ判定"""
        try:
            # Issue本文から期限情報抽出
            body = issue.body or ""
            for line in body.split('\n'):
                if "Expires At:" in line:
                    expires_str = line.split('`')[1]
                    expires_at = datetime.fromisoformat(expires_str)
                    return datetime.utcnow() > expires_at
            
            # 期限情報がない場合は期限切れと判定
            return True
            
        except Exception:
            return True
    
    async def schedule_github_lock_release(self, issue_number: int, ttl: int):
        """GitHub ロック自動解放スケジュール"""
        await asyncio.sleep(ttl)
        try:
            if self.github_client and self.repo_name:
                repo = self.github_client.get_repo(self.repo_name)
                issue = repo.get_issue(issue_number)
                issue.edit(state="closed")
                issue.create_comment("🔓 Lock automatically released due to TTL expiration.")
                
        except Exception as e:
            self.logger.warning(f"GitHub ロック自動解放失敗: {e}")
    
    async def release_lock(self, lock_id: str) -> bool:
        """ロック解放"""
        if lock_id not in self.active_locks:
            self.logger.warning(f"未知のロック解放試行: {lock_id}")
            return False
        
        lock_info = self.active_locks[lock_id]
        success = False
        
        try:
            # 各バックエンドからロック解放
            if self.redis_client:
                await self.release_redis_lock(lock_info.resource, lock_info.owner)
            
            await self.release_file_lock(lock_info.resource)
            
            if self.github_client:
                await self.release_github_lock(lock_info.resource)
            
            # アクティブロックから除去
            del self.active_locks[lock_id]
            success = True
            
            self.logger.info(f"ロック解放成功: {lock_id}")
            
        except Exception as e:
            self.logger.error(f"ロック解放失敗: {lock_id} - {e}")
        
        return success
    
    async def release_redis_lock(self, resource: str, owner: str) -> bool:
        """Redis ロック解放"""
        if not self.redis_client:
            return False
        
        try:
            lock_key = f"kaggle_lock:{resource}"
            # 所有者確認後削除
            current_value = await self.redis_client.get(lock_key)
            if current_value and owner in current_value.decode():
                await self.redis_client.delete(lock_key)
                return True
            
        except Exception as e:
            self.logger.error(f"Redis ロック解放失敗: {e}")
        
        return False
    
    async def release_file_lock(self, resource: str) -> bool:
        """ファイルロック解放"""
        try:
            lock_file = os.path.join(
                self.lock_dir,
                f"{hashlib.md5(resource.encode()).hexdigest()}.lock"
            )
            
            if os.path.exists(lock_file):
                file_lock = FileLock(lock_file)
                if file_lock.is_locked:
                    file_lock.release()
                
                # 情報ファイル削除
                lock_info_file = lock_file + ".info"
                if os.path.exists(lock_info_file):
                    os.remove(lock_info_file)
                
                return True
            
        except Exception as e:
            self.logger.error(f"ファイルロック解放失敗: {e}")
        
        return False
    
    async def release_github_lock(self, resource: str) -> bool:
        """GitHub ロック解放"""
        if not self.github_client or not self.repo_name:
            return False
        
        try:
            repo = self.github_client.get_repo(self.repo_name)
            lock_title = f"[LOCK] {resource}"
            
            issues = repo.get_issues(
                state="open",
                labels=["system:lock"]
            )
            
            for issue in issues:
                if lock_title in issue.title:
                    issue.edit(state="closed")
                    issue.create_comment("🔓 Lock manually released.")
                    return True
            
        except Exception as e:
            self.logger.error(f"GitHub ロック解放失敗: {e}")
        
        return False
    
    async def list_active_locks(self) -> List[LockInfo]:
        """アクティブロック一覧取得"""
        return list(self.active_locks.values())
    
    async def cleanup_expired_locks(self):
        """期限切れロック cleanup"""
        current_time = datetime.utcnow()
        expired_locks = []
        
        for lock_id, lock_info in self.active_locks.items():
            if current_time > lock_info.expires_at:
                expired_locks.append(lock_id)
        
        for lock_id in expired_locks:
            await self.release_lock(lock_id)
            self.logger.info(f"期限切れロック cleanup: {lock_id}")
    
    async def is_resource_locked(self, resource: str) -> bool:
        """リソースロック状態確認"""
        for lock_info in self.active_locks.values():
            if (lock_info.resource == resource and 
                datetime.utcnow() < lock_info.expires_at):
                return True
        
        return False