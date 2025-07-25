"""
指数バックオフ・リトライメカニズム

GitHub API操作の信頼性を向上させる高度なリトライシステム。
指数バックオフ・ジッター・条件別リトライを統合提供。
"""

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, TypeVar, Awaitable
from dataclasses import dataclass
from enum import Enum
import logging

# GitHubライブラリ（テスト環境では模擬）
try:
    from github.GithubException import (
        GithubException, 
        RateLimitExceededException,
        UnknownObjectException,
        BadCredentialsException
    )
except ImportError:
    # テスト環境用の模擬クラス
    class GithubException(Exception):
        def __init__(self, status=None, data=None):
            self.status = status
            super().__init__(str(data) if data else "GitHub API error")
    
    class RateLimitExceededException(GithubException):
        def __init__(self):
            super().__init__(status=403, data="Rate limit exceeded")
    
    class UnknownObjectException(GithubException):
        def __init__(self):
            super().__init__(status=404, data="Unknown object")
    
    class BadCredentialsException(GithubException):
        def __init__(self, status=401, data=None):
            super().__init__(status=status, data=data or "Bad credentials")


T = TypeVar('T')


class RetryStrategy(Enum):
    """リトライ戦略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    JITTERED_EXPONENTIAL = "jittered_exponential"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    SERVER_ERROR = "server_error"  # 5xx
    CLIENT_ERROR = "client_error"  # 4xx
    AUTHENTICATION = "authentication"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """リトライ設定"""
    max_attempts: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.JITTERED_EXPONENTIAL
    
    # エラー別設定
    rate_limit_max_delay: float = 300.0  # 5分
    network_timeout: float = 30.0
    
    # リトライ対象エラー
    retryable_exceptions: List[type] = None
    non_retryable_exceptions: List[type] = None

    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [
                RateLimitExceededException,
                ConnectionError,
                TimeoutError,
                Exception  # 5xx server errors
            ]
        
        if self.non_retryable_exceptions is None:
            self.non_retryable_exceptions = [
                BadCredentialsException,
                UnknownObjectException,
                ValueError,
                TypeError
            ]


@dataclass
class RetryAttempt:
    """リトライ試行記録"""
    attempt_number: int
    timestamp: datetime
    error: Optional[Exception]
    delay_seconds: float
    error_category: ErrorCategory
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp.isoformat(),
            "error": str(self.error) if self.error else None,
            "error_type": type(self.error).__name__ if self.error else None,
            "delay_seconds": self.delay_seconds,
            "error_category": self.error_category.value,
            "success": self.success
        }


@dataclass
class RetryResult:
    """リトライ結果"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    attempts: List[RetryAttempt] = None
    total_time: float = 0.0
    operation_name: str = ""

    def __post_init__(self):
        if self.attempts is None:
            self.attempts = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "error": str(self.error) if self.error else None,
            "error_type": type(self.error).__name__ if self.error else None,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "total_attempts": len(self.attempts),
            "total_time": self.total_time,
            "operation_name": self.operation_name
        }


class RetryExhaustedException(Exception):
    """リトライ上限到達エラー"""
    def __init__(self, message: str, retry_result: RetryResult):
        super().__init__(message)
        self.retry_result = retry_result


class RetryMechanism:
    """指数バックオフ・リトライメカニズム"""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.retry_statistics: Dict[str, List[RetryResult]] = {}
    
    async def execute_with_retry(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str = "unknown_operation",
        custom_config: Optional[RetryConfig] = None
    ) -> T:
        """リトライ付き操作実行"""
        
        config = custom_config or self.config
        start_time = time.time()
        attempts = []
        
        for attempt_num in range(1, config.max_attempts + 1):
            attempt_start = datetime.utcnow()
            
            try:
                # 操作実行
                result = await operation()
                
                # 成功記録
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=attempt_start,
                    error=None,
                    delay_seconds=0.0,
                    error_category=ErrorCategory.UNKNOWN,
                    success=True
                )
                attempts.append(attempt)
                
                # 統計記録
                retry_result = RetryResult(
                    success=True,
                    result=result,
                    attempts=attempts,
                    total_time=time.time() - start_time,
                    operation_name=operation_name
                )
                self.record_retry_statistics(operation_name, retry_result)
                
                if attempt_num > 1:
                    self.logger.info(
                        f"リトライ成功: {operation_name} (試行{attempt_num}/{config.max_attempts})"
                    )
                
                return result
                
            except Exception as e:
                # エラー分類
                error_category = self.classify_error(e)
                
                # リトライ可能性判定
                if not self.is_retryable_error(e, config):
                    # 非リトライエラー
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        timestamp=attempt_start,
                        error=e,
                        delay_seconds=0.0,
                        error_category=error_category,
                        success=False
                    )
                    attempts.append(attempt)
                    
                    retry_result = RetryResult(
                        success=False,
                        error=e,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        operation_name=operation_name
                    )
                    self.record_retry_statistics(operation_name, retry_result)
                    
                    self.logger.error(f"非リトライエラー: {operation_name} - {e}")
                    raise e
                
                # 最後の試行の場合
                if attempt_num >= config.max_attempts:
                    attempt = RetryAttempt(
                        attempt_number=attempt_num,
                        timestamp=attempt_start,
                        error=e,
                        delay_seconds=0.0,
                        error_category=error_category,
                        success=False
                    )
                    attempts.append(attempt)
                    
                    retry_result = RetryResult(
                        success=False,
                        error=e,
                        attempts=attempts,
                        total_time=time.time() - start_time,
                        operation_name=operation_name
                    )
                    self.record_retry_statistics(operation_name, retry_result)
                    
                    self.logger.error(
                        f"リトライ上限到達: {operation_name} (試行{config.max_attempts}) - {e}"
                    )
                    raise RetryExhaustedException(
                        f"Max retry attempts ({config.max_attempts}) exceeded for {operation_name}",
                        retry_result
                    )
                
                # 遅延時間計算
                delay = self.calculate_delay(
                    attempt_num, error_category, config
                )
                
                # 試行記録
                attempt = RetryAttempt(
                    attempt_number=attempt_num,
                    timestamp=attempt_start,
                    error=e,
                    delay_seconds=delay,
                    error_category=error_category,
                    success=False
                )
                attempts.append(attempt)
                
                self.logger.warning(
                    f"リトライ実行: {operation_name} (試行{attempt_num}/{config.max_attempts}) "
                    f"- {e} - {delay:.1f}秒後に再試行"
                )
                
                # 遅延実行
                await asyncio.sleep(delay)
        
        # ここには到達しないはず
        raise Exception(f"Unexpected retry loop exit for {operation_name}")
    
    def classify_error(self, error: Exception) -> ErrorCategory:
        """エラー分類"""
        
        if isinstance(error, RateLimitExceededException):
            return ErrorCategory.RATE_LIMIT
        elif isinstance(error, BadCredentialsException):
            return ErrorCategory.AUTHENTICATION
        elif isinstance(error, UnknownObjectException):
            return ErrorCategory.NOT_FOUND
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, GithubException):
            # HTTPステータスコードによる分類
            if hasattr(error, 'status'):
                if 400 <= error.status < 500:
                    return ErrorCategory.CLIENT_ERROR
                elif 500 <= error.status < 600:
                    return ErrorCategory.SERVER_ERROR
        
        return ErrorCategory.UNKNOWN
    
    def is_retryable_error(self, error: Exception, config: RetryConfig) -> bool:
        """リトライ可能エラー判定"""
        
        # 明示的非リトライエラー
        for non_retryable in config.non_retryable_exceptions:
            if isinstance(error, non_retryable):
                return False
        
        # 明示的リトライエラー
        for retryable in config.retryable_exceptions:
            if isinstance(error, retryable):
                return True
        
        # GitHub例外の詳細判定
        if isinstance(error, GithubException):
            if hasattr(error, 'status'):
                # 4xx (クライアントエラー) は通常リトライしない
                if 400 <= error.status < 500:
                    # ただし、レート制限は例外
                    return error.status == 429
                # 5xx (サーバーエラー) はリトライ対象
                elif 500 <= error.status < 600:
                    return True
        
        # ネットワークエラーはリトライ対象
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        # デフォルトは非リトライ
        return False
    
    def calculate_delay(
        self,
        attempt_number: int,
        error_category: ErrorCategory,
        config: RetryConfig
    ) -> float:
        """遅延時間計算"""
        
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.exponential_base ** (attempt_number - 1))
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt_number
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.JITTERED_EXPONENTIAL:
            base_delay = config.base_delay * (config.exponential_base ** (attempt_number - 1))
            # ジッター追加（±25%のランダム）
            jitter_factor = 0.75 + (random.random() * 0.5)  # 0.75 to 1.25
            delay = base_delay * jitter_factor
        else:
            delay = config.base_delay
        
        # エラーカテゴリ別調整
        if error_category == ErrorCategory.RATE_LIMIT:
            # レート制限の場合は最大遅延を拡張
            delay = min(delay, config.rate_limit_max_delay)
        else:
            delay = min(delay, config.max_delay)
        
        # ジッター適用（設定有効時）
        if config.jitter and config.strategy != RetryStrategy.JITTERED_EXPONENTIAL:
            jitter = random.uniform(-0.1, 0.1) * delay
            delay += jitter
        
        return max(delay, 0.1)  # 最小0.1秒
    
    def record_retry_statistics(
        self,
        operation_name: str,
        retry_result: RetryResult
    ):
        """リトライ統計記録"""
        
        if operation_name not in self.retry_statistics:
            self.retry_statistics[operation_name] = []
        
        self.retry_statistics[operation_name].append(retry_result)
        
        # 統計データの制限（最新1000件まで）
        if len(self.retry_statistics[operation_name]) > 1000:
            self.retry_statistics[operation_name] = self.retry_statistics[operation_name][-1000:]
    
    def get_retry_statistics(
        self,
        operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """リトライ統計取得"""
        
        if operation_name:
            # 特定操作の統計
            if operation_name not in self.retry_statistics:
                return {"operation_name": operation_name, "no_data": True}
            
            results = self.retry_statistics[operation_name]
        else:
            # 全操作の統計
            results = []
            for operation_results in self.retry_statistics.values():
                results.extend(operation_results)
        
        if not results:
            return {"no_data": True}
        
        # 統計計算
        total_operations = len(results)
        successful_operations = sum(1 for r in results if r.success)
        failed_operations = total_operations - successful_operations
        
        total_attempts = sum(len(r.attempts) for r in results)
        avg_attempts = total_attempts / total_operations if total_operations > 0 else 0
        
        total_time = sum(r.total_time for r in results)
        avg_time = total_time / total_operations if total_operations > 0 else 0
        
        # エラーカテゴリ別統計
        error_categories = {}
        for result in results:
            for attempt in result.attempts:
                if not attempt.success:
                    category = attempt.error_category.value
                    error_categories[category] = error_categories.get(category, 0) + 1
        
        # 成功率計算
        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0
        
        statistics = {
            "operation_name": operation_name or "all_operations",
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate_percent": round(success_rate, 2),
            "average_attempts": round(avg_attempts, 2),
            "total_attempts": total_attempts,
            "average_time_seconds": round(avg_time, 2),
            "total_time_seconds": round(total_time, 2),
            "error_categories": error_categories
        }
        
        return statistics
    
    def create_rate_limit_config(self) -> RetryConfig:
        """レート制限専用設定作成"""
        
        return RetryConfig(
            max_attempts=3,
            base_delay=60.0,  # 1分
            max_delay=300.0,  # 5分
            exponential_base=1.5,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            rate_limit_max_delay=600.0,  # 10分
            retryable_exceptions=[RateLimitExceededException]
        )
    
    def create_network_error_config(self) -> RetryConfig:
        """ネットワークエラー専用設定作成"""
        
        return RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=30.0,
            exponential_base=2.0,
            strategy=RetryStrategy.JITTERED_EXPONENTIAL,
            network_timeout=30.0,
            retryable_exceptions=[ConnectionError, TimeoutError]
        )
    
    def create_conservative_config(self) -> RetryConfig:
        """保守的設定作成（重要操作用）"""
        
        return RetryConfig(
            max_attempts=3,
            base_delay=5.0,
            max_delay=60.0,
            exponential_base=2.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            retryable_exceptions=[
                RateLimitExceededException,
                ConnectionError,
                TimeoutError
            ]
        )
    
    def create_aggressive_config(self) -> RetryConfig:
        """積極的設定作成（高頻度操作用）"""
        
        return RetryConfig(
            max_attempts=7,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=1.8,
            strategy=RetryStrategy.JITTERED_EXPONENTIAL,
            retryable_exceptions=[
                RateLimitExceededException,
                ConnectionError,
                TimeoutError,
                Exception  # Catch-all for server errors
            ]
        )
    
    def reset_statistics(self, operation_name: Optional[str] = None):
        """統計リセット"""
        
        if operation_name:
            self.retry_statistics.pop(operation_name, None)
        else:
            self.retry_statistics.clear()
        
        self.logger.info(f"リトライ統計リセット: {operation_name or 'all'}")
    
    async def test_retry_mechanism(self) -> Dict[str, Any]:
        """リトライメカニズムテスト"""
        
        test_results = {}
        
        # テスト1: 成功操作
        async def success_operation():
            return "success"
        
        try:
            result = await self.execute_with_retry(
                success_operation, "test_success"
            )
            test_results["success_test"] = {"passed": True, "result": result}
        except Exception as e:
            test_results["success_test"] = {"passed": False, "error": str(e)}
        
        # テスト2: リトライ成功操作
        retry_count = 0
        async def retry_success_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise ConnectionError("Temporary network error")
            return "retry_success"
        
        try:
            result = await self.execute_with_retry(
                retry_success_operation, "test_retry_success"
            )
            test_results["retry_success_test"] = {
                "passed": True, 
                "result": result, 
                "attempts": retry_count
            }
        except Exception as e:
            test_results["retry_success_test"] = {"passed": False, "error": str(e)}
        
        # テスト3: 非リトライエラー
        async def non_retryable_operation():
            raise BadCredentialsException(401, "Invalid credentials")
        
        try:
            await self.execute_with_retry(
                non_retryable_operation, "test_non_retryable"
            )
            test_results["non_retryable_test"] = {"passed": False, "unexpected_success": True}
        except BadCredentialsException:
            test_results["non_retryable_test"] = {"passed": True, "expected_error": True}
        except Exception as e:
            test_results["non_retryable_test"] = {"passed": False, "error": str(e)}
        
        return {
            "test_results": test_results,
            "statistics": self.get_retry_statistics(),
            "test_completed_at": datetime.utcnow().isoformat()
        }