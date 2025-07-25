"""
Issue安全連携システム

GitHub Issue APIによる安全なエージェント間連携システム。
原子性操作・競合回避・デッドロック防止を統合提供。
"""

__version__ = "0.1.0"

# コアコンポーネント
from .concurrency_control.atomic_operations import AtomicIssueOperations, IssueOperationResult
from .concurrency_control.deadlock_prevention import DeadlockPreventionSystem, DeadlockDetectionResult
from .concurrency_control.lock_manager import LockManager, LockInfo

# 依存関係管理
from .dependency_trackers.agent_dependency_graph import AgentDependencyGraph, AgentNode, DependencyEdge
from .state_machines.agent_state_tracker import AgentStateTracker, AgentLifecycleRecord, StateChange
from .state_machines.workflow_orchestrator import WorkflowOrchestrator, HandoffTransaction, WorkflowDefinition

# ユーティリティ
from .utils.github_api_wrapper import GitHubApiWrapper, RateLimitInfo
from .utils.retry_mechanism import RetryMechanism, RetryConfig
from .utils.audit_logger import AuditLogger, AuditEvent


__all__ = [
    # 原子性操作
    "AtomicIssueOperations", 
    "IssueOperationResult",
    
    # デッドロック防止
    "DeadlockPreventionSystem", 
    "DeadlockDetectionResult",
    
    # ロック管理
    "LockManager", 
    "LockInfo",
    
    # 依存関係管理
    "AgentDependencyGraph", 
    "AgentNode", 
    "DependencyEdge",
    
    # 状態管理
    "AgentStateTracker", 
    "AgentLifecycleRecord", 
    "StateChange",
    
    # ワークフロー
    "WorkflowOrchestrator", 
    "HandoffTransaction", 
    "WorkflowDefinition",
    
    # API ラッパー
    "GitHubApiWrapper", 
    "RateLimitInfo",
    
    # リトライ機構
    "RetryMechanism", 
    "RetryConfig",
    
    # 監査ログ
    "AuditLogger", 
    "AuditEvent"
]