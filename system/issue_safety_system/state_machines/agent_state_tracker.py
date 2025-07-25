"""
エージェント状態追跡システム

各エージェントの生存期間・状態遷移を完全追跡し、
安全な状態管理と異常検出を提供。
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json


class AgentState(Enum):
    """エージェント状態定義"""
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    SUSPENDED = "suspended"


class StateTransitionTrigger(Enum):
    """状態遷移トリガー"""
    USER_REQUEST = "user_request"
    DEPENDENCY_MET = "dependency_met"
    TASK_COMPLETED = "task_completed"
    ERROR_OCCURRED = "error_occurred"
    TIMEOUT_REACHED = "timeout_reached"
    SYSTEM_SUSPEND = "system_suspend"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class StateChange:
    """状態変更イベント"""
    agent_id: str
    from_state: AgentState
    to_state: AgentState
    trigger_event: StateTransitionTrigger
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    issue_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger_event": self.trigger_event.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "issue_number": self.issue_number
        }


@dataclass
class AgentLifecycleRecord:
    """エージェント生存期間記録"""
    agent_id: str
    competition: str
    agent_type: str
    start_time: datetime
    current_state: AgentState = AgentState.IDLE
    state_history: List[StateChange] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    blocking_agents: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)
    timeout_deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "competition": self.competition,
            "agent_type": self.agent_type,
            "start_time": self.start_time.isoformat(),
            "current_state": self.current_state.value,
            "state_history": [change.to_dict() for change in self.state_history],
            "dependencies": self.dependencies,
            "blocking_agents": self.blocking_agents,
            "blocked_by": self.blocked_by,
            "timeout_deadline": self.timeout_deadline.isoformat() if self.timeout_deadline else None,
            "metadata": self.metadata
        }


class InvalidStateTransitionError(Exception):
    """無効な状態遷移エラー"""
    def __init__(self, message: str, agent_id: str, from_state: str, to_state: str):
        super().__init__(message)
        self.agent_id = agent_id
        self.from_state = from_state
        self.to_state = to_state


class AgentStateTracker:
    """エージェント状態追跡システム"""
    
    def __init__(self):
        self.state_transitions = {
            AgentState.IDLE: {
                AgentState.STARTING,
                AgentState.ERROR,
                AgentState.SUSPENDED
            },
            AgentState.STARTING: {
                AgentState.RUNNING,
                AgentState.ERROR,
                AgentState.TIMEOUT,
                AgentState.SUSPENDED
            },
            AgentState.RUNNING: {
                AgentState.COMPLETED,
                AgentState.ERROR,
                AgentState.TIMEOUT,
                AgentState.SUSPENDED
            },
            AgentState.COMPLETED: {
                AgentState.IDLE
            },
            AgentState.ERROR: {
                AgentState.IDLE,
                AgentState.STARTING,  # リトライ可能
                AgentState.SUSPENDED
            },
            AgentState.TIMEOUT: {
                AgentState.IDLE,
                AgentState.STARTING,  # リスタート可能
                AgentState.SUSPENDED
            },
            AgentState.SUSPENDED: {
                AgentState.IDLE,
                AgentState.STARTING
            }
        }
        
        self.lifecycle_records: Dict[str, AgentLifecycleRecord] = {}
        self.state_listeners: Dict[AgentState, List[Callable]] = {}
        self.transition_listeners: List[Callable] = []
        self.logger = logging.getLogger(__name__)
        
        # エージェント別タイムアウト設定
        self.default_timeouts = {
            "planner": 1800,        # 30分
            "analyzer": 7200,       # 2時間
            "executor": 14400,      # 4時間
            "monitor": 28800,       # 8時間
            "retrospective": 3600   # 1時間
        }
    
    def register_agent(
        self, 
        agent_id: str, 
        competition: str, 
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentLifecycleRecord:
        """エージェント登録・追跡開始"""
        
        lifecycle_record = AgentLifecycleRecord(
            agent_id=agent_id,
            competition=competition,
            agent_type=agent_type,
            start_time=datetime.utcnow(),
            current_state=AgentState.IDLE,
            metadata=metadata or {}
        )
        
        # タイムアウト設定
        timeout_seconds = self.default_timeouts.get(agent_type, 3600)
        lifecycle_record.timeout_deadline = (
            lifecycle_record.start_time + timedelta(seconds=timeout_seconds)
        )
        
        self.lifecycle_records[agent_id] = lifecycle_record
        
        self.logger.info(f"エージェント登録: {agent_id} ({agent_type}@{competition})")
        return lifecycle_record
    
    async def track_agent_lifecycle(
        self, 
        agent_id: str, 
        competition_name: str
    ) -> AgentLifecycleRecord:
        """エージェント生存期間の完全追跡"""
        
        if agent_id not in self.lifecycle_records:
            self.logger.error(f"未登録エージェント追跡試行: {agent_id}")
            raise ValueError(f"Agent {agent_id} not registered")
        
        lifecycle_record = self.lifecycle_records[agent_id]
        
        try:
            # 状態変更の監視・記録
            while not self.is_final_state(lifecycle_record.current_state):
                # 状態変更イベント待機
                state_change = await self.wait_for_state_change(agent_id)
                
                # 状態遷移の妥当性検証
                if self.is_valid_transition(
                    lifecycle_record.current_state, 
                    state_change.to_state
                ):
                    # 状態更新・履歴記録
                    await self.apply_state_change(lifecycle_record, state_change)
                    
                    # 依存関係の動的更新
                    await self.update_agent_dependencies(agent_id, state_change)
                    
                    # 監視イベント発火
                    await self.notify_state_listeners(state_change)
                    
                else:
                    # 無効な状態遷移・エラー処理
                    await self.handle_invalid_state_transition(agent_id, state_change)
            
            self.logger.info(f"エージェント生存期間追跡完了: {agent_id}")
            return lifecycle_record
            
        except Exception as e:
            self.logger.error(f"エージェント追跡失敗: {agent_id} - {e}")
            raise
    
    def is_final_state(self, state: AgentState) -> bool:
        """最終状態判定"""
        return state in {AgentState.COMPLETED, AgentState.ERROR, AgentState.TIMEOUT}
    
    async def wait_for_state_change(
        self, 
        agent_id: str, 
        timeout: int = 300
    ) -> StateChange:
        """状態変更イベント待機"""
        
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            # エージェント状態ポーリング
            if agent_id in self.lifecycle_records:
                record = self.lifecycle_records[agent_id]
                
                # タイムアウトチェック
                if (record.timeout_deadline and 
                    datetime.utcnow() > record.timeout_deadline and
                    record.current_state == AgentState.RUNNING):
                    
                    return StateChange(
                        agent_id=agent_id,
                        from_state=record.current_state,
                        to_state=AgentState.TIMEOUT,
                        trigger_event=StateTransitionTrigger.TIMEOUT_REACHED,
                        metadata={"timeout_deadline": record.timeout_deadline.isoformat()}
                    )
                
                # Issue状態変更チェック（外部システム統合時に実装）
                # 現在はモック実装
                await asyncio.sleep(1)
        
        # タイムアウト時のデフォルト状態変更
        return StateChange(
            agent_id=agent_id,
            from_state=self.lifecycle_records[agent_id].current_state,
            to_state=AgentState.TIMEOUT,
            trigger_event=StateTransitionTrigger.TIMEOUT_REACHED,
            metadata={"reason": "wait_timeout"}
        )
    
    def is_valid_transition(
        self, 
        from_state: AgentState, 
        to_state: AgentState
    ) -> bool:
        """状態遷移妥当性検証"""
        return to_state in self.state_transitions.get(from_state, set())
    
    async def apply_state_change(
        self, 
        lifecycle_record: AgentLifecycleRecord, 
        state_change: StateChange
    ):
        """状態変更適用"""
        
        # 状態履歴に追加
        lifecycle_record.state_history.append(state_change)
        
        # 現在状態更新
        old_state = lifecycle_record.current_state
        lifecycle_record.current_state = state_change.to_state
        
        self.logger.info(
            f"状態遷移適用: {state_change.agent_id} "
            f"({old_state.value} -> {state_change.to_state.value})"
        )
        
        # 遷移リスナー通知
        for listener in self.transition_listeners:
            try:
                await listener(state_change)
            except Exception as e:
                self.logger.warning(f"遷移リスナー通知失敗: {e}")
    
    async def update_agent_dependencies(
        self, 
        agent_id: str, 
        state_change: StateChange
    ):
        """依存関係の動的更新"""
        
        if agent_id not in self.lifecycle_records:
            return
        
        record = self.lifecycle_records[agent_id]
        
        # 完了状態の場合、ブロックされているエージェントの依存関係解決
        if state_change.to_state == AgentState.COMPLETED:
            for blocked_agent in record.blocking_agents:
                if blocked_agent in self.lifecycle_records:
                    blocked_record = self.lifecycle_records[blocked_agent]
                    if agent_id in blocked_record.dependencies:
                        blocked_record.dependencies.remove(agent_id)
                        
                        self.logger.info(f"依存関係解決: {blocked_agent} no longer depends on {agent_id}")
    
    async def notify_state_listeners(self, state_change: StateChange):
        """状態監視リスナー通知"""
        
        listeners = self.state_listeners.get(state_change.to_state, [])
        
        for listener in listeners:
            try:
                await listener(state_change)
            except Exception as e:
                self.logger.warning(f"状態リスナー通知失敗: {e}")
    
    async def handle_invalid_state_transition(
        self, 
        agent_id: str, 
        state_change: StateChange
    ):
        """無効な状態遷移処理"""
        
        error_msg = (
            f"無効な状態遷移: {agent_id} "
            f"({state_change.from_state.value} -> {state_change.to_state.value})"
        )
        
        self.logger.error(error_msg)
        
        # エラー状態に遷移
        error_change = StateChange(
            agent_id=agent_id,
            from_state=state_change.from_state,
            to_state=AgentState.ERROR,
            trigger_event=StateTransitionTrigger.ERROR_OCCURRED,
            metadata={
                "original_transition": state_change.to_dict(),
                "error_reason": "invalid_transition"
            }
        )
        
        if agent_id in self.lifecycle_records:
            await self.apply_state_change(
                self.lifecycle_records[agent_id], 
                error_change
            )
        
        raise InvalidStateTransitionError(
            error_msg, agent_id, 
            state_change.from_state.value, 
            state_change.to_state.value
        )
    
    async def manage_inter_agent_dependencies(
        self, 
        agents: List[str]
    ) -> Dict[str, bool]:
        """エージェント間依存関係の動的管理"""
        
        dependency_status = {}
        
        for agent_id in agents:
            if agent_id not in self.lifecycle_records:
                dependency_status[agent_id] = False
                continue
            
            record = self.lifecycle_records[agent_id]
            
            # 依存関係チェック・実行可能性判定
            dependencies_met = await self.check_dependencies_satisfied(record)
            dependency_status[agent_id] = dependencies_met
            
            if dependencies_met:
                # 実行可能・起動許可
                await self.authorize_agent_execution(agent_id)
            else:
                # 依存関係未満足・待機状態継続
                await self.maintain_waiting_state(agent_id)
                
                # 依存先エージェントの進捗監視
                await self.monitor_dependency_progress(agent_id)
        
        return dependency_status
    
    async def check_dependencies_satisfied(
        self, 
        record: AgentLifecycleRecord
    ) -> bool:
        """依存関係満足度チェック"""
        
        for dependency_id in record.dependencies:
            if dependency_id in self.lifecycle_records:
                dep_record = self.lifecycle_records[dependency_id]
                if dep_record.current_state != AgentState.COMPLETED:
                    return False
            else:
                # 依存先エージェントが存在しない
                return False
        
        return True
    
    async def authorize_agent_execution(self, agent_id: str):
        """エージェント実行許可"""
        
        if agent_id in self.lifecycle_records:
            record = self.lifecycle_records[agent_id]
            
            if record.current_state == AgentState.IDLE:
                # STARTING状態に遷移
                state_change = StateChange(
                    agent_id=agent_id,
                    from_state=AgentState.IDLE,
                    to_state=AgentState.STARTING,
                    trigger_event=StateTransitionTrigger.DEPENDENCY_MET,
                    metadata={"authorization": "dependencies_satisfied"}
                )
                
                await self.apply_state_change(record, state_change)
                
                self.logger.info(f"エージェント実行許可: {agent_id}")
    
    async def maintain_waiting_state(self, agent_id: str):
        """待機状態維持"""
        
        if agent_id in self.lifecycle_records:
            record = self.lifecycle_records[agent_id]
            
            # 待機状態のメタデータ更新
            record.metadata["last_dependency_check"] = datetime.utcnow().isoformat()
            record.metadata["waiting_reason"] = "dependencies_not_satisfied"
    
    async def monitor_dependency_progress(self, agent_id: str):
        """依存先エージェント進捗監視"""
        
        if agent_id not in self.lifecycle_records:
            return
        
        record = self.lifecycle_records[agent_id]
        progress_info = {}
        
        for dependency_id in record.dependencies:
            if dependency_id in self.lifecycle_records:
                dep_record = self.lifecycle_records[dependency_id]
                progress_info[dependency_id] = {
                    "state": dep_record.current_state.value,
                    "progress": self.calculate_progress_percentage(dep_record)
                }
        
        record.metadata["dependency_progress"] = progress_info
    
    def calculate_progress_percentage(
        self, 
        record: AgentLifecycleRecord
    ) -> float:
        """進捗パーセンテージ計算"""
        
        state_progress = {
            AgentState.IDLE: 0.0,
            AgentState.STARTING: 10.0,
            AgentState.RUNNING: 50.0,
            AgentState.COMPLETED: 100.0,
            AgentState.ERROR: 0.0,
            AgentState.TIMEOUT: 0.0,
            AgentState.SUSPENDED: 0.0
        }
        
        return state_progress.get(record.current_state, 0.0)
    
    def add_state_listener(
        self, 
        target_state: AgentState, 
        listener: Callable
    ):
        """状態監視リスナー追加"""
        
        if target_state not in self.state_listeners:
            self.state_listeners[target_state] = []
        
        self.state_listeners[target_state].append(listener)
        
        self.logger.info(f"状態リスナー追加: {target_state.value}")
    
    def add_transition_listener(self, listener: Callable):
        """遷移監視リスナー追加"""
        
        self.transition_listeners.append(listener)
        self.logger.info("遷移リスナー追加")
    
    def get_agent_state(self, agent_id: str) -> Optional[AgentState]:
        """エージェント現在状態取得"""
        
        if agent_id in self.lifecycle_records:
            return self.lifecycle_records[agent_id].current_state
        
        return None
    
    def get_competition_agents_by_state(
        self, 
        competition: str, 
        target_state: AgentState
    ) -> List[str]:
        """コンペ別・状態別エージェント取得"""
        
        matching_agents = []
        
        for agent_id, record in self.lifecycle_records.items():
            if (record.competition == competition and 
                record.current_state == target_state):
                matching_agents.append(agent_id)
        
        return matching_agents
    
    def export_lifecycle_records(self) -> Dict[str, Any]:
        """生存期間記録エクスポート"""
        
        return {
            "records": {
                agent_id: record.to_dict() 
                for agent_id, record in self.lifecycle_records.items()
            },
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_agents": len(self.lifecycle_records)
        }
    
    async def force_state_transition(
        self, 
        agent_id: str, 
        target_state: AgentState, 
        reason: str = "manual_intervention"
    ) -> bool:
        """強制状態遷移（緊急用）"""
        
        if agent_id not in self.lifecycle_records:
            return False
        
        record = self.lifecycle_records[agent_id]
        
        state_change = StateChange(
            agent_id=agent_id,
            from_state=record.current_state,
            to_state=target_state,
            trigger_event=StateTransitionTrigger.MANUAL_INTERVENTION,
            metadata={"force_reason": reason}
        )
        
        try:
            await self.apply_state_change(record, state_change)
            
            self.logger.warning(f"強制状態遷移実行: {agent_id} -> {target_state.value} (理由: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"強制状態遷移失敗: {e}")
            return False