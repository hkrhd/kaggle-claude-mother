"""
ワークフロー・オーケストレーションシステム

エージェント間の安全な引き継ぎ・コンペ分離・ワークフロー制御を提供。
原子的トランザクションによる確実な状態管理を実現。
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from .agent_state_tracker import AgentStateTracker, AgentState, StateChange, StateTransitionTrigger
from ..dependency_trackers.agent_dependency_graph import AgentDependencyGraph


class HandoffPhase(Enum):
    """引き継ぎフェーズ"""
    PREPARATION = "preparation"
    DATA_TRANSFER = "data_transfer"
    TARGET_ACTIVATION = "target_activation"
    SOURCE_FINALIZATION = "source_finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowState(Enum):
    """ワークフロー状態"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"


@dataclass
class HandoffTransaction:
    """エージェント引き継ぎトランザクション"""
    transaction_id: str
    source_agent: str
    target_agent: str
    competition: str
    context: Dict[str, Any]
    phase: HandoffPhase = HandoffPhase.PREPARATION
    start_time: datetime = field(default_factory=datetime.utcnow)
    data_payload: Dict[str, Any] = field(default_factory=dict)
    rollback_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "competition": self.competition,
            "context": self.context,
            "phase": self.phase.value,
            "start_time": self.start_time.isoformat(),
            "data_payload": self.data_payload,
            "rollback_info": self.rollback_info,
            "metadata": self.metadata
        }


@dataclass
class WorkflowDefinition:
    """ワークフロー定義"""
    workflow_id: str
    competition: str
    agent_sequence: List[str]
    state: WorkflowState = WorkflowState.PENDING
    current_agent_index: int = 0
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "competition": self.competition,
            "agent_sequence": self.agent_sequence,
            "state": self.state.value,
            "current_agent_index": self.current_agent_index,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "error_info": self.error_info,
            "metadata": self.metadata
        }


class HandoffFailureError(Exception):
    """引き継ぎ失敗エラー"""
    def __init__(self, message: str, transaction: HandoffTransaction):
        super().__init__(message)
        self.transaction = transaction


class IsolationViolationError(Exception):
    """分離違反エラー"""
    def __init__(self, message: str, operation: Dict[str, Any]):
        super().__init__(message)
        self.operation = operation


class WorkflowOrchestrator:
    """ワークフロー・オーケストレーションシステム"""
    
    def __init__(
        self, 
        state_tracker: AgentStateTracker,
        dependency_graph: AgentDependencyGraph
    ):
        self.state_tracker = state_tracker
        self.dependency_graph = dependency_graph
        self.active_handoffs: Dict[str, HandoffTransaction] = {}
        self.active_workflows: Dict[str, WorkflowDefinition] = {}
        self.competition_contexts: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # 標準エージェント順序
        self.standard_workflow_sequence = [
            "planner", "analyzer", "executor", "monitor", "retrospective"
        ]
    
    def generate_transaction_id(self) -> str:
        """トランザクションID生成"""
        return f"handoff_{uuid.uuid4().hex[:12]}"
    
    def generate_workflow_id(self, competition: str) -> str:
        """ワークフローID生成"""
        return f"workflow_{competition}_{uuid.uuid4().hex[:8]}"
    
    async def orchestrate_safe_agent_handoff(
        self, 
        source_agent: str, 
        target_agent: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """エージェント間の安全な引き継ぎ実行"""
        
        # 引き継ぎトランザクション作成
        handoff_transaction = HandoffTransaction(
            transaction_id=self.generate_transaction_id(),
            source_agent=source_agent,
            target_agent=target_agent,
            competition=context.get("competition", "unknown"),
            context=context
        )
        
        self.active_handoffs[handoff_transaction.transaction_id] = handoff_transaction
        
        try:
            # Phase 1: 引き継ぎ準備・検証
            await self.prepare_handoff(handoff_transaction)
            
            # Phase 2: 原子的データ転送
            await self.execute_atomic_data_transfer(handoff_transaction)
            
            # Phase 3: ターゲットエージェント起動
            await self.activate_target_agent(handoff_transaction)
            
            # Phase 4: ソースエージェント完了処理
            await self.finalize_source_agent(handoff_transaction)
            
            handoff_transaction.phase = HandoffPhase.COMPLETED
            
            self.logger.info(f"エージェント引き継ぎ成功: {source_agent} -> {target_agent}")
            
            return {
                "handoff_success": True,
                "transaction_id": handoff_transaction.transaction_id,
                "completion_time": datetime.utcnow(),
                "source_agent": source_agent,
                "target_agent": target_agent
            }
            
        except Exception as e:
            # 引き継ぎ失敗・ロールバック実行
            await self.rollback_handoff_transaction(handoff_transaction)
            handoff_transaction.phase = HandoffPhase.FAILED
            
            self.logger.error(f"エージェント引き継ぎ失敗: {source_agent} -> {target_agent} - {e}")
            raise HandoffFailureError(f"Agent handoff failed: {str(e)}", handoff_transaction)
        
        finally:
            # アクティブ引き継ぎから削除
            if handoff_transaction.transaction_id in self.active_handoffs:
                del self.active_handoffs[handoff_transaction.transaction_id]
    
    async def prepare_handoff(self, transaction: HandoffTransaction):
        """引き継ぎ準備・検証"""
        
        transaction.phase = HandoffPhase.PREPARATION
        
        # ソースエージェント状態確認
        source_state = self.state_tracker.get_agent_state(transaction.source_agent)
        if source_state not in {AgentState.RUNNING, AgentState.COMPLETED}:
            raise ValueError(f"ソースエージェント状態不正: {source_state}")
        
        # ターゲットエージェント状態確認
        target_state = self.state_tracker.get_agent_state(transaction.target_agent)
        if target_state not in {AgentState.IDLE, AgentState.STARTING}:
            raise ValueError(f"ターゲットエージェント状態不正: {target_state}")
        
        # 依存関係確認
        if not self.dependency_graph.is_agent_ready_to_execute(transaction.target_agent):
            raise ValueError(f"ターゲットエージェント依存関係未満足: {transaction.target_agent}")
        
        # コンペ分離確認
        source_comp = self.get_agent_competition(transaction.source_agent)
        target_comp = self.get_agent_competition(transaction.target_agent)
        if source_comp != target_comp:
            raise IsolationViolationError(
                f"コンペ分離違反: {source_comp} != {target_comp}",
                transaction.to_dict()
            )
        
        # ロールバック情報保存
        transaction.rollback_info = {
            "source_state": source_state.value if source_state else None,
            "target_state": target_state.value if target_state else None,
            "preparation_time": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"引き継ぎ準備完了: {transaction.transaction_id}")
    
    async def execute_atomic_data_transfer(self, transaction: HandoffTransaction):
        """原子的データ転送"""
        
        transaction.phase = HandoffPhase.DATA_TRANSFER
        
        try:
            # ソースエージェントからデータ抽出
            source_data = await self.extract_agent_data(
                transaction.source_agent, 
                transaction.context
            )
            
            # データ変換・検証
            validated_data = await self.validate_transfer_data(
                source_data, 
                transaction.target_agent
            )
            
            # データペイロード設定
            transaction.data_payload = validated_data
            
            # 転送データのバックアップ
            transaction.rollback_info["data_backup"] = source_data
            
            self.logger.info(f"データ転送完了: {transaction.transaction_id}")
            
        except Exception as e:
            self.logger.error(f"データ転送失敗: {transaction.transaction_id} - {e}")
            raise
    
    async def activate_target_agent(self, transaction: HandoffTransaction):
        """ターゲットエージェント起動"""
        
        transaction.phase = HandoffPhase.TARGET_ACTIVATION
        
        try:
            # ターゲットエージェント状態をRUNNINGに遷移
            await self.state_tracker.force_state_transition(
                transaction.target_agent,
                AgentState.RUNNING,
                f"handoff_activation_{transaction.transaction_id}"
            )
            
            # データペイロード注入
            await self.inject_agent_data(
                transaction.target_agent,
                transaction.data_payload
            )
            
            # 依存関係グラフ更新
            self.dependency_graph.update_agent_state(
                transaction.target_agent,
                "running"
            )
            
            self.logger.info(f"ターゲットエージェント起動完了: {transaction.target_agent}")
            
        except Exception as e:
            self.logger.error(f"ターゲットエージェント起動失敗: {e}")
            raise
    
    async def finalize_source_agent(self, transaction: HandoffTransaction):
        """ソースエージェント完了処理"""
        
        transaction.phase = HandoffPhase.SOURCE_FINALIZATION
        
        try:
            # ソースエージェント状態をCOMPLETEDに遷移
            await self.state_tracker.force_state_transition(
                transaction.source_agent,
                AgentState.COMPLETED,
                f"handoff_completion_{transaction.transaction_id}"
            )
            
            # 依存関係グラフ更新
            self.dependency_graph.update_agent_state(
                transaction.source_agent,
                "completed"
            )
            
            # クリーンアップ
            await self.cleanup_source_agent_resources(transaction.source_agent)
            
            self.logger.info(f"ソースエージェント完了処理: {transaction.source_agent}")
            
        except Exception as e:
            self.logger.error(f"ソースエージェント完了処理失敗: {e}")
            raise
    
    async def rollback_handoff_transaction(self, transaction: HandoffTransaction):
        """引き継ぎトランザクションロールバック"""
        
        try:
            rollback_info = transaction.rollback_info
            
            # ターゲットエージェント状態復元
            if "target_state" in rollback_info and rollback_info["target_state"]:
                await self.state_tracker.force_state_transition(
                    transaction.target_agent,
                    AgentState(rollback_info["target_state"]),
                    f"rollback_{transaction.transaction_id}"
                )
            
            # ソースエージェント状態復元
            if "source_state" in rollback_info and rollback_info["source_state"]:
                await self.state_tracker.force_state_transition(
                    transaction.source_agent,
                    AgentState(rollback_info["source_state"]),
                    f"rollback_{transaction.transaction_id}"
                )
            
            # データ復元
            if "data_backup" in rollback_info:
                await self.restore_agent_data(
                    transaction.source_agent,
                    rollback_info["data_backup"]
                )
            
            self.logger.info(f"引き継ぎロールバック完了: {transaction.transaction_id}")
            
        except Exception as e:
            self.logger.error(f"引き継ぎロールバック失敗: {e}")
    
    async def enforce_competition_isolation(
        self, 
        operation_request: Dict[str, Any]
    ) -> Any:
        """コンペ間操作の完全分離保証"""
        
        # 操作対象コンペの特定・検証
        target_competition = self.extract_competition_context(operation_request)
        
        # 他コンペIssueへの誤操作防止チェック
        isolation_violations = await self.check_isolation_violations(
            operation_request, target_competition
        )
        
        if isolation_violations:
            raise IsolationViolationError(
                f"Operation would violate competition isolation: {isolation_violations}",
                operation_request
            )
        
        # 安全な分離環境での操作実行
        async with self.competition_isolation_context(target_competition):
            return await self.execute_isolated_operation(operation_request)
    
    def extract_competition_context(self, operation: Dict[str, Any]) -> str:
        """操作からコンペコンテキスト抽出"""
        
        # 複数の方法でコンペ名抽出試行
        if "competition" in operation:
            return operation["competition"]
        
        if "agent_id" in operation:
            agent_id = operation["agent_id"]
            if agent_id in self.state_tracker.lifecycle_records:
                return self.state_tracker.lifecycle_records[agent_id].competition
        
        if "issue_labels" in operation:
            for label in operation["issue_labels"]:
                if label.startswith("comp:"):
                    return label[5:]  # "comp:" を除去
        
        raise ValueError("コンペコンテキスト特定失敗")
    
    async def check_isolation_violations(
        self, 
        operation: Dict[str, Any], 
        target_competition: str
    ) -> List[str]:
        """分離違反チェック"""
        
        violations = []
        
        # エージェントのコンペ所属確認
        if "agent_id" in operation:
            agent_id = operation["agent_id"]
            agent_comp = self.get_agent_competition(agent_id)
            if agent_comp and agent_comp != target_competition:
                violations.append(f"agent_competition_mismatch:{agent_id}@{agent_comp}")
        
        # Issue操作のラベル確認
        if "issue_operation" in operation:
            issue_op = operation["issue_operation"]
            if "labels" in issue_op:
                for label in issue_op["labels"]:
                    if label.startswith("comp:") and not label.endswith(target_competition):
                        violations.append(f"issue_label_mismatch:{label}")
        
        # ファイルパス操作確認
        if "file_operations" in operation:
            for file_op in operation["file_operations"]:
                if "path" in file_op:
                    path = file_op["path"]
                    if f"competitions/{target_competition}" not in path:
                        violations.append(f"file_path_violation:{path}")
        
        return violations
    
    async def competition_isolation_context(self, competition: str):
        """コンペ分離コンテキスト"""
        
        class CompetitionIsolationContext:
            def __init__(self, comp: str, orchestrator):
                self.competition = comp
                self.orchestrator = orchestrator
                self.original_context = None
            
            async def __aenter__(self):
                # 現在のコンテキスト保存
                self.original_context = self.orchestrator.competition_contexts.get(
                    "current", {}
                )
                
                # 分離コンテキスト設定
                self.orchestrator.competition_contexts["current"] = {
                    "competition": self.competition,
                    "isolation_active": True,
                    "start_time": datetime.utcnow().isoformat()
                }
                
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                # 元のコンテキスト復元
                if self.original_context:
                    self.orchestrator.competition_contexts["current"] = self.original_context
                else:
                    self.orchestrator.competition_contexts.pop("current", None)
        
        return CompetitionIsolationContext(competition, self)
    
    async def execute_isolated_operation(self, operation: Dict[str, Any]) -> Any:
        """分離環境での操作実行"""
        
        # 操作種別による処理分岐
        operation_type = operation.get("type", "unknown")
        
        if operation_type == "agent_state_change":
            return await self.execute_agent_state_operation(operation)
        elif operation_type == "issue_operation":
            return await self.execute_issue_operation(operation)
        elif operation_type == "file_operation":
            return await self.execute_file_operation(operation)
        else:
            self.logger.warning(f"未知の操作種別: {operation_type}")
            return {"status": "unknown_operation", "operation": operation}
    
    async def create_competition_workflow(
        self, 
        competition: str, 
        custom_sequence: Optional[List[str]] = None
    ) -> WorkflowDefinition:
        """コンペ用ワークフロー作成"""
        
        workflow_id = self.generate_workflow_id(competition)
        agent_sequence = custom_sequence or self.standard_workflow_sequence.copy()
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            competition=competition,
            agent_sequence=agent_sequence,
            state=WorkflowState.PENDING,
            metadata={
                "created_at": datetime.utcnow().isoformat(),
                "agent_count": len(agent_sequence)
            }
        )
        
        self.active_workflows[workflow_id] = workflow
        
        self.logger.info(f"ワークフロー作成: {workflow_id} ({competition})")
        return workflow
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """ワークフロー実行"""
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"ワークフローが見つかりません: {workflow_id}")
        
        workflow = self.active_workflows[workflow_id]
        
        try:
            workflow.state = WorkflowState.RUNNING
            workflow.start_time = datetime.utcnow()
            
            # エージェント順次実行
            for i, agent_type in enumerate(workflow.agent_sequence):
                workflow.current_agent_index = i
                
                # エージェント起動・完了待機
                await self.execute_workflow_step(workflow, agent_type)
                
                # 次のエージェントへの引き継ぎ（最後以外）
                if i < len(workflow.agent_sequence) - 1:
                    next_agent_type = workflow.agent_sequence[i + 1]
                    await self.orchestrate_workflow_handoff(
                        workflow, agent_type, next_agent_type
                    )
            
            # ワークフロー完了
            workflow.state = WorkflowState.COMPLETED
            workflow.completion_time = datetime.utcnow()
            
            self.logger.info(f"ワークフロー実行完了: {workflow_id}")
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "completion_time": workflow.completion_time,
                "total_agents": len(workflow.agent_sequence)
            }
            
        except Exception as e:
            workflow.state = WorkflowState.FAILED
            workflow.error_info = {
                "error_message": str(e),
                "failed_at": datetime.utcnow().isoformat(),
                "current_agent_index": workflow.current_agent_index
            }
            
            self.logger.error(f"ワークフロー実行失敗: {workflow_id} - {e}")
            raise
    
    async def execute_workflow_step(self, workflow: WorkflowDefinition, agent_type: str):
        """ワークフローステップ実行"""
        
        agent_id = f"{workflow.competition}:{agent_type}"
        
        # エージェント登録（未登録の場合）
        if agent_id not in self.state_tracker.lifecycle_records:
            self.state_tracker.register_agent(
                agent_id, workflow.competition, agent_type
            )
        
        # エージェント実行開始
        await self.state_tracker.force_state_transition(
            agent_id, AgentState.STARTING, f"workflow_step_{workflow.workflow_id}"
        )
        
        # 完了まで待機（実際の実装では外部システム連携）
        await self.wait_for_agent_completion(agent_id)
    
    async def wait_for_agent_completion(self, agent_id: str, timeout: int = 3600):
        """エージェント完了待機"""
        
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            agent_state = self.state_tracker.get_agent_state(agent_id)
            
            if agent_state == AgentState.COMPLETED:
                return True
            elif agent_state in {AgentState.ERROR, AgentState.TIMEOUT}:
                raise Exception(f"エージェント異常終了: {agent_id} ({agent_state.value})")
            
            await asyncio.sleep(5)  # 5秒間隔でポーリング
        
        raise TimeoutError(f"エージェント完了タイムアウト: {agent_id}")
    
    async def orchestrate_workflow_handoff(
        self, 
        workflow: WorkflowDefinition, 
        current_agent: str, 
        next_agent: str
    ):
        """ワークフロー内引き継ぎ"""
        
        current_agent_id = f"{workflow.competition}:{current_agent}"
        next_agent_id = f"{workflow.competition}:{next_agent}"
        
        context = {
            "competition": workflow.competition,
            "workflow_id": workflow.workflow_id,
            "handoff_type": "workflow_step"
        }
        
        await self.orchestrate_safe_agent_handoff(
            current_agent_id, next_agent_id, context
        )
    
    def get_agent_competition(self, agent_id: str) -> Optional[str]:
        """エージェントのコンペ取得"""
        
        if agent_id in self.state_tracker.lifecycle_records:
            return self.state_tracker.lifecycle_records[agent_id].competition
        
        # agent_idからコンペ抽出試行
        if ":" in agent_id:
            return agent_id.split(":")[0]
        
        return None
    
    async def extract_agent_data(
        self, 
        agent_id: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """エージェントデータ抽出（実装依存）"""
        
        # Mock implementation
        return {
            "agent_id": agent_id,
            "extracted_at": datetime.utcnow().isoformat(),
            "context": context,
            "data_size": 1024  # bytes
        }
    
    async def validate_transfer_data(
        self, 
        data: Dict[str, Any], 
        target_agent: str
    ) -> Dict[str, Any]:
        """転送データ検証"""
        
        # Basic validation
        if not isinstance(data, dict):
            raise ValueError("データ形式不正")
        
        # サイズ制限チェック
        data_size = data.get("data_size", 0)
        if data_size > 10 * 1024 * 1024:  # 10MB
            raise ValueError("データサイズ制限超過")
        
        return data
    
    async def inject_agent_data(
        self, 
        agent_id: str, 
        data: Dict[str, Any]
    ):
        """エージェントデータ注入（実装依存）"""
        
        # Mock implementation
        self.logger.info(f"データ注入: {agent_id} ({data.get('data_size', 0)} bytes)")
    
    async def cleanup_source_agent_resources(self, agent_id: str):
        """ソースエージェントリソースクリーンアップ"""
        
        # Mock implementation
        self.logger.info(f"リソースクリーンアップ: {agent_id}")
    
    async def restore_agent_data(
        self, 
        agent_id: str, 
        backup_data: Dict[str, Any]
    ):
        """エージェントデータ復元"""
        
        # Mock implementation
        self.logger.info(f"データ復元: {agent_id}")
    
    async def execute_agent_state_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """エージェント状態操作実行"""
        
        # Mock implementation
        return {"status": "executed", "operation_type": "agent_state"}
    
    async def execute_issue_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """Issue操作実行"""
        
        # Mock implementation
        return {"status": "executed", "operation_type": "issue"}
    
    async def execute_file_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """ファイル操作実行"""
        
        # Mock implementation
        return {"status": "executed", "operation_type": "file"}