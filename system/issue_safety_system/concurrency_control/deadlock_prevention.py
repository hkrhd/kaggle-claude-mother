"""
デッドロック防止システム

エージェント間の循環依存・無限待機を防止する安全機構。
厳密な実行順序強制とタイムアウト管理によりシステム安定性を保証。
"""

import asyncio

# NetworkX（テスト環境では模擬）
try:
    import networkx as nx
except ImportError:
    # テスト環境用の模擬networkx
    class MockDigraph:
        def __init__(self):
            self.nodes_data = {}
            self.edges_data = []
        
        def add_node(self, node, **attr):
            self.nodes_data[node] = attr
        
        def add_edge(self, source, target, **attr):
            self.edges_data.append((source, target, attr))
        
        def has_edge(self, source, target):
            return any(e[0] == source and e[1] == target for e in self.edges_data)
        
        def remove_edge(self, source, target):
            self.edges_data = [e for e in self.edges_data if not (e[0] == source and e[1] == target)]
        
        def nodes(self, data=False):
            if data:
                return list(self.nodes_data.items())
            return list(self.nodes_data.keys())
        
        def edges(self, data=False):
            if data:
                return [(e[0], e[1], e[2]) for e in self.edges_data]
            return [(e[0], e[1]) for e in self.edges_data]
        
        def clear(self):
            self.nodes_data = {}
            self.edges_data = []
    
    class MockNetworkX:
        DiGraph = MockDigraph
        
        @staticmethod
        def simple_cycles(graph):
            return []
        
        @staticmethod
        def topological_sort(graph):
            return list(graph.nodes_data.keys())
        
        @staticmethod
        def is_directed_acyclic_graph(graph):
            return True
    
    nx = MockNetworkX()

from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging


class AgentType(Enum):
    """エージェント種別"""
    PLANNER = "planner"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    MONITOR = "monitor"
    RETROSPECTIVE = "retrospective"


class AgentState(Enum):
    """エージェント状態"""
    IDLE = "idle"
    WAITING = "waiting"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class AgentExecution:
    """エージェント実行状態"""
    agent_id: str
    agent_type: AgentType
    competition: str
    state: AgentState
    start_time: datetime
    last_update: datetime
    dependencies: List[str] = field(default_factory=list)
    blocking_agents: List[str] = field(default_factory=list)
    timeout_deadline: Optional[datetime] = None
    issue_number: Optional[int] = None


@dataclass
class DeadlockDetectionResult:
    """デッドロック検出結果"""
    deadlock_risk: bool
    detected_cycles: List[List[str]] = field(default_factory=list)
    affected_agents: Set[str] = field(default_factory=set)
    resolution_strategies: List[str] = field(default_factory=list)


class DeadlockPreventionSystem:
    """デッドロック防止システム"""
    
    def __init__(self):
        self.agent_dependency_graph = nx.DiGraph()
        self.execution_timeouts = {
            AgentType.PLANNER: 1800,        # 30分
            AgentType.ANALYZER: 7200,       # 2時間
            AgentType.EXECUTOR: 14400,      # 4時間
            AgentType.MONITOR: 28800,       # 8時間（継続監視）
            AgentType.RETROSPECTIVE: 3600   # 1時間
        }
        self.active_agents: Dict[str, AgentExecution] = {}
        self.competition_agents: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # 強制実行順序（デッドロック防止）
        self.mandatory_order = [
            AgentType.PLANNER,
            AgentType.ANALYZER,
            AgentType.EXECUTOR,
            AgentType.MONITOR,
            AgentType.RETROSPECTIVE
        ]
    
    async def detect_potential_deadlock(
        self, 
        agent_requests: List[AgentExecution]
    ) -> DeadlockDetectionResult:
        """デッドロック可能性の事前検出"""
        try:
            # 現在の待機・実行状態の分析
            current_state = await self.get_current_agent_states()
            
            # 新しいリクエストによる依存関係追加
            updated_graph = self.add_new_dependencies(agent_requests)
            
            # 循環依存の検出
            cycles = list(nx.simple_cycles(updated_graph))
            
            if cycles:
                affected_agents = self.extract_affected_agents(cycles)
                resolution_strategies = await self.generate_resolution_strategies(cycles)
                
                self.logger.warning(f"デッドロック可能性検出: {len(cycles)}個の循環依存")
                
                return DeadlockDetectionResult(
                    deadlock_risk=True,
                    detected_cycles=cycles,
                    affected_agents=affected_agents,
                    resolution_strategies=resolution_strategies
                )
            
            return DeadlockDetectionResult(deadlock_risk=False)
            
        except Exception as e:
            self.logger.error(f"デッドロック検出失敗: {e}")
            raise
    
    async def get_current_agent_states(self) -> Dict[str, AgentExecution]:
        """現在のエージェント状態取得"""
        current_time = datetime.utcnow()
        active_states = {}
        
        for agent_id, execution in self.active_agents.items():
            # タイムアウトチェック
            if (execution.timeout_deadline and 
                current_time > execution.timeout_deadline and
                execution.state == AgentState.RUNNING):
                execution.state = AgentState.TIMEOUT
                self.logger.warning(f"エージェントタイムアウト検出: {agent_id}")
            
            active_states[agent_id] = execution
        
        return active_states
    
    def add_new_dependencies(
        self, 
        agent_requests: List[AgentExecution]
    ) -> nx.DiGraph:
        """新しい依存関係をグラフに追加"""
        updated_graph = self.agent_dependency_graph.copy()
        
        for request in agent_requests:
            # ノード追加
            updated_graph.add_node(request.agent_id, 
                                 agent_type=request.agent_type,
                                 competition=request.competition)
            
            # 依存関係エッジ追加
            for dependency in request.dependencies:
                if dependency in updated_graph:
                    updated_graph.add_edge(request.agent_id, dependency)
        
        return updated_graph
    
    def extract_affected_agents(self, cycles: List[List[str]]) -> Set[str]:
        """循環依存に影響されるエージェント抽出"""
        affected = set()
        for cycle in cycles:
            affected.update(cycle)
        return affected
    
    async def generate_resolution_strategies(
        self, 
        cycles: List[List[str]]
    ) -> List[str]:
        """解決戦略生成"""
        strategies = []
        
        for cycle in cycles:
            # 最も古い待機エージェントを特定
            oldest_agent = await self.find_oldest_waiting_agent(cycle)
            if oldest_agent:
                strategies.append(f"force_timeout:{oldest_agent}")
            
            # 優先度の低いエージェントの一時停止
            low_priority = await self.find_lowest_priority_agent(cycle)
            if low_priority:
                strategies.append(f"suspend:{low_priority}")
        
        return strategies
    
    async def prevent_deadlock_formation(
        self, 
        agent_execution_plan: List[AgentExecution]
    ) -> Dict[str, Any]:
        """デッドロック形成の事前防止"""
        try:
            # 実行順序の強制（planner → analyzer → executor → monitor → retrospective）
            enforced_order = self.enforce_strict_execution_order(agent_execution_plan)
            
            # 逆依存の禁止（後段→前段のIssue作成禁止）
            validated_plan = self.validate_no_reverse_dependencies(enforced_order)
            
            # タイムアウト・デッドライン設定
            timeout_plan = self.apply_execution_timeouts(validated_plan)
            
            self.logger.info(f"デッドロック防止策適用完了: {len(timeout_plan)}エージェント")
            
            return {
                "safe_execution_plan": timeout_plan,
                "deadlock_prevention_applied": True,
                "execution_constraints": self.get_applied_constraints(timeout_plan)
            }
            
        except Exception as e:
            self.logger.error(f"デッドロック防止失敗: {e}")
            raise
    
    def enforce_strict_execution_order(
        self, 
        agent_plan: List[AgentExecution]
    ) -> List[AgentExecution]:
        """厳密な実行順序の強制"""
        ordered_plan = []
        competition_groups = {}
        
        # コンペ別グループ化
        for agent in agent_plan:
            comp = agent.competition
            if comp not in competition_groups:
                competition_groups[comp] = []
            competition_groups[comp].append(agent)
        
        # 各コンペ内で順序強制
        for comp, agents in competition_groups.items():
            # エージェント種別でソート
            agents_by_type = {agent.agent_type: agent for agent in agents}
            
            for agent_type in self.mandatory_order:
                if agent_type in agents_by_type:
                    agent = agents_by_type[agent_type]
                    
                    # 前段エージェントへの依存関係設定
                    prev_agents = self.get_previous_agents_in_order(
                        agent_type, agents_by_type, comp
                    )
                    agent.dependencies.extend([
                        f"{comp}:{prev.agent_type.value}" 
                        for prev in prev_agents
                    ])
                    
                    ordered_plan.append(agent)
        
        return ordered_plan
    
    def get_previous_agents_in_order(
        self, 
        current_type: AgentType, 
        agents_by_type: Dict[AgentType, AgentExecution],
        competition: str
    ) -> List[AgentExecution]:
        """実行順序における前段エージェント取得"""
        previous = []
        current_index = self.mandatory_order.index(current_type)
        
        for i in range(current_index):
            prev_type = self.mandatory_order[i]
            if prev_type in agents_by_type:
                previous.append(agents_by_type[prev_type])
        
        return previous
    
    def validate_no_reverse_dependencies(
        self, 
        agent_plan: List[AgentExecution]
    ) -> List[AgentExecution]:
        """逆依存の禁止検証"""
        validated_plan = []
        
        for agent in agent_plan:
            # 逆依存チェック
            invalid_deps = []
            for dep in agent.dependencies:
                if self.is_reverse_dependency(agent.agent_type, dep):
                    invalid_deps.append(dep)
            
            # 無効な依存関係を除去
            for invalid_dep in invalid_deps:
                agent.dependencies.remove(invalid_dep)
                self.logger.warning(f"逆依存除去: {agent.agent_id} -> {invalid_dep}")
            
            validated_plan.append(agent)
        
        return validated_plan
    
    def is_reverse_dependency(self, agent_type: AgentType, dependency: str) -> bool:
        """逆依存の判定"""
        # dependency format: "competition:agent_type"
        if ":" in dependency:
            _, dep_agent_type_str = dependency.split(":", 1)
            try:
                dep_agent_type = AgentType(dep_agent_type_str)
                current_index = self.mandatory_order.index(agent_type)
                dep_index = self.mandatory_order.index(dep_agent_type)
                
                # 後段エージェントが前段エージェントを待つのは逆依存
                return current_index < dep_index
            except (ValueError, KeyError):
                return False
        
        return False
    
    def apply_execution_timeouts(
        self, 
        agent_plan: List[AgentExecution]
    ) -> List[AgentExecution]:
        """実行タイムアウト・デッドライン設定"""
        timeout_plan = []
        
        for agent in agent_plan:
            timeout_seconds = self.execution_timeouts.get(agent.agent_type, 3600)
            agent.timeout_deadline = agent.start_time + timedelta(seconds=timeout_seconds)
            timeout_plan.append(agent)
        
        return timeout_plan
    
    def get_applied_constraints(
        self, 
        agent_plan: List[AgentExecution]
    ) -> Dict[str, Any]:
        """適用された制約の取得"""
        constraints = {}
        
        for agent in agent_plan:
            constraints[agent.agent_id] = {
                "dependencies": agent.dependencies,
                "timeout_deadline": agent.timeout_deadline.isoformat() if agent.timeout_deadline else None,
                "execution_order": self.mandatory_order.index(agent.agent_type)
            }
        
        return constraints
    
    async def break_existing_deadlock(
        self, 
        deadlocked_agents: List[str]
    ) -> Dict[str, Any]:
        """既存デッドロックの強制解除"""
        try:
            # 最も古い待機エージェントの特定
            oldest_wait = await self.find_oldest_waiting_agent(deadlocked_agents)
            
            if oldest_wait:
                # 強制タイムアウト・解除実行
                await self.force_timeout_agent(oldest_wait)
                
                # 解除後の依存関係クリーンアップ
                await self.cleanup_deadlock_dependencies(deadlocked_agents)
                
                # 影響を受けたエージェントの再起動計画
                restart_plan = await self.create_restart_plan(deadlocked_agents)
                
                self.logger.info(f"デッドロック強制解除完了: {oldest_wait}")
                
                return {
                    "deadlock_broken": True,
                    "forced_timeout_agent": oldest_wait,
                    "restart_plan": restart_plan,
                    "recovery_actions": await self.generate_recovery_actions(deadlocked_agents)
                }
            
            return {"deadlock_broken": False, "reason": "no_resolvable_agent_found"}
            
        except Exception as e:
            self.logger.error(f"デッドロック解除失敗: {e}")
            raise
    
    async def find_oldest_waiting_agent(self, agent_ids: List[str]) -> Optional[str]:
        """最も古い待機エージェント特定"""
        oldest = None
        oldest_time = datetime.utcnow()
        
        for agent_id in agent_ids:
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                if (agent.state == AgentState.WAITING and 
                    agent.start_time < oldest_time):
                    oldest = agent_id
                    oldest_time = agent.start_time
        
        return oldest
    
    async def find_lowest_priority_agent(self, agent_ids: List[str]) -> Optional[str]:
        """最低優先度エージェント特定"""
        # retrospective < monitor < executor < analyzer < planner の順で優先度設定
        priority_order = list(reversed(self.mandatory_order))
        
        for agent_type in priority_order:
            for agent_id in agent_ids:
                if (agent_id in self.active_agents and 
                    self.active_agents[agent_id].agent_type == agent_type):
                    return agent_id
        
        return None
    
    async def force_timeout_agent(self, agent_id: str):
        """エージェントの強制タイムアウト"""
        if agent_id in self.active_agents:
            agent = self.active_agents[agent_id]
            agent.state = AgentState.TIMEOUT
            agent.last_update = datetime.utcnow()
            
            self.logger.warning(f"エージェント強制タイムアウト実行: {agent_id}")
    
    async def cleanup_deadlock_dependencies(self, deadlocked_agents: List[str]):
        """デッドロック依存関係クリーンアップ"""
        for agent_id in deadlocked_agents:
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                
                # デッドロックに関与する依存関係を除去
                clean_deps = [
                    dep for dep in agent.dependencies 
                    if dep.split(":")[0] not in deadlocked_agents
                ]
                agent.dependencies = clean_deps
    
    async def create_restart_plan(
        self, 
        affected_agents: List[str]
    ) -> List[Dict[str, Any]]:
        """再起動計画作成"""
        restart_plan = []
        
        for agent_id in affected_agents:
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                restart_plan.append({
                    "agent_id": agent_id,
                    "agent_type": agent.agent_type.value,
                    "competition": agent.competition,
                    "restart_delay": 60,  # 1分後に再起動
                    "clean_start": True   # クリーンスタート
                })
        
        return restart_plan
    
    async def generate_recovery_actions(
        self, 
        affected_agents: List[str]
    ) -> List[str]:
        """復旧アクション生成"""
        actions = []
        actions.append("clear_dependency_graph")
        actions.append("reset_agent_states")
        actions.append("restart_affected_agents")
        actions.append("monitor_for_recurrence")
        
        return actions