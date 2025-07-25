"""
エージェント依存関係グラフ管理システム

エージェント間の複雑な依存関係を有向グラフで管理し、
安全な実行順序とデッドロック防止を提供。
"""

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
import json


class DependencyType(Enum):
    """依存関係種別"""
    EXECUTION_ORDER = "execution_order"     # 実行順序依存
    DATA_DEPENDENCY = "data_dependency"     # データ依存
    RESOURCE_LOCK = "resource_lock"         # リソースロック依存
    COMPETITION_ISOLATION = "competition_isolation"  # コンペ分離依存


@dataclass
class DependencyEdge:
    """依存関係エッジ情報"""
    source_agent: str
    target_agent: str
    dependency_type: DependencyType
    weight: float = 1.0
    condition: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_agent": self.source_agent,
            "target_agent": self.target_agent,
            "dependency_type": self.dependency_type.value,
            "weight": self.weight,
            "condition": self.condition,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class AgentNode:
    """エージェントノード情報"""
    agent_id: str
    agent_type: str
    competition: str
    state: str = "idle"
    priority: int = 5
    resources: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "competition": self.competition,
            "state": self.state,
            "priority": self.priority,
            "resources": list(self.resources),
            "capabilities": list(self.capabilities),
            "metadata": self.metadata
        }


class AgentDependencyGraph:
    """エージェント依存関係グラフ管理システム"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, AgentNode] = {}
        self.edges: Dict[Tuple[str, str], DependencyEdge] = {}
        self.competition_groups: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # 標準エージェント実行順序
        self.standard_agent_order = [
            "planner", "analyzer", "executor", "monitor", "retrospective"
        ]
    
    def add_agent_node(self, agent_node: AgentNode) -> bool:
        """エージェントノード追加"""
        try:
            agent_id = agent_node.agent_id
            
            # グラフにノード追加
            self.graph.add_node(
                agent_id,
                agent_type=agent_node.agent_type,
                competition=agent_node.competition,
                state=agent_node.state,
                priority=agent_node.priority
            )
            
            # ノード情報保存
            self.nodes[agent_id] = agent_node
            
            # コンペ別グループ管理
            if agent_node.competition not in self.competition_groups:
                self.competition_groups[agent_node.competition] = set()
            self.competition_groups[agent_node.competition].add(agent_id)
            
            self.logger.info(f"エージェントノード追加: {agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"ノード追加失敗: {e}")
            return False
    
    def add_dependency_edge(self, edge: DependencyEdge) -> bool:
        """依存関係エッジ追加"""
        try:
            source = edge.source_agent
            target = edge.target_agent
            
            # 循環依存チェック
            if self.would_create_cycle(source, target):
                self.logger.warning(f"循環依存検出、エッジ追加拒否: {source} -> {target}")
                return False
            
            # グラフにエッジ追加
            self.graph.add_edge(
                source, target,
                weight=edge.weight,
                dependency_type=edge.dependency_type.value,
                condition=edge.condition,
                created_at=edge.created_at
            )
            
            # エッジ情報保存
            edge_key = (source, target)
            self.edges[edge_key] = edge
            
            self.logger.info(f"依存関係エッジ追加: {source} -> {target} ({edge.dependency_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"エッジ追加失敗: {e}")
            return False
    
    def would_create_cycle(self, source: str, target: str) -> bool:
        """循環依存の事前検出"""
        # 一時的にエッジを追加してサイクルチェック
        temp_graph = self.graph.copy()
        temp_graph.add_edge(source, target)
        
        try:
            # サイクル検出
            cycles = list(nx.simple_cycles(temp_graph))
            return len(cycles) > 0
        except:
            return True  # エラー時は安全側に倒す
    
    def build_standard_competition_dependencies(self, competition: str) -> List[DependencyEdge]:
        """標準コンペ内依存関係構築"""
        edges = []
        competition_agents = self.get_competition_agents(competition)
        
        # 実行順序による依存関係
        agent_by_type = {}
        for agent_id in competition_agents:
            if agent_id in self.nodes:
                agent_type = self.nodes[agent_id].agent_type
                agent_by_type[agent_type] = agent_id
        
        # 順次依存関係作成
        for i in range(len(self.standard_agent_order) - 1):
            current_type = self.standard_agent_order[i]
            next_type = self.standard_agent_order[i + 1]
            
            if current_type in agent_by_type and next_type in agent_by_type:
                edge = DependencyEdge(
                    source_agent=agent_by_type[next_type],  # 後段が前段に依存
                    target_agent=agent_by_type[current_type],
                    dependency_type=DependencyType.EXECUTION_ORDER,
                    weight=1.0,
                    metadata={"competition": competition, "auto_generated": True}
                )
                edges.append(edge)
        
        return edges
    
    def get_competition_agents(self, competition: str) -> Set[str]:
        """コンペ所属エージェント取得"""
        return self.competition_groups.get(competition, set())
    
    def get_ready_agents(self, competition: Optional[str] = None) -> List[str]:
        """実行可能エージェント取得"""
        ready_agents = []
        
        # 対象エージェント特定
        target_agents = (
            self.get_competition_agents(competition) 
            if competition 
            else set(self.nodes.keys())
        )
        
        for agent_id in target_agents:
            if self.is_agent_ready_to_execute(agent_id):
                ready_agents.append(agent_id)
        
        # 優先度順でソート
        ready_agents.sort(
            key=lambda x: self.nodes[x].priority if x in self.nodes else 5
        )
        
        return ready_agents
    
    def is_agent_ready_to_execute(self, agent_id: str) -> bool:
        """エージェント実行可能性判定"""
        if agent_id not in self.nodes:
            return False
        
        agent = self.nodes[agent_id]
        
        # 状態チェック
        if agent.state not in ["idle", "waiting"]:
            return False
        
        # 依存関係チェック
        predecessors = list(self.graph.predecessors(agent_id))
        for pred_id in predecessors:
            if pred_id in self.nodes:
                pred_agent = self.nodes[pred_id]
                if pred_agent.state != "completed":
                    return False
        
        return True
    
    def get_execution_path(self, competition: str) -> List[List[str]]:
        """実行パス計算（トポロジカルソート）"""
        competition_agents = self.get_competition_agents(competition)
        
        if not competition_agents:
            return []
        
        # コンペ内サブグラフ作成
        subgraph = self.graph.subgraph(competition_agents)
        
        try:
            # トポロジカルソートで実行順序決定
            execution_order = list(nx.topological_sort(subgraph))
            
            # 並列実行可能グループに分割
            execution_levels = []
            remaining_agents = set(execution_order)
            
            while remaining_agents:
                current_level = []
                
                # 依存関係が満たされたエージェントを特定
                for agent_id in list(remaining_agents):
                    predecessors = set(subgraph.predecessors(agent_id))
                    if predecessors.issubset(set(execution_order) - remaining_agents):
                        current_level.append(agent_id)
                
                if not current_level:
                    # デッドロック検出
                    self.logger.error(f"実行パス計算でデッドロック検出: {competition}")
                    break
                
                execution_levels.append(current_level)
                remaining_agents -= set(current_level)
            
            return execution_levels
            
        except nx.NetworkXError as e:
            self.logger.error(f"実行パス計算失敗: {e}")
            return []
    
    def detect_deadlocks(self, competition: Optional[str] = None) -> List[List[str]]:
        """デッドロック検出"""
        # 対象グラフ決定
        if competition:
            agents = self.get_competition_agents(competition)
            target_graph = self.graph.subgraph(agents)
        else:
            target_graph = self.graph
        
        try:
            cycles = list(nx.simple_cycles(target_graph))
            if cycles:
                self.logger.warning(f"デッドロック検出: {len(cycles)}個の循環")
            return cycles
        except Exception as e:
            self.logger.error(f"デッドロック検出失敗: {e}")
            return []
    
    def suggest_deadlock_resolution(self, cycles: List[List[str]]) -> List[Dict[str, Any]]:
        """デッドロック解決提案"""
        suggestions = []
        
        for cycle in cycles:
            # 最低優先度エージェント特定
            min_priority_agent = min(
                cycle, 
                key=lambda x: self.nodes[x].priority if x in self.nodes else 10
            )
            
            # 解決策提案
            suggestion = {
                "cycle": cycle,
                "resolution_type": "remove_lowest_priority",
                "target_agent": min_priority_agent,
                "action": f"一時的に{min_priority_agent}の実行を遅延",
                "affected_edges": [
                    (src, tgt) for src, tgt in self.edges.keys()
                    if src in cycle and tgt in cycle
                ]
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def update_agent_state(self, agent_id: str, new_state: str) -> bool:
        """エージェント状態更新"""
        if agent_id in self.nodes:
            old_state = self.nodes[agent_id].state
            self.nodes[agent_id].state = new_state
            
            # グラフ属性も更新
            if self.graph.has_node(agent_id):
                self.graph.nodes[agent_id]["state"] = new_state
            
            self.logger.info(f"エージェント状態更新: {agent_id} ({old_state} -> {new_state})")
            return True
        
        return False
    
    def remove_completed_agents(self, competition: str) -> int:
        """完了エージェントの削除・クリーンアップ"""
        competition_agents = self.get_competition_agents(competition)
        completed_agents = []
        
        for agent_id in competition_agents:
            if (agent_id in self.nodes and 
                self.nodes[agent_id].state == "completed"):
                completed_agents.append(agent_id)
        
        removed_count = 0
        for agent_id in completed_agents:
            if self.remove_agent_node(agent_id):
                removed_count += 1
        
        return removed_count
    
    def remove_agent_node(self, agent_id: str) -> bool:
        """エージェントノード削除"""
        try:
            if agent_id in self.nodes:
                competition = self.nodes[agent_id].competition
                
                # グラフからノード削除
                if self.graph.has_node(agent_id):
                    self.graph.remove_node(agent_id)
                
                # エッジ情報削除
                edges_to_remove = [
                    key for key in self.edges.keys()
                    if key[0] == agent_id or key[1] == agent_id
                ]
                for edge_key in edges_to_remove:
                    del self.edges[edge_key]
                
                # ノード情報削除
                del self.nodes[agent_id]
                
                # コンペグループから削除
                if competition in self.competition_groups:
                    self.competition_groups[competition].discard(agent_id)
                
                self.logger.info(f"エージェントノード削除: {agent_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"ノード削除失敗: {e}")
            return False
    
    def get_dependency_matrix(self, competition: str) -> Dict[str, Dict[str, Any]]:
        """依存関係マトリクス生成"""
        agents = list(self.get_competition_agents(competition))
        matrix = {}
        
        for agent in agents:
            matrix[agent] = {}
            for other_agent in agents:
                if self.graph.has_edge(agent, other_agent):
                    edge_data = self.graph.edges[agent, other_agent]
                    matrix[agent][other_agent] = {
                        "has_dependency": True,
                        "type": edge_data.get("dependency_type", "unknown"),
                        "weight": edge_data.get("weight", 1.0)
                    }
                else:
                    matrix[agent][other_agent] = {"has_dependency": False}
        
        return matrix
    
    def export_graph_state(self) -> Dict[str, Any]:
        """グラフ状態エクスポート"""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": {f"{k[0]}->{k[1]}": edge.to_dict() for k, edge in self.edges.items()},
            "competition_groups": {k: list(v) for k, v in self.competition_groups.items()},
            "export_timestamp": datetime.utcnow().isoformat()
        }
    
    def import_graph_state(self, state_data: Dict[str, Any]) -> bool:
        """グラフ状態インポート"""
        try:
            # 既存状態クリア
            self.graph.clear()
            self.nodes.clear()
            self.edges.clear()
            self.competition_groups.clear()
            
            # ノード復元
            for node_id, node_data in state_data.get("nodes", {}).items():
                agent_node = AgentNode(
                    agent_id=node_data["agent_id"],
                    agent_type=node_data["agent_type"],
                    competition=node_data["competition"],
                    state=node_data.get("state", "idle"),
                    priority=node_data.get("priority", 5),
                    resources=set(node_data.get("resources", [])),
                    capabilities=set(node_data.get("capabilities", [])),
                    metadata=node_data.get("metadata", {})
                )
                self.add_agent_node(agent_node)
            
            # エッジ復元
            for edge_key, edge_data in state_data.get("edges", {}).items():
                edge = DependencyEdge(
                    source_agent=edge_data["source_agent"],
                    target_agent=edge_data["target_agent"],
                    dependency_type=DependencyType(edge_data["dependency_type"]),
                    weight=edge_data.get("weight", 1.0),
                    condition=edge_data.get("condition"),
                    created_at=datetime.fromisoformat(edge_data["created_at"]),
                    metadata=edge_data.get("metadata", {})
                )
                self.add_dependency_edge(edge)
            
            self.logger.info("グラフ状態インポート完了")
            return True
            
        except Exception as e:
            self.logger.error(f"グラフ状態インポート失敗: {e}")
            return False