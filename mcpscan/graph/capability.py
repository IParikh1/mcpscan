"""Tool capability graph data structures for MCP analysis.

This module defines the graph data structures for modeling MCP tool
capabilities and their relationships, enabling privilege escalation
path analysis and attack surface mapping.

Patent-relevant innovation: Novel graph-based representation of MCP tool
capabilities enabling automated detection of privilege escalation paths
and attack chains specific to AI agent tool ecosystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any


class CapabilityType(str, Enum):
    """Categories of tool capabilities."""

    # Data access capabilities
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    LIST_DIRECTORY = "list_directory"

    # Code execution capabilities
    EXECUTE_CODE = "execute_code"
    SHELL_ACCESS = "shell_access"
    PROCESS_SPAWN = "process_spawn"

    # Network capabilities
    HTTP_REQUEST = "http_request"
    DNS_LOOKUP = "dns_lookup"
    SOCKET_ACCESS = "socket_access"

    # Database capabilities
    DB_READ = "db_read"
    DB_WRITE = "db_write"
    DB_ADMIN = "db_admin"

    # Cloud/API capabilities
    API_CALL = "api_call"
    CLOUD_RESOURCE = "cloud_resource"
    SECRET_ACCESS = "secret_access"

    # System capabilities
    ENV_ACCESS = "env_access"
    SYSTEM_INFO = "system_info"
    PRIVILEGED_OP = "privileged_op"

    # AI/LLM capabilities
    LLM_INVOKE = "llm_invoke"
    PROMPT_INJECT = "prompt_inject"
    CONTEXT_ACCESS = "context_access"


class RiskCategory(str, Enum):
    """Risk categories for capability combinations."""

    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    PERSISTENCE = "persistence"
    CREDENTIAL_THEFT = "credential_theft"
    DENIAL_OF_SERVICE = "denial_of_service"
    SUPPLY_CHAIN = "supply_chain"


@dataclass
class ToolNode:
    """Represents a tool in the capability graph.

    A tool node captures the capabilities and metadata of a single
    MCP tool, serving as a vertex in the capability graph.
    """

    id: str  # Unique identifier (server_name.tool_name)
    name: str
    server_name: str
    description: Optional[str] = None
    capabilities: Set[CapabilityType] = field(default_factory=set)
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Input/output schema if available
    input_schema: Optional[Dict[str, Any]] = None
    output_types: Set[str] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolNode):
            return False
        return self.id == other.id


@dataclass
class CapabilityEdge:
    """Represents a relationship between tools in the capability graph.

    Edges represent how one tool's output can enable or enhance
    another tool's capabilities, forming potential attack chains.
    """

    source: str  # Source tool ID
    target: str  # Target tool ID
    edge_type: str  # Type of relationship
    weight: float = 1.0  # Edge weight for path analysis
    risk_amplification: float = 1.0  # How much this edge amplifies risk
    description: str = ""

    def __hash__(self) -> int:
        return hash((self.source, self.target, self.edge_type))


@dataclass
class AttackPath:
    """Represents a potential attack path through the capability graph."""

    nodes: List[str]  # Tool IDs in order
    edges: List[CapabilityEdge]
    total_risk: float
    risk_category: RiskCategory
    description: str
    mitigations: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.nodes,
            "length": self.length,
            "total_risk": round(self.total_risk, 3),
            "category": self.risk_category.value,
            "description": self.description,
            "mitigations": self.mitigations,
        }


class CapabilityGraph:
    """Graph representation of MCP tool capabilities.

    This graph enables analysis of:
    - Privilege escalation paths
    - Data flow between tools
    - Attack surface mapping
    - Least privilege violations
    """

    def __init__(self) -> None:
        """Initialize an empty capability graph."""
        self.nodes: Dict[str, ToolNode] = {}
        self.edges: List[CapabilityEdge] = []
        self._adjacency: Dict[str, List[str]] = {}  # Forward edges
        self._reverse_adj: Dict[str, List[str]] = {}  # Backward edges

    def add_node(self, node: ToolNode) -> None:
        """Add a tool node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []
        if node.id not in self._reverse_adj:
            self._reverse_adj[node.id] = []

    def add_edge(self, edge: CapabilityEdge) -> None:
        """Add a capability edge to the graph."""
        self.edges.append(edge)

        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        self._adjacency[edge.source].append(edge.target)

        if edge.target not in self._reverse_adj:
            self._reverse_adj[edge.target] = []
        self._reverse_adj[edge.target].append(edge.source)

    def get_node(self, node_id: str) -> Optional[ToolNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_neighbors(self, node_id: str) -> List[str]:
        """Get outgoing neighbors of a node."""
        return self._adjacency.get(node_id, [])

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get incoming neighbors of a node."""
        return self._reverse_adj.get(node_id, [])

    def get_edges_from(self, node_id: str) -> List[CapabilityEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source == node_id]

    def get_edges_to(self, node_id: str) -> List[CapabilityEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target == node_id]

    def get_nodes_with_capability(
        self,
        capability: CapabilityType
    ) -> List[ToolNode]:
        """Find all nodes with a specific capability."""
        return [
            node for node in self.nodes.values()
            if capability in node.capabilities
        ]

    def get_high_risk_nodes(self, threshold: float = 0.6) -> List[ToolNode]:
        """Get nodes with risk score above threshold."""
        return [
            node for node in self.nodes.values()
            if node.risk_score >= threshold
        ]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "server": n.server_name,
                    "capabilities": [c.value for c in n.capabilities],
                    "risk_score": round(n.risk_score, 3),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
            "stats": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
            },
        }
