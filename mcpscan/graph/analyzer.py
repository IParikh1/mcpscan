"""Graph analysis for MCP tool capability security.

This module implements graph algorithms for detecting privilege escalation
paths, attack chains, and security risks in MCP tool configurations.

Patent-relevant innovation: Novel graph-based security analysis algorithms
specifically designed for MCP tool ecosystems, identifying attack paths
and privilege escalation vectors unique to AI agent architectures.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple, Any

from mcpscan.graph.capability import (
    CapabilityGraph,
    ToolNode,
    CapabilityEdge,
    AttackPath,
    CapabilityType,
    RiskCategory,
)
from mcpscan.models import MCPConfig, MCPServer


@dataclass
class GraphAnalysisResult:
    """Results from capability graph analysis."""

    graph: CapabilityGraph
    attack_paths: List[AttackPath]
    high_risk_nodes: List[ToolNode]
    privilege_escalation_vectors: List[Dict[str, Any]]
    total_attack_surface: float
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "attack_paths": [p.to_dict() for p in self.attack_paths],
            "high_risk_nodes": [
                {"id": n.id, "risk": n.risk_score}
                for n in self.high_risk_nodes
            ],
            "privilege_escalation_vectors": self.privilege_escalation_vectors,
            "total_attack_surface": round(self.total_attack_surface, 3),
            "recommendations": self.recommendations,
        }


class GraphAnalyzer:
    """Analyze MCP configurations using capability graph analysis.

    This analyzer builds a capability graph from MCP configurations
    and applies security-focused graph algorithms to detect:
    - Privilege escalation paths
    - Data exfiltration vectors
    - Credential theft opportunities
    - Attack chain formations
    """

    # Tool name patterns to capability mappings
    CAPABILITY_PATTERNS = {
        # File operations
        r'read|get_file|load|fetch_file': [CapabilityType.READ_FILE],
        r'write|save|put_file|store': [CapabilityType.WRITE_FILE],
        r'delete|remove|rm|unlink': [CapabilityType.DELETE_FILE],
        r'list|ls|dir|browse': [CapabilityType.LIST_DIRECTORY],

        # Code execution
        r'exec|execute|run|eval': [CapabilityType.EXECUTE_CODE],
        r'shell|bash|cmd|terminal|powershell': [CapabilityType.SHELL_ACCESS],
        r'spawn|fork|process': [CapabilityType.PROCESS_SPAWN],

        # Network
        r'http|request|fetch|curl|wget|api': [CapabilityType.HTTP_REQUEST],
        r'dns|resolve|lookup': [CapabilityType.DNS_LOOKUP],
        r'socket|tcp|udp|connect': [CapabilityType.SOCKET_ACCESS],

        # Database
        r'query|select|db_read|find': [CapabilityType.DB_READ],
        r'insert|update|db_write|save': [CapabilityType.DB_WRITE],
        r'create_table|drop|migrate|admin': [CapabilityType.DB_ADMIN],

        # Cloud/Secrets
        r'secret|credential|vault|key': [CapabilityType.SECRET_ACCESS],
        r'cloud|aws|gcp|azure|s3': [CapabilityType.CLOUD_RESOURCE],

        # System
        r'env|environment|config': [CapabilityType.ENV_ACCESS],
        r'system|os|platform|info': [CapabilityType.SYSTEM_INFO],
        r'sudo|admin|root|privileged': [CapabilityType.PRIVILEGED_OP],

        # AI/LLM
        r'llm|gpt|claude|generate|complete': [CapabilityType.LLM_INVOKE],
        r'prompt|message|context': [CapabilityType.CONTEXT_ACCESS],
    }

    # Dangerous capability combinations indicating attack paths
    ATTACK_PATTERNS = [
        # Data exfiltration: read + network
        {
            "required": {CapabilityType.READ_FILE, CapabilityType.HTTP_REQUEST},
            "category": RiskCategory.DATA_EXFILTRATION,
            "risk": 0.8,
            "description": "File read combined with network access enables data exfiltration",
        },
        # Privilege escalation: shell + file write
        {
            "required": {CapabilityType.SHELL_ACCESS, CapabilityType.WRITE_FILE},
            "category": RiskCategory.PRIVILEGE_ESCALATION,
            "risk": 0.9,
            "description": "Shell access with file write enables persistence and privilege escalation",
        },
        # Credential theft: secret access + network
        {
            "required": {CapabilityType.SECRET_ACCESS, CapabilityType.HTTP_REQUEST},
            "category": RiskCategory.CREDENTIAL_THEFT,
            "risk": 0.95,
            "description": "Secret access with network enables credential exfiltration",
        },
        # Supply chain: code execution + cloud
        {
            "required": {CapabilityType.EXECUTE_CODE, CapabilityType.CLOUD_RESOURCE},
            "category": RiskCategory.SUPPLY_CHAIN,
            "risk": 0.85,
            "description": "Code execution with cloud access enables supply chain attacks",
        },
        # Lateral movement: shell + socket
        {
            "required": {CapabilityType.SHELL_ACCESS, CapabilityType.SOCKET_ACCESS},
            "category": RiskCategory.LATERAL_MOVEMENT,
            "risk": 0.8,
            "description": "Shell with socket access enables lateral movement",
        },
        # Database compromise: db_admin + shell
        {
            "required": {CapabilityType.DB_ADMIN, CapabilityType.SHELL_ACCESS},
            "category": RiskCategory.PRIVILEGE_ESCALATION,
            "risk": 0.9,
            "description": "Database admin with shell enables full database compromise",
        },
        # Prompt injection amplification
        {
            "required": {CapabilityType.CONTEXT_ACCESS, CapabilityType.EXECUTE_CODE},
            "category": RiskCategory.PRIVILEGE_ESCALATION,
            "risk": 0.85,
            "description": "Context access with code execution amplifies prompt injection impact",
        },
    ]

    # Edge type definitions for capability flow
    EDGE_TYPES = {
        "data_flow": 1.0,       # One tool's output feeds another
        "capability_grant": 1.5, # One tool grants capabilities to another
        "privilege_chain": 2.0,  # Tools chain to escalate privileges
        "implicit": 0.5,        # Implicit relationship through shared resources
    }

    def __init__(self) -> None:
        """Initialize the graph analyzer."""
        pass

    def analyze(self, config: MCPConfig) -> GraphAnalysisResult:
        """Perform complete graph analysis on an MCP configuration.

        Args:
            config: Parsed MCP configuration

        Returns:
            GraphAnalysisResult with all analysis findings
        """
        # Build capability graph
        graph = self._build_graph(config)

        # Find attack paths
        attack_paths = self._find_attack_paths(graph)

        # Identify high-risk nodes
        high_risk_nodes = graph.get_high_risk_nodes(threshold=0.6)

        # Find privilege escalation vectors
        priv_esc_vectors = self._find_privilege_escalation(graph)

        # Calculate total attack surface
        attack_surface = self._calculate_attack_surface(graph)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            graph, attack_paths, priv_esc_vectors
        )

        return GraphAnalysisResult(
            graph=graph,
            attack_paths=attack_paths,
            high_risk_nodes=high_risk_nodes,
            privilege_escalation_vectors=priv_esc_vectors,
            total_attack_surface=attack_surface,
            recommendations=recommendations,
        )

    def _build_graph(self, config: MCPConfig) -> CapabilityGraph:
        """Build a capability graph from MCP configuration.

        This method:
        1. Creates nodes for each tool/server
        2. Infers capabilities from tool names and metadata
        3. Creates edges based on capability relationships
        """
        graph = CapabilityGraph()

        # Create nodes for each server and infer tools
        for server_name, server in config.servers.items():
            # Create server-level node
            server_node = self._create_server_node(server_name, server)
            graph.add_node(server_node)

            # Create nodes for individual tools if defined
            for tool in server.tools:
                tool_node = self._create_tool_node(server_name, tool.name, tool.description)
                graph.add_node(tool_node)

                # Add edge from server to tool
                graph.add_edge(CapabilityEdge(
                    source=server_node.id,
                    target=tool_node.id,
                    edge_type="provides",
                    weight=1.0,
                ))

        # Infer edges based on capability relationships
        self._infer_edges(graph)

        # Calculate risk scores for all nodes
        self._calculate_node_risks(graph)

        return graph

    def _create_server_node(
        self,
        server_name: str,
        server: MCPServer
    ) -> ToolNode:
        """Create a node for an MCP server."""
        capabilities = set()

        # Infer capabilities from server configuration
        config_str = str(server.raw_config).lower()

        for pattern, caps in self.CAPABILITY_PATTERNS.items():
            if re.search(pattern, config_str):
                capabilities.update(caps)

        # Check for network capability (remote server)
        if server.url:
            capabilities.add(CapabilityType.HTTP_REQUEST)

        # Check command for execution capabilities
        if server.command:
            cmd = server.command.lower()
            if any(sh in cmd for sh in ['bash', 'sh', 'cmd', 'powershell']):
                capabilities.add(CapabilityType.SHELL_ACCESS)
            if any(ex in cmd for ex in ['python', 'node', 'ruby', 'exec']):
                capabilities.add(CapabilityType.EXECUTE_CODE)

        return ToolNode(
            id=f"server.{server_name}",
            name=server_name,
            server_name=server_name,
            description=f"MCP Server: {server_name}",
            capabilities=capabilities,
            metadata={"type": "server", "has_url": bool(server.url)},
        )

    def _create_tool_node(
        self,
        server_name: str,
        tool_name: str,
        description: Optional[str] = None
    ) -> ToolNode:
        """Create a node for an individual tool."""
        capabilities = set()

        # Infer capabilities from tool name
        name_lower = tool_name.lower()
        for pattern, caps in self.CAPABILITY_PATTERNS.items():
            if re.search(pattern, name_lower):
                capabilities.update(caps)

        # Also check description if available
        if description:
            desc_lower = description.lower()
            for pattern, caps in self.CAPABILITY_PATTERNS.items():
                if re.search(pattern, desc_lower):
                    capabilities.update(caps)

        return ToolNode(
            id=f"{server_name}.{tool_name}",
            name=tool_name,
            server_name=server_name,
            description=description,
            capabilities=capabilities,
            metadata={"type": "tool"},
        )

    def _infer_edges(self, graph: CapabilityGraph) -> None:
        """Infer edges between nodes based on capability relationships."""
        nodes = list(graph.nodes.values())

        for i, source in enumerate(nodes):
            for target in nodes[i + 1:]:
                edge = self._check_edge_relationship(source, target)
                if edge:
                    graph.add_edge(edge)

                # Check reverse direction too
                reverse_edge = self._check_edge_relationship(target, source)
                if reverse_edge:
                    graph.add_edge(reverse_edge)

    def _check_edge_relationship(
        self,
        source: ToolNode,
        target: ToolNode
    ) -> Optional[CapabilityEdge]:
        """Check if there should be an edge between two nodes."""
        if source.id == target.id:
            return None

        # Data flow: read capabilities flow to network capabilities
        if (CapabilityType.READ_FILE in source.capabilities and
            CapabilityType.HTTP_REQUEST in target.capabilities):
            return CapabilityEdge(
                source=source.id,
                target=target.id,
                edge_type="data_flow",
                weight=self.EDGE_TYPES["data_flow"],
                risk_amplification=1.5,
                description="File data can flow to network requests",
            )

        # Privilege chain: shell enables other capabilities
        if (CapabilityType.SHELL_ACCESS in source.capabilities and
            any(c in target.capabilities for c in [
                CapabilityType.WRITE_FILE,
                CapabilityType.EXECUTE_CODE,
                CapabilityType.DB_ADMIN,
            ])):
            return CapabilityEdge(
                source=source.id,
                target=target.id,
                edge_type="privilege_chain",
                weight=self.EDGE_TYPES["privilege_chain"],
                risk_amplification=2.0,
                description="Shell access can amplify other capabilities",
            )

        # Secret access enables authentication
        if (CapabilityType.SECRET_ACCESS in source.capabilities and
            CapabilityType.CLOUD_RESOURCE in target.capabilities):
            return CapabilityEdge(
                source=source.id,
                target=target.id,
                edge_type="capability_grant",
                weight=self.EDGE_TYPES["capability_grant"],
                risk_amplification=1.8,
                description="Secrets can grant cloud access",
            )

        return None

    def _calculate_node_risks(self, graph: CapabilityGraph) -> None:
        """Calculate risk scores for all nodes."""
        # Capability risk weights
        capability_risks = {
            CapabilityType.SHELL_ACCESS: 0.9,
            CapabilityType.EXECUTE_CODE: 0.85,
            CapabilityType.SECRET_ACCESS: 0.8,
            CapabilityType.DB_ADMIN: 0.75,
            CapabilityType.PRIVILEGED_OP: 0.9,
            CapabilityType.WRITE_FILE: 0.6,
            CapabilityType.DELETE_FILE: 0.65,
            CapabilityType.HTTP_REQUEST: 0.4,
            CapabilityType.READ_FILE: 0.3,
            CapabilityType.CLOUD_RESOURCE: 0.5,
            CapabilityType.SOCKET_ACCESS: 0.5,
        }

        for node in graph.nodes.values():
            risk = 0.0

            # Sum capability risks
            for cap in node.capabilities:
                risk += capability_risks.get(cap, 0.2)

            # Normalize to 0-1
            node.risk_score = min(1.0, risk / 2.0)

            # Amplify if node has dangerous combinations
            for pattern in self.ATTACK_PATTERNS:
                if pattern["required"].issubset(node.capabilities):
                    node.risk_score = min(1.0, node.risk_score * 1.3)

    def _find_attack_paths(self, graph: CapabilityGraph) -> List[AttackPath]:
        """Find potential attack paths through the capability graph.

        Uses BFS to find paths that combine capabilities into attack chains.
        """
        attack_paths = []

        # For each attack pattern, find paths that satisfy it
        for pattern in self.ATTACK_PATTERNS:
            paths = self._find_paths_for_pattern(graph, pattern)
            attack_paths.extend(paths)

        # Sort by risk
        attack_paths.sort(key=lambda p: p.total_risk, reverse=True)

        return attack_paths[:20]  # Top 20 paths

    def _find_paths_for_pattern(
        self,
        graph: CapabilityGraph,
        pattern: Dict[str, Any]
    ) -> List[AttackPath]:
        """Find all paths that satisfy an attack pattern."""
        paths = []
        required_caps = pattern["required"]

        # Find nodes that have any of the required capabilities
        relevant_nodes = [
            n for n in graph.nodes.values()
            if n.capabilities & required_caps
        ]

        # Check each node pair for path
        for source in relevant_nodes:
            for target in relevant_nodes:
                if source.id == target.id:
                    continue

                # Check if this pair satisfies the pattern
                combined_caps = source.capabilities | target.capabilities
                if required_caps.issubset(combined_caps):
                    # Find actual path in graph
                    path_nodes = self._bfs_path(graph, source.id, target.id)
                    if path_nodes:
                        path_edges = [
                            e for e in graph.edges
                            if any(
                                e.source == path_nodes[i] and e.target == path_nodes[i+1]
                                for i in range(len(path_nodes) - 1)
                            )
                        ]

                        attack_path = AttackPath(
                            nodes=path_nodes,
                            edges=path_edges,
                            total_risk=pattern["risk"],
                            risk_category=pattern["category"],
                            description=pattern["description"],
                            mitigations=self._get_mitigations(pattern["category"]),
                        )
                        paths.append(attack_path)

        return paths

    def _bfs_path(
        self,
        graph: CapabilityGraph,
        start: str,
        end: str
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes using BFS."""
        if start not in graph.nodes or end not in graph.nodes:
            return None

        if start == end:
            return [start]

        visited = {start}
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            for neighbor in graph.get_neighbors(current):
                if neighbor == end:
                    return path + [neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No direct path, but they're in same config (implicit connection)
        return [start, end]

    def _find_privilege_escalation(
        self,
        graph: CapabilityGraph
    ) -> List[Dict[str, Any]]:
        """Find privilege escalation vectors in the graph."""
        vectors = []

        # Look for nodes that can escalate privileges
        shell_nodes = graph.get_nodes_with_capability(CapabilityType.SHELL_ACCESS)
        exec_nodes = graph.get_nodes_with_capability(CapabilityType.EXECUTE_CODE)
        privileged_nodes = graph.get_nodes_with_capability(CapabilityType.PRIVILEGED_OP)

        for node in shell_nodes + exec_nodes:
            # Check what this node can reach
            reachable = self._get_reachable_capabilities(graph, node.id)

            if CapabilityType.SECRET_ACCESS in reachable:
                vectors.append({
                    "source": node.id,
                    "type": "credential_access",
                    "description": f"'{node.name}' can reach secret access capabilities",
                    "risk": 0.85,
                })

            if CapabilityType.DB_ADMIN in reachable:
                vectors.append({
                    "source": node.id,
                    "type": "database_takeover",
                    "description": f"'{node.name}' can reach database admin capabilities",
                    "risk": 0.8,
                })

            if CapabilityType.CLOUD_RESOURCE in reachable:
                vectors.append({
                    "source": node.id,
                    "type": "cloud_compromise",
                    "description": f"'{node.name}' can reach cloud resources",
                    "risk": 0.75,
                })

        return vectors

    def _get_reachable_capabilities(
        self,
        graph: CapabilityGraph,
        start: str,
        max_depth: int = 5
    ) -> Set[CapabilityType]:
        """Get all capabilities reachable from a starting node."""
        reachable = set()
        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            current, depth = queue.popleft()

            node = graph.get_node(current)
            if node:
                reachable.update(node.capabilities)

            if depth < max_depth:
                for neighbor in graph.get_neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

        return reachable

    def _calculate_attack_surface(self, graph: CapabilityGraph) -> float:
        """Calculate the total attack surface of the configuration.

        Attack surface = weighted sum of exposed capabilities and paths.
        """
        surface = 0.0

        # Sum node risks
        for node in graph.nodes.values():
            surface += node.risk_score * 0.1

        # Add edge risk contributions
        for edge in graph.edges:
            surface += edge.risk_amplification * 0.05

        # Normalize to 0-1
        return min(1.0, surface)

    def _get_mitigations(self, category: RiskCategory) -> List[str]:
        """Get mitigation recommendations for a risk category."""
        mitigations = {
            RiskCategory.DATA_EXFILTRATION: [
                "Implement network egress filtering",
                "Add data loss prevention (DLP) controls",
                "Monitor and log network requests",
                "Restrict file access to specific directories",
            ],
            RiskCategory.PRIVILEGE_ESCALATION: [
                "Apply principle of least privilege",
                "Sandbox shell/exec capabilities",
                "Implement mandatory access controls",
                "Add human-in-the-loop for sensitive operations",
            ],
            RiskCategory.CREDENTIAL_THEFT: [
                "Use short-lived, scoped credentials",
                "Implement secret rotation",
                "Add credential access logging",
                "Use hardware security modules where possible",
            ],
            RiskCategory.LATERAL_MOVEMENT: [
                "Implement network segmentation",
                "Restrict inter-service communication",
                "Monitor for unusual connection patterns",
                "Apply zero-trust networking principles",
            ],
            RiskCategory.SUPPLY_CHAIN: [
                "Verify tool/package integrity",
                "Use signed packages only",
                "Implement dependency scanning",
                "Monitor for supply chain indicators",
            ],
        }

        return mitigations.get(category, [
            "Review and restrict tool capabilities",
            "Implement comprehensive logging",
            "Add security monitoring",
        ])

    def _generate_recommendations(
        self,
        graph: CapabilityGraph,
        attack_paths: List[AttackPath],
        priv_esc_vectors: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate security recommendations based on analysis."""
        recommendations = []

        # High-level recommendations based on findings
        if len(attack_paths) > 5:
            recommendations.append(
                "CRITICAL: Multiple attack paths detected. Review tool combinations "
                "and implement capability isolation."
            )

        if priv_esc_vectors:
            recommendations.append(
                "HIGH: Privilege escalation vectors found. Apply least privilege "
                "and sandbox dangerous capabilities."
            )

        # Capability-specific recommendations
        shell_nodes = graph.get_nodes_with_capability(CapabilityType.SHELL_ACCESS)
        if shell_nodes:
            recommendations.append(
                f"MEDIUM: {len(shell_nodes)} tool(s) with shell access. "
                "Consider sandboxing or removing shell capabilities."
            )

        secret_nodes = graph.get_nodes_with_capability(CapabilityType.SECRET_ACCESS)
        if secret_nodes:
            recommendations.append(
                f"HIGH: {len(secret_nodes)} tool(s) with secret access. "
                "Ensure secrets are scoped and rotated regularly."
            )

        # Network exposure
        network_nodes = graph.get_nodes_with_capability(CapabilityType.HTTP_REQUEST)
        if network_nodes:
            recommendations.append(
                f"MEDIUM: {len(network_nodes)} tool(s) with network access. "
                "Implement egress filtering and URL allowlisting."
            )

        if not recommendations:
            recommendations.append(
                "LOW: No critical issues detected. Continue monitoring for "
                "configuration drift and new vulnerabilities."
            )

        return recommendations
