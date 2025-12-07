"""MCP configuration models."""

from __future__ import annotations

from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field


class MCPTool(BaseModel):
    """Represents a tool defined in an MCP server."""

    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="Input JSON schema")


class MCPServer(BaseModel):
    """Represents an MCP server configuration."""

    name: str = Field(..., description="Server name/identifier")
    command: Optional[str] = Field(None, description="Command to run the server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    url: Optional[str] = Field(None, description="Server URL for remote servers")
    tools: List[MCPTool] = Field(default_factory=list, description="Tools provided by server")
    raw_config: Dict[str, Any] = Field(default_factory=dict, description="Original raw config")


class MCPConfig(BaseModel):
    """Represents a complete MCP configuration file."""

    file_path: str = Field(..., description="Path to the config file")
    servers: Dict[str, MCPServer] = Field(
        default_factory=dict, description="Server configurations keyed by name"
    )
    raw_content: str = Field("", description="Raw file content")
    parse_errors: List[str] = Field(default_factory=list, description="Any parsing errors")
