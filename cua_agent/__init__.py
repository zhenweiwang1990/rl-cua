"""CUA (Computer Use Agent) for GRPO training."""

from cua_agent.config import CUAConfig, GBoxConfig
from gbox_cua.gbox_client import GBoxClient
from cua_agent.actions import ActionType, Target, CUAAction, parse_action, action_to_dict
from cua_agent.tools import CUA_TOOLS_SCHEMA, get_tools_schema
from cua_agent.agent import CUAAgent

__all__ = [
    "CUAConfig",
    "GBoxConfig",
    "GBoxClient",
    "ActionType",
    "Target",
    "CUAAction",
    "parse_action",
    "action_to_dict",
    "CUA_TOOLS_SCHEMA",
    "get_tools_schema",
    "CUAAgent",
]

