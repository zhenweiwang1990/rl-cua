"""Tool definitions for CUA Agent.

This module re-exports shared tool definitions from gbox-cua package.
Install with: pip install git+https://github.com/babelcloud/gbox-cua.git
"""

from gbox_cua.tools import (
    get_tools_schema,
    tool_call_to_action_dict,
    PERFORM_ACTION_TOOL,
    SLEEP_TOOL,
    REPORT_TASK_COMPLETE_TOOL,
    GBOX_CUA_TOOLS_SCHEMA as CUA_TOOLS_SCHEMA,
)

__all__ = [
    "get_tools_schema",
    "tool_call_to_action_dict",
    "PERFORM_ACTION_TOOL",
    "SLEEP_TOOL",
    "REPORT_TASK_COMPLETE_TOOL",
    "CUA_TOOLS_SCHEMA",
]

