"""GBox 环境适配器，用于 AReaL 异步 Rollout。

这个模块将 GBox Android 环境适配为可由训练中的 actor 驱动的接口。
核心功能：
- tool_call → CUAAction → gbox 执行
- 截图以 base64 data URL 格式返回
- 上下文管理（保留系统提示 + 工具 schema + 最近 N 轮）
- 终局奖励计算（1/0）
- Trace 落盘（JSON + 截图）
"""

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from gbox_cua.gbox_client import GBoxClient
from gbox_cua.prompts import create_system_prompt

from cua_agent.tasks import CUATask
from cua_agent.actions import parse_action, ActionType
from cua_agent.tools import get_tools_schema, tool_call_to_action_dict

logger = logging.getLogger(__name__)


@dataclass
class GBoxEnvConfig:
    """GBox 环境配置"""
    gbox_api_key: str = ""
    gbox_model: str = "gbox-handy-1"
    box_type: str = "android"
    timeout: str = "60s"
    wait: int = 30
    expires_in: str = "15m"
    action_delay: float = 0.5
    screenshot_delay: float = 0.3
    max_turns: int = 15
    context_window: int = 5  # 保留最近 N 轮对话
    trace_dir: Optional[str] = None  # Trace 落盘目录


@dataclass
class TurnRecord:
    """单轮记录，用于上下文管理和 trace"""
    turn: int
    messages: List[Dict]  # 该轮的 messages（assistant + tool response）
    tool_call: Optional[Dict] = None
    action_type: Optional[str] = None
    action_result: Optional[Dict] = None
    screenshot_path: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    timestamp: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "turn": self.turn,
            "tool_call": self.tool_call,
            "action_type": self.action_type,
            "action_result": self.action_result,
            "screenshot_path": self.screenshot_path,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class GBoxActorEnv:
    """GBox 环境，由训练中的 actor 驱动。
    
    核心功能：
    1. 接收 actor 的 tool_call 输出
    2. 转换为 CUAAction 并执行 gbox
    3. 返回包含截图的 observation（base64 data URL）
    4. 管理对话上下文（保留系统提示 + 工具 schema + 最近 N 轮）
    5. 终局计算 reward（1/0）
    6. 落盘 trace（JSON + 截图）
    """
    
    def __init__(self, config: GBoxEnvConfig, task: CUATask, rollout_id: str = ""):
        """初始化环境。
        
        Args:
            config: 环境配置
            task: 任务定义
            rollout_id: Rollout ID，用于 trace 落盘
        """
        self.config = config
        self.task = task
        self.rollout_id = rollout_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # GBox 客户端
        self.gbox_client: Optional[GBoxClient] = None
        
        # 状态
        self.num_turns: int = 0
        self.turn_records: List[TurnRecord] = []
        self.task_completed: bool = False
        self.task_success: bool = False
        self.errors: List[str] = []
        
        # 系统提示和工具
        self.system_prompt = create_system_prompt(
            task_description=task.description,
            max_turns=config.max_turns,
        )
        self.tools_schema = get_tools_schema()
        
        # Trace 目录
        self.trace_path: Optional[Path] = None
        if config.trace_dir:
            self.trace_path = Path(config.trace_dir) / task.id / self.rollout_id
            self.trace_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GBoxActorEnv initialized for task: {task.id}, rollout: {self.rollout_id}")
    
    async def reset(self) -> Dict[str, Any]:
        """重置环境，创建 GBox 并获取初始截图。
        
        Returns:
            初始观察：包含 messages（含系统提示和初始用户消息 + 截图）
        """
        # 关闭之前的 GBox（如果有）
        await self.close()
        
        # 重置状态
        self.num_turns = 0
        self.turn_records = []
        self.task_completed = False
        self.task_success = False
        self.errors = []
        
        # 创建 GBox
        self.gbox_client = GBoxClient(
            api_key=self.config.gbox_api_key,
            model=self.config.gbox_model,
            box_type=self.config.box_type,
            timeout=self.config.timeout,
            wait=self.config.wait,
            expires_in=self.config.expires_in,
            labels=None,
            envs=None,
        )
        await self.gbox_client.create_box(self.config.box_type)
        
        logger.info(f"GBox created: {self.gbox_client.box_id}")
        
        # 等待 box 就绪
        await asyncio.sleep(2.0)
        
        # 获取初始截图
        screenshot_bytes, _ = await self.gbox_client.take_screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        screenshot_data_url = f"data:image/png;base64,{screenshot_b64}"
        
        # 保存初始截图
        if self.trace_path:
            screenshot_path = self.trace_path / "turn_0_initial.png"
            screenshot_path.write_bytes(screenshot_bytes)
        
        # 构造初始 messages
        initial_messages = self._build_initial_messages(screenshot_data_url)
        
        return {
            "messages": initial_messages,
            "tools": self.tools_schema,
            "task_id": self.task.id,
            "task_description": self.task.description,
            "turn": 0,
            "done": False,
        }
    
    def _build_initial_messages(self, screenshot_data_url: str) -> List[Dict]:
        """构造初始 messages。
        
        Args:
            screenshot_data_url: 截图的 base64 data URL
            
        Returns:
            初始 messages 列表
        """
        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Turn 1/{self.config.max_turns}. Please analyze the screenshot and take the next action to complete the task.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_data_url},
                    },
                ],
            },
        ]
    
    async def step(self, tool_call: Dict) -> Tuple[Dict, float, bool, Dict]:
        """执行一步：接收 tool_call，执行动作，返回 observation。
        
        Args:
            tool_call: 模型输出的 tool_call，格式：
                {
                    "id": "call_xxx",
                    "function": {
                        "name": "click",
                        "arguments": "{\"target\": \"Settings icon\"}"
                    }
                }
                
        Returns:
            Tuple of (observation, reward, done, info)
            - observation: 包含下一轮 messages（裁剪后的上下文 + 新截图）
            - reward: 中间奖励（始终为 0，终局时计算）
            - done: 是否结束
            - info: 额外信息
        """
        self.num_turns += 1
        turn_start = time.time()
        
        # 解析 tool_call
        try:
            function_name = tool_call.get("function", {}).get("name", "")
            function_args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                function_args = json.loads(function_args_str)
            except json.JSONDecodeError:
                function_args = {}
            
            logger.info(f"[Turn {self.num_turns}] Tool call: {function_name}, args: {function_args}")
            
        except Exception as e:
            logger.error(f"[Turn {self.num_turns}] Failed to parse tool_call: {e}")
            self.errors.append(f"Parse error: {e}")
            return self._make_error_observation(str(e))
        
        # 转换为 CUAAction
        try:
            action_dict = tool_call_to_action_dict(function_name, function_args)
            action = parse_action(action_dict)
            
            if action is None:
                raise ValueError(f"Failed to parse action from tool_call: {function_name}")
            
            logger.info(f"[Turn {self.num_turns}] Parsed action: {action.action_type}")
            
        except Exception as e:
            logger.error(f"[Turn {self.num_turns}] Failed to convert tool_call to action: {e}")
            self.errors.append(f"Action parse error: {e}")
            return self._make_error_observation(str(e))
        
        # 检查是否是 task_complete
        if action.action_type == ActionType.TASK_COMPLETE:
            self.task_completed = True
            self.task_success = getattr(action, 'success', False)
            result_message = getattr(action, 'result_message', '')
            
            logger.info(f"[Turn {self.num_turns}] Task complete: success={self.task_success}, message={result_message}")
            
            # 记录这一轮
            record = TurnRecord(
                turn=self.num_turns,
                messages=[],
                tool_call=tool_call,
                action_type="task_complete",
                action_result={"success": self.task_success, "message": result_message},
                success=True,
                timestamp=datetime.now().isoformat(),
            )
            self.turn_records.append(record)
            
            # 保存 trace
            await self._save_trace()
            
            # 计算终局奖励
            reward = 1.0 if self.task_success else 0.0
            
            return {
                "messages": [],
                "tools": self.tools_schema,
                "task_id": self.task.id,
                "turn": self.num_turns,
                "done": True,
            }, reward, True, {
                "task_completed": True,
                "task_success": self.task_success,
                "result_message": result_message,
                "num_turns": self.num_turns,
                "errors": self.errors,
            }
        
        # 执行动作
        action_result = {}
        action_success = True
        action_error = None
        
        try:
            await asyncio.sleep(self.config.action_delay)
            
            # 根据动作类型执行
            if action.action_type == ActionType.CLICK:
                target_desc = action.target.to_description() if action.target else ""
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                result = await self.gbox_client.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="click",
                    target=target_desc,
                )
                coords = result.get("response", {}).get("coordinates", {})
                x = coords.get("x", 0)
                y = coords.get("y", 0)
                if x or y:
                    await self.gbox_client.click(x=x, y=y, button="left", double_click=False)
                    action_result = {"type": "click", "target": target_desc, "coords": {"x": x, "y": y}}
                else:
                    raise ValueError(f"Could not find coordinates for: {target_desc}")
                    
            elif action.action_type == ActionType.INPUT:
                text = getattr(action, 'text', '') or ""
                target_desc = action.target.to_description() if action.target else ""
                if target_desc:
                    screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                    result = await self.gbox_client.generate_coordinates(
                        screenshot_uri=screenshot_uri,
                        action_type="click",
                        target=target_desc,
                    )
                    coords = result.get("response", {}).get("coordinates", {})
                    x = coords.get("x", 0)
                    y = coords.get("y", 0)
                    if x or y:
                        await self.gbox_client.click(x=x, y=y, button="left", double_click=False)
                        await asyncio.sleep(0.3)
                await self.gbox_client.type_text(text)
                action_result = {"type": "input", "text": text, "target": target_desc}
                    
            elif action.action_type == ActionType.SCROLL:
                direction = getattr(action, 'direction', 'down')
                # scroll 使用 generate_coordinates 获取坐标
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                result = await self.gbox_client.generate_coordinates(
                    screenshot_uri=screenshot_uri,
                    action_type="scroll",
                    location=direction,
                )
                await self.gbox_client.scroll(direction=direction)
                action_result = {"type": "scroll", "direction": direction}
                
            elif action.action_type == ActionType.SWIPE:
                direction = getattr(action, 'direction', 'up')
                start_target = action.start_target.to_description() if hasattr(action, 'start_target') and action.start_target else "screen center"
                end_target = action.end_target.to_description() if hasattr(action, 'end_target') and action.end_target else "screen center"
                screenshot_bytes, screenshot_uri = await self.gbox_client.take_screenshot()
                await self.gbox_client.swipe(
                    start_x=0, start_y=0, end_x=0, end_y=0, 
                    direction=direction
                )
                action_result = {"type": "swipe", "direction": direction}
                
            elif action.action_type == ActionType.KEY_PRESS:
                keys = getattr(action, 'keys', None) or getattr(action, 'key', 'home')
                if isinstance(keys, str):
                    keys = [keys]
                await self.gbox_client.press_key(keys=keys)
                action_result = {"type": "key_press", "keys": keys}
                
            elif action.action_type == ActionType.BUTTON_PRESS:
                button = getattr(action, 'button', 'home')
                await self.gbox_client.press_button(button=button)
                action_result = {"type": "button_press", "button": button}
                
            elif action.action_type == ActionType.WAIT:
                duration = getattr(action, 'duration', 1.0)
                await asyncio.sleep(duration)
                action_result = {"type": "wait", "duration": duration}
                
            else:
                raise ValueError(f"Unknown action type: {action.action_type}")
            
            logger.info(f"[Turn {self.num_turns}] Action executed: {action_result}")
            
        except Exception as e:
            action_success = False
            action_error = str(e)
            self.errors.append(f"Action error: {e}")
            logger.error(f"[Turn {self.num_turns}] Action failed: {e}")
        
        # 等待动作生效
        await asyncio.sleep(self.config.screenshot_delay)
        
        # 获取新截图
        try:
            screenshot_bytes, _ = await self.gbox_client.take_screenshot()
            screenshot_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            screenshot_data_url = f"data:image/png;base64,{screenshot_b64}"
            
            # 保存截图
            screenshot_path = None
            if self.trace_path:
                screenshot_path = self.trace_path / f"turn_{self.num_turns}.png"
                screenshot_path.write_bytes(screenshot_bytes)
                screenshot_path = str(screenshot_path)
                
        except Exception as e:
            logger.error(f"[Turn {self.num_turns}] Failed to take screenshot: {e}")
            screenshot_data_url = None
            screenshot_path = None
        
        # 记录这一轮
        record = TurnRecord(
            turn=self.num_turns,
            messages=[],
            tool_call=tool_call,
            action_type=str(action.action_type),
            action_result=action_result,
            screenshot_path=screenshot_path,
            success=action_success,
            error=action_error,
            timestamp=datetime.now().isoformat(),
        )
        self.turn_records.append(record)
        
        # 检查是否超过最大轮数
        done = self.num_turns >= self.config.max_turns
        
        if done:
            logger.info(f"[Turn {self.num_turns}] Max turns reached")
            await self._save_trace()
        
        # 构造下一轮的 messages（裁剪上下文）
        next_messages = self._build_next_messages(
            tool_call=tool_call,
            action_result=action_result,
            action_success=action_success,
            action_error=action_error,
            screenshot_data_url=screenshot_data_url,
        )
        
        # 中间奖励始终为 0
        intermediate_reward = 0.0
        
        # 如果结束，计算终局奖励
        if done:
            intermediate_reward = 1.0 if self.task_success else 0.0
        
        return {
            "messages": next_messages,
            "tools": self.tools_schema,
            "task_id": self.task.id,
            "turn": self.num_turns,
            "done": done,
        }, intermediate_reward, done, {
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "action_result": action_result,
            "action_success": action_success,
            "action_error": action_error,
            "num_turns": self.num_turns,
            "errors": self.errors,
        }
    
    def _build_next_messages(
        self,
        tool_call: Dict,
        action_result: Dict,
        action_success: bool,
        action_error: Optional[str],
        screenshot_data_url: Optional[str],
    ) -> List[Dict]:
        """构造下一轮的 messages，保留系统提示 + 工具 schema + 最近 N 轮。
        
        Args:
            tool_call: 刚执行的 tool_call
            action_result: 动作执行结果
            action_success: 动作是否成功
            action_error: 错误信息（如果失败）
            screenshot_data_url: 新截图的 data URL
            
        Returns:
            裁剪后的 messages 列表
        """
        messages = []
        
        # 1. 系统提示（始终保留）
        messages.append({
            "role": "system",
            "content": self.system_prompt,
        })
        
        # 2. 最近 N 轮的摘要
        recent_turns = self.turn_records[-self.config.context_window:]
        for record in recent_turns:
            # 添加 assistant 的 tool_call
            if record.tool_call:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [record.tool_call],
                })
                
                # 添加 tool 响应
                tool_response = {
                    "status": "success" if record.success else "error",
                    "action": record.action_type,
                }
                if record.action_result:
                    tool_response["result"] = record.action_result
                if record.error:
                    tool_response["error"] = record.error
                    
                messages.append({
                    "role": "tool",
                    "tool_call_id": record.tool_call.get("id", "call_0"),
                    "content": json.dumps(tool_response),
                })
        
        # 3. 当前截图和下一步提示
        if screenshot_data_url:
            next_turn = self.num_turns + 1
            user_content = [
                {
                    "type": "text",
                    "text": f"Turn {next_turn}/{self.config.max_turns}. "
                            f"Previous action: {'succeeded' if action_success else 'failed'}. "
                            f"Please analyze the new screenshot and take the next action.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": screenshot_data_url},
                },
            ]
            
            if action_error:
                user_content[0]["text"] += f" Error: {action_error}"
                
            messages.append({
                "role": "user",
                "content": user_content,
            })
        
        return messages
    
    def _make_error_observation(self, error: str) -> Tuple[Dict, float, bool, Dict]:
        """构造错误 observation。"""
        return {
            "messages": [],
            "tools": self.tools_schema,
            "task_id": self.task.id,
            "turn": self.num_turns,
            "done": True,
        }, 0.0, True, {
            "task_completed": False,
            "task_success": False,
            "error": error,
            "num_turns": self.num_turns,
            "errors": self.errors,
        }
    
    async def compute_reward(self) -> float:
        """计算终局奖励。
        
        Returns:
            奖励值：成功 1.0，失败 0.0
        """
        reward = 1.0 if self.task_success else 0.0
        logger.info(f"Reward computed for task {self.task.id}: {reward} (success={self.task_success})")
        return reward
    
    async def _save_trace(self):
        """保存 trace 到文件。"""
        if not self.trace_path:
            return
            
        trace_data = {
            "task_id": self.task.id,
            "task_description": self.task.description,
            "rollout_id": self.rollout_id,
            "num_turns": self.num_turns,
            "task_completed": self.task_completed,
            "task_success": self.task_success,
            "reward": 1.0 if self.task_success else 0.0,
            "errors": self.errors,
            "turns": [record.to_dict() for record in self.turn_records],
            "timestamp": datetime.now().isoformat(),
        }
        
        trace_file = self.trace_path / "trace.json"
        with open(trace_file, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Trace saved to: {trace_file}")
    
    async def close(self):
        """关闭环境，释放资源。"""
        if self.gbox_client is not None:
            try:
                await self.gbox_client.terminate_box()
                logger.info(f"GBox terminated: {self.gbox_client.box_id}")
            except Exception as e:
                logger.warning(f"Error terminating GBox: {e}")
            finally:
                self.gbox_client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def create_env_factory(config: GBoxEnvConfig):
    """创建环境工厂函数。
    
    Args:
        config: GBox 环境配置
        
    Returns:
        环境工厂函数
    """
    def factory(task: CUATask, rollout_id: str = "") -> GBoxActorEnv:
        return GBoxActorEnv(config, task, rollout_id)
    
    return factory


__all__ = [
    "GBoxEnvConfig",
    "GBoxActorEnv",
    "TurnRecord",
    "create_env_factory",
]
