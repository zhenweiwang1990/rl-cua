"""GBox 环境适配器，用于 AReaL 异步 Rollout Collector。

这个模块将 GBox Android 环境适配到 AReaL 的 BaseEnv 接口。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

# AReaL imports - 注意：这些可能需要根据实际 AReaL API 调整
try:
    from areal.env import BaseEnv, EnvConfig
    from areal.rollout import RolloutResult
except ImportError:
    # 如果 AReaL 未安装，定义占位符
    class BaseEnv:
        pass
    class EnvConfig:
        pass
    class RolloutResult:
        pass

from cua_agent.tasks import CUATask
from cua_agent.reward import simple_reward_function, CUARolloutResult

# 直接使用 gbox-cua 的提示语、工具和上下文管理
from gbox_cua.prompts import create_system_prompt, create_user_message_with_screenshot
from gbox_cua.tools import get_tools_schema
from gbox_cua.agent import StandaloneGBoxCUAAgent

logger = logging.getLogger(__name__)


@dataclass
class GBoxEnvConfig(EnvConfig):
    """GBox 环境配置"""
    gbox_api_key: str = ""
    box_type: str = "android"
    timeout: str = "60s"
    expires_in: str = "15m"
    action_delay: float = 0.5
    screenshot_delay: float = 0.3
    max_turns: int = 15


class GBoxAReaLEnv(BaseEnv):
    """GBox 环境，适配 AReaL 的 BaseEnv 接口。
    
    这个类实现了 AReaL 期望的环境接口：
    - reset(): 初始化任务
    - step(): 执行动作
    - compute_reward(): 计算奖励
    - close(): 清理资源
    """
    
    def __init__(self, config: GBoxEnvConfig):
        """初始化 GBox 环境。
        
        Args:
            config: GBox 环境配置
        """
        self.config = config
        self.agent: Optional[StandaloneGBoxCUAAgent] = None
        self.task: Optional[CUATask] = None
        self.conversation: List[Dict] = []
        self.action_history: List[Dict] = []
        self.num_turns: int = 0
        
        logger.info(f"GBoxAReaLEnv initialized with box_type={config.box_type}")
    
    async def reset(self, task: CUATask) -> Dict[str, Any]:
        """重置环境，准备新任务。
        
        Args:
            task: CUA 任务
            
        Returns:
            初始观察（包含任务描述）
        """
        # 关闭之前的 agent（如果有）
        if self.agent is not None:
            await self.close()
        
        self.task = task
        self.conversation = []
        self.action_history = []
        self.num_turns = 0
        
        # 创建新的 agent（使用 gbox-cua 的 StandaloneGBoxCUAAgent）
        self.agent = StandaloneGBoxCUAAgent(
            gbox_api_key=self.config.gbox_api_key,
            box_type=self.config.box_type,
        )
        
        # 使用 gbox-cua 的提示语生成逻辑
        system_prompt = create_system_prompt(
            task_description=task.description,
            max_turns=self.config.max_turns,
        )
        self.conversation.append({
            "role": "system",
            "content": system_prompt,
        })
        
        logger.info(f"Environment reset for task: {task.id}")
        
        return {
            "task_id": task.id,
            "task_description": task.description,
            "system_prompt": system_prompt,
        }
    
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        """执行一个动作。
        
        Args:
            action: 要执行的动作（来自模型）
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.num_turns += 1
        
        try:
            # 通过 agent 执行动作
            # 注意：StandaloneGBoxCUAAgent 的接口可能需要根据实际情况调整
            result = await self.agent.execute_action(action)
            
            # 更新对话历史
            self.conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [action] if isinstance(action, dict) else action,
            })
            
            # 更新动作历史
            self.action_history.append({
                "action": action,
                "result": result,
                "success": result.get("success", False),
                "turn": self.num_turns,
            })
            
            # 检查是否完成
            done = result.get("done", False) or self.num_turns >= self.config.max_turns
            
            # 获取截图作为观察
            screenshot = await self.agent.take_screenshot() if not done else None
            
            observation = {
                "screenshot": screenshot,
                "action_result": result,
                "turn": self.num_turns,
            }
            
            # 中间奖励（可选）
            intermediate_reward = 0.0
            
            info = {
                "task_completed": result.get("task_completed", False),
                "task_success": result.get("task_success", False),
                "errors": result.get("errors", []),
            }
            
            return observation, intermediate_reward, done, info
            
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            
            self.action_history.append({
                "action": action,
                "error": str(e),
                "success": False,
                "turn": self.num_turns,
            })
            
            observation = {"error": str(e)}
            done = True
            info = {"error": str(e)}
            
            return observation, 0.0, done, info
    
    async def compute_reward(self, trajectory: List[Dict]) -> float:
        """计算轨迹的最终奖励。
        
        Args:
            trajectory: 完整的轨迹（动作和观察序列）
            
        Returns:
            奖励值
        """
        # 构建 rollout 结果
        errors = [
            step.get("error", "")
            for step in self.action_history
            if not step.get("success", True)
        ]
        
        # 检查任务是否成功
        task_completed = any(
            step.get("result", {}).get("task_completed", False)
            for step in self.action_history
        )
        task_success = any(
            step.get("result", {}).get("task_success", False)
            for step in self.action_history
        )
        
        rollout_result = CUARolloutResult(
            task_id=self.task.id,
            task_completed=task_completed,
            task_success=task_success,
            num_turns=self.num_turns,
            max_turns=self.config.max_turns,
            errors=errors,
        )
        
        # 使用现有的奖励函数
        reward = simple_reward_function(rollout_result, self.task)
        
        logger.debug(
            f"Reward computed for task {self.task.id}: "
            f"reward={reward:.3f}, success={task_success}"
        )
        
        return reward
    
    def get_conversation(self) -> List[Dict]:
        """获取完整的对话历史。"""
        return self.conversation.copy()
    
    def get_tools_schema(self) -> List[Dict]:
        """获取工具定义。"""
        return get_tools_schema()
    
    async def close(self):
        """关闭环境，释放资源。"""
        if self.agent is not None:
            try:
                await self.agent.close()
            except Exception as e:
                logger.warning(f"Error closing agent: {e}")
            finally:
                self.agent = None
        
        logger.debug("Environment closed")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def create_env_factory(config: GBoxEnvConfig):
    """创建环境工厂函数，用于 AReaL Rollout Collector。
    
    Args:
        config: GBox 环境配置
        
    Returns:
        环境工厂函数
    """
    def factory() -> GBoxAReaLEnv:
        return GBoxAReaLEnv(config)
    
    return factory


__all__ = [
    "GBoxEnvConfig",
    "GBoxAReaLEnv",
    "create_env_factory",
]

