#!/usr/bin/env python3
"""AReaL GRPO 训练脚本 - Actor 驱动环境版本。

使用 AReaL 框架训练 CUA Agent。
训练中的 actor（通过 AReaL 管理的 vLLM）直接驱动 GBox 环境交互。

用法：
    # 单节点多 GPU
    python -m areal.launcher.local train_areal.py --config configs/cua_grpo.yaml
    
    # 多节点（使用 Ray）
    python -m areal.launcher.ray train_areal.py --config configs/cua_grpo.yaml
    
    # 直接运行（用于调试）
    python train_areal.py --config configs/cua_grpo.yaml
"""

import os
import sys
import warnings
import logging
import asyncio
import json
from copy import deepcopy
from datetime import datetime

import torch
import torch.distributed as dist
from datasets import Dataset
from openai import AsyncOpenAI

# Configure logging for CUA Agent
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True,
)

logging.getLogger('cua_agent').setLevel(logging.INFO)
logging.getLogger('gbox_cua').setLevel(logging.INFO)
logging.getLogger('areal').setLevel(logging.INFO)

# ============ Patch AReaL dataset module to support CUA dataset ============
import areal.dataset as areal_dataset_module

_original_get_custom_dataset = areal_dataset_module._get_custom_dataset

def _get_cua_rl_dataset(
    path: str,
    split: str,
    tokenizer,
    max_length: int | None = None,
    **kwargs,
):
    """Load CUA dataset for RL training."""
    if ":" in path:
        module_path, func_name = path.split(":", 1)
    else:
        module_path = "cua_agent.tasks"
        func_name = "get_areal_train_dataset" if split == "train" else "get_areal_eval_dataset"
    
    import importlib
    module = importlib.import_module(module_path)
    get_dataset_func = getattr(module, func_name)
    raw_data = get_dataset_func()
    
    def process_item(item):
        prompt = item.get("prompt", "")
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        
        task = item.get("task")
        task_dict = task.to_dict() if hasattr(task, "to_dict") else (task if task else {})
        metadata = item.get("metadata", {})
        
        result = {"messages": messages}
        
        if task_dict:
            result["answer"] = task_dict.get("description", "")
            result["task_id"] = item.get("id", "")
            # 将 task 的所有可序列化字段保存到 task_metadata 中
            # 这样 workflow 可以重新构建 task 对象
            # 注意：将 task_metadata 序列化为 JSON 字符串，避免 PyArrow 类型推断问题
            task_metadata = metadata.copy() if metadata else {}
            # 添加 task 的完整信息到 metadata（使用 task_dict 中的所有字段）
            task_metadata.update({
                "task_id": task_dict.get("id", item.get("id", "")),
                "task_name": task_dict.get("name", ""),
                "task_description": task_dict.get("description", ""),
                "task_difficulty": task_dict.get("difficulty", "medium"),
                "task_category": task_dict.get("category", "system"),
                "max_steps": task_dict.get("max_steps", 15),
                "validation_type": task_dict.get("validation_type", "state"),
                "validation_query": task_dict.get("validation_query"),
                "expected_result": task_dict.get("expected_result"),
                "tags": task_dict.get("tags", []),
                "prerequisites": task_dict.get("prerequisites", []),
            })
            # 序列化为 JSON 字符串，避免 PyArrow 类型推断问题
            result["task_metadata"] = json.dumps(task_metadata, ensure_ascii=False)
        
        return result
    
    processed_data = [process_item(item) for item in raw_data]
    dataset = Dataset.from_list(processed_data)
    
    if max_length is not None and tokenizer is not None:
        def filter_length(sample):
            messages = sample.get("messages", [])
            if messages:
                content = messages[0].get("content", "")
                if content:
                    tokens = tokenizer.encode(content, add_special_tokens=False)
                    return len(tokens) <= max_length
            return True
        dataset = dataset.filter(filter_length)
    
    return dataset


def _patched_get_custom_dataset(
    path: str,
    type: str = "sft",
    split: str | None = None,
    max_length: int | None = None,
    tokenizer=None,
    processor=None,
    **kwargs,
):
    """Patched version of _get_custom_dataset that supports CUA dataset."""
    if ("cua" in path.lower() or "cua_agent" in path) and type == "rl":
        return _get_cua_rl_dataset(
            path=path,
            split=split or "train",
            tokenizer=tokenizer,
            max_length=max_length,
            **kwargs,
        )
    
    return _original_get_custom_dataset(
        path=path,
        type=type,
        split=split,
        max_length=max_length,
        tokenizer=tokenizer,
        processor=processor,
        **kwargs,
    )

areal_dataset_module._get_custom_dataset = _patched_get_custom_dataset
if "cua" not in areal_dataset_module.VALID_DATASETS:
    areal_dataset_module.VALID_DATASETS.append("cua")
# ============ End of patch ============

# AReaL imports
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.dataset import get_custom_dataset
from areal.engine.ppo.actor import FSDPPPOActor
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.platforms import current_platform
from areal.utils import seeding, stats_tracker
from areal.utils.dataloader import create_dataloader
from areal.utils.device import log_gpu_stats
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger
from areal.workflow.rlvr import RLVRWorkflow

# 项目 imports
from cua_agent.tasks import CUATask, TaskDifficulty, TaskCategory
from cua_agent.areal_env import GBoxEnvConfig, GBoxActorEnv

warnings.filterwarnings("ignore", category=DeprecationWarning)


def cua_reward_fn(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """CUA 奖励函数 - 直接使用 workflow 传入的终局奖励。"""
    logger = logging.getLogger(__name__)
    
    if isinstance(completions, list):
        completion = completions[0] if completions else ""
    else:
        completion = completions
    
    # 直接使用 workflow 传入的 reward
    reward = kwargs.get("reward")
    if reward is None:
        task_success = kwargs.get("task_success", False)
        task_completed = kwargs.get("task_completed", False)
        reward = 1.0 if (task_success or task_completed) else 0.0
    
    try:
        reward = float(reward)
    except Exception:
        reward = 0.0
    
    rank = int(os.getenv("RANK", "0"))
    if rank == 0:
        task_id = kwargs.get("task_id", "N/A")
        num_turns = kwargs.get("num_turns", 0)
        max_turns = kwargs.get("max_turns", 15)
        task_success = kwargs.get("task_success", False)
        errors = kwargs.get("errors", [])
        
        logger.info("=" * 80)
        logger.info(f"[CUA Rollout Complete] Task ID: {task_id}")
        logger.info(f"  Turns: {num_turns}/{max_turns}")
        logger.info(f"  Task Success: {task_success}")
        logger.info(f"  Reward: {reward:.4f}")
        if errors:
            logger.info(f"  Errors ({len(errors)}): {errors[:3]}")
        logger.info("=" * 80)
    
    return reward


class CUAEnvRolloutWorkflow(RLVRWorkflow):
    """Workflow：训练中的 actor 直接驱动 GBox 环境交互。
    
    核心流程：
    1. 为每个 sample 创建 GBoxActorEnv
    2. 使用 AsyncOpenAI 客户端调用 AReaL 管理的 vLLM
    3. 多轮交互：构造 messages → vLLM 生成 tool_call → 环境执行 → 更新上下文
    4. 终局计算 reward（1/0）
    5. 返回带有 reward 的 sample
    """

    def __init__(self, *args, **kwargs):
        # 提取自定义参数
        self.gbox_api_key = kwargs.pop("gbox_api_key", None) or os.getenv("GBOX_API_KEY", "")
        self.gbox_model = kwargs.pop("gbox_model", None) or os.getenv("GBOX_MODEL", "gbox-handy-1")
        self.max_turns = kwargs.pop("max_turns", None) or int(os.getenv("CUA_MAX_TURNS", "15"))
        self.context_window = kwargs.pop("context_window", None) or int(os.getenv("CUA_CONTEXT_WINDOW", "5"))
        self.trace_dir = kwargs.pop("trace_dir", None)
        
        # vLLM 端点：从 AReaL 环境变量获取
        # 注意：AReaL 在启动时会设置 AREAL_LLM_SERVER_ADDRS 环境变量
        # 格式为 "host:port" 或 "host1:port1,host2:port2"
        areal_llm_addrs = os.getenv("AREAL_LLM_SERVER_ADDRS", "")
        if areal_llm_addrs:
            # 格式可能是 "host:port" 或 "host1:port1,host2:port2"，取第一个
            first_addr = areal_llm_addrs.split(",")[0].strip()
            if ":" in first_addr:
                host, port = first_addr.rsplit(":", 1)
                self.vllm_api_base = f"http://{host}:{port}/v1"
            else:
                self.vllm_api_base = f"http://{first_addr}/v1"
        else:
            # 如果环境变量不存在，尝试从 VLLM_API_BASE 获取（用户手动设置）
            vllm_api_base = os.getenv("VLLM_API_BASE", "")
            if vllm_api_base:
                self.vllm_api_base = vllm_api_base if vllm_api_base.endswith("/v1") else f"{vllm_api_base}/v1"
            else:
                # 最后回退到默认值
                self.vllm_api_base = "http://localhost:8000/v1"
                self._logger.warning(
                    "AREAL_LLM_SERVER_ADDRS not set, using default vLLM API base: "
                    f"{self.vllm_api_base}. This may not work correctly."
                )
        
        self.model_name = kwargs.pop("model_name", None) or ""
        
        super().__init__(*args, **kwargs)
        
        self._logger = logging.getLogger(__name__)
        
        if not self.gbox_api_key:
            self._logger.warning(
                "CUAEnvRolloutWorkflow initialized without GBOX_API_KEY; "
                "will fall back to text-only workflow."
            )
    
    def _build_env_config(self) -> GBoxEnvConfig:
        """构建环境配置。"""
        return GBoxEnvConfig(
            gbox_api_key=self.gbox_api_key,
            gbox_model=self.gbox_model,
            box_type="android",
            timeout="60s",
            wait=30,
            expires_in="15m",
            action_delay=0.5,
            screenshot_delay=0.3,
            max_turns=self.max_turns,
            context_window=self.context_window,
            trace_dir=self.trace_dir,
        )
    
    def _parse_action_from_text(self, text: str, turn: int) -> dict:
        """从模型输出的文本中解析动作（备选模式）。
        
        支持以下格式：
        1. JSON 格式: {"action": "click", "target": "..."}
        2. 函数调用格式: click(target="...")
        3. Markdown 代码块中的 JSON
        """
        import re
        
        # 尝试从 Markdown 代码块中提取 JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                action_data = json.loads(json_match.group(1))
                return self._action_data_to_tool_call(action_data, turn)
            except json.JSONDecodeError:
                pass
        
        # 尝试直接解析 JSON
        try:
            # 查找第一个 { 到最后一个 }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and start < end:
                json_str = text[start:end+1]
                action_data = json.loads(json_str)
                return self._action_data_to_tool_call(action_data, turn)
        except json.JSONDecodeError:
            pass
        
        # 尝试解析函数调用格式
        func_match = re.search(r'(\w+)\s*\(([^)]*)\)', text)
        if func_match:
            func_name = func_match.group(1).lower()
            args_str = func_match.group(2)
            
            # 简单解析参数
            args = {}
            for arg_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                args[arg_match.group(1)] = arg_match.group(2)
            
            return {
                "id": f"call_{turn}",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args),
                },
            }
        
        # 检查是否包含 task_complete 相关关键词
        if any(kw in text.lower() for kw in ["task complete", "completed", "done", "finished"]):
            success = any(kw in text.lower() for kw in ["success", "successfully", "accomplished"])
            return {
                "id": f"call_{turn}_complete",
                "function": {
                    "name": "task_complete",
                    "arguments": json.dumps({"success": success, "message": text[:200]}),
                },
            }
        
        # 无法解析，默认返回 task_complete with failure
        self._logger.warning(
            f"[Turn {turn}] Could not parse action from text: {text[:100]}..."
        )
        return {
            "id": f"call_{turn}_parse_failed",
            "function": {
                "name": "task_complete",
                "arguments": json.dumps({
                    "success": False,
                    "message": f"Failed to parse action from model output: {text[:100]}",
                }),
            },
        }
    
    def _action_data_to_tool_call(self, action_data: dict, turn: int) -> dict:
        """将解析的动作数据转换为 tool_call 格式。"""
        action_type = action_data.get("action", action_data.get("type", ""))
        
        # 构建参数
        args = {}
        for key in ["target", "text", "direction", "duration", "key", "button", "success", "message"]:
            if key in action_data:
                args[key] = action_data[key]
        
        return {
            "id": f"call_{turn}",
            "function": {
                "name": action_type or "task_complete",
                "arguments": json.dumps(args),
            },
        }
    
    def _build_task(self, sample: dict) -> CUATask:
        """从 sample 构建 CUATask。"""
        # task_metadata 现在是 JSON 字符串，需要反序列化
        task_metadata_str = sample.get("task_metadata", "")
        if isinstance(task_metadata_str, str):
            try:
                task_metadata = json.loads(task_metadata_str) if task_metadata_str else {}
            except (json.JSONDecodeError, TypeError):
                task_metadata = {}
        else:
            task_metadata = task_metadata_str or {}
        
        # 从 task_metadata 中获取所有信息
        task_id = task_metadata.get("task_id") or sample.get("task_id", "") or f"task_{datetime.now().strftime('%H%M%S')}"
        task_name = task_metadata.get("task_name", task_id)
        task_description = (
            task_metadata.get("task_description") or 
            sample.get("answer") or 
            sample.get("messages", [{}])[0].get("content", "")
        )
        
        # 解析 difficulty 和 category（可能是字符串）
        difficulty_str = task_metadata.get("task_difficulty", "medium")
        try:
            difficulty = TaskDifficulty(difficulty_str)
        except (ValueError, TypeError):
            difficulty = TaskDifficulty.MEDIUM
        
        category_str = task_metadata.get("task_category", "system")
        try:
            category = TaskCategory(category_str)
        except (ValueError, TypeError):
            category = TaskCategory.SYSTEM
        
        return CUATask(
            id=task_id,
            name=task_name,
            description=task_description,
            difficulty=difficulty,
            category=category,
            max_steps=task_metadata.get("max_steps", self.max_turns),
            validation_query=task_metadata.get("validation_query"),
            expected_result=task_metadata.get("expected_result"),
        )
    
    async def _run_env_rollout(self, sample: dict) -> dict:
        """运行环境 rollout：actor 驱动的多轮交互。
        
        Args:
            sample: 输入 sample
            
        Returns:
            更新后的 sample，包含 reward 和执行信息
        """
        task = self._build_task(sample)
        env_config = self._build_env_config()
        rollout_id = f"{task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self._logger.info(f"[CUAEnvRollout] Starting task: {task.id}, rollout: {rollout_id}")
        self._logger.info(f"[CUAEnvRollout] vLLM API base: {self.vllm_api_base}")
        
        # 创建 vLLM 客户端（指向 AReaL 管理的 vLLM）
        vllm_client = AsyncOpenAI(
            base_url=self.vllm_api_base,
            api_key="not-needed",  # vLLM 不需要 API key
        )
        
        # 确定模型名称
        model_name = self.model_name
        if not model_name:
            # 尝试从 vLLM 获取
            try:
                models = await vllm_client.models.list()
                if models.data:
                    model_name = models.data[0].id
            except Exception:
                model_name = "default"
        
        env = GBoxActorEnv(env_config, task, rollout_id)
        
        try:
            # 重置环境，获取初始 observation
            obs = await env.reset()
            messages = obs["messages"]
            tools = obs["tools"]
            done = obs["done"]
            
            # 多轮交互循环
            while not done:
                # 调用 vLLM 生成 tool_call
                try:
                    # 首先尝试使用 tools 参数（需要 vLLM 支持工具调用）
                    try:
                        response = await vllm_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            tools=tools,
                            tool_choice="auto",
                            max_tokens=1024,
                            temperature=0.7,
                        )
                        use_tool_calls = True
                    except Exception as tool_error:
                        # 如果 vLLM 不支持工具调用，使用普通模式并解析 JSON
                        self._logger.warning(
                            f"[Turn {env.num_turns + 1}] vLLM tool calling failed: {tool_error}, "
                            "falling back to JSON parsing mode"
                        )
                        response = await vllm_client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            max_tokens=1024,
                            temperature=0.7,
                        )
                        use_tool_calls = False
                    
                    choice = response.choices[0]
                    
                    # 检查是否有 tool_call
                    if use_tool_calls and choice.message.tool_calls:
                        tool_call = choice.message.tool_calls[0]
                        tool_call_dict = {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        
                        self._logger.info(
                            f"[Turn {env.num_turns + 1}] vLLM generated tool_call: "
                            f"{tool_call.function.name}"
                        )
                    else:
                        # 尝试从文本内容中解析 JSON 格式的动作
                        content = choice.message.content or ""
                        tool_call_dict = self._parse_action_from_text(content, env.num_turns + 1)
                    
                    # 执行环境步骤
                    obs, reward, done, info = await env.step(tool_call_dict)
                    
                    if not done:
                        messages = obs["messages"]
                        tools = obs["tools"]
                        
                except Exception as e:
                    self._logger.error(f"[Turn {env.num_turns + 1}] vLLM call failed: {e}")
                    # 强制结束
                    done = True
                    info = {
                        "task_completed": False,
                        "task_success": False,
                        "error": str(e),
                        "num_turns": env.num_turns,
                        "errors": env.errors + [str(e)],
                    }
                    reward = 0.0
            
            # 计算终局奖励
            final_reward = await env.compute_reward()
            
            # 更新 sample
            sample["task_completed"] = info.get("task_completed", False)
            sample["task_success"] = info.get("task_success", False)
            sample["num_turns"] = info.get("num_turns", env.num_turns)
            sample["max_turns"] = self.max_turns
            sample["errors"] = info.get("errors", env.errors)
            sample["reward"] = final_reward
            sample["completion"] = info.get("result_message", "")
            
            self._logger.info(
                f"[CUAEnvRollout] Task {task.id} completed: "
                f"success={sample['task_success']}, turns={sample['num_turns']}, "
                f"reward={final_reward}"
            )
            
        except Exception as e:
            self._logger.error(f"[CUAEnvRollout] Task {task.id} failed: {e}", exc_info=True)
            sample["task_completed"] = False
            sample["task_success"] = False
            sample["num_turns"] = env.num_turns if env else 0
            sample["max_turns"] = self.max_turns
            sample["errors"] = [str(e)]
            sample["reward"] = 0.0
            sample["completion"] = ""
            
        finally:
            await env.close()
        
        return sample
    
    def __call__(self, sample, **kwargs):
        """执行 workflow。
        
        如果有 GBOX_API_KEY，则运行环境 rollout；否则回退到文本 workflow。
        """
        self._logger.info(f"[CUAEnvRolloutWorkflow.__call__] Called with sample keys: {list(sample.keys())}")
        self._logger.info(f"[CUAEnvRolloutWorkflow.__call__] GBOX_API_KEY present: {bool(self.gbox_api_key)}")
        
        if not self.gbox_api_key:
            self._logger.warning("[CUAEnvRolloutWorkflow.__call__] No GBOX_API_KEY, falling back to text-only workflow")
            return super().__call__(sample, **kwargs)
        
        # 运行环境 rollout
        self._logger.info("[CUAEnvRolloutWorkflow.__call__] Starting environment rollout")
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        sample = loop.run_until_complete(self._run_env_rollout(sample))
        self._logger.info(f"[CUAEnvRolloutWorkflow.__call__] Rollout completed, reward: {sample.get('reward', 'N/A')}")
        
        return sample


def main(args):
    """主函数。"""
    config, _ = load_expr_config(args, GRPOConfig)
    config: GRPOConfig

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 1:
            device_id = 0
        else:
            device_id = local_rank % num_gpus
        
        torch.cuda.set_device(device_id)
        
        if rank == 0:
            print(f"[Rank {rank}] CUDA device assignment: device_id={device_id}, "
                  f"local_rank={local_rank}, world_size={world_size}, num_gpus={num_gpus}")
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None

    # Initialize train engine
    actor = FSDPPPOActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    actual_world_size = dist.get_world_size() if dist.is_initialized() else 1
    actual_rank = dist.get_rank() if dist.is_initialized() else 0
    
    if actual_rank == 0:
        print(f"[Rank {actual_rank}] Detected world_size={actual_world_size}, "
              f"data_parallel_world_size={actor.data_parallel_world_size}")

    # Create dataset and dataloaders
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )

    train_dataloader = create_dataloader(
        train_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.train_dataset,
    )
    valid_dataloader = create_dataloader(
        valid_dataset,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        dataset_config=config.valid_dataset,
    )
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * config.train_dataset.batch_size,
        train_batch_size=config.train_dataset.batch_size,
    )

    # Initialize inference engine
    rollout = RemotevLLMEngine(config.rollout)
    eval_rollout = RemotevLLMEngine(deepcopy(config.rollout))
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()

    # Determine weight update mode
    effective_world_size = actual_world_size if dist.is_initialized() else 1
    effective_dp_size = actor.data_parallel_world_size if hasattr(actor, 'data_parallel_world_size') else 1
    num_visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    if effective_world_size == 1 or effective_dp_size == 1 or num_visible_gpus == 1:
        if actual_rank == 0:
            print(f"[Rank {actual_rank}] Single GPU mode detected, using 'disk' mode for weight updates")
        weight_update_mode = "disk"
    else:
        weight_update_mode = config.actor.weight_update_mode
        if actual_rank == 0:
            print(f"[Rank {actual_rank}] Multi-GPU mode detected, using '{weight_update_mode}' mode")
    
    if weight_update_mode == "xccl":
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
            allocation_mode,
            use_lora=config.actor.use_lora,
            lora_name=getattr(config.gconfig, "lora_name", None),
            lora_int_id=1,
            base_model_name=config.actor.path,
        )
    elif weight_update_mode == "disk":
        weight_update_meta = WeightUpdateMeta.from_disk(
            config.saver.experiment_name,
            config.saver.trial_name,
            config.saver.fileroot,
            use_lora=config.actor.use_lora,
            lora_name=getattr(config.gconfig, "lora_name", None),
            lora_int_id=1,
            base_model_name=config.actor.path,
        )
    else:
        raise ValueError(f"Invalid weight_update_mode: {weight_update_mode}")

    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)

    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
        ref = FSDPPPOActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)

    # Create rollout workflow - 使用新的环境驱动 workflow
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Trace 目录
    trace_dir = os.path.join(
        StatsLogger.get_log_path(config.stats_logger), "trace"
    )
    
    workflow = CUAEnvRolloutWorkflow(
        reward_fn=cua_reward_fn,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        trace_dir=trace_dir,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
    )
    eval_workflow = CUAEnvRolloutWorkflow(
        reward_fn=cua_reward_fn,
        gconfig=config.gconfig.new(temperature=0.6),
        tokenizer=tokenizer,
        enable_thinking=False,
        trace_dir=os.path.join(trace_dir, "eval"),
        rollout_stat_scope="eval-rollout",
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated-eval"
        ),
    )

    # Run training
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)

    recover_handler = RecoverHandler(config.recover, ft_spec)
    recover_info = recover_handler.load(
        actor,
        saver,
        evaluator,
        stats_logger,
        train_dataloader,
        inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = (
        recover_info.last_step_info.next().global_step
        if recover_info is not None
        else 0
    )

    total_epochs = config.total_train_epochs
    steps_per_epoch = len(train_dataloader)
    max_steps = total_epochs * steps_per_epoch

    if actual_rank == 0:
        # 获取 vLLM 地址
        areal_llm_addrs = os.getenv("AREAL_LLM_SERVER_ADDRS", "")
        vllm_api_base = "N/A"
        if areal_llm_addrs:
            first_addr = areal_llm_addrs.split(",")[0].strip()
            if ":" in first_addr:
                host, port = first_addr.rsplit(":", 1)
                vllm_api_base = f"http://{host}:{port}/v1"
            else:
                vllm_api_base = f"http://{first_addr}/v1"
        
        print(f"\n{'='*60}")
        print(f"CUA GRPO Training - Actor Driven Environment")
        print(f"{'='*60}")
        print(f"  vLLM API: {vllm_api_base} (AReaL managed)")
        print(f"  AREAL_LLM_SERVER_ADDRS: {areal_llm_addrs or 'NOT SET'}")
        print(f"  GBox API Key: {'***' + os.getenv('GBOX_API_KEY', '')[-4:] if os.getenv('GBOX_API_KEY') else 'NOT SET'}")
        print(f"  Max Turns: {int(os.getenv('CUA_MAX_TURNS', '15'))}")
        print(f"  Context Window: {int(os.getenv('CUA_CONTEXT_WINDOW', '5'))}")
        print(f"  Trace Dir: {trace_dir}")
        print(f"  Total Steps: {max_steps}")
        print(f"{'='*60}\n")

    for global_step in range(start_step, max_steps):
        epoch = global_step // steps_per_epoch
        step = global_step % steps_per_epoch
        step_info = StepInfo(
            global_step=global_step,
            epoch=epoch,
            epoch_step=step,
            steps_per_epoch=steps_per_epoch,
        )

        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                granularity=actor.config.group_size,
                workflow=workflow,
                should_accept_fn=lambda sample: True,
            )

        if config.actor.recompute_logprob or config.actor.use_decoupled_loss:
            with stats_tracker.record_timing("recompute_logp"):
                logp = actor.compute_logp(batch)
                batch["prox_logp"] = logp
                log_gpu_stats("recompute logp")

        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
                log_gpu_stats("ref logp")

        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
            log_gpu_stats("compute advantages")

        with (
            stats_tracker.record_timing("train_step"),
            stats_tracker.scope("grpo_actor"),
        ):
            stats = actor.ppo_update(batch)
            actor.step_lr_scheduler()
            log_gpu_stats("ppo update")

        rollout.pause()

        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)

        with stats_tracker.record_timing("save"):
            saver.save(actor, epoch, step, global_step, tokenizer=tokenizer)

        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor,
                step_info,
                saver,
                evaluator,
                stats_logger,
                train_dataloader,
                tokenizer=tokenizer,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        with stats_tracker.record_timing("eval"):

            def evaluate_fn():
                if actor.is_data_parallel_head():
                    cnt = 0
                    for data in valid_dataloader:
                        for item in data:
                            eval_rollout.submit(item, eval_workflow)
                            cnt += 1
                    eval_rollout.wait(cnt, timeout=None)
                dist.barrier(device_ids=[actor.device.index])
                current_platform.synchronize()

            evaluator.evaluate(
                evaluate_fn,
                epoch,
                step,
                global_step,
            )

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        stats = actor.export_stats()
        stats_logger.commit(epoch, step, global_step, stats)

        dist.barrier(device_ids=[actor.device.index])
        current_platform.synchronize()

        rollout.resume()

    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
