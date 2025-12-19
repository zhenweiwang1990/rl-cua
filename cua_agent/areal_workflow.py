"""AReaL Workflow for CUA Agent training.

This module implements CUAEnvRolloutWorkflow, which handles multi-turn
environment interactions using AReaL's InferenceEngine.
"""

import os
import json
import logging
import uuid
import aiofiles
import aiofiles.os
import colorama
import torch
from datetime import datetime
from typing import Any

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.image import image2base64
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)
from transformers import PreTrainedTokenizerFast, AutoProcessor

from cua_agent.tasks import CUATask, TaskDifficulty, TaskCategory
from cua_agent.areal_env import GBoxEnvConfig, GBoxActorEnv

logger = logging.getLogger(__name__)


class CUAEnvRolloutWorkflow(RolloutWorkflow):
    """Workflow：训练中的 actor 直接驱动 GBox 环境交互。
    
    核心流程：
    1. 为每个 sample 创建 GBoxActorEnv
    2. 使用 InferenceEngine.agenerate 获取模型输出
    3. 多轮交互：构造 messages → 模型生成 tool_call → 环境执行 → 更新上下文
    4. 处理截图（类似 VisionRLVRWorkflow）
    5. 终局计算 reward（1/0）
    6. 返回符合 AReaL 格式的 tensor 数据
    """

    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: AutoProcessor | None = None,
        enable_thinking: bool = True,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        **kwargs,
    ):
        # 提取自定义参数
        self.gbox_api_key = kwargs.pop("gbox_api_key", None) or os.getenv("GBOX_API_KEY", "")
        self.gbox_model = kwargs.pop("gbox_model", None) or os.getenv("GBOX_MODEL", "gbox-handy-1")
        self.max_turns = kwargs.pop("max_turns", None) or int(os.getenv("CUA_MAX_TURNS", "15"))
        self.context_window = kwargs.pop("context_window", None) or int(os.getenv("CUA_CONTEXT_WINDOW", "5"))
        self.trace_dir = kwargs.pop("trace_dir", None)
        
        # 初始化 tokenizer
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(self.tokenizer)
        
        # 初始化 processor（用于处理图像）
        self.processor = processor
        
        # 初始化 gconfig
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.rollout_stat_scope = rollout_stat_scope
        self.dump_dir = dump_dir
        
        # 初始化 reward function
        self.reward_fn = reward_fn
        
        # 初始化 logger
        self._logger = logging.getLogger(__name__)
        
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
        
        if not self.gbox_api_key:
            self._logger.warning(
                "CUAEnvRolloutWorkflow initialized without GBOX_API_KEY; "
                "environment interaction will not work."
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
    
    def _extract_images_from_messages(self, messages: list) -> list[bytes]:
        """从 messages 中提取图像数据（base64 data URL -> bytes）。"""
        import base64
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_url = item.get("image_url", {}).get("url", "")
                        if image_url.startswith("data:image"):
                            # 提取 base64 数据
                            header, data = image_url.split(",", 1)
                            try:
                                image_bytes = base64.b64decode(data)
                                images.append(image_bytes)
                            except Exception as e:
                                self._logger.warning(f"Failed to decode image: {e}")
        return images
    
    def _build_model_request(
        self,
        messages: list,
        rid: str,
        n_samples: int = 1,
    ) -> ModelRequest:
        """构建 ModelRequest，处理多模态输入（文本 + 图像）。
        
        参考 VisionRLVRWorkflow 的实现，使用 image_data 和 vision_msg_vllm 参数。
        对于 CUA agent，图像是必需的，如果处理失败会直接抛出异常。
        """
        # 提取图像数据（从 messages 中的 image_url）
        images = self._extract_images_from_messages(messages)
        
        # CUA agent 必须要有图像，如果没有图像则报错
        if not images:
            raise ValueError(
                "CUAEnvRolloutWorkflow: No image found in messages. "
                "CUA agent requires screenshots to function properly."
            )
        
        # 转换为 base64（参考 VisionRLVRWorkflow）
        try:
            # 使用 PIL Image 处理图像
            from PIL import Image
            import io
            pil_images = []
            for image_bytes in images:
                pil_image = Image.open(io.BytesIO(image_bytes))
                pil_images.append(pil_image)
            
            # 转换为 base64（使用 AReaL 的工具函数）
            byte_images = image2base64(pil_images)
        except Exception as e:
            raise RuntimeError(
                f"Failed to process images for CUA agent: {e}. "
                "Image processing is required and cannot be skipped."
            ) from e
        
        # 使用 tokenizer 处理 messages（提取文本部分用于 tokenization）
        # 注意：对于多模态模型，tokenizer 会处理包含 image_url 的 messages
        if hasattr(self.tokenizer, "apply_chat_template"):
            # 直接使用原始 messages，tokenizer 会处理图像 token
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
            input_ids = list(input_ids)
        else:
            # Fallback: 简单拼接文本（不应该发生，但保留作为安全措施）
            text = " ".join([msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)])
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        
        # 验证 input_ids 不为空
        if not input_ids or len(input_ids) == 0:
            raise ValueError(
                f"CUAEnvRolloutWorkflow: input_ids is empty after tokenization. "
                f"Messages: {messages[:2] if len(messages) > 2 else messages}"
            )
        
        # 构建 ModelRequest，参考 VisionRLVRWorkflow
        # 传递 image_data 和 vision_msg_vllm 以支持多模态
        # vision_msg_vllm 应该是包含 messages 的列表，用于 vLLM 的多模态处理
        return ModelRequest(
            rid=rid,
            input_ids=input_ids,
            image_data=byte_images,
            vision_msg_vllm=[messages] if messages else None,
            gconfig=self.gconfig.new(n_samples=n_samples),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
    
    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[float, str]:
        """计算奖励。"""
        completions_str = self.tokenizer.decode(resp.output_tokens)
        
        # 调用 reward function
        if callable(self.reward_fn):
            import inspect
            # 获取 answer（从 task_data 或使用 task_description 作为默认值）
            answer = task_data.get("answer") or task_data.get("task_description", "")
            
            # 创建 task_data 的副本，移除 answer 避免重复传递
            task_data_copy = {k: v for k, v in task_data.items() if k != "answer"}
            
            if inspect.iscoroutinefunction(self.reward_fn):
                reward = await self.reward_fn(
                    prompt_str,
                    completions_str,
                    resp.input_tokens,
                    resp.output_tokens,
                    answer,
                    **task_data_copy,
                )
            else:
                reward = self.reward_fn(
                    prompt_str,
                    completions_str,
                    resp.input_tokens,
                    resp.output_tokens,
                    answer,
                    **task_data_copy,
                )
        else:
            reward = task_data.get("reward", 0.0)
        
        return reward, completions_str
    
    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float, str]:
        """生成一个 sample 并计算奖励。"""
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)
        
        reward, completions_str = await self._compute_rewards(
            resp, prompt_str, task_data
        )
        
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
        
        return resp, reward, completions_str
    
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """运行一个完整的 episode：多轮环境交互。
        
        Args:
            engine: InferenceEngine 实例
            data: 输入数据，包含 task_metadata 等信息
            
        Returns:
            符合 AReaL 格式的 tensor 字典
        """
        # 如果没有 GBOX_API_KEY，返回空结果
        if not self.gbox_api_key:
            self._logger.warning("No GBOX_API_KEY, skipping environment rollout")
            # 返回空 tensor
            return {
                "input_ids": torch.tensor([[]], dtype=torch.int32),
                "loss_mask": torch.tensor([[]], dtype=torch.int32),
                "logprobs": torch.tensor([[]], dtype=torch.float32),
                "versions": torch.tensor([[]], dtype=torch.int32),
                "attention_mask": torch.tensor([[]], dtype=torch.bool),
                "rewards": torch.tensor([0.0], dtype=torch.float32),
            }
        
        # 构建任务
        task = self._build_task(data)
        env_config = self._build_env_config()
        rollout_id = f"{task.id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self._logger.info(f"[CUAEnvRollout] Starting task: {task.id}, rollout: {rollout_id}")
        
        # 创建环境
        env = GBoxActorEnv(env_config, task, rollout_id)
        
        # 存储所有轮次的响应和奖励
        all_resps: list[ModelResponse] = []
        all_rewards: list[float] = []
        all_completions: list[str] = []
        
        try:
            # 重置环境
            obs = await env.reset()
            messages = obs["messages"]
            tools = obs["tools"]
            done = obs["done"]
            
            # 多轮交互循环
            while not done:
                turn = env.num_turns + 1
                self._logger.info(f"[Turn {turn}/{self.max_turns}] Starting turn")
                
                # 构建 ModelRequest
                req = self._build_model_request(
                    messages=messages,
                    rid=uuid.uuid4().hex,
                    n_samples=1,
                )
                
                # 准备 task_data
                prompt_str = self.tokenizer.decode(req.input_ids)
                task_data = {
                    "task_id": task.id,
                    "task_description": task.description,
                    "answer": task.description,  # 用于 reward function
                    "num_turns": turn,
                    "max_turns": self.max_turns,
                }
                
                # 生成响应
                resp, reward, completion_str = await self._collect_samples(
                    engine, req, prompt_str, task_data
                )
                
                all_resps.append(resp)
                all_rewards.append(reward)
                all_completions.append(completion_str)
                
                # 从响应中解析 tool_call
                output_text = completion_str
                tool_call_dict = self._parse_action_from_text(output_text, turn)
                
                # 执行环境步骤
                obs, env_reward, done, info = await env.step(tool_call_dict)
                
                # 如果结束，计算终局奖励并更新
                if done:
                    final_reward = await env.compute_reward()
                    # 更新最后一轮的奖励
                    if all_rewards:
                        all_rewards[-1] = final_reward
                    task_data.update({
                        "task_completed": info.get("task_completed", False),
                        "task_success": info.get("task_success", False),
                        "num_turns": info.get("num_turns", env.num_turns),
                        "errors": info.get("errors", env.errors),
                        "reward": final_reward,
                    })
                else:
                    # 更新 messages 和 tools
                    messages = obs["messages"]
                    tools = obs["tools"]
                    # 中间轮次奖励为 0
                    if all_rewards:
                        all_rewards[-1] = 0.0
                
                self._logger.info(
                    f"[Turn {turn}] Generated tool_call: {tool_call_dict.get('function', {}).get('name', 'unknown')}, "
                    f"env_reward: {env_reward}, done: {done}"
                )
            
            final_reward = task_data.get("reward", 0.0)
            self._logger.info(
                f"[CUAEnvRollout] Task {task.id} completed: "
                f"success={task_data.get('task_success', False)}, turns={task_data.get('num_turns', 0)}, "
                f"reward={final_reward}"
            )
            
        except Exception as e:
            self._logger.error(f"[CUAEnvRollout] Task {task.id} failed: {e}", exc_info=True)
            # 如果出错，至少返回一个空响应
            if not all_resps:
                # 创建一个空的响应
                empty_req = ModelRequest(
                    rid=uuid.uuid4().hex,
                    input_ids=[],
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
                try:
                    empty_resp = await engine.agenerate(empty_req)
                    all_resps.append(empty_resp)
                    all_rewards.append(0.0)
                    all_completions.append("")
                except:
                    pass
            
        finally:
            await env.close()
        
        # 构建结果 tensors
        if not all_resps:
            # 返回空结果
            return {
                "input_ids": torch.tensor([[]], dtype=torch.int32),
                "loss_mask": torch.tensor([[]], dtype=torch.int32),
                "logprobs": torch.tensor([[]], dtype=torch.float32),
                "versions": torch.tensor([[]], dtype=torch.int32),
                "attention_mask": torch.tensor([[]], dtype=torch.bool),
                "rewards": torch.tensor([0.0], dtype=torch.float32),
            }
        
        results = []
        version = engine.get_version()
        
        for resp, reward in zip(all_resps, all_rewards):
            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions
            
            res = {
                "input_ids": torch.tensor(seq, dtype=torch.int32),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32),
                "versions": torch.tensor(versions, dtype=torch.int32),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool),
                "rewards": torch.tensor(reward, dtype=torch.float32),
            }
            res = {k: v.unsqueeze(0) for k, v in res.items()}
            results.append(res)
        
        # Dump to file if needed
        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            
            qid = task.id or uuid.uuid4().hex
            file_path = os.path.join(dump_path, f"{qid}.txt")
            
            async with aiofiles.open(file_path, "a") as f:
                for i, (completion, reward) in enumerate(zip(all_completions, all_rewards)):
                    info = "\n".join([
                        f"idx: {i + 1} / {len(all_completions)}, reward is {reward}.",
                        f"completion: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{completion}{colorama.Style.RESET_ALL}",
                    ])
                    await f.write(info + "\n")
        
        return concat_padded_tensors(results)

