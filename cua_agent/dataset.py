"""Dataset handling for CUA Agent AReaL training.

This module patches AReaL's dataset module to support CUA dataset loading.
"""

import json
import logging
from datasets import Dataset

logger = logging.getLogger(__name__)


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


def patch_areal_dataset_module():
    """Patch AReaL's dataset module to support CUA dataset."""
    import areal.dataset as areal_dataset_module
    
    _original_get_custom_dataset = areal_dataset_module._get_custom_dataset
    
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
    
    logger.info("Patched AReaL dataset module to support CUA dataset")

