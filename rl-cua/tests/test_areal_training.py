"""测试 AReaL 训练流程。"""

import pytest
from pathlib import Path


def test_config_loading():
    """测试配置文件加载。"""
    try:
        from areal.config import GRPOConfig
        
        config = GRPOConfig.from_yaml("configs/cua_grpo.yaml")
        
        assert config.model.name is not None
        assert config.training.learning_rate > 0
        assert config.rollout.num_rollouts > 0
    except ImportError:
        pytest.skip("AReaL not installed")


def test_model_loading():
    """测试模型加载（需要 GPU）。"""
    pytest.skip("Requires GPU")
    
    # 测试代码...


# 更多测试...

