"""测试 AReaL 环境适配器。"""

import pytest
import asyncio
from cua_agent.areal_env import GBoxAReaLEnv, GBoxEnvConfig
from cua_agent.tasks import get_training_tasks


@pytest.fixture
def env_config():
    return GBoxEnvConfig(
        gbox_api_key="test_key",
        box_type="android",
        max_turns=5,
    )


@pytest.fixture
def env(env_config):
    return GBoxAReaLEnv(env_config)


@pytest.fixture
def sample_task():
    return get_training_tasks()[0]


def test_env_creation(env):
    """测试环境创建。"""
    assert env is not None
    assert env.agent is None


@pytest.mark.asyncio
async def test_env_reset(env, sample_task):
    """测试环境重置。"""
    observation = await env.reset(sample_task)
    
    assert "task_id" in observation
    assert "task_description" in observation
    assert observation["task_id"] == sample_task.id


# 更多测试...

