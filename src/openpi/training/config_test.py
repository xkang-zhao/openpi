import numpy as np

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config as _config


def test_lerobot_auto_stack_data_config(tmp_path):
    factory = _config.LeRobotAutoStackDataConfig(
        repo_id="local/auto_stack",
        assets=_config.AssetsConfig(asset_id="auto_stack"),
        base_config=_config.DataConfig(prompt_from_task=True),
    )
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=10)
    data_config = factory.create(tmp_path, model_config)

    assert data_config.action_sequence_keys == ("action",)
    assert data_config.prompt_from_task

    sample = {
        "observation.state": np.arange(7, dtype=np.float32),
        "observation.images.top_camera": np.zeros((3, 32, 32), dtype=np.uint8),
        "observation.images.wrist_camera": np.ones((3, 32, 32), dtype=np.uint8),
        "action": np.zeros((10, 7), dtype=np.float32),
        "prompt": "stack the block",
    }

    repacked = transforms.compose(data_config.repack_transforms.inputs)(sample)
    assert repacked["prompt"] == "stack the block"

    transformed = transforms.compose(data_config.data_transforms.inputs)(repacked)
    assert transformed["state"].shape == (7,)
    assert transformed["actions"].shape == (10, 7)
    assert transformed["prompt"] == "stack the block"
    assert transformed["image"]["base_0_rgb"].shape == (32, 32, 3)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (32, 32, 3)
    assert not transformed["image_mask"]["right_wrist_0_rgb"]


def test_pi0_auto_stack_uses_lerobot_config_and_base_action_dim():
    train_config = _config.get_config("pi0_auto_stack")

    assert isinstance(train_config.data, _config.LeRobotAutoStackDataConfig)
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 10
