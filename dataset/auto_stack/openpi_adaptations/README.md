# OpenPI AutoStack 适配

本目录包含为 UR10e_2f85_mujoco 机器人 `auto_stack` 数据集适配 OpenPI 的所有文件。

## 机器人规格

- **机器人类型**: UR10e_2f85_mujoco
- **状态维度**: 13（前7维：6关节 + 夹爪位置）
- **动作维度**: 7（Delta 动作）
- **相机**: 2个（top_camera, wrist_camera）
- **动作类型**: Delta（相对变化量，无需额外转换）

## 文件说明

- `auto_stack_policy.py` - 策略文件，已移至 `src/openpi/policies/`
- `test_policy.py` - 测试脚本，验证策略输入输出转换
- `config_additions.py` - 参考配置（已直接添加到 config.py，无需使用）

## 适配步骤

### 1. 准备数据集

确保你的数据集位于 LeRobot 格式：
```
dataset/auto_stack/
├── data/
│   └── chunk-000/
│       └── file-000.parquet
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── tasks.parquet
│   └── episodes/
└── videos/
    ├── observation.images.top_camera/
    └── observation.images.wrist_camera/
```

### 2. 配置已添加

`pi0_auto_stack` 配置已添加到 `src/openpi/training/config.py` 中。

### 3. 计算归一化统计

```bash
python -m openpi.train --config pi0_auto_stack --exp_name run_001
```

### 5. 推理

使用训练好的策略进行推理：

```python
from openpi.policies.policy_config import create_trained_policy
from openpi.training import config as _config

config = _config.get_config("pi0_auto_stack")
policy = create_trained_policy(config, "checkpoints/pi0_auto_stack/run_001")

# 推理
obs = {
    "observation.state": np.random.rand(13),
    "observation.images.top_camera": top_cam_image,
    "observation.images.wrist_camera": wrist_cam_image,
    "task": "stack the block",
}
actions = policy.infer(obs)
```

## 相机映射

| LeRobot 字段 | 模型字段 | 说明 |
|-------------|---------|-----|
| observation.images.top_camera | base_0_rgb | 第三人称视角 |
| observation.images.wrist_camera | left_wrist_0_rgb | 左手腕视角 |
| - | right_wrist_0_rgb | 填充零，mask=False |

## 动作空间

- **维度**: 7
- **类型**: Delta（相对变化量）
- **内容**: 前6维是关节变化量，第7维是夹爪动作

## 注意事项

1. **相机分辨率**: 原始数据 720x1280 会被 ResizeImages transform 自动调整为 224x224
2. **状态截取**: 策略自动使用前7维状态（6关节 + 夹爪）
3. **无坐标系转换**: UR10e 的坐标系与训练数据一致，无需额外转换
4. **LeRobot 数据集**: 如果数据集在 HuggingFace 上，需要修改 `repo_id` 为实际的 repo 路径

## 故障排查

### 归一化错误
```
ValueError: norm stats not found
```
检查 `assets/auto_stack/norm_stats.json` 是否存在且格式正确。

### 形状不匹配
```
ValueError: shapes don't match
```
检查 `action_dim` 配置是否为 7。

### 图像错误
```
ValueError: Expected images to contain ('top_camera', 'wrist_camera')
```
检查 LeRobot 数据集的图像键名是否正确。