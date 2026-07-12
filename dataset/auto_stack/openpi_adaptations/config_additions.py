"""
配置代码片段 - 需要添加到 src/openpi/training/config.py 中

import 已在文件顶部添加：
```python
import openpi.policies.auto_stack_policy as auto_stack_policy
```

添加到 _CONFIGS 列表中：
"""

# Pi0 AutoStack 配置
TrainConfig(
    name="pi0_auto_stack",
    model=pi0_config.Pi0Config(action_dim=7, action_horizon=10),
    data=SimpleDataConfig(
        repo_id="local/auto_stack",  # 使用本地数据集
        assets=AssetsConfig(asset_id="auto_stack"),
        data_transforms=lambda model: _transforms.Group(
            inputs=[auto_stack_policy.AutoStackInputs(model_type=model.model_type.value)],
            outputs=[auto_stack_policy.AutoStackOutputs()],
        ),
        base_config=DataConfig(
            prompt_from_task=True,  # 从 tasks.parquet 中提取 prompt
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
),