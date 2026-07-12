"""
测试脚本 - 验证 auto_stack_policy.py 是否正确工作

运行此脚本来检查策略文件的输入输出转换是否正确。
"""

import sys
sys.path.insert(0, "dataset/auto_stack/openpi_adaptations")

import numpy as np
from auto_stack_policy import make_auto_stack_example, AutoStackInputs, AutoStackOutputs


def test_example():
    """测试 make_auto_stack_example"""
    example = make_auto_stack_example()
    print("=== Example Input ===")
    print(f"Keys: {example.keys()}")
    print(f"State shape: {example['observation.state'].shape}")
    print(f"Top camera shape: {example['observation.images.top_camera'].shape}")
    print(f"Wrist camera shape: {example['observation.images.wrist_camera'].shape}")
    print(f"Task: {example['task']}")
    print()


def test_inputs():
    """测试 AutoStackInputs"""
    example = make_auto_stack_example()

    print("=== Testing AutoStackInputs ===")
    for model_type in ["pi0", "pi05", "pi0_fast"]:
        print(f"Model type: {model_type}")
        inputs_transform = AutoStackInputs(model_type=model_type)
        result = inputs_transform(example)

        print(f"  Image keys: {result['image'].keys()}")
        print(f"  Image masks: {result['image_mask']}")
        print(f"  State shape: {result['state'].shape}")
        print(f"  Prompt: {result.get('prompt', 'N/A')}")
        print()


def test_inputs_with_action():
    """测试带动作的输入"""
    example = make_auto_stack_example()
    example["action"] = np.random.randn(7)

    print("=== Testing with actions ===")
    inputs_transform = AutoStackInputs(model_type="pi0")
    result = inputs_transform(example)

    print(f"  Actions shape: {result['actions'].shape}")
    print()


def test_outputs():
    """测试 AutoStackOutputs"""
    model_output = {
        "actions": np.random.randn(10, 7, 2),  # batch, horizon, padded_dim
    }

    print("=== Testing AutoStackOutputs ===")
    outputs_transform = AutoStackOutputs()
    result = outputs_transform(model_output)

    print(f"  Output actions shape: {result['actions'].shape}")
    print()


def test_full_pipeline():
    """测试完整流程"""
    print("=== Full Pipeline Test ===")

    example = {
        "observation.images": {
            "top_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "wrist_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "observation.state": np.random.rand(13),  # 原始13维
        "action": np.random.randn(7),
        "task": "stack the block",
    }

    inputs_transform = AutoStackInputs(model_type="pi0")
    model_input = inputs_transform(example)

    print(f"  State (first 7 dims): {model_input['state']}")
    print(f"  Image keys: {list(model_input['image'].keys())}")
    print(f"  Actions shape: {model_input['actions'].shape}")
    print(f"  Prompt: {model_input['prompt']}")
    print()


if __name__ == "__main__":
    test_example()
    test_inputs()
    test_inputs_with_action()
    test_outputs()
    test_full_pipeline()
    print("All tests passed!")