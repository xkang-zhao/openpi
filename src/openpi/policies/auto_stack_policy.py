import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_auto_stack_example() -> dict:
    """Creates a random input example for the AutoStack policy."""
    return {
        "observation.state": np.random.rand(7),
        "observation.images.top_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "observation.images.wrist_camera": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        "task": "stack the block",
    }


@dataclasses.dataclass(frozen=True)
class AutoStackInputs(transforms.DataTransformFn):
    """Inputs for the AutoStack policy.

    Expected inputs (LeRobot format):
    - observation.images.top_camera: [channel, height, width] RGB image
    - observation.images.wrist_camera: [channel, height, width] RGB image
    - observation.state: [7] - first 7 dims are 6 joints + gripper position
    - action: [7] - delta actions (relative changes)
    - task: string - task description

    Output (model format):
    - image: dict with base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    - image_mask: dict with True for valid cameras, False for padding
    - state: [7] robot state
    - actions: [action_horizon, 7] action sequence
    - prompt: task instruction
    """

    model_type: str

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_camera", "wrist_camera")

    def __call__(self, data: dict) -> dict:
        in_images = data.get("observation.images", {})
        if set(in_images.keys()) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = _parse_image(in_images["top_camera"])
        wrist_image = _parse_image(in_images["wrist_camera"])

        match self.model_type:
            case "pi0" | "pi05":
                images = {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.False_,
                }
            case "pi0_fast":
                images = {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": wrist_image,
                    "right_wrist_0_rgb": np.zeros_like(base_image),
                }
                image_masks = {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_,
                }
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        state = np.asarray(data["observation.state"])
        if state.shape[0] > 7:
            state = state[:7]

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "action" in data:
            inputs["actions"] = np.asarray(data["action"])

        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AutoStackOutputs(transforms.DataTransformFn):
    """Outputs for the AutoStack policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][..., :7])
        return {"actions": actions}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image