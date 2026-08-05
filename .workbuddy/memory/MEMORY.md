# openpi 项目长期记忆

## 项目概述
- openpi = Physical Intelligence 开源机器人 VLA (Vision-Language-Action) 模型仓库
- 三种模型: pi0 (flow matching), pi0-FAST (自回归), pi0.5 (知识隔离 + adaRMS)
- 双后端: JAX (主) + PyTorch (新, 已在 LIBERO 上验证)
- 核心架构: PaliGemma (SigLIP ViT + Gemma 2B LLM) + Action Expert (Gemma 300M)

## 用户自定义工作
- 分支: feature/auto_stack-adapter
- 新增 AutoStack 适配器 (auto_stack_policy.py) 用于 UR10e_2f85_mujoco 机器人
- 新增配置: pi0_auto_stack, pi05_auto_stack
- 修改了 train.py, train_pytorch.py, checkpoints.py
- pi05_auto_stack 配置特点: batch_size=32, save_interval=50, save_weights_only=True, 从本地路径加载权重

## 关键文件路径
- 模型: src/openpi/models/ (pi0.py, pi0_fast.py, gemma.py, siglip.py, pi0_config.py)
- PyTorch模型: src/openpi/models_pytorch/ (pi0_pytorch.py, gemma_pytorch.py)
- 策略: src/openpi/policies/ (aloha_policy.py, droid_policy.py, libero_policy.py, auto_stack_policy.py)
- 训练: src/openpi/training/ (config.py=所有配置, data_loader.py, optimizer.py, checkpoints.py)
- 数据变换: src/openpi/transforms.py
- 推理服务: src/openpi/serving/websocket_policy_server.py
- 脚本: scripts/ (train.py, train_pytorch.py, serve_policy.py, compute_norm_stats.py)
- 客户端: packages/openpi-client/

## 数据流水线
Repack(键名重映射) -> Data transforms(机器人适配) -> Normalize(分位数/z-score) -> Model transforms(Tokenize)
反变换: Model outputs -> Unnormalize -> Data outputs -> Repack outputs

## 技术栈
- JAX 0.5.3 + Flax 0.10.2 + Orbax (主后端)
- PyTorch 2.7.1 + transformers 4.53.2 (需 patch transformers 库)
- uv 包管理, LeRobot 数据集格式, W&B 日志
- 需要 NVIDIA GPU: 推理>8GB, LoRA微调>22.5GB, 全量微调>70GB
