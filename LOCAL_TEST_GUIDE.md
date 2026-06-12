
# 本地测试指南

## 📋 前置条件

| 项目 | 推荐版本 |
|------|----------|
| Python | 3.8 ~ 3.11 |
| PyTorch | 1.12+ 或 2.x |
| CUDA | 11.x/12.x (如使用GPU) |

---

## 🚀 快速开始（本地运行）

### 1. 安装依赖

```bash
cd "e:\pj3 RL\DQN-Atari_Games"
pip install torch gymnasium ale-py tensorboard imageio-ffmpeg
```

### 2. 运行快速测试（10k步，验证代码能跑通+视频能生成）

```bash
python quick_test.py
```

### 3. 运行完整训练（500万步）

```bash
python dqn_atari.py --exp-name MsPacman-v5 --capture-video --env-id ALE/MsPacman-v5 --total-timesteps 5000000 --save-model
```

---

## 📝 环境说明

| 工具 | 说明 |
|------|------|
| `dqn_atari.py` | 完整训练脚本 |
| `dqn_eval.py` | 模型评估脚本 |
| `quick_test.py` | 快速测试脚本（仅10k步） |
| `check_env.py` | 环境依赖检查脚本 |

---

## 🎬 验证输出

测试运行后，检查以下内容：

1. **视频文件**：查看 `videos/` 目录下是否有生成的 `.mp4` 视频
2. **模型文件**：查看 `runs/` 目录下是否有保存的 `.pth` 模型
3. **训练日志**：控制台输出正常，显示步数、回合奖励等

---

## 🛠️ 参数说明（`quick_test.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | 10000 | 总训练步数 |
| `--capture-video` | True | 是否录制视频 |
| `--save-model` | True | 是否保存模型 |
| `--amp` | False | 是否启用FP16混合精度训练 |

---

## 🎯 ModelScope 上运行完整训练

如果你在 ModelScope 上（8核32G + A10 24G）：

```bash
# 1. 安装依赖
pip install torch gymnasium ale-py tensorboard imageio-ffmpeg

# 2. 运行完整训练（约1~2小时）
python dqn_atari.py --exp-name MsPacman-v5 --capture-video --env-id ALE/MsPacman-v5 --total-timesteps 5000000 --save-model --amp
```

---

## 📚 问题排查

### 问题：`ModuleNotFoundError: No module named 'xxx'`
**解决**：运行 `pip install torch gymnasium ale-py tensorboard imageio-ffmpeg`

### 问题：视频无法正常录制
**解决**：确认安装了 `imageio-ffmpeg`

### 问题：训练慢
**解决**：开启 GPU/混合精度训练
