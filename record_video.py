
"""
录制训练好的模型玩 MsPacman 的演示视频
"""
import argparse
import os
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import ale_py
gym.register_envs(ale_py)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


def make_env(env_id, seed):
    env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env.action_space.seed(seed)
    return env


class QNetwork(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def record_video(
    model_path: str,
    env_id: str = "ALE/MsPacman-v5",
    num_episodes: int = 3,
    epsilon: float = 0.01,
    seed: int = 42
):
    run_name = f"video-{env_id.replace('/', '_')}-{int(time.time())}"
    video_dir = f"videos/{run_name}"
    os.makedirs(video_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[录制] 使用设备: {device}")
    print(f"[录制] 加载模型: {model_path}")
    print(f"[录制] 录制 {num_episodes} 个视频")

    # 创建环境并包装 RecordVideo
    base_env = make_env(env_id, seed)
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=video_dir,
        episode_trigger=lambda x: x < num_episodes,
        name_prefix="rl-video"
    )

    model = QNetwork(env.action_space).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    episodic_returns = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        step_count = 0

        while True:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
                q_values = model(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                print(f"[游戏] 第 {episode+1}/{num_episodes} 集 | 奖励: {episode_reward:.0f} | 步数: {step_count}")
                episodic_returns.append(episode_reward)
                break

    env.close()

    print(f"\n[成功] 视频已保存到: {video_dir}/")
    print(f"[统计] 平均奖励: {np.mean(episodic_returns):.1f}")
    print(f"[统计] 最高奖励: {np.max(episodic_returns):.1f}")
    print(f"[统计] 最低奖励: {np.min(episodic_returns):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="训练好的模型路径 (.pth文件)")
    parser.add_argument("--env-id", type=str, default="ALE/MsPacman-v5", help="环境ID")
    parser.add_argument("--num-episodes", type=int, default=3, help="录制多少个视频")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    record_video(
        model_path=args.model_path,
        env_id=args.env_id,
        num_episodes=args.num_episodes,
        seed=args.seed
    )
