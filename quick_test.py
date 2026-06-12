
"""
快速测试脚本 - 确保代码能跑通，视频能生成
仅运行 10,000 步，适合快速验证
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import ale_py
gym.register_envs(ale_py)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="quick_test",
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="ALE/MsPacman-v5",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.1,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=1000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=4,
        help="the frequency of training")
    parser.add_argument("--amp", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, automatic mixed precision (FP16) training will be enabled")
    args = parser.parse_args()
    # fmt: on
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    return args


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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
            # 只录制第1集、最后一集和中间间隔5集
            def episode_trigger_func(episode_id):
                return episode_id == 0 or episode_id % 5 == 0
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=episode_trigger_func)
        else:
            env = gym.make(env_id, frameskip=1)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=True,
            grayscale_obs=True,
            scale_obs=False,
        )

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        
        env = gym.wrappers.ClipReward(env, min_reward=-1.0, max_reward=1.0)
        env = gym.wrappers.FrameStackObservation(env, stack_size=4)
        env.action_space.seed(seed)

        return env
    
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
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
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class ReplayBuffer:
    def __init__(self, buffer_size, observation_space, action_space, device, optimize_memory_usage=True, handle_timeout_termination=False):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        obs_shape = observation_space.shape
        action_shape = action_space.shape

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=observation_space.dtype)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=observation_space.dtype)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos=None):
        batch_size = obs.shape[0]
        for i in range(batch_size):
            self.observations[self.pos] = obs[i]
            self.next_observations[self.pos] = next_obs[i]
            self.actions[self.pos] = action[i]
            self.rewards[self.pos] = reward[i]
            self.dones[self.pos] = done[i]
            self.pos += 1
            if self.pos >= self.buffer_size:
                self.full = True
                self.pos = 0

    def sample(self, batch_size):
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        data = type('Data', (), {})()
        data.observations = torch.from_numpy(self.observations[indices]).float().to(self.device)
        data.next_observations = torch.from_numpy(self.next_observations[indices]).float().to(self.device)
        data.actions = torch.from_numpy(self.actions[indices]).long().to(self.device)
        data.rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        data.dones = torch.from_numpy(self.dones[indices]).float().to(self.device)
        return data

    def __len__(self):
        return self.buffer_size if self.full else self.pos


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # AMP 自动混合精度
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    print(f"[INFO] Running quick test on {device}")
    print(f"[INFO] Total timesteps: {args.total_timesteps}")
    print(f"[INFO] Capture video: {args.capture_video}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None or "episode" not in info:
                    continue
                print(f"[INFO] Step {global_step}, Episode return: {info['episode']['r']}")

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.autocast(device.type, enabled=args.amp):
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    # Ensure actions have correct shape (batch_size, 1) for gather
                    actions = data.actions
                    if actions.dim() == 1:
                        actions = actions.unsqueeze(1)
                    old_val = q_network(data.observations).gather(1, actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        
        # Print progress every 1000 steps
        if global_step % 1000 == 0 and global_step > 0:
            print(f"[PROGRESS] Step {global_step}/{args.total_timesteps}, SPS: {int(global_step / (time.time() - start_time))}")

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"[SUCCESS] Model saved to {model_path}")

    print(f"\n[SUCCESS] Test completed!")
    print(f"[INFO] Check 'videos/{run_name}' for the recorded video")
    print(f"[INFO] Total time: {int(time.time() - start_time)} seconds")
       
    envs.close()
