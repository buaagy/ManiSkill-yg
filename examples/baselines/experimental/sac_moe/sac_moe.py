from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs


@dataclass
class Args:
    # 实验配置参数
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC_MOE"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    # 环境相关参数
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 150
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 150
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    # 算法相关参数
    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    training_freq: int = 64
    """training frequency (in steps)"""
    utd: float = 0.5
    """update to data ratio"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always' or 'never'"""

    # 运行时填充的参数
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


# 经验回放缓冲区样本类
@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
            
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )


# 算法逻辑: 初始化智能体
# =====================================================================
# 门控网络 (Gating Network)
# ---------------------------------------------------------------------
# 在混合专家 (Mixture of Experts, MoE) 架构中，门控网络负责根据当前
# 输入（状态或状态-动作对）为每个专家网络分配一个权重（重要性分数）。
#
# 工作流程：
#   1. 接收与专家网络相同的输入特征；
#   2. 通过多层全连接网络对输入进行非线性特征变换；
#   3. 在输出层使用 softmax 将 logits 归一化为概率分布；
#   4. 输出的权重向量将与各专家的输出逐一相乘并求和，得到最终结果。
#
# 网络结构（共 4 层全连接）：
#   input_dim -> 128  (ReLU + Dropout)
#          -> 256  (LeakyReLU + Dropout)
#          -> 128  (LeakyReLU + Dropout)
#          -> num_experts (Softmax)
# 使用 Dropout 进行正则化，避免门控网络对某个专家过度依赖。
# =====================================================================
class Gating(nn.Module):
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        """
        门控网络的构造函数。

        参数:
            input_dim (int): 输入特征的维度。
                - 对于价值网络 (VNetwork) 场景: 输入为观测，维度 = obs.shape 的乘积
                - 对于 Q 网络场景: 输入为观测与动作的拼接，维度 = obs.shape + action.shape
            num_experts (int): 专家网络的数量，决定输出层的神经元个数。
                每个输出对应一个专家的权重（重要性分数）。
            dropout_rate (float): Dropout 概率，用于正则化并防止过拟合。
                默认 0.1, 表示每个神经元有 10% 的概率在训练时被置零。
        """
        super(Gating, self).__init__()

        # 第 1 层: 输入层 -> 128 维
        # 使用 ReLU 激活函数引入非线性，随后接 Dropout 进行正则化
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第 2 层: 128 维 -> 256 维
        # 使用 LeakyReLU（允许负值有微小梯度，缓解神经元"死亡"问题）
        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        # 第 3 层: 256 维 -> 128 维
        # 继续特征压缩，同样使用 LeakyReLU + Dropout
        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        # 第 4 层: 输出层，128 维 -> num_experts
        # 输出每个专家的 logit（未归一化的分数），后续在 forward 中通过 softmax 转为权重
        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        """
        门控网络的前向传播。

        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_dim)。

        返回:
            torch.Tensor: 各专家的权重（概率分布），形状为 (batch_size, num_experts)。
                所有权重沿 dim=1 求和为 1, 表示每个样本对所有专家的归一化分配比例。
        """
        # 第 1 层: 全连接 -> ReLU -> Dropout
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        # 第 2 层: 全连接 -> LeakyReLU -> Dropout
        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        # 第 3 层: 全连接 -> LeakyReLU -> Dropout
        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        # 输出层: 全连接 -> Softmax
        # 在专家维度 (dim=1) 上做 softmax，得到每个专家的归一化权重
        return torch.softmax(self.layer4(x), dim=1)


# =====================================================================
# 混合专家网络 (Mixture of Experts, MoE)
# ---------------------------------------------------------------------
# MoE 通过门控网络 (Gating Network) 动态地为每个输入样本分配多个专家
# 网络的权重，并将各专家输出加权求和作为最终输出。
#
# 在本实现中，MoE 被用于构建价值网络 (V) 和 Q 网络 (Q) 的集成：
#   - 价值网络场景: 输入为观测 obs，输出 V(s)
#   - Q 网络场景: 输入为观测与动作的拼接 [obs, act]，输出 Q(s, a)
#
# 结构说明：
#   - experts: 由 num_experts 个独立的专家网络组成的列表
#   - gating: 门控网络，根据输入为每个专家生成归一化权重
#
# 前向传播流程：
#   1. 若提供了动作 a，则将观测与动作在最后一维拼接
#   2. 门控网络根据输入计算各专家权重 (batch_size, num_experts)
#   3. 所有专家分别对同一输入进行计算，输出堆叠为 (batch_size, 1, num_experts)
#   4. 将权重扩展到与专家输出相同的形状，逐元素相乘后求和
#   5. 最终输出形状为 (batch_size, 1)，即加权后的集成结果
# =====================================================================
class MoE(nn.Module):
    def __init__(self, num_experts, expert_module, env):
        """
        混合专家网络的构造函数。

        参数:
            num_experts (int): 专家网络的数量。每个专家是一个独立的全连接网络，
                通过门控网络的权重进行集成。
            expert_module (nn.Module): 专家网络的类构造器，可以是 VNetwork 或 SoftQNetwork。
                根据该类型自动推断门控网络的输入维度。
            env: 环境对象，用于获取观测空间和动作空间的维度信息。
        """
        super(MoE, self).__init__()

        # 根据专家类型确定门控网络的输入维度
        # - 价值网络 (VNetwork): 输入仅为观测，维度 = obs.shape 的乘积
        # - Q 网络 (SoftQNetwork): 输入为观测 + 动作，维度 = obs.shape + action.shape
        if expert_module == VNetwork:
            input_dim = np.prod(env.single_observation_space.shape)
        else:
            input_dim = np.prod(env.single_observation_space.shape) + np.prod(env.single_action_space.shape)

        # 创建 num_experts 个独立的专家网络实例
        # 每个专家网络结构相同但参数独立，通过各自的学习形成多样化的策略
        self.experts = nn.ModuleList([expert_module(env) for _ in range(num_experts)])

        # 门控网络，根据输入特征为每个专家分配权重
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x, a=None):
        """
        混合专家网络的前向传播。

        参数:
            x (torch.Tensor): 观测张量，形状为 (batch_size, obs_dim)。
            a (torch.Tensor, optional): 动作张量，形状为 (batch_size, action_dim)。
                若提供(Q 网络场景)，则与观测在最后一维拼接后送入专家网络。

        返回:
            torch.Tensor: 加权集成后的输出，形状为 (batch_size, 1)。
        """
        # 若提供了动作，将观测与动作在最后一维拼接，形成 (batch_size, obs_dim + action_dim)
        if a is not None:
            x = torch.cat([x, a], dim=-1)

        # 门控网络计算各专家权重: (batch_size, num_experts)
        weights = self.gating(x)

        # 所有专家分别对输入进行计算
        # 每个 expert(x) 输出形状为 (batch_size, 1)
        # 堆叠后形状为 (batch_size, 1, num_experts)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)

        # 将权重扩展到与 outputs 相同的形状，以便逐元素相乘
        # weights: (batch_size, num_experts) -> (batch_size, 1, num_experts)
        weights = weights.unsqueeze(1).expand_as(outputs)

        # 加权求和: 在专家维度 (dim=-1) 上求和
        # 最终输出形状: (batch_size, 1)
        return torch.sum(outputs * weights, dim=-1)


# =====================================================================
# 价值网络 (Value Network, VNetwork)
# ---------------------------------------------------------------------
# 价值网络用于估计状态的价值函数 V(s)，即从当前状态出发，遵循当前策略
# 所能获得的期望累计回报。
#
# 在 SAC (Soft Actor-Critic) 算法中，价值网络作为额外的价值估计器，
# 与 Q 网络的输出进行加权融合，用于稳定训练并加速收敛。
#
# 网络结构（4 层全连接）：
#   obs_dim -> 256 -> 256 -> 256 -> 1
# 每层之间使用 ReLU 激活函数，输出层不使用激活函数（输出实数）。
# =====================================================================
class VNetwork(nn.Module):
    def __init__(self, env):
        """
        价值网络的构造函数。

        参数:
            env: 环境对象，用于获取观测空间的维度信息。
        """
        super().__init__()
        # 构建全连接网络
        # 输入维度: 观测空间形状的乘积 (obs_dim)
        # 隐藏层: 3 层 256 维，使用 ReLU 激活函数
        # 输出层: 1 维，表示状态价值 V(s)
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        价值网络的前向传播。

        参数:
            x (torch.Tensor): 观测张量，形状为 (batch_size, obs_dim)。

        返回:
            torch.Tensor: 状态价值估计，形状为 (batch_size, 1)。
        """
        return self.net(x)


# =====================================================================
# 软 Q 网络 (Soft Q Network)
# ---------------------------------------------------------------------
# 软 Q 网络用于估计状态-动作对的软 Q 值 Q(s, a)，即在最大熵强化学习
# 框架下，从当前状态执行给定动作后，遵循最优策略所能获得的期望累计
# 回报（包含熵奖励项）。
#
# 在 SAC 算法中，通常使用两个独立的 Q 网络（qf1, qf2）构成 Critic，
# 并取两者的较小值来缓解 Q 值过估计问题（Clipped Double-Q Trick）。
# 本实现中每个 Q 网络都是 MoE 结构，由多个专家网络组成。
#
# 网络结构（4 层全连接）：
#   (obs_dim + action_dim) -> 256 -> 256 -> 256 -> 1
# 每层之间使用 ReLU 激活函数，输出层不使用激活函数（输出实数）。
# =====================================================================
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        """
        软 Q 网络的构造函数。

        参数:
            env: 环境对象，用于获取观测空间和动作空间的维度信息。
        """
        super().__init__()
        # 构建全连接网络
        # 输入维度: 观测空间形状乘积 + 动作空间形状乘积 (obs_dim + action_dim)
        # 隐藏层: 3 层 256 维，使用 ReLU 激活函数
        # 输出层: 1 维，表示软 Q 值 Q(s, a)
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """
        软 Q 网络的前向传播。

        参数:
            x (torch.Tensor): 状态-动作对张量，形状为 (batch_size, obs_dim + action_dim)。
                通常由观测和动作在最后一维拼接而成。

        返回:
            torch.Tensor: 软 Q 值估计，形状为 (batch_size, 1)。
        """
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


# Actor网络 (策略网络)
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


# 日志记录器
class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
        
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
        
    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### 环境设置 #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["sac_moe", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    # 初始化网络
    actor = Actor(envs).to(device)
    value_predictor = MoE(4, VNetwork, envs).to(device)
    qf1 = MoE(4, SoftQNetwork, envs).to(device)
    qf2 = MoE(4, SoftQNetwork, envs).to(device)
    qf1_target = MoE(4, SoftQNetwork, envs).to(device)
    qf2_target = MoE(4, SoftQNetwork, envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    predictor_optimizer = optim.Adam(list(value_predictor.parameters()), lr=args.q_lr)

    # 自动熵调优
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )


    # 开始训练
    obs, info = envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # 评估
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")

        # 从环境中收集样本
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # 算法逻辑: 放置动作选择逻辑
            if not learning_has_started:
                actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # 执行游戏并记录数据
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # always need final obs when episode ends
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
                else: # bootstrap at truncated
                    need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                    stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

            # 关键步骤: 更新观测
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # 算法逻辑: 训练
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            # 更新价值网络
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                target_v_values = value_predictor(data.next_obs)
                target_q_values = torch.min(qf1_next_target, qf2_next_target) 
                target_values = 0.8 * target_q_values  + 0.2 * target_v_values
                min_qf_next_target = target_values - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done
            
            qf1_a_values = qf1(data.obs, data.actions).view(-1)
            qf2_a_values = qf2(data.obs, data.actions).view(-1)
            V = value_predictor(data.obs)
            
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()
            
            with torch.no_grad():
                qf1_a_values = qf1(data.obs, data.actions)
                qf2_a_values = qf2(data.obs, data.actions)
                Q = torch.min(qf1_a_values, qf2_a_values)
            
            vf_err = V - Q
            vf_sign = (vf_err > 0).float()
            vf_weight = (1 - vf_sign) * 0.7 + vf_sign * 0.3
            predictor_loss = (vf_weight * (vf_err**2)).mean()
            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()

            # 更新策略网络
            if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                pi, log_pi, _ = actor.get_action(data.obs)
                qf1_pi = qf1(data.obs, pi)
                qf2_pi = qf2(data.obs, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # 更新目标网络
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # 记录训练相关数据
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
            logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha", alpha, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor': actor.state_dict(),
            'qf1': qf1_target.state_dict(),
            'qf2': qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()
    envs.close()
