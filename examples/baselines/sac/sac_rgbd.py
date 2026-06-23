
# 导入必要的库
from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm  # 进度条库

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro  # 命令行参数解析库

import mani_skill.envs


@dataclass
class Args:
    # 实验相关参数
    exp_name: Optional[str] = None
    """实验名称"""
    seed: int = 1
    """随机种子"""
    torch_deterministic: bool = True
    """如果启用, 设置 torch.backends.cudnn.deterministic=False"""
    cuda: bool = True
    """如果启用, 默认使用 cuda"""
    track: bool = False
    """如果启用, 使用 Weights and Biases 跟踪实验"""
    wandb_project_name: str = "ManiSkill"
    """wandb 项目名称"""
    wandb_entity: Optional[str] = None
    """wandb 项目的实体 (团队)"""
    wandb_group: str = "SAC"
    """wandb 运行的分组"""
    capture_video: bool = True
    """是否捕获智能体性能视频 (查看 `videos` 文件夹)"""
    save_trajectory: bool = False
    """是否将轨迹数据保存到 `videos` 文件夹"""
    save_model: bool = True
    """是否将模型保存到 `runs/{run_name}` 文件夹"""
    evaluate: bool = False
    """如果启用, 仅使用给定的模型检查点运行评估并保存评估轨迹"""
    checkpoint: Optional[str] = None
    """预训练检查点文件的路径, 用于开始评估/训练"""
    log_freq: int = 1_000
    """日志记录频率 (环境步数)"""

    # 环境特定参数
    env_id: str = "PickCube-v1"
    """环境 ID"""
    obs_mode: str = "rgb"
    """使用的观测模式"""
    include_state: bool = True
    """是否在观测中包含状态"""
    env_vectorization: str = "gpu"
    """环境向量化的类型"""
    num_envs: int = 16
    """并行环境数量"""
    num_eval_envs: int = 16
    """并行评估环境数量"""
    partial_reset: bool = False
    """是否让并行环境在终止时重置而不是截断时重置"""
    eval_partial_reset: bool = False
    """是否让并行评估环境在终止时重置而不是截断时重置"""
    num_steps: int = 50
    """每次策略 rollout 在每个环境中运行的步数"""
    num_eval_steps: int = 50
    """评估期间在每个评估环境中运行的步数"""
    reconfiguration_freq: Optional[int] = None
    """训练期间重新配置环境的频率"""
    eval_reconfiguration_freq: Optional[int] = 1
    """为了基准测试, 我们希望在每次重置时重新配置评估环境, 以确保某些任务中的对象随机化"""
    eval_freq: int = 25
    """评估频率 (迭代次数)"""
    save_train_video_freq: Optional[int] = None
    """保存训练视频的频率 (迭代次数)"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """环境的控制模式"""
    render_mode: str = "all"
    """环境渲染模式"""

    # 算法特定参数
    total_timesteps: int = 1_000_000
    """实验总时间步数"""
    buffer_size: int = 1_000_000
    """经验回放缓冲区大小"""
    buffer_device: str = "cuda"
    """经验回放缓冲区存储位置. 可以是 'cpu' 或 'cuda' 用于 GPU"""
    gamma: float = 0.8
    """折扣因子 gamma"""
    tau: float = 0.01
    """目标网络平滑系数"""
    batch_size: int = 512
    """从经验回放缓冲区采样的批次大小"""
    learning_starts: int = 4_000
    """开始学习的时间步"""
    policy_lr: float = 3e-4
    """策略网络优化器的学习率"""
    q_lr: float = 3e-4
    """Q 网络优化器的学习率"""
    policy_frequency: int = 1
    """训练策略的频率 (延迟更新)"""
    target_network_frequency: int = 1  # Denis Yarats 的实现将此延迟 2
    """目标网络更新的频率"""
    alpha: float = 0.2
    """熵正则化系数"""
    autotune: bool = True
    """自动调整熵系数"""
    training_freq: int = 64
    """训练频率 (步数)"""
    utd: float = 0.25
    """更新与数据的比率"""
    partial_reset: bool = False
    """是否让并行环境在终止时重置而不是截断时重置"""
    bootstrap_at_done: str = "always"
    """收到 done 信号时使用的 bootstrap 方法. 可以是 'always' 或 'never'"""
    camera_width: Optional[int] = None
    """相机图像的宽度. 如果为 None, 将使用环境指定的默认值"""
    camera_height: Optional[int] = None
    """相机图像的高度. 如果为 None, 将使用环境指定的默认值"""

    # 运行时填充的参数
    grad_steps_per_iteration: int = 0
    """每次迭代的梯度更新次数"""
    steps_per_env: int = 0
    """每次迭代每个并行环境执行的步数"""
# 字典数组类, 用于存储字典形式的观测数据
class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    # 根据 numpy dtype 转换为对应的 torch dtype
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

# 经验回放缓冲区采样数据结构
@dataclass
class ReplayBufferSample:
    obs: torch.Tensor  # 当前观测
    next_obs: torch.Tensor  # 下一步观测
    actions: torch.Tensor  # 执行的动作
    rewards: torch.Tensor  # 获得的奖励
    dones: torch.Tensor  # 是否终止

# 经验回放缓冲区类 (支持字典观测)
class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size  # 缓冲区总大小
        self.pos = 0  # 当前写入位置
        self.full = False  # 缓冲区是否已满
        self.num_envs = num_envs  # 并行环境数量
        self.storage_device = storage_device  # 存储设备
        self.sample_device = sample_device  # 采样设备
        self.per_env_buffer_size = buffer_size // num_envs  # 每个环境的缓冲区大小
        # 注意: 128x128x3 的 RGB 数据, 回放缓冲区大小为 100_000 时约占 4.7GB GPU 内存
        # 32 个并行环境使用渲染约占 2.2GB GPU 内存
        self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        # TODO (stao): 优化最终观测存储
        self.next_obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)
        self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)

    # 添加经验到缓冲区
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        # 如果存储设备是 CPU, 将数据移到 CPU
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        # 如果到达缓冲区末尾, 重新开始
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0
    
    # 从缓冲区采样一个批次
    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))
        obs_sample = self.obs[batch_inds, env_inds]
        next_obs_sample = self.next_obs[batch_inds, env_inds]
        # 将采样数据移到采样设备
        obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
        next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}
        return ReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

# 算法逻辑: 在此初始化智能体
# 平面卷积网络, 用于从图像中提取特征
class PlainConv(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_dim=256,
                 pool_feature_map=False,
                 last_act=True, # True for ConvBody, False for CNN
                 image_size=[128, 128]
                 ):
        super().__init__()
        # 假设输入图像尺寸为 128x128 或 64x64

        self.out_dim = out_dim
        # CNN 结构: 逐步下采样提取特征
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4) if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # [32, 32]
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [16, 16]
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [8, 8]
            nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # [4, 4]
            nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        # 根据是否池化特征图选择不同的输出方式
        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None
            self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=last_act)

        self.reset_parameters()

    # 重置参数
    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        x = self.cnn(image)
        if self.pool is not None:
            x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

# class Encoder(nn.Module):
#     def __init__(self, sample_obs):
#         super().__init__()

#         extractors = {}

#         self.out_features = 0
#         feature_size = 256
#         in_channels=sample_obs["rgb"].shape[-1]
#         image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


#         # here we use a NatureCNN architecture to process images, but any architecture is permissble here
#         cnn = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=in_channels,
#                 out_channels=32,
#                 kernel_size=8,
#                 stride=4,
#                 padding=0,
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
#             ),
#             nn.ReLU(),
#             nn.Conv2d(
#                 in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
#             ),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # to easily figure out the dimensions after flattening, we pass a test tensor
#         with torch.no_grad():
#             n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
#             fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
#         extractors["rgb"] = nn.Sequential(cnn, fc)
#         self.out_features += feature_size
#         self.extractors = nn.ModuleDict(extractors)

#     def forward(self, observations) -> torch.Tensor:
#         encoded_tensor_list = []
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             obs = observations[key]
#             if key == "rgb":
#                 obs = obs.float().permute(0,3,1,2)
#                 obs = obs / 255
#             encoded_tensor_list.append(extractor(obs))
#         return torch.cat(encoded_tensor_list, dim=1)
# 编码器观测包装器, 用于处理字典形式的观测
class EncoderObsWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, obs):
        # 处理 RGB 观测
        if "rgb" in obs:
            rgb = obs['rgb'].float() / 255.0 # (B, H, W, 3*k)
        # 处理深度观测
        if "depth" in obs:
            depth = obs['depth'].float() # (B, H, W, 1*k)
        # 合并 RGB 和深度信息
        if "rgb" and "depth" in obs:
            img = torch.cat([rgb, depth], dim=3) # (B, H, W, C)
        elif "rgb" in obs:
            img = rgb
        elif "depth" in obs:
            img = depth
        else:
            raise ValueError(f"Observation dict must contain 'rgb' or 'depth'")
        # 转换为 PyTorch 需要的格式 (B, C, H, W)
        img = img.permute(0, 3, 1, 2) # (B, C, H, W)
        return self.encoder(img)

# 创建多层感知机 (MLP)
def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)

# Soft Q 网络 (用于估计状态-动作对的价值)
class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder: EncoderObsWrapper):
        super().__init__()
        self.encoder = encoder
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        # 网络结构: 视觉特征 + 状态 + 动作 -> 512 -> 256 -> 1 (Q值)
        self.mlp = make_mlp(encoder.encoder.out_dim+action_dim+state_dim, [512, 256, 1], last_act=False)

    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        # 如果没有提供视觉特征, 使用编码器提取
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        # 如果需要分离编码器梯度 (用于策略更新时防止梯度流回编码器)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        # 拼接视觉特征、状态和动作
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        return self.mlp(x)


# 对数标准差的最大值和最小值 (用于限制策略网络输出的范围)
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Actor 网络 (策略网络)
class Actor(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape)
        state_dim = envs.single_observation_space['state'].shape[0]
        # 计算通道数和图像尺寸
        in_channels = 0
        if "rgb" in sample_obs:
            in_channels += sample_obs["rgb"].shape[-1]
            image_size = sample_obs["rgb"].shape[1:3]
        if "depth" in sample_obs:
            in_channels += sample_obs["depth"].shape[-1]
            image_size = sample_obs["depth"].shape[1:3]

        # 视觉编码器
        self.encoder = EncoderObsWrapper(
            PlainConv(in_channels=in_channels, out_dim=256, image_size=image_size) # 假设图像是 64x64
        )
        # MLP: 视觉特征 + 状态 -> 512 -> 256
        self.mlp = make_mlp(self.encoder.encoder.out_dim+state_dim, [512, 256], last_act=True)
        # 输出均值和对数标准差
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # 动作缩放
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)

    # 获取特征
    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs['state']], dim=1)
        return self.mlp(x), visual_feature

    def forward(self, obs, detach_encoder=False):
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        # 使用 tanh 将 log_std 限制在 [LOG_STD_MIN, LOG_STD_MAX] 范围内
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # 来自 SpinUp / Denis Yarats

        return mean, log_std, visual_feature

    # 获取评估动作 (确定性策略)
    def get_eval_action(self, obs):
        mean, log_std, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    # 获取训练动作 (随机策略)
    def get_action(self, obs, detach_encoder=False):
        mean, log_std, visual_feature = self(obs, detach_encoder)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        # 使用重参数化技巧采样 (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # 强制动作边界
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, visual_feature

    # 将模型移动到指定设备
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

# 日志记录器类
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

# 主函数
if __name__ == "__main__":
    # 解析命令行参数
    args = tyro.cli(Args)
    # 计算运行时参数
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    # 设置实验名称
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

    # 设置设备 (CPU 或 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### 环境设置 #######
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend="gpu", sensor_configs=dict())
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    # 设置相机图像尺寸
    if args.camera_width is not None:
        # 这会覆盖用于观测生成的每个传感器
        env_kwargs["sensor_configs"]["width"] = args.camera_width
    if args.camera_height is not None:
        env_kwargs["sensor_configs"]["height"] = args.camera_height
    # 创建训练环境和评估环境
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)

    # RGBD 观测模式返回字典数据, 我们将其展平为只有 rgbd 键和 state 键
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    # 如果动作空间是字典类型, 使用 FlattenActionSpaceWrapper 展平
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    # 设置视频/轨迹记录
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        # 如果需要保存训练视频
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    # 使用向量环境包装器
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # 获取最大回合步数
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    # 如果不是评估模式, 设置日志记录
    if not args.evaluate:
        print("Running training")
        # 如果启用 wandb 跟踪
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
                tags=["sac", "walltime_efficient"]
            )
        # 创建 TensorBoard writer
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # 设置观测空间数据类型
    envs.single_observation_space.dtype = np.float32
    # 初始化经验回放缓冲区
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )


    # 开始训练
    obs, info = envs.reset(seed=args.seed) # 在 Gymnasium 中, seed 传给 reset() 而不是 seed()
    eval_obs, _ = eval_envs.reset(seed=args.seed)

    # 架构说明: 所有 actor 和 q-network 共享相同的视觉编码器. 编码器的输出与任何状态数据拼接, 后面跟随单独的 MLPs
    actor = Actor(envs, sample_obs=obs).to(device)
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)
    # 如果有检查点, 加载模型
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])
    # 初始化目标网络
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # 初始化优化器 (包含 Q 网络的 MLP 和共享的编码器)
    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) +
        list(qf2.mlp.parameters()) +
        list(qf1.encoder.parameters()),
        lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # 自动熵调整
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    # 主训练循环
    while global_step < args.total_timesteps:
        # 评估逻辑: 定期评估智能体性能
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # 切换到评估模式
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            # 运行评估
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            # 计算评估指标的平均值
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            # 更新进度条描述
            pbar.set_description(
                f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            # 如果是评估模式, 退出循环
            if args.evaluate:
                break
            # 切换回训练模式
            actor.train()

            # 保存模型检查点
            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1': qf1_target.state_dict(),
                    'qf2': qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")

        # 从环境收集样本 (数据收集阶段)
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += 1 * args.num_envs

            # 算法逻辑: 在此放置动作选择逻辑
            if not learning_has_started:
                # 学习开始前使用随机动作
                actions = 2 * torch.rand(size=envs.action_space.shape, dtype=torch.float32, device=device) - 1
            else:
                # 使用策略网络选择动作
                actions, _, _, _ = actor.get_action(obs)
                actions = actions.detach()

            # 执行动作并记录数据
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = {k:v.clone() for k, v in next_obs.items()}
            # 根据配置决定如何处理终止状态
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations # 回合结束时总是停止 bootstrap
            else:
                if args.bootstrap_at_done == 'always':
                    need_final_obs = truncations | terminations # 回合结束时总是需要最终观测
                    stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # 从不停止 bootstrap
                else: # 在截断时 bootstrap
                    need_final_obs = truncations & (~terminations) # 仅在截断且未终止时需要最终观测
                    stop_bootstrap = terminations # 仅在终止时停止 bootstrap, 截断时不停止
            # 处理最终观测信息
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k in real_next_obs.keys():
                    real_next_obs[k][need_final_obs] = infos["final_observation"][k][need_final_obs].clone()
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            # 添加经验到回放缓冲区
            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

            # 关键步骤: 更新当前观测
            obs = next_obs
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # 算法逻辑: 训练阶段
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        # 执行多次梯度更新
        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            # 从回放缓冲区采样一个批次
            data = rb.sample(args.batch_size)

            # 更新价值网络 (Q 网络)
            with torch.no_grad():
                # 获取下一状态的动作和策略熵
                next_state_actions, next_state_log_pi, _, visual_feature = actor.get_action(data.next_obs)
                # 计算目标 Q 值
                qf1_next_target = qf1_target(data.next_obs, next_state_actions, visual_feature)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions, visual_feature)
                # 取最小值并减去熵正则化项
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # 计算目标 Q 值: reward + gamma * (1 - done) * min_qf_next
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones 是 "stop_bootstrap", 根据前面的 args.bootstrap_at_done 计算
            # 提取当前状态的特征
            visual_feature = actor.encoder(data.obs)
            # 计算 Q 值
            qf1_a_values = qf1(data.obs, data.actions, visual_feature).view(-1)
            qf2_a_values = qf2(data.obs, data.actions, visual_feature).view(-1)
            # 计算 Q 网络损失
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # 优化 Q 网络
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # 更新策略网络
            if global_update % args.policy_frequency == 0:  # TD3 延迟更新支持
                pi, log_pi, _, visual_feature = actor.get_action(data.obs)
                # 计算 Q 值 (detach_encoder=True 防止梯度流回编码器)
                qf1_pi = qf1(data.obs, pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data.obs, pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                # 计算策略损失: alpha * log_pi - min_qf_pi
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                # 优化策略网络
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 自动调整熵系数
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _, _ = actor.get_action(data.obs)
                    # if args.correct_alpha:
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                    # else:
                    #     alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()
                    # log_alpha 的历史原因: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                    # 优化熵系数
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # 更新目标网络 (软更新)
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

    # 保存最终模型
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
