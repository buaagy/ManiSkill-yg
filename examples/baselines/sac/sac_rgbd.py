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

# 复用 sac_moe 文件夹中已创建的 jepa.py 辅助模块
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "experimental" / "sac_moe"))
from jepa import JEPA, JEPAConfig


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
    num_steps: int = 150
    """单 episode 最大截断 horizon, 与ppo算法中的意义不同(ppo是每次策略 rollout 在每个环境中运行的步数)"""
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
    bootstrap_at_done: str = "always"
    """收到 done 信号时使用的 bootstrap 方法. 可以是 'always' 或 'never'"""
    camera_width: Optional[int] = None
    """相机图像的宽度. 如果为 None, 将使用环境指定的默认值"""
    camera_height: Optional[int] = None
    """相机图像的高度. 如果为 None, 将使用环境指定的默认值"""

    # JEPA 辅助模块参数
    use_jepa: bool = False
    """如果启用, 在训练期间开启动作条件的 JEPA 辅助损失模块."""
    jepa_lr: float = 3e-4
    """JEPA predictor 优化器的学习率."""
    jepa_hidden_dims: str = "256,256"
    """JEPA predictor MLP 的隐藏层维度 (逗号分隔)."""
    jepa_update_encoder: bool = False
    """如果启用, JEPA 优化器也会更新共享 encoder. 默认 False (encoder 仅由 SAC 更新)."""

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

        """返回字典中所有的键
        
        Returns:
            dict_keys: 包含字典中所有键的视图对象
        """
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

# ============================================================================
# 经验回放缓冲区类 (Experience Replay Buffer, 支持字典观测)
# ----------------------------------------------------------------------------
# 作用:
#   在 SAC 等 off-policy 强化学习算法中, 用于存储智能体与环境交互产生的
#   转移样本 (s, a, r, s', done), 并支持从中随机均匀采样小批量数据进行训练.
#   随机采样可以打破样本间的时间相关性, 使训练数据更接近 i.i.d. 假设,
#   从而提升梯度估计的无偏性与训练稳定性.
#
# 设计要点:
#   1. 并行环境: 同时管理 num_envs 个并行环境的转移, 每个环境独立维护时间轴.
#      缓冲区形状为 (per_env_buffer_size, num_envs), 其中
#      per_env_buffer_size = buffer_size // num_envs 为每个环境可存储的最大步数.
#   2. 字典观测: 观测 (obs/next_obs) 为嵌套字典结构 (如 {'rgb':..., 'state':...}),
#      使用 DictArray 存储; 动作、奖励、终止标志等为普通张量.
#   3. 设备分离: 存储设备 (storage_device) 与采样设备 (sample_device) 可不同.
#      - 存储设备通常为 CPU (节省显存) 或 GPU (加速采样);
#      - 采样设备为模型所在设备 (一般为 GPU), 采样时自动迁移数据.
#   4. 环形缓冲区: 写满后从头部覆盖旧数据, 保证缓冲区大小恒定且无需动态分配.
#   5. 未使用字段: logprobs / values 在本 SAC 脚本中未实际写入 (add 未传入),
#      保留是为了与 PPO 等其他算法共享同一缓冲区结构, 方便代码复用.
# ============================================================================
class ReplayBuffer:
    # ------------------------------------------------------------------------
    # 构造函数: 根据环境信息预分配所有存储张量
    #
    # 参数:
    #   env             : 向量环境实例, 用于读取 single_observation_space
    #                     和 single_action_space 以推断各张量的形状与数据类型.
    #   num_envs        : 并行环境数量, 决定缓冲区第 1 维 (环境维) 的大小.
    #   buffer_size     : 缓冲区总容量 (跨所有环境的总步数), 会按 num_envs 均分.
    #   storage_device  : 实际存放张量的设备 (如 torch.device('cpu') 或 'cuda').
    #   sample_device   : 采样后返回数据所在设备, 通常与模型参数所在设备一致.
    # ------------------------------------------------------------------------
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size  # 缓冲区总容量 (所有环境合计的步数)
        self.pos = 0  # 当前写入的时间步位置 (指向 per_env_buffer_size 维度的索引)
        self.full = False  # 缓冲区是否已写满至少一轮 (用于采样范围判断)
        self.num_envs = num_envs  # 并行环境数量
        self.storage_device = storage_device  # 数据存储设备 (CPU/GPU)
        self.sample_device = sample_device  # 采样返回数据的目标设备
        # 每个环境可存储的最大步数: 总容量按环境数均分
        # 例如 buffer_size=1_000_000, num_envs=16 -> per_env_buffer_size=62_500
        self.per_env_buffer_size = buffer_size // num_envs  # 每个环境的缓冲区大小

        # ------------------------------------------------------------------
        # 显存占用提示 (仅作参考, 实际取决于观测分辨率与缓冲区大小):
        #   - 128x128x3 的 RGB 数据, 回放缓冲区大小为 100_000 时约占 4.7GB GPU 内存
        #   - 32 个并行环境使用渲染约占 2.2GB GPU 内存
        # 若显存不足, 可将 storage_device 设为 'cpu', 以时间换空间.
        # ------------------------------------------------------------------

        # 观测存储: 使用 DictArray 以支持嵌套字典观测 (如 rgb / state / depth)
        # 形状: (per_env_buffer_size, num_envs, *obs_shape)
        self.obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)
        # TODO (stao): 优化最终观测存储
        # 下一时刻观测: 同样使用 DictArray, 与 obs 形状一致
        # 注意: next_obs 独立存储而非通过 obs[t+1] 推导, 是为了正确处理
        #       回合边界 (done) 时 next_obs 来自 final_observation 的特殊情况.
        self.next_obs = DictArray((self.per_env_buffer_size, num_envs), env.single_observation_space, device=storage_device)

        # 动作存储: 普通张量, 形状 (per_env_buffer_size, num_envs, *action_shape)
        self.actions = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape, device=storage_device)
        # 以下字段在当前 SAC 脚本中未实际使用, 保留以兼容其他算法 (如 PPO) 的缓冲区接口
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)  # 动作对数概率 (PPO 用)
        self.rewards = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)  # 奖励
        self.dones = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)  # 终止/截断标志 (实际语义见 add 的 done 参数)
        self.values = torch.zeros((self.per_env_buffer_size, num_envs), device=storage_device)  # 价值估计 (PPO 用)

    # ------------------------------------------------------------------------
    # 添加一条 (跨所有并行环境的) 转移经验到缓冲区
    #
    # 参数:
    #   obs      : 当前观测, 字典形式, 每个键对应的张量形状为 (num_envs, *obs_shape)
    #   next_obs : 下一时刻观测, 字典形式, 形状同 obs
    #   action   : 执行的动作, 张量, 形状 (num_envs, *action_shape)
    #   reward   : 获得的奖励, 张量, 形状 (num_envs,)
    #   done     : 是否停止 bootstrap 的标志, 张量, 形状 (num_envs,).
    #              注意: 此处 done 的语义为 "stop_bootstrap", 而非简单的终止信号.
    #              其具体含义由调用方根据 args.bootstrap_at_done 配置决定:
    #                - 'always' : 永不停止 bootstrap (全 False)
    #                - 'never'  : 终止或截断都停止 (terminations | truncations)
    #                - 其他     : 仅终止时停止, 截断时继续 bootstrap
    # ------------------------------------------------------------------------
    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        # 设备迁移: 若存储设备为 CPU, 需将可能位于 GPU 的输入数据迁移到 CPU
        # 这样可以释放 GPU 显存, 代价是增加一次跨设备拷贝开销
        if self.storage_device == torch.device("cpu"):
            obs = {k: v.cpu() for k, v in obs.items()}
            next_obs = {k: v.cpu() for k, v in next_obs.items()}
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        # 将当前时间步的所有并行环境数据写入缓冲区的 self.pos 位置
        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        # 注意: logprobs / values 在本 SAC 脚本中未写入, 保持为 0

        # 写入指针后移一位 (沿时间步维度前进)
        self.pos += 1
        # 环形缓冲区: 若已写满, 标记 full 并回绕到起始位置, 后续写入将覆盖最旧数据
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    # ------------------------------------------------------------------------
    # 从缓冲区中随机均匀采样一个小批量的转移数据
    #
    # 采样策略:
    #   - 时间维度: 若缓冲区已满 (self.full), 从 [0, per_env_buffer_size) 均匀采样;
    #               否则仅从已写入区域 [0, self.pos) 采样, 避免取到未初始化的零值.
    #   - 环境维度: 从 [0, num_envs) 均匀采样, 与时间维度独立.
    #   - 两个维度独立采样构成 (batch_size,) 个 (time_idx, env_idx) 索引对,
    #     相当于在所有已存储的 (时间步, 环境) 二维网格上做均匀随机抽样.
    #
    # 参数:
    #   batch_size : 采样批次大小, 即返回的样本数量
    #
    # 返回:
    #   ReplayBufferSample : 包含 obs / next_obs / actions / rewards / dones
    #                        的数据结构, 所有张量已迁移到 sample_device.
    # ------------------------------------------------------------------------
    def sample(self, batch_size: int):
        # 1. 生成时间步索引: 根据缓冲区是否写满选择采样范围
        if self.full:
            # 已写满: 从整个时间维度 [0, per_env_buffer_size) 采样
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size, ))
        else:
            # 未写满: 仅从已写入区域 [0, self.pos) 采样, 避免取到占位零值
            batch_inds = torch.randint(0, self.pos, size=(batch_size, ))
        # 2. 生成环境索引: 从所有并行环境 [0, num_envs) 采样
        env_inds = torch.randint(0, self.num_envs, size=(batch_size, ))

        # 3. 使用二维索引 (batch_inds, env_inds) 从缓冲区取数据
        #    对于 DictArray, 返回值为字典; 对于普通张量, 返回值为张量
        obs_sample = self.obs[batch_inds, env_inds]
        next_obs_sample = self.next_obs[batch_inds, env_inds]

        # 4. 将采样数据从存储设备迁移到采样设备 (通常为 GPU), 供模型前向计算使用
        obs_sample = {k: v.to(self.sample_device) for k, v in obs_sample.items()}
        next_obs_sample = {k: v.to(self.sample_device) for k, v in next_obs_sample.items()}

        # 5. 打包为 ReplayBufferSample 返回 (动作/奖励/dones 同样迁移到采样设备)
        return ReplayBufferSample(
            obs=obs_sample,
            next_obs=next_obs_sample,
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device)
        )

# 朴素卷积网络, 用于从图像中提取特征
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
        # CNN特征提取主干网络: 逐步下采样提取特征
        # ----------------------------------------------------------------
        # 设计思路: 通过 4 个 "Conv2d + MaxPool2d" 下采样块逐步提取视觉特征,
        # 最后接一个 1x1 卷积进行通道维度的特征融合.
        # 输入图像尺寸支持 128x128 或 64x64, 经网络后均下采样至 4x4 的特征图.
        #
        # 通道数变化: in_channels -> 16 -> 32 -> 64 -> 64 -> 64
        # 空间尺寸变化 (128x128 输入): 128 -> 32 -> 16 -> 8 -> 4 -> 4
        # 空间尺寸变化 (64x64   输入):  64 -> 32 -> 16 -> 8 -> 4 -> 4
        #
        # 最终输出: (B, 64, 4, 4) 的特征图, 后续展平为 (B, 1024) 送入 FC 层
        # ----------------------------------------------------------------
        self.cnn = nn.Sequential(
            # ---- 第 1 个卷积块: 浅层特征提取 (边缘、颜色、纹理等) ----
            # Conv2d: 3x3 卷积核, padding=1 保持空间尺寸不变, stride=1 (默认)
            # in_channels -> 16: 将输入图像 (RGB/RGBD) 映射到 16 维特征空间
            # ReLU(inplace=True): 原地操作节省显存, 引入非线性
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            # MaxPool2d: 自适应下采样, 统一不同输入尺寸到 32x32
            #   - 128x128 输入: 4x4 池化核 stride=4, 输出 128/4 = 32x32
            #   - 64x64   输入: 2x2 池化核 stride=2, 输出  64/2 = 32x32
            # MaxPool 选取局部最大值, 提供平移不变性并减少计算量
            nn.MaxPool2d(4, 4) if image_size[0] == 128 and image_size[1] == 128 else nn.MaxPool2d(2, 2),  # 图像尺寸: [32, 32]

            # ---- 第 2 个卷积块: 中层特征提取 (局部形状、部件组合等) ----
            # 16 -> 32: 加倍通道数以提取更丰富的特征, 3x3 卷积 + padding=1 保持尺寸
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            # 2x2 最大池化: 空间尺寸减半, 32x32 -> 16x16
            nn.MaxPool2d(2, 2),  # 图像尺寸: [16, 16]

            # ---- 第 3 个卷积块: 高层特征提取 (物体部件、空间关系等) ----
            # 32 -> 64: 继续加倍通道数, 3x3 卷积 + padding=1 保持尺寸
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            # 2x2 最大池化: 空间尺寸减半, 16x16 -> 8x8
            nn.MaxPool2d(2, 2),  # 图像尺寸: [8, 8]

            # ---- 第 4 个卷积块: 高层语义特征提取 ----
            # 64 -> 64: 保持通道数不变, 进一步精炼特征表示
            nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            # 2x2 最大池化: 空间尺寸减半, 8x8 -> 4x4
            nn.MaxPool2d(2, 2),  # 图像尺寸: [4, 4]

            # ---- 第 5 个卷积层: 1x1 卷积, 通道维度特征融合 ----
            # 1x1 卷积不改变空间尺寸, 仅在通道维度上做线性组合
            # 作用: 跨通道信息整合, 类似于在每个像素位置施加一个共享 MLP
            # 输出: (B, 64, 4, 4) — 64 个 4x4 的特征图
            nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True),
        )

        # 根据是否池化特征图选择不同的输出方式
        if pool_feature_map:
            self.pool = nn.AdaptiveMaxPool2d((1, 1)) # 自适应池化到 1x1, 输出 (B, 64, 1, 1)
            self.fc = make_mlp(128, [out_dim], last_act=last_act)
        else:
            self.pool = None # 不池化, 输出 (B, 64, 4, 4)
            self.fc = make_mlp(64 * 4 * 4, [out_dim], last_act=last_act)
            
        # 重置参数
        self.reset_parameters()
        
        # 打印 PlainConv 结构与维度信息
        print(f"[PlainConv] 模型结构与维度信息:")
        print(f"  - in_channels (输入通道数): {in_channels}")
        print(f"  - out_dim (输出特征维度): {out_dim}")
        print(f"  - image_size (输入图像尺寸): {image_size}")
        print(f"  - pool_feature_map: {pool_feature_map}")
        print(f"  - last_act: {last_act}")
        print(f"  - CNN 网络结构:\n{self.cnn}")
        if self.pool is not None:
            print(f"  - pool: {self.pool}")
        print(f"  - FC 网络结构:\n{self.fc}")
        print(f"  - 总参数量: {sum(p.numel() for p in self.parameters())}")

    # 重置参数
    def reset_parameters(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, image):
        # 打印前向传播中各阶段特征图维度 (仅第一次)
        if not getattr(self, "_printed_fwd_dims", False):
            print(f"[PlainConv.forward] 特征图维度变化 (仅打印一次):")
            print(f"  - input image.shape: {image.shape} (B, C, H, W)")
        x = self.cnn(image) # CNN 特征提取
        if not getattr(self, "_printed_fwd_dims", False):
            print(f"  - after cnn.shape: {x.shape} (B, 64, 4, 4)")
        if self.pool is not None:
            x = self.pool(x) # 特征图池化
            if not getattr(self, "_printed_fwd_dims", False):
                print(f"  - after pool.shape: {x.shape} (B, 64, 1, 1)")
        x = x.flatten(1) # 展平为 (B, 64*4*4) 或 (B, 64*1*1)
        if not getattr(self, "_printed_fwd_dims", False):
            print(f"  - after flatten.shape: {x.shape} (B, {x.shape[1]})")
        x = self.fc(x) # 全连接层映射到最终输出维度
        if not getattr(self, "_printed_fwd_dims", False):
            print(f"  - output.shape: {x.shape} (B, {self.out_dim})")
            self._printed_fwd_dims = True
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
        # 处理 Depth 观测
        if "depth" in obs:
            depth = obs['depth'].float() # (B, H, W, 1*k)
            
        # 合并 RGB 和深度信息
        if "rgb" in obs and "depth" in obs:
            img = torch.cat([rgb, depth], dim=3) # (B, H, W, C), dim=3表示在图像通道维度上拼接
        elif "rgb" in obs:
            img = rgb
        elif "depth" in obs:
            img = depth
        else:
            raise ValueError(f"Observation dict must contain 'rgb' or 'depth'")
        
        # 重排维度, 转换为 PyTorch 需要的格式 (B, C, H, W)
        img = img.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        return self.encoder(img)

# 创建多层感知机 (MLP)
#
# 该函数根据给定的输入维度和各层输出维度列表, 依次堆叠 nn.Linear 线性层和激活函数,
# 返回一个 nn.Sequential 容器. 常用于构建策略网络、Q 网络以及视觉编码器末端的特征投影模块.
#
# 参数:
#   in_channels  (int)               : 输入特征维度 (第一层 Linear 的 in_features).
#   mlp_channels (list[int])         : 各层输出维度列表, 列表长度即为 MLP 的层数.
#   act_builder  (callable, 可选)    : 激活函数构造器, 默认为 nn.ReLU, 传入构造器 (而非实例) 以便每层创建独立的激活函数模块.
#   last_act     (bool, 可选)        : 是否在最后一层 Linear 之后也添加激活函数.
#                                       - True : 每层 Linear 后都接激活函数 (适用于特征提取/编码器主体).
#                                       - False: 最后一层 Linear 不接激活函数 (适用于输出 Q 值等回归任务).
#
# 返回:
#   nn.Sequential : 由 Linear 层和激活函数交替组成的顺序模型.
def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels  # 当前层的输入维度, 初始化为整体输入维度
    module_list = []    # 用于按顺序存放每一层的模块
    # 遍历每一层的输出维度, 逐层构建 Linear + 激活函数
    for idx, c_out in enumerate(mlp_channels):
        # 添加当前层的线性变换: y = Wx + b
        module_list.append(nn.Linear(c_in, c_out))
        # 判断是否添加激活函数:
        #   - 若 last_act 为 True, 则所有层 (包括最后一层) 均添加激活函数;
        #   - 若 last_act 为 False, 则除最后一层外均添加激活函数 (idx < len(mlp_channels) - 1).
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out  # 更新下一层的输入维度为当前层的输出维度
    # 将所有模块封装为 nn.Sequential, 以便像单模块一样前向调用
    return nn.Sequential(*module_list)

# Soft Q 网络 (用于估计状态-动作对的价值)
# Q(s, a) 近似软 Bellman 方程中的 Q 值函数: Q(s, a) ≈ r + γ * E_{s'~p, a'~π}[Q(s', a') - α * log π(a'|s')]
class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder: EncoderObsWrapper):
        super().__init__()
        self.encoder = encoder # 视觉编码器, 用于提取视觉特征
        action_dim = np.prod(envs.single_action_space.shape) # 动作空间维度, np.prod()函数计算数组中所有元素的乘积
        state_dim = envs.single_observation_space['state'].shape[0] # 状态空间维度
        # Soft Q 网络(self.mlp)结构: 视觉特征(self.encoder) + 状态 + 动作 -> 512 -> 256 -> 1 (Q值)
        self.mlp = make_mlp(encoder.encoder.out_dim + action_dim + state_dim, [512, 256, 1], last_act=False)
        # 打印 SoftQNetwork 结构与维度信息
        print(f"[SoftQNetwork] 模型结构与维度信息:")
        print(f"  - 视觉特征维度 (encoder.out_dim): {encoder.encoder.out_dim}")
        print(f"  - 动作维度 (action_dim): {action_dim}")
        print(f"  - 状态维度 (state_dim): {state_dim}")
        print(f"  - 输入总维度 (visual + state + action): {encoder.encoder.out_dim + action_dim + state_dim}")
        print(f"  - MLP 隐藏层维度: [512, 256, 1]")
        print(f"  - MLP 网络结构:\n{self.mlp}")
        print(f"  - MLP 参数量: {sum(p.numel() for p in self.mlp.parameters())}")
        
    # 前向传播: 视觉特征(观测) + 状态 + 动作 -> Q 值
    def forward(self, obs, action, visual_feature=None, detach_encoder=False):
        # 如果没有提供视觉特征, 使用编码器提取
        if visual_feature is None:
            visual_feature = self.encoder(obs)
        # 如果需要分离编码器梯度 (用于策略更新时防止梯度流回编码器)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        # 拼接视觉特征、状态和动作
        x = torch.cat([visual_feature, obs["state"], action], dim=1)
        # 打印前向传播维度 (仅第一次)
        if not getattr(self, "_printed_fwd_dims", False):
            print(f"[SoftQNetwork.forward] 维度信息 (仅打印一次):")
            print(f"  - visual_feature.shape: {visual_feature.shape}")
            print(f"  - obs['state'].shape: {obs['state'].shape}")
            print(f"  - action.shape: {action.shape}")
            print(f"  - 拼接后 x.shape: {x.shape}")
            out = self.mlp(x)
            print(f"  - Q 值输出 out.shape: {out.shape}")
            self._printed_fwd_dims = True
            return out
        return self.mlp(x)


# 对数标准差的最大值和最小值 (用于限制策略网络输出的范围)
LOG_STD_MAX = 2
LOG_STD_MIN = -5

# Actor 网络 (策略网络)
# 策略网络 π(a|s) 输出动作分布的均值 μ(s) 和标准差 σ(s)
# 使用 tanh 高斯策略: a = tanh(μ(s) + σ(s) * ε), ε ~ N(0, I)
# 对应的概率密度为: π(a|s) = N(u; μ(s), σ(s)) * |det(da/du)|^{-1}
class Actor(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        action_dim = np.prod(envs.single_action_space.shape) # 动作空间维度, np.prod()函数计算数组中所有元素的乘积
        state_dim = envs.single_observation_space['state'].shape[0] # 状态空间维度
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
        # Actor 网络(self.mlp)结构: 视觉特征(self.encoder) + 状态 -> 512 -> 256
        self.mlp = make_mlp(self.encoder.encoder.out_dim + state_dim, [512, 256], last_act=True)
        # 输出均值和对数标准差
        self.fc_mean = nn.Linear(256, action_dim)   # MLP的输出维度=256
        self.fc_logstd = nn.Linear(256, action_dim) # MLP的输出维度=256
        # 动作缩放
        self.action_scale = torch.FloatTensor((envs.single_action_space.high - envs.single_action_space.low) / 2.0)
        self.action_bias = torch.FloatTensor((envs.single_action_space.high + envs.single_action_space.low) / 2.0)

        # 打印 Actor 结构与维度信息
        print(f"[Actor] 模型结构与维度信息:")
        print(f"  - action_dim (动作维度): {action_dim}")
        print(f"  - state_dim (状态维度): {state_dim}")
        print(f"  - in_channels (输入通道数): {in_channels}")
        print(f"  - image_size (图像尺寸): {image_size}")
        print(f"  - encoder.out_dim (视觉特征维度): {self.encoder.encoder.out_dim}")
        print(f"  - MLP 输入维度 (visual + state): {self.encoder.encoder.out_dim + state_dim}")
        print(f"  - MLP 隐藏层维度: [512, 256]")
        print(f"  - MLP 网络结构:\n{self.mlp}")
        print(f"  - fc_mean: {self.fc_mean}")
        print(f"  - fc_logstd: {self.fc_logstd}")
        print(f"  - action_scale.shape: {self.action_scale.shape}")
        print(f"  - action_bias.shape: {self.action_bias.shape}")
        print(f"  - 总参数量: {sum(p.numel() for p in self.parameters())}")

    # ------------------------------------------------------------------------
    # 获取策略网络的共享特征表示 (Actor 前向传播核心)
    #
    # 作用:
    #   将视觉观测 (RGB/RGBD) 与状态观测 (state) 融合为统一的特征向量,
    #   供后续 fc_mean / fc_logstd 头部生成动作分布参数 (均值 μ(s) 与对数标准差 log σ(s)).
    #   该方法是 Actor 网络前向传播的核心, 被 forward / get_action / get_eval_action 复用.
    #
    # 数据流:
    #   1. 视觉特征: obs['rgb'] (或 'depth') -> self.encoder -> visual_feature (B, out_dim)
    #   2. 状态特征: obs['state'] (B, state_dim)
    #   3. 拼接: [visual_feature, state] -> x (B, out_dim + state_dim)
    #   4. MLP 投影: x -> self.mlp -> mlp_out (B, 256)
    #
    # 参数:
    #   obs            : 字典形式观测, 至少包含 'state' 键以及 'rgb'/'depth' 之一.
    #   detach_encoder : 是否切断编码器梯度回传.
    #                    - True : 用于策略更新阶段, 防止策略损失梯度流入共享编码器
    #                            (SAC 中编码器主要由 Q 网络损失更新, 以保持视觉表征稳定).
    #                    - False: 编码器随策略损失一起更新 (默认).
    #
    # 返回:
    #   mlp_out        : MLP 输出的共享特征, 形状 (B, 256), 用于生成 μ(s) 与 log σ(s).
    #   visual_feature : 视觉编码器输出, 形状 (B, out_dim). 额外返回以便外部 (如 Q 网络)
    #                    复用同一份视觉特征, 避免重复前向计算 (SAC 中 Q 网络与 Actor 共享编码器).
    # ------------------------------------------------------------------------
    def get_feature(self, obs, detach_encoder=False):
        # 1. 视觉编码: 通过 EncoderObsWrapper 处理 RGB/Depth, 输出 (B, out_dim) 视觉特征
        visual_feature = self.encoder(obs)
        # 2. 可选: 切断梯度, 防止策略损失更新共享编码器 (SAC 常见做法, 编码器由 Q 损失驱动)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        # 3. 特征融合: 沿特征维度 (dim=1) 拼接视觉特征与状态向量 -> (B, out_dim + state_dim)
        x = torch.cat([visual_feature, obs['state']], dim=1)
        # 4. (仅首次) 打印前向传播各阶段维度, 便于调试与核对网络结构
        if not getattr(self, "_printed_feature_dims", False):
            print(f"[Actor.get_feature] 维度信息 (仅打印一次):")
            print(f"  - visual_feature.shape: {visual_feature.shape}")
            print(f"  - obs['state'].shape: {obs['state'].shape}")
            print(f"  - 拼接后 x.shape: {x.shape}")
            mlp_out = self.mlp(x)
            print(f"  - MLP 输出 shape: {mlp_out.shape}")
            self._printed_feature_dims = True
            return mlp_out, visual_feature
        # 5. MLP 投影到 256 维共享特征空间, 供 fc_mean / fc_logstd 使用
        return self.mlp(x), visual_feature

    # ------------------------------------------------------------------------
    # 前向传播: 由共享特征生成动作分布参数 (均值 μ(s) 与对数标准差 log σ(s))
    #
    # 作用:
    #   调用 get_feature 获取视觉+状态融合后的共享特征 x, 再分别通过 fc_mean / fc_logstd
    #   两个线性头输出高斯策略的均值和对数标准差. 为了避免 log_std 数值过大/过小导致
    #   采样不稳定, 使用 tanh 将其映射到 [LOG_STD_MIN, LOG_STD_MAX] 区间.
    #
    # 数据流:
    #   obs -> get_feature -> x (B, 256)
    #   x -> fc_mean    -> mean    (B, action_dim)        动作分布均值
    #   x -> fc_logstd  -> log_std (B, action_dim)        动作分布对数标准差 (未裁剪)
    #   log_std -> tanh + 线性映射 -> log_std ∈ [LOG_STD_MIN, LOG_STD_MAX]
    #
    # 参数:
    #   obs            : 字典形式观测, 至少包含 'state' 键以及 'rgb'/'depth' 之一.
    #   detach_encoder : 是否切断编码器梯度回传 (透传给 get_feature).
    #
    # 返回:
    #   mean           : 动作分布均值 μ(s), 形状 (B, action_dim).
    #   log_std        : 裁剪后的对数标准差 log σ(s), 形状 (B, action_dim).
    #   visual_feature : 视觉编码器输出, 形状 (B, out_dim). 额外返回以便 Q 网络复用,
    #                    避免重复前向计算 (SAC 中 Q 网络与 Actor 共享编码器).
    # ------------------------------------------------------------------------
    def forward(self, obs, detach_encoder=False):
        # 1. 获取共享特征表示 (视觉 + 状态 -> MLP -> 256 维特征)
        x, visual_feature = self.get_feature(obs, detach_encoder)
        # 2. 通过线性头输出动作分布均值 μ(s)
        mean = self.fc_mean(x)
        # 3. 通过线性头输出动作分布对数标准差 log σ(s) (尚未裁剪)
        log_std = self.fc_logstd(x)
        # 4. 使用 tanh 将 log_std 限制在 [LOG_STD_MIN, LOG_STD_MAX] 范围内
        #    该变换来自 SpinUp / Denis Yarats 实现:
        #      - torch.tanh(log_std) ∈ (-1, 1)
        #      - 再线性映射到 [LOG_STD_MIN, LOG_STD_MAX]: 0.5*(MAX-MIN)*(tanh+1) + MIN
        #    相比简单的 torch.clamp, 该映射处处可导, 梯度更平滑, 有利于训练稳定.
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # 来自 SpinUp / Denis Yarats

        return mean, log_std, visual_feature

    # ------------------------------------------------------------------------
    # 获取评估动作 (确定性策略)
    #
    # 作用:
    #   在评估/测试阶段使用策略的均值动作 (而非采样动作), 以获得确定性的行为输出.
    #   由于策略输出的是未缩放的高斯均值, 需先经过 tanh 压缩到 (-1, 1), 再通过
    #   action_scale / action_bias 线性映射回实际动作空间范围.
    #
    # 动作映射公式:
    #   a = tanh(μ(s)) * action_scale + action_bias
    #   其中 action_scale = (high - low) / 2, action_bias = (high + low) / 2
    #   可将 tanh 输出的 (-1, 1) 映射到动作空间 [low, high].
    #
    # 参数:
    #   obs : 字典形式观测, 至少包含 'state' 键以及 'rgb'/'depth' 之一.
    #
    # 返回:
    #   action : 确定性评估动作, 形状 (B, action_dim), 已映射到动作空间范围.
    # ------------------------------------------------------------------------
    def get_eval_action(self, obs):
        # 1. 前向传播获取动作分布参数 (仅需 mean, 忽略 log_std 和 visual_feature)
        mean, _, _ = self(obs)
        # 2. 对均值施加 tanh 并线性映射到动作空间 [low, high], 得到确定性评估动作
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    # 获取训练动作 (随机策略)
    # 使用重参数化技巧 (Reparameterization Trick):
    #   u = μ(s) + σ(s) * ε, ε ~ N(0, I)
    #   a = tanh(u) * scale + bias
    # 对数概率 (考虑 tanh 变换的 Jacobian 修正):
    #   log π(a|s) = log N(u; μ(s), σ(s)) - Σ log(scale * (1 - tanh²(u)) + ε)
    def get_action(self, obs, detach_encoder=False):
        mean, log_std, visual_feature = self(obs, detach_encoder) # 获取动作分布参数
        std = log_std.exp() # 计算标准差 σ(s) = exp(log σ(s))
        normal = torch.distributions.Normal(mean, std) # 创建高斯分布 N(μ(s), σ(s))
        # 使用重参数化技巧采样 (mean + std * N(0,1))
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # 强制动作边界, 修正 tanh 变换的对数概率: log|da/du| = log(scale * (1 - tanh²(u)))
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, visual_feature

    # ------------------------------------------------------------------------
    # 将模型移动到指定设备 (重写 nn.Module.to)
    #
    # 作用:
    #   将 Actor 网络的所有张量 (包括网络参数和动作缩放张量) 迁移到目标设备
    #   (如 'cpu' / 'cuda'), 以便后续在该设备上执行前向计算.
    #
    # 为何需要重写:
    #   self.action_scale 和 self.action_bias 在 __init__ 中通过 torch.FloatTensor
    #   直接创建, 它们既不是 nn.Parameter (不参与梯度优化), 也没有通过
    #   register_buffer 注册为模块缓冲区. 因此父类 nn.Module.to(device) 无法
    #   自动感知并迁移这两个张量, 会导致它们停留在 CPU 上, 而网络参数已迁移到
    #   GPU, 前向计算时因设备不一致而报错.
    #
    #   解决方案: 在调用父类 to() 之前, 先手动将这两个张量迁移到目标设备,
    #   再通过 super().to(device) 迁移所有已注册的参数 (parameters) 与缓冲区
    #   (buffers), 确保整个模块的所有张量都位于同一设备.
    #
    # 参数:
    #   device : 目标设备, 可以是 torch.device 对象或字符串 (如 'cuda' / 'cpu').
    #
    # 返回:
    #   self : 返回模块自身 (与 nn.Module.to 的行为一致, 支持链式调用).
    # ------------------------------------------------------------------------
    def to(self, device):
        # 1. 手动迁移动作缩放张量 (普通 Tensor, 不被父类 to() 管理)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        # 2. 调用父类 to() 迁移所有注册的参数与缓冲区, 并返回 self
        return super().to(device)

# ============================================================================
# 日志记录器类 (Logger)
# ----------------------------------------------------------------------------
# 作用:
#   统一封装训练/评估过程中标量指标 (如 loss、return、success_once 等) 的记录逻辑,
#   对上层屏蔽底层日志后端的具体实现. 当前支持两种后端:
#     1. TensorBoard (通过 torch.utils.tensorboard.SummaryWriter): 本地可视化,
#        数据写入 runs/{run_name} 目录, 可用 `tensorboard --logdir runs` 查看.
#     2. Weights & Biases (wandb, 可选): 云端实验跟踪平台, 便于跨实验对比与协作.
#
# 设计要点:
#   - 双后端兼容: 通过 log_wandb 开关控制是否同时向 wandb 上报数据. 无论是否启用
#     wandb, 始终会向 TensorBoard 写入, 保证本地始终留存完整日志.
#   - 轻量包装: 该类不缓存/聚合数据, 仅做透传, 实际的步数对齐与时间戳由调用方
#     (主训练循环) 通过 step 参数显式传入, 保证不同指标的时间轴一致.
#   - 生命周期管理: 提供 close() 方法在训练结束时关闭 TensorBoard writer,
#     确保缓冲数据落盘; 注意 wandb 的退出由 wandb.finish() 单独管理 (本脚本未调用).
# ============================================================================
class Logger:
    # ------------------------------------------------------------------------
    # 构造函数: 初始化日志后端
    #
    # 参数:
    #   log_wandb   (bool, 可选)        : 是否同时向 Weights & Biases 上报数据.
    #                                     - True : add_scalar 时额外调用 wandb.log;
    #                                     - False: 仅写入 TensorBoard.
    #                                     注意: 启用前需在外部已完成 wandb.init(),
    #                                     否则调用 add_scalar 会因 wandb 未初始化而报错.
    #   tensorboard (SummaryWriter, 可选): 已创建的 TensorBoard SummaryWriter 实例,
    #                                     用于本地标量/文本数据的写入. 若为 None 则
    #                                     add_scalar / close 将因 writer 为 None 而失败,
    #                                     因此调用方应确保在训练模式下传入有效 writer.
    # ------------------------------------------------------------------------
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard   # TensorBoard 写入器 (本地日志后端)
        self.log_wandb = log_wandb  # 是否同时启用 wandb (云端日志后端)

    # ------------------------------------------------------------------------
    # 记录一个标量指标
    #
    # 作用:
    #   将单个标量值 (如损失、奖励、成功率等) 同时写入启用的日志后端.
    #   - 若 log_wandb 为 True, 先调用 wandb.log 上报到云端, step 用于对齐时间轴;
    #   - 无论 log_wandb 取值, 始终调用 self.writer.add_scalar 写入 TensorBoard.
    #
    # 参数:
    #   tag          (str)   : 指标名称, 通常使用 "/" 分层 (如 "losses/qf1_loss",
    #                          "eval/return"), TensorBoard 会据此自动分组展示.
    #   scalar_value (float) : 指标数值, 可为 Python 标量或 0 维张量.
    #   step         (int)   : 对应的全局步数 (一般为 global_step), 用于确定曲线的 x 轴.
    #                          不同指标共用同一 step 以保证时间轴对齐.
    # ------------------------------------------------------------------------
    def add_scalar(self, tag, scalar_value, step):
        # 1. (可选) 上报到 Weights & Biases 云端
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        # 2. 始终写入本地 TensorBoard
        self.writer.add_scalar(tag, scalar_value, step)

    # ------------------------------------------------------------------------
    # 关闭日志记录器
    #
    # 作用:
    #   关闭内部的 TensorBoard SummaryWriter, 刷新并释放底层文件资源,
    #   确保所有缓冲的标量数据已落盘到 runs/{run_name} 目录.
    #   通常在训练结束 (或异常退出前) 调用一次.
    #
    # 注意:
    #   - 本方法仅关闭 TensorBoard writer, 不会调用 wandb.finish();
    #     wandb 的资源释放需由调用方另行处理.
    #   - 重复调用 close() 可能因 writer 已关闭而抛出异常, 应避免.
    # ------------------------------------------------------------------------
    def close(self):
        self.writer.close()

# 主函数
if __name__ == "__main__":
    # 解析命令行参数
    args = tyro.cli(Args)
    # 计算运行时参数
    args.grad_steps_per_iteration = int(args.training_freq * args.utd) # 64 * 0.25 = 16
    args.steps_per_env = args.training_freq // args.num_envs # 64 / 32 = 2
    # 设置实验名称
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
        
    # 打印输入的和计算得到的关键参数值
    print("=" * 80)
    print("[参数配置] 输入的和计算得到的关键参数值:")
    print("=" * 80)
    # --- 实验相关参数 ---
    print("【实验相关参数】")
    print(f"  - exp_name: {args.exp_name}")
    print(f"  - run_name: {run_name}")
    print(f"  - seed: {args.seed}")
    print(f"  - torch_deterministic: {args.torch_deterministic}")
    print(f"  - cuda: {args.cuda}")
    print(f"  - track (wandb): {args.track}")
    print(f"  - capture_video: {args.capture_video}")
    print(f"  - save_trajectory: {args.save_trajectory}")
    print(f"  - save_model: {args.save_model}")
    print(f"  - evaluate: {args.evaluate}")
    print(f"  - checkpoint: {args.checkpoint}")
    print(f"  - log_freq: {args.log_freq}")
    # --- 环境特定参数 ---
    print("【环境特定参数】")
    print(f"  - env_id: {args.env_id}")
    print(f"  - obs_mode: {args.obs_mode}")
    print(f"  - include_state: {args.include_state}")
    print(f"  - env_vectorization: {args.env_vectorization}")
    print(f"  - num_envs (训练): {args.num_envs}")
    print(f"  - num_eval_envs (评估): {args.num_eval_envs}")
    print(f"  - num_steps: {args.num_steps}")
    print(f"  - num_eval_steps: {args.num_eval_steps}")
    print(f"  - eval_freq: {args.eval_freq}")
    print(f"  - control_mode: {args.control_mode}")
    print(f"  - render_mode: {args.render_mode}")
    print(f"  - camera_width: {args.camera_width}")
    print(f"  - camera_height: {args.camera_height}")
    # --- 算法特定参数 ---
    print("【算法特定参数】")
    print(f"  - total_timesteps: {args.total_timesteps}")
    print(f"  - buffer_size: {args.buffer_size}")
    print(f"  - buffer_device: {args.buffer_device}")
    print(f"  - gamma: {args.gamma}")
    print(f"  - tau: {args.tau}")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - learning_starts: {args.learning_starts}")
    print(f"  - policy_lr: {args.policy_lr}")
    print(f"  - q_lr: {args.q_lr}")
    print(f"  - policy_frequency: {args.policy_frequency}")
    print(f"  - target_network_frequency: {args.target_network_frequency}")
    print(f"  - alpha: {args.alpha}")
    print(f"  - autotune: {args.autotune}")
    print(f"  - training_freq: {args.training_freq}")
    print(f"  - utd: {args.utd}")
    print(f"  - bootstrap_at_done: {args.bootstrap_at_done}")
    # --- JEPA 辅助模块参数 ---
    print("【JEPA 辅助模块参数】")
    print(f"  - use_jepa: {args.use_jepa}")
    print(f"  - jepa_lr: {args.jepa_lr}")
    print(f"  - jepa_hidden_dims: {args.jepa_hidden_dims}")
    print(f"  - jepa_update_encoder: {args.jepa_update_encoder}")
    # --- 计算得到的运行时参数 ---
    print("【计算得到的运行时参数】")
    print(f"  - grad_steps_per_iteration = training_freq * utd = {args.training_freq} * {args.utd} = {args.grad_steps_per_iteration}")
    print(f"  - steps_per_env = training_freq // num_envs = {args.training_freq} // {args.num_envs} = {args.steps_per_env}")
    print(f"  - global_steps_per_iteration = num_envs * steps_per_env = {args.num_envs} * {args.steps_per_env} = {args.num_envs * args.steps_per_env}")
    print("=" * 80)

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

    # 打印环境观测/动作空间信息
    print(f"[Env] 环境空间信息:")
    print(f"  - single_observation_space: {envs.single_observation_space}")
    print(f"  - single_action_space: {envs.single_action_space}")
    print(f"  - action_space.shape: {envs.action_space.shape}")
    
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
    actor = Actor(envs, sample_obs=obs).to(device) # 创建 Actor 网络, 共享视觉编码器
    qf1 = SoftQNetwork(envs, actor.encoder).to(device)  # 第一个软Q网络
    qf2 = SoftQNetwork(envs, actor.encoder).to(device)  # 第二个软Q网络，用于双重Q学习以减少过拟合
    qf1_target = SoftQNetwork(envs, actor.encoder).to(device)  # 第一个目标软Q网络，用于稳定训练
    qf2_target = SoftQNetwork(envs, actor.encoder).to(device)  # 第二个目标软Q网络，同样用于双重Q学习
    
    # 如果有检查点, 加载模型
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)   # 加载检查点文件
        actor.load_state_dict(ckpt['actor']) # 加载actor模型
        qf1.load_state_dict(ckpt['qf1'])     # 加载第一个Q网络
        qf2.load_state_dict(ckpt['qf2'])     # 加载第二个Q网络
        
    # 初始化目标网络
    qf1_target.load_state_dict(qf1.state_dict()) # 将第一个Q网络的状态字典复制到目标网络
    qf2_target.load_state_dict(qf2.state_dict()) # 将第二个Q网络的状态字典复制到目标网络
    
    # 初始化优化器 (包含 Q 网络的 MLP 和共享的编码器)
    q_optimizer = optim.Adam(
        list(qf1.mlp.parameters()) +
        list(qf2.mlp.parameters()) +
        list(qf1.encoder.parameters()),
        lr=args.q_lr)
    # 初始化 actor 优化器
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    
    # JEPA 辅助模块 (可选): 与 actor.encoder 共享同一个 Encoder 实例, 独立 Predictor 和 Optimizer
    jepa = None # 初始化jepa变量为None
    # 如果参数中指定使用jepa
    if args.use_jepa:
        # 将jepa隐藏维度参数字符串转换为整数元组
        jepa_hidden_dims = tuple(int(x) for x in args.jepa_hidden_dims.split(",") if x.strip() != "")
        # 创建jepa配置对象，包含以下参数：
        # latent_dim: 编码器的输出维度
        # hidden_dims: jepa的隐藏层维度
        # lr: 学习率
        # update_encoder: 是否更新编码器
        jepa_config = JEPAConfig(
            latent_dim=actor.encoder.encoder.out_dim,
            hidden_dims=jepa_hidden_dims,
            lr=args.jepa_lr,
            update_encoder=args.jepa_update_encoder,
        )
        # 创建JEPA模型实例，传入以下参数：
        # encoder: 编码器
        # action_dim: 动作维度（通过环境动作空间形状计算）
        # config: jepa配置
        # device: 计算设备
        jepa = JEPA(
            encoder=actor.encoder,
            action_dim=int(np.prod(envs.single_action_space.shape)),
            config=jepa_config,
            device=device,
        )

    # 自动熵调整(可选)
    # 目标熵: H_target = -|A| (动作维度的负数)
    # 熵系数 α 通过以下损失自动调整:
    #   J(α) = E[-α * (log π(a|s) + H_target)]
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    global_step = 0  # 全局环境交互步数计数器 (所有并行环境合计), 用于控制训练进度与日志 x 轴
    global_update = 0  # 全局梯度更新次数计数器, 用于控制策略/目标网络的更新频率
    learning_has_started = False  # 标记是否已过 learning_starts 阶段 (区分随机探索与策略采样)

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)  # 每次迭代收集的总环境步数, 用于计算 rollout FPS 等指标
    pbar = tqdm.tqdm(range(args.total_timesteps))  # 创建训练进度条, 可视化总步数推进情况
    cumulative_times = defaultdict(float)  # 累计各阶段 (rollout/update/eval) 耗时的字典, 用于统计时间开销

    # 主训练循环
    while global_step < args.total_timesteps:
        # 触发评估的逻辑: 刚完成一轮完整训练, 跨过了评估间隔的整数边界, 需要触发评估
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
                # f"success_once: {eval_metrics_mean['success_once']:.2f}, "
                # f"return: {eval_metrics_mean['return']:.2f}"
                f"success_once: {eval_metrics_mean.get('success_once', 0.0):.2f}, "
                f"return: {eval_metrics_mean.get('return', 0.0):.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
                
            # 如果是评估模式, 则退出循环
            if args.evaluate:
                break
            # 否则, 切换回训练模式
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
            global_step += 1 * args.num_envs # 每个环境步增加 num_envs 个全局步数

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

            # 打印第一次采样的批次维度信息
            if not learning_has_started or global_update == 1:
                print(f"[ReplayBuffer.sample] 第一次采样批次维度信息 (global_update={global_update}):")
                print(f"  - batch_size: {args.batch_size}")
                for k, v in data.obs.items():
                    print(f"  - data.obs['{k}'].shape: {v.shape}")
                for k, v in data.next_obs.items():
                    print(f"  - data.next_obs['{k}'].shape: {v.shape}")
                print(f"  - data.actions.shape: {data.actions.shape}")
                print(f"  - data.rewards.shape: {data.rewards.shape}")
                print(f"  - data.dones.shape: {data.dones.shape}")

            # 更新价值网络 (Q 网络)
            # 软 Bellman 目标值:
            # y = r + γ * (1 - done) * (min(Q₁'(s', a'), Q₂'(s', a')) - α * log π(a'|s'))
            # 其中 a' ~ π(·|s'), Q' 为目标网络
            with torch.no_grad():
                # 获取下一状态的动作和策略熵
                next_state_actions, next_state_log_pi, _, visual_feature = actor.get_action(data.next_obs)
                # 计算目标 Q 值
                qf1_next_target = qf1_target(data.next_obs, next_state_actions, visual_feature)
                qf2_next_target = qf2_target(data.next_obs, next_state_actions, visual_feature)
                # 取最小值并减去熵正则化项: min(Q₁', Q₂') - α * log π
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # 计算目标 Q 值: y = r + γ * (1 - done) * (min_qf_next - α * log π)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)
                # data.dones 是 "stop_bootstrap", 根据前面的 args.bootstrap_at_done 计算
            # 提取当前状态的特征
            visual_feature = actor.encoder(data.obs)
            # 计算 Q 值
            qf1_a_values = qf1(data.obs, data.actions, visual_feature).view(-1)
            qf2_a_values = qf2(data.obs, data.actions, visual_feature).view(-1)
            # 计算 Q 网络损失 (均方误差): L_Q = E[(Q(s,a) - y)²]
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # 优化 Q 网络
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            # 更新策略网络
            # 策略损失 (最大化期望回报 + 熵):
            # J_π = E_{s~D, a~π}[α * log π(a|s) - min(Q₁(s,a), Q₂(s,a))]
            # 即最小化: L_π = E[α * log π(a|s) - min(Q₁(s,a), Q₂(s,a))]
            if global_update % args.policy_frequency == 0:  # TD3 延迟更新支持
                pi, log_pi, _, visual_feature = actor.get_action(data.obs)
                # 计算 Q 值 (detach_encoder=True 防止梯度流回编码器)
                qf1_pi = qf1(data.obs, pi, visual_feature, detach_encoder=True)
                qf2_pi = qf2(data.obs, pi, visual_feature, detach_encoder=True)
                min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                # 计算策略损失: L_π = E[α * log π(a|s) - min(Q₁(s,a), Q₂(s,a))]
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                # 优化策略网络
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 自动调整熵系数
                # 熵系数损失: J(α) = E[-α * (log π(a|s) + H_target)]
                # 其中 H_target 为目标熵, 通常取 -|A|
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
            # 目标网络软更新公式: θ' ← τ * θ + (1 - τ) * θ'
            # 其中 τ 为平滑系数 (较小的值, 如 0.01)
            if global_update % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # JEPA 辅助损失更新 (独立 Predictor / Optimizer, 不影响 SAC Loss)
            if args.use_jepa and jepa is not None:
                jepa.update(data.obs, data.next_obs, data.actions)
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
            if args.use_jepa and jepa is not None:
                logger.add_scalar("losses/jepa_loss", jepa.get_last_loss(), global_step)

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
