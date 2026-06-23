"""
Action-Conditioned JEPA (Joint Embedding Predictive Architecture) 辅助模块.

本模块作为 SAC-MoE 的独立辅助损失 (Auxiliary Loss) 存在:
  - 与 actor.encoder 共享同一个 EncoderObsWrapper 实例 (不复制权重).
  - 使用独立的 Predictor 网络与独立的 Optimizer.
  - 仅通过 update() 接口在训练循环中被调用, 不介入 SAC Loss / MoE Forward / Replay Buffer.

预测目标:
    z_t     = encoder(obs)            # 不 detach, 梯度回传到共享 encoder
    z_next  = encoder(next_obs).detach()
    z_pred  = predictor(z_t, actions)
    loss    = MSE(z_pred, z_next)

注意: 由于 encoder 被 SAC 与 JEPA 共享, JEPA 的梯度会同时影响 encoder.
为避免干扰 SAC 主流程的梯度图, update() 内部会在反向传播后只 step JEPA 自己的
optimizer (仅含 predictor 参数), encoder 参数仍由 SAC 的 optimizer 负责更新.
若希望 encoder 也被 JEPA 优化, 可通过 update_encoder=True 开启 (默认关闭).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    """与 sac_moe_rgbd.py 中保持一致的简易 MLP 构造函数."""
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(act_builder())
        c_in = c_out
    return nn.Sequential(*module_list)


class JEPAPredictor(nn.Module):
    """动作条件的潜在状态转移预测器.

    输入: (z_t, action) -> 拼接后送入 MLP
    输出: z_pred (与 z_t 同维度的潜在表示)
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dims=(256, 256), last_act=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.net = make_mlp(latent_dim + action_dim, list(hidden_dims) + [latent_dim], last_act=last_act)

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, action], dim=-1)
        return self.net(x)


@dataclass
class JEPAConfig:
    """JEPA 辅助模块的超参数."""
    latent_dim: int = 256
    """编码器输出维度 (与 PlainConv.out_dim 对齐)."""
    hidden_dims: tuple = (256, 256)
    """Predictor 隐藏层维度."""
    lr: float = 3e-4
    """JEPA 独立 optimizer 的学习率."""
    update_encoder: bool = False
    """是否让 JEPA optimizer 也更新共享 encoder 参数. 默认 False, 即 encoder 仅由 SAC 更新."""


class JEPA(nn.Module):
    """Action-Conditioned JEPA 辅助模块.

    Args:
        encoder: 共享的 EncoderObsWrapper 实例 (来自 actor.encoder).
        action_dim: 动作空间维度.
        config: JEPA 超参数.
        device: 计算设备.
    """

    def __init__(self, encoder: nn.Module, action_dim: int, config: Optional[JEPAConfig] = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.config = config if config is not None else JEPAConfig()
        self.device = device
        # 共享 encoder: 不复制, 直接引用 actor.encoder
        self.encoder = encoder
        # 独立 Predictor
        self.predictor = JEPAPredictor(
            latent_dim=self.config.latent_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            last_act=True,
        ).to(device)

        # 独立 Optimizer: 默认只包含 predictor 参数
        if self.config.update_encoder:
            params = list(self.predictor.parameters()) + list(self.encoder.parameters())
        else:
            params = list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)

        # 最近一次 loss, 便于外部日志记录
        self.last_loss: float = 0.0

    def encode(self, obs, detach: bool = False):
        """通过共享 encoder 获取潜在表示."""
        z = self.encoder(obs)
        if detach:
            z = z.detach()
        return z

    def predict(self, z_t, actions):
        """由 (z_t, actions) 预测下一时刻潜在表示."""
        return self.predictor(z_t, actions)

    def compute_loss(self, obs, next_obs, actions) -> torch.Tensor:
        """计算 JEPA MSE 损失.

        z_t    = encoder(obs)            # 不 detach
        z_next = encoder(next_obs).detach()
        z_pred = predictor(z_t, actions)
        loss   = MSE(z_pred, z_next)
        """
        z_t = self.encoder(obs)
        z_next = self.encoder(next_obs).detach()
        z_pred = self.predictor(z_t, actions)
        loss = F.mse_loss(z_pred, z_next)
        return loss

    @torch.no_grad()
    def get_last_loss(self) -> float:
        return self.last_loss

    def update(self, obs, next_obs, actions) -> float:
        """执行一次 JEPA 辅助梯度更新.

        Args:
            obs: 当前观测 (dict, 与 ReplayBuffer.sample 输出一致).
            next_obs: 下一时刻观测 (dict).
            actions: 动作张量 (B, action_dim).

        Returns:
            本次 JEPA loss 的标量值.
        """
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(obs, next_obs, actions)
        loss.backward()
        self.optimizer.step()
        self.last_loss = loss.item()
        return self.last_loss