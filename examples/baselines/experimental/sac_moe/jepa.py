"""
Action-Conditioned JEPA (Joint Embedding Predictive Architecture) 辅助模块,作为 SAC/SAC-MoE 的独立辅助损失 (Auxiliary Loss) 存在.
满足 JEPA 的核心三件事: 在 latent space 中, 用 predictor 去预测 target encoder 产生的 representation, 而不是预测像素或原始观测.

SAC 原本只想把 encoder 训练成"有利于回报最大化"的特征提取器;
JEPA 往梯度里额外增加一项约束: 提取出来的潜特征 z 必须能被 (z_t,a_t) 预测出下一时刻 z_{t+1};
相当于给表征增加了动力学一致性正则项, encoder 不再只服从奖励信号，同时还要满足潜空间状态转移可预测.
数学形式等价于: 总优化目标变成带正则的强化学习: Ltotal = Lsac + λ⋅Ljepa
只不过这个正则项不是直接加在 loss 里, 而是通过两次 backward 梯度叠加实现.

网络架构:
  - 与 actor.encoder 共享同一个 EncoderObsWrapper 实例 (不复制权重).
  - 使用独立的 Predictor 网络与独立的 Optimizer.
  - 仅通过 update() 接口在训练循环中被调用, 不介入 SAC Loss / MoE Forward / Replay Buffer.

预测目标 (Action-Conditioned Latent Dynamics):
  - 用 encoder 把观测压缩成 latent 表示 z, 然后训练一个 predictor 去预测下一时刻的 latent
  - 本质: 学习潜动力学模型(latent dynamics model), 但不直接在像素/观测空间做预测, 而是在 encoder 空间做预测

  符号定义:
    φ     : 共享 encoder g_φ(·) 的参数 (与 SAC 共享, 不复制权重)
    θ     : JEPA predictor f_θ(·) 的参数 (JEPA 独有)
    sg(·) : stop-gradient, 即 .detach(), 阻断梯度回传
    B     : batch 大小

  前向计算:
    z_t     = g_φ(o_t)                  # 当前状态潜表示, 不 detach, 梯度可回传到 φ
    z_{t+1} = sg(g_φ(o_{t+1}))          # 下一状态潜表示, detach, 不回传梯度到 encoder
    ẑ_{t+1} = f_θ(z_t, a_t)             # 动作条件预测, 预测下一状态潜表示

  JEPA 损失 (潜空间上的均方误差):
    L_JEPA = (1/B) · Σ_b || ẑ_{t+1}^b - z_{t+1}^b ||₂²

  优化:
    - 默认 (update_encoder=False): 仅更新 predictor        θ ← θ - η · ∇_θ L_JEPA
    - 开启 (update_encoder=True):  同时更新 predictor 与 encoder   θ,φ ← θ,φ - η · ∇_{θ,φ} L_JEPA

注意: 由于 encoder 被 SAC 与 JEPA 共享, JEPA 的梯度会同时影响 encoder.
为避免干扰 SAC 主流程的梯度图, update() 内部会在反向传播后只 step JEPA 自己的optimizer (仅含 predictor 参数).
encoder 参数仍由 SAC 的 optimizer 负责更新.
若希望 encoder 也被 JEPA 优化, 可通过 update_encoder=True 开启 (默认关闭).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    """简易多层感知机 (MLP) 构造函数.

    与 sac_moe_rgbd.py 中保持一致. 根据输入维度和各层输出维度列表,
    依次堆叠 nn.Linear 线性层与激活函数, 返回一个 nn.Sequential 容器.

    参数:
        in_channels (int): 输入特征维度 (第一层 Linear 的 in_features).
        mlp_channels (list[int]): 各层输出维度列表, 列表长度即为 MLP 的层数.
        act_builder (callable, 可选): 激活函数构造器, 默认为 nn.ReLU.
            传入构造器 (而非实例) 以便每层创建独立的激活函数模块.
        last_act (bool, 可选): 是否在最后一层 Linear 之后也添加激活函数.
            - True : 每层 Linear 后都接激活函数 (适用于特征提取/编码器主体).
            - False: 最后一层 Linear 不接激活函数 (适用于输出回归值等任务).

    返回:
        nn.Sequential: 由 Linear 层和激活函数交替组成的顺序模型.
    """
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
    return nn.Sequential(*module_list)


class JEPAPredictor(nn.Module):
    """动作条件的潜在状态转移预测器: 在 latent 空间里, 给定当前状态 + 动作, 下一状态什么样

    输入: (z_t, action) -> 拼接后送入 MLP
    输出: z_pred (与 z_t 同维度的潜在表示, 维度具体是 encoder 的输出维度 (batch_size, latent_dim))
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dims=(256, 256), last_act=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        # MLP 网络: 输入维度 = latent_dim + action_dim, 输出维度 = latent_dim
        self.net = make_mlp(latent_dim + action_dim, list(hidden_dims) + [latent_dim], last_act=last_act)
        # 打印 JEPAPredictor 结构与维度信息
        print(f"[JEPAPredictor] 模型结构与维度信息:")
        print(f"  - latent_dim (潜状态维度): {latent_dim}")
        print(f"  - action_dim (动作维度): {action_dim}")
        print(f"  - hidden_dims (隐藏层维度): {hidden_dims}")
        print(f"  - last_act: {last_act}")
        print(f"  - MLP 输入维度 (latent_dim + action_dim): {latent_dim + action_dim}")
        print(f"  - MLP 各层输出维度: {list(hidden_dims) + [latent_dim]}")
        print(f"  - Predictor 网络结构:\n{self.net}")

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # 动作条件潜状态预测: ẑ_{t+1} = f_θ(z_t, a_t)
        # 输入拼接 [z_t ; a_t] ∈ R^{latent_dim + action_dim}, 经 MLP 映射回 latent_dim 维
        # 打印输入张量维度 (仅在第一次前向时打印, 避免日志过多)
        if not getattr(self, "_printed_forward_dims", False):
            print(f"[JEPAPredictor.forward] 输入张量维度:")
            print(f"  - z_t.shape: {z_t.shape} (B, latent_dim={self.latent_dim})")
            print(f"  - action.shape: {action.shape} (B, action_dim={self.action_dim})")
            self._printed_forward_dims = True
        x = torch.cat([z_t, action], dim=-1) # 特征拼接(状态特征 + 动作特征)
        out = self.net(x)
        if not getattr(self, "_printed_forward_out", False):
            print(f"[JEPAPredictor.forward] 输出张量维度:")
            print(f"  - z_pred.shape: {out.shape} (B, latent_dim={self.latent_dim})")
            self._printed_forward_out = True
        return out


@dataclass
class JEPAConfig:
    """JEPA 辅助模块的超参数, 必须和encoder的输出维度对齐."""
    latent_dim: int = 256
    """编码器输出维度 (与 PlainConv.out_dim 对齐)."""
    hidden_dims: tuple = (256, 256)
    """Predictor 隐藏层维度."""
    lr: float = 3e-4
    """JEPA 独立 optimizer 的学习率."""
    update_encoder: bool = False
    """是否让 JEPA optimizer 也更新共享 encoder 参数. 默认 False, 即 encoder 仅由 SAC 更新."""
    """设置为True, 则 encoder + predictor 一起训练, 这决定 JEPA 是 "纯辅助头" 还是 "联合表示学习模块"."""
    
class JEPA(nn.Module):
    """Action-Conditioned JEPA 辅助模块.

    Args:
        encoder: 共享的 EncoderObsWrapper 实例 (来自 actor.encoder), JEPA 和 SAC 共用同一个 encoder.
        action_dim: 动作空间维度, 用于构建 predictor 网络.
        config: JEPA 超参数.
        device: 计算设备, 默认为 CPU.
    """

    def __init__(self, encoder: nn.Module, action_dim: int, config: Optional[JEPAConfig] = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.config = config if config is not None else JEPAConfig()
        self.device = device
        # 共享 encoder: 不复制, 直接引用 actor.encoder
        self.encoder = encoder
        # 独立 Predictor, 不共享权重
        self.predictor = JEPAPredictor(
            latent_dim=self.config.latent_dim,
            action_dim=action_dim,
            hidden_dims=self.config.hidden_dims,
            last_act=True,
        ).to(device)

        # 独立 Optimizer: 默认只包含 predictor 参数
        if self.config.update_encoder:
            # 把 predictor 和 encoder 里的所有可训练参数打包成一个列表，交给优化器统一管理, 即 encoder 也被 JEPA 优化
            params = list(self.predictor.parameters()) + list(self.encoder.parameters())
        else:
            params = list(self.predictor.parameters()) # encoder 不被JEPA优化
        # adams 优化器, 仅优化 predictor (和 encoder, 如果 update_encoder=True)
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)

        # 最近一次 loss, 便于外部日志记录
        self.last_loss: float = 0.0

        # 打印 JEPA 模块整体结构与配置信息
        print(f"[JEPA] 模块初始化完成, 结构与配置信息:")
        print(f"  - device: {self.device}")
        print(f"  - config.latent_dim: {self.config.latent_dim}")
        print(f"  - config.hidden_dims: {self.config.hidden_dims}")
        print(f"  - config.lr: {self.config.lr}")
        print(f"  - config.update_encoder: {self.config.update_encoder}")
        print(f"  - action_dim: {action_dim}")
        print(f"  - 共享 encoder 类型: {type(self.encoder).__name__}")
        print(f"  - Predictor 参数量: {sum(p.numel() for p in self.predictor.parameters())}")
        if self.config.update_encoder:
            enc_params = sum(p.numel() for p in self.encoder.parameters())
            print(f"  - Encoder 参数量 (JEPA 也会更新): {enc_params}")
        print(f"  - Optimizer 管理的参数组数: {len(self.optimizer.param_groups)}")
        print(f"  - Optimizer 管理的参数总数: {sum(p.numel() for p in params)}")
        print(f"  - Encoder 网络结构:\n{self.encoder}")

    def compute_loss(self, obs, next_obs, actions) -> torch.Tensor:
        """计算 JEPA MSE 损失.

        前向 (sg(·) = .detach() 表示 stop-gradient):
            z_t     = g_φ(o_t)                  # 当前状态潜表示 (不 detach, 梯度可回传到共享 encoder)
            z_{t+1} = sg(g_φ(o_{t+1}))          # 下一状态潜表示 (detach, 不回传梯度到 encoder)
            ẑ_{t+1} = f_θ(z_t, a_t)             # 动作条件预测

        损失 (潜空间均方误差, batch 大小 B):
            L_JEPA = (1/B) · Σ_b || ẑ_{t+1}^b - z_{t+1}^b ||₂²

        注意: 对 target z_{t+1} 使用 stop-gradient 是 JEPA 的关键, 可避免表征坍塌
              (predictor 不能通过把 target 拉向自己来作弊, encoder 仅由预测端梯度驱动).
        """
        # 打印输入观测维度 (仅在第一次调用时打印, 避免日志过多)
        if not getattr(self, "_printed_loss_dims", False):
            print(f"[JEPA.compute_loss] 输入维度信息 (仅打印一次):")
            if isinstance(obs, dict):
                for k, v in obs.items():
                    print(f"  - obs['{k}'].shape: {v.shape}")
            else:
                print(f"  - obs.shape: {obs.shape}")
            if isinstance(next_obs, dict):
                for k, v in next_obs.items():
                    print(f"  - next_obs['{k}'].shape: {v.shape}")
            else:
                print(f"  - next_obs.shape: {next_obs.shape}")
            print(f"  - actions.shape: {actions.shape}")
            self._printed_loss_dims = True

        # z_t: 当前状态潜表示, 梯度可回传到共享 encoder (φ)
        z_t = self.encoder(obs)
        # z_{t+1}: 下一状态潜表示, detach 实现 stop-gradient sg(·), 不回传梯度到 encoder
        z_next = self.encoder(next_obs).detach()
        # ẑ_{t+1} = f_θ(z_t, a_t): 动作条件潜状态预测
        z_pred = self.predictor(z_t, actions)
        # L_JEPA = (1/B) · Σ_b || ẑ_{t+1}^b - z_{t+1}^b ||₂²
        loss = F.mse_loss(z_pred, z_next)

        # 打印潜表示维度 (仅第一次)
        if not getattr(self, "_printed_latent_dims", False):
            print(f"[JEPA.compute_loss] 潜表示维度信息 (仅打印一次):")
            print(f"  - z_t.shape: {z_t.shape} (当前状态潜表示, 梯度可回传)")
            print(f"  - z_next.shape: {z_next.shape} (目标潜表示, detached)")
            print(f"  - z_pred.shape: {z_pred.shape} (预测潜表示)")
            print(f"  - loss: {loss.item():.6f}")
            self._printed_latent_dims = True
        return loss

    @torch.no_grad()
    def get_last_loss(self) -> float:
        """
        获取最后一次训练的损失值, 不会触发梯度计算.
        返回:
            float: 最后一次训练时的损失值
        """
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
        # 梯度更新: θ ← θ - η · ∇_θ L_JEPA  (update_encoder=True 时 φ 一并被更新)
        self.optimizer.zero_grad(set_to_none=True)       # 清空梯度缓存
        loss = self.compute_loss(obs, next_obs, actions) # 计算 L_JEPA
        loss.backward()                                  # 反向传播求 ∇ L_JEPA
        self.optimizer.step()                            # 按 Adam 步进更新参数
        self.last_loss = loss.item()                     # 更新最近一次 loss
        return self.last_loss
