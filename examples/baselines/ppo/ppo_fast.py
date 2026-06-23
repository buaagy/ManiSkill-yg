# 文档和实验结果可在 https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy 查看
# 这是一个优化版本的PPO实现, 使用torch.compile和CUDA Graphs来加速训练

# ManiSkill特定的导入
import os
from mani_skill.utils import gym_utils  # Gym工具函数
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper  # 扁平化动作空间包装器
from mani_skill.utils.wrappers.record import RecordEpisode  # 记录episode包装器
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # ManiSkill向量化环境

# 设置torch.dynamo的内联选项
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

# 导入必要的Python库
import math  # 数学函数
import os  # 操作系统接口
import random  # 随机数生成
import time  # 时间相关功能
from collections import defaultdict  # 用于创建默认字典
from dataclasses import dataclass  # 数据类装饰器
from typing import Optional, Tuple  # 类型提示

# 导入深度学习框架
import gymnasium as gym  # 强化学习环境库
import numpy as np  # 数值计算库
import tensordict  # TensorDict库, 用于高效数据管理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # PyTorch优化器模块
import tqdm  # 进度条库
import tyro  # 命令行参数解析库
from torch.utils.tensorboard import SummaryWriter  # TensorBoard记录器
import wandb  # Weights and Biases
from tensordict import from_module  # 从模块创建TensorDict
from tensordict.nn import CudaGraphModule  # CUDA图模块, 用于性能优化
from torch.distributions.normal import Normal  # 正态分布


@dataclass
class Args:
    # 实验配置参数
    exp_name: Optional[str] = None  # 实验名称
    """the name of this experiment"""
    seed: int = 1  # 实验随机种子
    """seed of the experiment"""
    torch_deterministic: bool = True  # 是否启用确定性模式
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True  # 是否使用CUDA
    """if toggled, cuda will be enabled by default"""
    track: bool = False  # 是否使用Weights and Biases跟踪实验
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"  # wandb项目名称
    """the wandb's project name"""
    wandb_entity: Optional[str] = None  # wandb实体(团队)
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"  # wandb运行组
    """the group of the run for wandb"""
    capture_video: bool = True  # 是否录制视频
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False  # 是否保存轨迹数据
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True  # 是否保存模型
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False  # 是否仅进行评估
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None  # 预训练检查点路径
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # 环境特定参数
    env_id: str = "PickCube-v1"  # 环境ID
    """the id of the environment"""
    env_vectorization: str = "gpu"  # 环境向量化类型
    """the type of environment vectorization to use"""
    num_envs: int = 512  # 并行环境数量
    """the number of parallel environments"""
    num_eval_envs: int = 16  # 并行评估环境数量
    """the number of parallel evaluation environments"""
    partial_reset: bool = True  # 是否在终止时重置环境
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False  # 评估环境是否在终止时重置
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50  # 每次rollout的步数
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50  # 评估时的步数
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None  # 重新配置环境的频率
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1  # 评估环境重新配置频率
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25  # 评估频率(迭代次数)
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None  # 训练视频保存频率
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"  # 控制模式
    """the control mode to use for the environment"""

    # 算法特定参数
    total_timesteps: int = 10000000  # 总训练步数
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4  # 学习率
    """the learning rate of the optimizer"""
    anneal_lr: bool = False  # 是否使用学习率退火
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8  # 折扣因子
    """the discount factor gamma"""
    gae_lambda: float = 0.9  # 广义优势估计lambda参数
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32  # 小批次数量
    """the number of mini-batches"""
    update_epochs: int = 4  # 策略更新轮数
    """the K epochs to update the policy"""
    norm_adv: bool = True  # 是否归一化优势
    """Toggles advantages normalization"""
    clip_coef: float = 0.2  # 裁剪系数
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False  # 是否使用裁剪值函数损失
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0  # 熵系数
    """coefficient of the entropy"""
    vf_coef: float = 0.5  # 值函数系数
    """coefficient of the value function"""
    max_grad_norm: float = 0.5  # 最大梯度范数
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1  # 目标KL散度阈值
    """the target KL divergence threshold"""
    reward_scale: float = 1.0  # 奖励缩放因子
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = False  # 是否使用有限视界GAE

    # 运行时计算的参数
    batch_size: int = 0  # 批次大小(运行时计算)
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0  # 小批次大小(运行时计算)
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0  # 迭代次数(运行时计算)
    """the number of iterations (computed in runtime)"""

    # PyTorch优化参数
    compile: bool = False  # 是否使用torch.compile
    """whether to use torch.compile."""
    cudagraphs: bool = False  # 是否使用CUDA图
    """whether to use cudagraphs on top of compile."""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """初始化神经网络层的权重和偏置
    
    Args:
        layer: 要初始化的神经网络层
        std: 权重的标准差
        bias_const: 偏置的常数值
    
    Returns:
        初始化后的层
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """PPO智能体类, 包含演员网络和评论家网络"""
    
    def __init__(self, n_obs, n_act, device=None):
        """初始化智能体
        
        Args:
            n_obs: 观测空间的维度
            n_act: 动作空间的维度
            device: 存储设备
        """
        super().__init__()
        # 评论家网络, 用于估计状态价值
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        # 演员网络均值部分, 用于生成动作均值
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        # 演员网络标准差对数, 作为可学习参数
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        """获取状态价值
        
        Args:
            x: 状态观测
        
        Returns:
            状态价值
        """
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        """获取动作、对数概率、熵和状态价值
        
        Args:
            obs: 状态观测
            action: 可选的动作, 如果为None则采样
        
        Returns:
            action: 动作
            logprob: 动作的对数概率
            entropy: 策略的熵
            value: 状态价值
        """
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

class Logger:
    """日志记录类, 用于记录训练过程中的各种指标"""
    
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        """初始化日志记录器
        
        Args:
            log_wandb: 是否使用wandb记录
            tensorboard: TensorBoard记录器
        """
        self.writer = tensorboard
        self.log_wandb = log_wandb
        
    def add_scalar(self, tag, scalar_value, step):
        """记录标量值
        
        Args:
            tag: 标签名称
            scalar_value: 标量值
            step: 训练步数
        """
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
        
    def close(self):
        """关闭日志记录器"""
        self.writer.close()

def gae(next_obs, next_done, container, final_values):
    """计算广义优势估计(GAE)
    
    Args:
        next_obs: 下一个观测
        next_done: 下一个完成状态
        container: 包含rollout数据的TensorDict
        final_values: 最终价值值
    
    Returns:
        更新后的container, 包含advantages和returns
    """
    # 如果未完成则bootstrap价值
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        # real_next_values = nextvalues * nextnonterminal
        real_next_values = nextnonterminal * nextvalues + final_values[t] # t instead of t+1
        delta = rewards[t] + args.gamma * real_next_values - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done):
    """在环境中执行rollout并收集数据
    
    Args:
        obs: 初始观测
        done: 初始完成状态
    
    Returns:
        next_obs: 下一个观测
        done: 完成状态
        container: 包含rollout数据的TensorDict
        final_values: 最终价值值
    """
    ts = []
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    for step in range(args.num_steps):
        # 算法逻辑: 获取动作
        action, logprob, _, value = policy(obs=obs)

        # 在环境中执行动作并记录数据
        next_obs, reward, next_done, infos = step_func(action)

        # 处理episode结束时的信息
        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
            with torch.no_grad():
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)

        # 创建TensorDict存储当前步的数据
        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo示例将done与前一个obs关联(不是由action产生的done)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )
        # 注意: 这里需要为GPU环境进行修改
        obs = next_obs = next_obs
        done = next_done
    # 注意: 需要执行.to(device), 否则container.device为None, 不确定这是否会影响任何东西
    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container, final_values


def update(obs, actions, logprobs, advantages, returns, vals):
    """更新策略和价值网络
    
    Args:
        obs: 观测
        actions: 动作
        logprobs: 对数概率
        advantages: 优势
        returns: 回报
        vals: 价值
    
    Returns:
        approx_kl: 近似KL散度
        v_loss: 价值损失
        pg_loss: 策略损失
        entropy_loss: 熵损失
        old_approx_kl: 旧近似KL散度
        clipfrac: 裁剪比例
        gn: 梯度范数
    """
    optimizer.zero_grad()
    # 计算新的动作概率、熵和价值
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # 计算近似KL散度 http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    # 归一化优势
    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 策略损失(使用PPO裁剪)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # 价值损失
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        # 使用裁剪的价值损失
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        # 标准价值损失
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    # 熵损失(鼓励探索)
    entropy_loss = entropy.mean()
    # 总损失
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    # 反向传播和优化
    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


# 将update函数包装为TensorDictModule, 允许使用CudaGraphModule
update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    # 解析命令行参数
    args = tyro.cli(Args)
    # if not args.evaluate: exit()

    # 计算运行时参数
    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    # 设置实验名称
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # 设置随机种子以确保可重复性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 设置设备(CPU或CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # 环境设置
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    # 创建训练环境和评估环境
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    # 如果动作空间是字典类型, 则扁平化
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    # 视频录制和轨迹保存设置
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    # 向量化环境
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # 获取最大episode步数
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    # 训练模式设置
    if not args.evaluate:
        print("Running training")
        # 初始化wandb(如果启用)
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient", f"GPU:{torch.cuda.get_device_name()}"]
            )
        # 初始化TensorBoard记录器
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")
        
    # 计算观测和动作空间维度
    n_act = math.prod(envs.single_action_space.shape)
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # 将step注册为特殊操作以避免图中断
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 注意: 这里需要为GPU环境进行修改
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return next_obs, reward, next_done, info

    # 智能体设置
    agent = Agent(n_obs, n_act, device=device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    # 创建一个带有分离参数的智能体版本用于推理
    agent_inference = Agent(n_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    # 优化器设置
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    # 可执行函数设置
    # 定义网络: 将策略包装在TensorDictModule中允许我们使用CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # 编译策略
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    # 使用CUDA图优化
    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    # 开始训练
    global_step = 0
    start_time = time.time()
    container_local = None
    next_obs = envs.reset()[0]
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    # 累计时间统计
    cumulative_times = defaultdict(float)

    for iteration in pbar:
        agent.eval()  # 设置为评估模式
        
        # 评估循环
        if iteration % args.eval_freq == 1:
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.actor_mean(eval_obs))
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
        
        # 保存模型检查点
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        
        # 学习率退火(如果启用)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # 标记CUDA图步骤开始
        torch.compiler.cudagraph_mark_step_begin()
        # Rollout阶段: 在环境中执行策略并收集数据
        rollout_time = time.perf_counter()
        next_obs, next_done, container, final_values = rollout(next_obs, next_done)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        global_step += container.numel()

        # 更新阶段
        update_time = time.perf_counter()
        # 计算优势(Advantage)和回报(Return)
        container = gae(next_obs, next_done, container, final_values)
        container_flat = container.view(-1)

        # 优化策略和价值网络
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=tensordict.TensorDict())
                clipfracs.append(out["clipfrac"])
                # 如果KL散度超过阈值, 提前停止更新
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # 记录训练指标
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", out["v_loss"].item(), global_step)
        logger.add_scalar("losses/policy_loss", out["pg_loss"].item(), global_step)
        logger.add_scalar("losses/entropy", out["entropy_loss"].item(), global_step)
        logger.add_scalar("losses/old_approx_kl", out["old_approx_kl"].item(), global_step)
        logger.add_scalar("losses/approx_kl", out["approx_kl"].item(), global_step)
        logger.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean().cpu().item(), global_step)
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        # 记录累计时间
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    
    # 保存最终模型和关闭环境
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()