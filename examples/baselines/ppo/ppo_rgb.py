# 文档和实验结果可在 https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy 查看

# 导入必要的Python库
from collections import defaultdict  # 用于创建默认字典
import os  # 操作系统接口
import random  # 随机数生成
import time  # 时间相关功能
from dataclasses import dataclass  # 数据类装饰器
from typing import Optional  # 类型提示

# 导入深度学习框架
import gymnasium as gym  # 强化学习环境库
import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as optim  # PyTorch优化器模块
import tyro  # 命令行参数解析库
from torch.distributions.normal import Normal  # 正态分布
from torch.utils.tensorboard import SummaryWriter  # TensorBoard记录器

# ManiSkill特定的导入
import mani_skill.envs  # ManiSkill环境模块
from mani_skill.utils import gym_utils  # Gym工具函数
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper  # 扁平化动作空间和RGBD观测包装器
from mani_skill.utils.wrappers.record import RecordEpisode  # 记录episode包装器
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv  # ManiSkill向量化环境

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
    save_model: bool = True  # 是否保存模型
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False  # 是否仅进行评估
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None  # 预训练检查点路径
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "all"  # 环境渲染模式
    """the environment rendering mode"""

    # 算法特定参数
    env_id: str = "PickCube-v1"  # 环境ID
    """the id of the environment"""
    include_state: bool = True  # 是否在观测中包含状态信息
    """whether to include state information in observations"""
    total_timesteps: int = 10000000  # 总训练步数
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4  # 学习率
    """the learning rate of the optimizer"""
    num_envs: int = 512  # 并行环境数量
    """the number of parallel environments"""
    num_eval_envs: int = 8  # 并行评估环境数量
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
    control_mode: Optional[str] = "pd_joint_delta_pos"  # 控制模式
    """the control mode to use for the environment"""
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
    target_kl: float = 0.2  # 目标KL散度阈值
    """the target KL divergence threshold"""
    reward_scale: float = 1.0  # 奖励缩放因子
    """Scale the reward by this factor"""
    eval_freq: int = 25  # 评估频率(迭代次数)
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None  # 训练视频保存频率
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False  # 是否使用有限视界GAE


    # 运行时计算的参数
    batch_size: int = 0  # 批次大小(运行时计算)
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0  # 小批次大小(运行时计算)
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0  # 迭代次数(运行时计算)
    """the number of iterations (computed in runtime)"""

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


class DictArray(object):
    """字典数组类, 用于存储和操作字典形式的观测数据"""
    
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        """初始化字典数组
        
        Args:
            buffer_shape: 缓冲区的形状
            element_space: 元素空间(通常是gym.spaces.dict.Dict)
            data_dict: 可选的初始数据字典
            device: 存储设备
        """
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
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        """返回字典的键"""
        return self.data.keys()

    def __getitem__(self, index):
        """获取索引处的值
        
        Args:
            index: 可以是字符串键或数组索引
        
        Returns:
            如果index是字符串, 返回对应键的值
            如果index是数组索引, 返回该索引处所有键的值组成的字典
        """
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        """设置索引处的值
        
        Args:
            index: 可以是字符串键或数组索引
            value: 要设置的值或值字典
        """
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        """返回缓冲区的形状"""
        return self.buffer_shape

    def reshape(self, shape):
        """重塑字典数组的形状
        
        Args:
            shape: 新的形状
        
        Returns:
            重塑后的新DictArray对象
        """
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)


class NatureCNN(nn.Module):
    """NatureCNN类, 用于处理RGB图像观测的卷积神经网络"""
    
    def __init__(self, sample_obs):
        """初始化NatureCNN
        
        Args:
            sample_obs: 样本观测数据, 用于确定网络结构
        """
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # 这里使用NatureCNN架构来处理图像, 但这里可以使用任何架构
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 为了容易计算展平后的维度, 我们通过一个测试张量
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # 对于状态数据, 我们简单地通过一个线性层
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        """前向传播
        
        Args:
            observations: 观测数据字典
        
        Returns:
            编码后的特征张量
        """
        encoded_tensor_list = []
        # self.extractors包含进行所有处理的nn.Module
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)  # 转换为NCHW格式
                obs = obs / 255  # 归一化到[0, 1]
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class Agent(nn.Module):
    """PPO智能体类, 包含特征提取网络、演员网络和评论家网络"""
    
    def __init__(self, envs, sample_obs):
        """初始化智能体
        
        Args:
            envs: 环境对象
            sample_obs: 样本观测数据, 用于初始化特征提取网络
        """
        super().__init__()
        # 特征提取网络
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        # latent_size = np.array(envs.unwrapped.single_observation_space.shape).prod()
        latent_size = self.feature_net.out_features
        
        # 评论家网络, 用于估计状态价值
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        
        # 演员网络均值部分, 用于生成动作均值
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.unwrapped.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        # 演员网络标准差对数, 作为可学习参数
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.unwrapped.single_action_space.shape)) * -0.5)
        
    def get_features(self, x):
        """获取特征表示
        
        Args:
            x: 输入观测
        
        Returns:
            特征向量
        """
        return self.feature_net(x)
        
    def get_value(self, x):
        """获取状态价值
        
        Args:
            x: 状态观测
        
        Returns:
            状态价值
        """
        x = self.feature_net(x)
        return self.critic(x)
        
    def get_action(self, x, deterministic=False):
        """获取动作
        
        Args:
            x: 状态观测
            deterministic: 是否使用确定性策略
        
        Returns:
            动作
        """
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
        
    def get_action_and_value(self, x, action=None):
        """获取动作、对数概率、熵和状态价值
        
        Args:
            x: 状态观测
            action: 可选的动作, 如果为None则采样
        
        Returns:
            action: 动作
            logprob: 动作的对数概率
            entropy: 策略的熵
            value: 状态价值
        """
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

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

if __name__ == "__main__":
    # 解析命令行参数
    args = tyro.cli(Args)
    # 计算运行时参数
    args.batch_size = int(args.num_envs * args.num_steps)  # 批次大小 = 环境数 * 每个环境的步数
    args.minibatch_size = int(args.batch_size // args.num_minibatches)  # 小批次大小
    args.num_iterations = args.total_timesteps // args.batch_size  # 总迭代次数
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
    env_kwargs = dict(obs_mode="rgb", render_mode=args.render_mode, sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    # 创建评估环境和训练环境
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # rgbd观测模式返回一个数据字典, 我们将其扁平化以便只有rgb键和state键
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)

    # 如果动作空间是字典类型, 则扁平化
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    # 视频录制设置
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
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
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
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

    # 存储设置, 用于存储rollout数据
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # 开始训练
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    # 初始化智能体和优化器
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # 加载预训练检查点(如果指定)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # 累计时间统计
    cumulative_times = defaultdict(float)

    # 主训练循环
    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()  # 设置为评估模式
        
        # 评估循环
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
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
            optimizer.param_groups[0]["lr"] = lrnow
            
        # Rollout阶段: 在环境中执行策略并收集数据
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # 使用当前策略获取动作
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # 在环境中执行动作并收集数据
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            # 处理episode结束时的信息
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                # 处理最终观测(用于bootstrap)
                for k in infos["final_observation"]:
                    infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"]).view(-1)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        
        # 计算优势(Advantage)和回报(Return)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                # 有限视界GAE计算
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # 初始化
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # 第二项的和
                        value_term_sum = 0. # 第三项的和
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    # 标准GAE计算
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # 这里实际上应该使用next_not_terminated, 但如果终止了我们没有lastgamlam
            returns = advantages + values

        # 扁平化批次数据
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # 优化策略和价值网络
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # 计算新的动作概率、熵和价值
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # 计算近似KL散度 http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                # 如果KL散度超过阈值, 提前停止更新
                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                # 归一化优势
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # 策略损失(使用PPO裁剪)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # 价值损失
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    # 使用裁剪的价值损失
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # 标准价值损失
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # 熵损失(鼓励探索)
                entropy_loss = entropy.mean()
                # 总损失
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # 如果KL散度超过阈值, 提前停止epoch
            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        
        # 计算解释方差
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # 记录训练指标
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
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
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()