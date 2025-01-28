import torch
import numpy as np
import argparse
import gym
import d4rl


def get_config(algorithm="A2PR_Diffusion"):
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default=algorithm)  # Policy name
    parser.add_argument(
        "--env_id", default="walker2d-medium-replay-v2"
    )  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=1024, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--mask", default=0.5, type=float
    )
    # 将vae_weight改为diffusion_weight
    parser.add_argument(
        "--diffusion_weight", default=1.0, type=float
    )
    # 添加扩散模型的特定参数 - 这些将在DiffusionModel类中使用，而不是在A2PR类中
    parser.add_argument(
        "--num_timesteps", default=100, type=int
    )  # Number of diffusion steps
    parser.add_argument(
        "--beta_start", default=1e-4, type=float
    )  # Initial beta value for diffusion
    parser.add_argument(
        "--beta_end", default=0.02, type=float
    )  # Final beta value for diffusion
    parser.add_argument(
        "--eval_freq", default=5000, type=int
    )  # How often (time steps) we evaluate
    parser.add_argument(
        "--save_model_freq", default=500000, type=int
    )  # How often (time steps) we save model
    parser.add_argument(
        "--max_timesteps", default=1e6, type=int
    )  # Max time steps to run environment
    parser.add_argument(
        "--save_model", default=True, action="store_false"
    )  # Save model and optimizer parameters
    parser.add_argument(
        "--load_model", default="default"
    )  # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--device", default="cuda", type=str)  # Use gpu or cpu
    parser.add_argument("--info", default="default")  # Additional information
    # TD3
    parser.add_argument("--actor_lr", default=3e-4, type=float)  # Actor learning rate
    parser.add_argument("--critic_lr", default=3e-4, type=float)  # Critic learning rate
    parser.add_argument(
        "--expl_noise", default=0.1
    )  # Std of Gaussian exploration noise
    parser.add_argument(
        "--batch_size", default=256, type=int
    )  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005)  # Target network update rate
    parser.add_argument(
        "--policy_noise", default=0.2
    )  # Noise added to target policy during critic update
    parser.add_argument(
        "--noise_clip", default=0.5
    )  # Range to clip target policy noise
    parser.add_argument(
        "--policy_freq", default=2, type=int
    )  # Frequency of delayed policy updates
    parser.add_argument(
        "--normalize", default=True, action="store_false"
    )  # Whether to normalize the states
    parser.add_argument(
        "--scale", default=1, type=int
    ) 
    parser.add_argument(
        "--shift", default=0, type=int
    )
    parser.add_argument("--load_emb_model", default="")
    # eigen value
    parser.add_argument("--save_eigen", action="store_true")
    parser.add_argument("--opt_predictor", default=True)
    parser.add_argument("--load_encoder_model", default=True)
    parser.add_argument("--encoder_model_path", default='', type=str)

    parser.add_argument("--alpha", default=2.5, type=float)
    parser.add_argument("--gamma", default=0.5, type=float)

    parser.add_argument("--eta", default=0.5, type=float)
    args = parser.parse_args()

    env = gym.make(args.env_id)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # 只包含A2PR类初始化需要的参数
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "device": args.device,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "alpha": args.alpha,
        "mask": args.mask,
        "diffusion_weight": args.diffusion_weight,  # 将vae_weight改为diffusion_weight
    }

    return args, env, kwargs


def save_config(args, arg_path):
    argsDict = args.__dict__
    with open(arg_path, "w") as f:
        for key, value in argsDict.items():
            f.write(key + " : " + str(value) + "\n")
