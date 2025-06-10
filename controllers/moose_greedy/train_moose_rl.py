import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from MooseRLGreedy import MooseRLGymEnv

if __name__ == '__main__':
    # env = MooseRLGymEnv() # For single environment
    env = make_vec_env(MooseRLGymEnv, n_envs=1) # parallel environment

    # Define the RL model
    model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./tensorboard_logs/")

    # Train the model
    model.learn(total_timesteps=500000)

    # Save the trained model
    model.save("moose_rl_model")

    # Load and test:
    # model = PPO.load("moose_rl_model")
    # obs = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = env.step(action)
    #     if dones:
    #         obs = env.reset()