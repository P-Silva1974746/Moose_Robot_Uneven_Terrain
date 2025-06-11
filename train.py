import os
from stable_baselines3 import A2C
from moose_gym import MooseEnv

train=False

env = MooseEnv()

save_dir = "models/a2c_moose"
model_path = f"{save_dir}/a2c_moose_model_v3"

os.makedirs(save_dir, exist_ok=True)

if os.path.exists(model_path + ".zip"):
    print("Modelo existente encontrado.")
    model = A2C.load(model_path, env=env, reset_timesteps=False)
else:
    print("Nenhum modelo encontrado. A iniciar treino de novo...")
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0001, tensorboard_log="./a2c_moose/")

if train:
    model.learn(total_timesteps=500000, tb_log_name="A2C_v3", reset_num_timesteps=False)
    model.save(model_path)
    print("Modelo guardado com sucesso.")
else:
    count=5
    start = count
    obs, info =env.reset()
    mile_stone = 0.1
    while count > 0:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        progress=round(1-(count / start), 2)

        if progress > mile_stone:
            print(f"Progress {progress}%")
            mile_stone+=0.05
        if terminated or truncated:
            count -= 1
            obs, info=env.reset()