import os
from stable_baselines3 import A2C
from moose_gym import MooseEnv

env = MooseEnv()

save_dir = "models/a2c_moose"
model_path = f"{save_dir}/a2c_moose_model"

os.makedirs(save_dir, exist_ok=True)

if os.path.exists(model_path + ".zip"):
    print("Modelo existente encontrado. A carregar e continuar treino...")
    model = A2C.load(model_path, env=env, reset_timesteps=False)
else:
    print("Nenhum modelo encontrado. A iniciar treino de novo...")
    model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0001, tensorboard_log="./a2c_moose/")

model.learn(total_timesteps=100000, tb_log_name="A2C")

model.save(model_path)
print("Modelo guardado com sucesso.")
