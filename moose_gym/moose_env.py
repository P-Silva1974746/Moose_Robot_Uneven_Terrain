import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import pickle

class MooseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # [velocidade_esq, velocidade_dir]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # exemplo: 6 sensores

        # Comunicação com Webots (TCP/IP)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", 10000))  # Porta igual ao lado Webots

    def reset(self, seed=None, options=None):
        self.sock.send(pickle.dumps({"cmd": "reset"}))
        obs = pickle.loads(self.sock.recv(4096))
        return obs, {}

    def step(self, action):
        self.sock.send(pickle.dumps({"cmd": "step", "action": action.tolist()}))
        data = pickle.loads(self.sock.recv(4096))
        return data["obs"], data["reward"], data["done"], False, {}

    def close(self):
        self.sock.close()
