import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import pickle
import  struct



class MooseEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # [velocidade_esq, velocidade_dir]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6153,), dtype=np.float32)  # exemplo: 7 sensores

        # Comunicação com Webots (TCP/IP)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("localhost", 10000))  # Porta igual ao lado Webots

    def reset(self, seed=None, options=None):
        self.sock.send(pickle.dumps({"cmd": "reset"}))
        obs = self.recv_msg(self.sock)
        return obs, {}

    def step(self, action):
        self.sock.send(pickle.dumps({"cmd": "step", "action": action.tolist()}))
        data = self.recv_msg(self.sock)
        return data["obs"], data["reward"], data["done"], False, {}

    def close(self):
        self.sock.close()

    def recvall(self, sock, n):
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def recv_msg(self, sock):
        raw_len = self.recvall(sock, 4)
        if not raw_len:
            return None
        msg_len = struct.unpack("!I", raw_len)[0]

        return pickle.loads(self.recvall(sock, msg_len))
