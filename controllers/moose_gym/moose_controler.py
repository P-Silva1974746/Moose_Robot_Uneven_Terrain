from controller import Robot, Motor, GPS, InertialUnit
import socket
import pickle
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motors = [robot.getDevice(f"left_motor_{i}") for i in range(1, 5)]
right_motors = [robot.getDevice(f"right_motor_{i}") for i in range(1, 5)]

gps = robot.getDevice("gps")
gps.enable(timestep)
imu = robot.getDevice("inertial unit")
imu.enable(timestep)

for m in left_motors + right_motors:
    m.setPosition(float('inf'))
    m.setVelocity(0)

# TCP/IP socket setup
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 10000))
server.listen(1)
conn, _ = server.accept()

# parâmetros
goal = [5.0, 0.0, 5.0]
d_g = 0.5
mu = 1.0
stuck_counter = 0
max_stuck_steps = 20

# variáveis de estado
min_dist = float("inf")
start_pos = gps.getValues()
start_time = robot.getTime()
last_time = start_time
last_pos = start_pos
total_steps = 0

while robot.step(timestep) != -1:
    try:
        data = pickle.loads(conn.recv(4096))
        if data["cmd"] == "step":
            total_steps += 1
            current_time = robot.getTime()
            delta_t = current_time - last_time
            last_time = current_time

            action = data["action"]
            for m in left_motors:
                m.setVelocity(action[0] * 10)
            for m in right_motors:
                m.setVelocity(action[1] * 10)

            # Estado atual
            pos = gps.getValues()
            rpy = imu.getRollPitchYaw()
            obs = [pos[0], pos[1], pos[2], rpy[0], rpy[1], rpy[2]]

            # Distância atual ao objetivo
            d_t = math.sqrt(sum((pos[i] - goal[i]) ** 2 for i in range(3)))
            if d_t < min_dist:
                min_dist = d_t

            done = False

            # -------- Reward Function -------- #
            if d_t <= d_g:
                total_time = current_time - start_time
                d_start = math.sqrt(sum((start_pos[i] - goal[i]) ** 2 for i in range(3)))
                r_v = d_start / (total_time + 1e-8)
                reward = 0.5 + r_v
                done = True
            elif d_t < min_dist:
                reward = mu * (min_dist - d_t) / (delta_t + 1e-8)
            elif stuck_counter > max_stuck_steps:
                reward = -0.5
                done = True
            else:
                reward = 0.0
            # -------- ---------------- -------- #

            # Detecção de "stuck"
            delta = math.sqrt(sum((pos[i] - last_pos[i]) ** 2 for i in range(3)))
            if delta < 0.01:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_pos = pos

            conn.send(pickle.dumps({"obs": obs, "reward": reward, "done": done}))

        elif data["cmd"] == "reset":
            # Reset do estado
            start_pos = gps.getValues()
            min_dist = float("inf")
            stuck_counter = 0
            start_time = robot.getTime()
            last_time = start_time
            last_pos = start_pos
            total_steps = 0
            conn.send(pickle.dumps([0.0]*6))

    except:
        break
