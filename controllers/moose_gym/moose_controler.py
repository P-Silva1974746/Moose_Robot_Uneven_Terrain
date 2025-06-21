import struct

from controller import Supervisor, Motor, GPS, InertialUnit
import socket
import pickle
import math
import random  # para gerar objetivo aleatório
import numpy as np
import pandas as pd
import os
#futuramente quando adicionar função de mudar mapas, se durante 10 tentativas nos primeiros 10 segindos ficar com problemas de não pousar corretamnet mudar mundo


def send_msg(conn, data):
    """This function sends the message in chunks so this is so that heavier object can be passed"""
    pickled = pickle.dumps(data)
    length = struct.pack("!I", len(pickled))
    conn.sendall(length + pickled)

def sanitize_observation(obs):
    return  np.nan_to_num(np.array(obs), nan=0.0, posinf=1e5, neginf=-1e5).tolist()

def height_at(x, y, h_v):
    x = int(x)
    y = int(y)
    x = max(0, min(x, 200 - 1))  # Garantir que x esteja dentro do grid
    y = max(0, min(y, 200 - 1))  # Garantir que y esteja dentro do grid
    index = y * 200 + x
    return h_v[index]


#creating one supervisor instead of robot so that we can get more information out of it
supervisor = Supervisor()
robot = supervisor.getFromDef('MOOSEROBOT')

timestep = int(supervisor.getBasicTimeStep())

left_motors = [supervisor.getDevice(f"left_motor_{i}") for i in range(1, 5)]
right_motors = [supervisor.getDevice(f"right_motor_{i}") for i in range(1, 5)]

# get the sensors activated
gps = supervisor.getDevice("gps")
gps.enable(timestep)
imu = supervisor.getDevice("inertial unit")
imu.enable(timestep)
lidar = supervisor.getDevice("lidar")
lidar.enable(timestep)
lidar.enablePointCloud()

for m in left_motors + right_motors:
    m.setPosition(float('inf'))
    m.setVelocity(0)

elevation_grid = supervisor.getFromDef('EG')
height_field = elevation_grid.getField('height')
height_values = [height_field.getMFFloat(i) for i in range(height_field.getCount())]

# TCP/IP socket setup
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 10000))
server.listen(1)
conn, _ = server.accept()


d_g = 5
mu = 0.05
stuck_counter = 0
max_stuck_steps = 10000 # since each timeset are 4 mms this is equivalent to 40 sec

# variáveis de estado
min_dist = float("inf")
start_pos = robot.getField("translation").getSFVec3f()
start_rotation= robot.getField("rotation").getSFRotation()
start_time = supervisor.getTime()
last_time = start_time
last_pos = start_pos
total_steps = 0

# goal setup at a maximum of 50
max_dist_travel = 35
goal = [
        random.uniform(max(20,start_pos[0]-max_dist_travel), min(180, start_pos[0]+max_dist_travel)),  # x aleatório
        random.uniform(max(20,start_pos[1]-max_dist_travel), min(180, start_pos[1]+max_dist_travel)),  # x aleatório
        0.0                             # Altura Z definida separadamente
]
goal[2] = height_at(x=goal[0], y=goal[1], h_v=height_values) + 0.2 # +0.2 so that it isn't on the ground
# to limit clutter of prints in of progress
count=0

# log information
logging = True
total_distance = 0
avg_speed=0
completed=False
heights=[]
log_path="controllers/moose_gym/logs/performance_v5_map2.csv"

while supervisor.step(timestep) != -1:
    try:
        count+=1

        data = pickle.loads(conn.recv(4096))
        if data["cmd"] == "step":
            total_steps += 1

            current_time = supervisor.getTime()
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
            point_cloud=lidar.getPointCloud()
            # point_cloud = lidar.getPointCloud()[::5]  # downsample if training is not fast/good performance
            flat_point_cloud = [coord for pt in point_cloud for coord in (pt.x, pt.y, pt.z)]
            obs = list(goal) + list(pos) + list(rpy) + flat_point_cloud


            # Distância atual ao objetivo
            d_t = math.sqrt(sum((pos[i] - goal[i]) ** 2 for i in range(3)))

            # -------- Reward Function -------- #
            if d_t <= d_g:
                total_time = current_time - start_time
                d_start = math.sqrt(sum((start_pos[i] - goal[i]) ** 2 for i in range(3)))
                r_v = d_start / (total_time + 1e-8)
                reward = 1000 + r_v
                print("Goal reached!!!")
                done = True
                completed=True
            elif d_t < min_dist:
                reward = mu * (min_dist - d_t) / (delta_t + 1e-8)
                done = False
            elif stuck_counter > max_stuck_steps or current_time - start_time > 600 or pos[2]<-100 :#or abs(rpy[0])>(math.pi/2):
                reward = -100
                if stuck_counter > max_stuck_steps or abs(rpy[0])>(math.pi/2):
                    print("Robot is stuck")
                elif current_time - start_time > 600:
                    print("Timeout exceeded")
                elif pos[2]<-100:
                    print("Robot fall of the side of the map")
                    reward = -1000
                done = True
                completed=False
            else:
                reward = 0.0
                done = False
            # -------- ---------------- -------- #

            # Update min_dist
            if d_t < min_dist:
                min_dist = d_t

            # at each 1000 steps 4sec we will say how many meters to goal
            if count % 1000 == 0:
                print(f"{d_t:.2f}m to the goal")
                print(f"Current pos: {pos}")
                count = 0

            # Detecção de "stuck"
            delta = math.sqrt(sum((pos[i] - last_pos[i]) ** 2 for i in range(3)))

            # this means that the robot must be moving more than 0.9 km/h in order to not be considered stuck
            if delta < 0.001:
                stuck_counter += 1
                #print(f"STUCK: {stuck_counter}")
            else:
                stuck_counter = 0
            last_pos = pos

            if logging:
                total_distance += delta
                heights.append(height_at(x=pos[0], y=goal[1], h_v=height_values))
            obs = sanitize_observation(obs)
            reward =float(np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))
            #print(reward)

            if logging and done:
                euclidean_dist= math.hypot(goal[0] - start_pos[0], goal[1] - start_pos[1], goal[2] - start_pos[2])
                total_time = current_time - start_time
                avg_speed = total_distance / total_time
                height_difference= np.max(heights) - np.min(heights)
                height_std = np.std(heights)
                print(f"Distância total percorrida: {total_distance} metros")
                print(f"Tempo total: {total_time} segundos")
                print(f"Velocidade media: {avg_speed} m/s")
                print(f"Diferenca de altitude: {height_difference} m")
                print(f"Std da altitude: {height_std} m")
                print(f"Distancia minima ao goal: {min_dist} m")
                print(f"Completou a run: {completed}")

                data = {
                    "Distance": [euclidean_dist],
                    "Distance traveled": [total_distance],
                    "Time": [total_time],
                    "Avg Speed": [avg_speed],
                    "Height Difference": [height_difference],
                    "Height Std": [height_std],
                    "Minimum distance to goal": [min_dist],
                    "Completion": [completed],
                }

                df = pd.DataFrame(data=data)
                try:
                    df.to_csv(path_or_buf=log_path, mode='a', header=not os.path.exists(log_path), index=False)
                except Exception as e:
                    print(f"Failed to write the log: {e}")

            send_msg(conn, {"obs": obs, "reward": reward, "done": done})

        elif data["cmd"] == "reset":
            # Reset do estado
            print("Resetting robot")

            # goal setup at a maximum of 50
            max_dist_travel = 35
            goal = [
                random.uniform(max(20, start_pos[0] - max_dist_travel), min(180, start_pos[0] + max_dist_travel)),
                # x aleatório
                random.uniform(max(20, start_pos[1] - max_dist_travel), min(180, start_pos[1] + max_dist_travel)),
                # x aleatório
                0.0  # Altura Z definida separadamente
            ]
            goal[2] = height_at(x=goal[0], y=goal[1], h_v=height_values) + 0.2 # +0.2 so that it isn't on the ground

            print(f"Goal: {goal}")
            robot.getField("translation").setSFVec3f(start_pos)
            robot.getField("rotation").setSFRotation(start_rotation)

            # waiting a few timesteps so that the robot settles and sensors update
            for _ in range(5):
                supervisor.step(timestep)

            # creating the new observation space
            pos = gps.getValues()
            rpy = imu.getRollPitchYaw()
            point_cloud = lidar.getPointCloud()
            #point_cloud = lidar.getPointCloud()[::5]  # downsample if training is not fast/good performance
            flat_point_cloud = [coord for pt in point_cloud for coord in (pt.x, pt.y, pt.z)]
            obs = list(goal) + list(pos) + list(rpy) + flat_point_cloud

            min_dist = float("inf")
            stuck_counter = 0
            start_time = supervisor.getTime()
            last_time = start_time
            last_pos = start_pos
            total_steps = 0

            # log information
            logging = True
            total_distance = 0
            avg_speed = 0
            completed = False
            heights = []
            log_path = "controllers/moose_gym/logs/performance_v5_map2.csv"

            obs = sanitize_observation(obs)
            send_msg(conn, obs)

    except Exception as e:
        print(e)
        break
