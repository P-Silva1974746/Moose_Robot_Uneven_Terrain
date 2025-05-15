from controller import Robot, Motor, GPS, InertialUnit
import socket
import pickle

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

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("localhost", 10000))
server.listen(1)
conn, _ = server.accept()

while robot.step(timestep) != -1:
    try:
        data = pickle.loads(conn.recv(4096))
        if data["cmd"] == "step":
            action = data["action"]  # [vel_esq, vel_dir]
            for m in left_motors:
                m.setVelocity(action[0] * 10)
            for m in right_motors:
                m.setVelocity(action[1] * 10)

            obs = [
                gps.getValues()[0],
                gps.getValues()[1],
                gps.getValues()[2],
                imu.getRollPitchYaw()[0],
                imu.getRollPitchYaw()[1],
                imu.getRollPitchYaw()[2],
            ]
            reward = -1.0  # exemplo: penalização constante
            done = False

            conn.send(pickle.dumps({"obs": obs, "reward": reward, "done": done}))
        elif data["cmd"] == "reset":
            gps.getValues()
            conn.send(pickle.dumps([0.0]*6))
    except:
        break
