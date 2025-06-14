import math
import random
import csv
import os
import numpy as np
from controller import Supervisor, InertialUnit

class MooseNavigator:
    def __init__(self):
        self.robot = Supervisor()
        # super().__init__()
        # Setup
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getFromDef("MOOSEROBOT")
        if self.robot_node is None:
            print("ERROR: Could not find robot node with DEF 'MOOSEROBOT'")
        self.translation_field = self.robot_node.getField("translation")
        #self.translation_field.setSFVec3f([10.00, 10.00, 3.10])

        # Motores
        self.left_motors = [self.robot.getDevice(f"left_motor_{i}") for i in range(1, 5)]
        self.right_motors = [self.robot.getDevice(f"right_motor_{i}") for i in range(1, 5)]

        for motor in self.left_motors + self.right_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Sensores
        self.gps = self.robot.getDevice("gps")  # posição do robo
        self.gps.enable(self.timestep)
        self.imu = self.robot.getDevice("inertial unit")  # orientação do robo
        self.imu.enable(self.timestep)

        # Mapa (grid)
        self.elevation_grid = self.robot.getFromDef("EG")
        self.height_field = self.elevation_grid.getField("height")
        self.height_values = [self.height_field.getMFFloat(i) for i in range(self.height_field.getCount())]
        self.x_dim = self.elevation_grid.getField("xDimension").getSFInt32()
        self.y_dim = self.elevation_grid.getField("yDimension").getSFInt32()
        #self.z_dim = self.elevation_grid.getField("zDimension").getSFInt32()
        self.x_spacing = self.elevation_grid.getField("xSpacing").getSFFloat()
        self.y_spacing = self.elevation_grid.getField("ySpacing").getSFFloat()
        #self.z_spacing = self.elevation_grid.getField("zSpacing").getSFFloat()

        self.max_speed = 6.28
        self.total_distance = 0.0
        self.speeds_history = []

        #self.randomize_positions2()
        #self.set_robot_position()
        #self.set_goal_position()

    def set_motor_velocity(self, left_speed, right_speed):
        for m in self.left_motors:
            m.setVelocity(left_speed)
        for m in self.right_motors:
            m.setVelocity(right_speed)

    def get_terrain_height(self, x, y):
        # Convert world coords (x,y) to grid indices (ix, iy)
        ix = int(x / self.x_spacing)
        iy = int(y / self.y_spacing)

        # Clamp indices to valid range
        ix = max(0, min(self.x_dim - 1, ix))
        iy = max(0, min(self.y_dim - 1, iy))

        # Height field is a 1D list, row-major order
        idx = iy * self.x_dim + ix

        # Get height value
        return self.height_values[idx]


    def get_slope_value(self, x, y, scale=10):
        #slope magnitude and surface normal at a grid position (x, y).
        #scale: size of the local area to calculate gradient from (must be odd). normal: np.array([x, y, z])

        half_scale = scale // 2
        heights = np.zeros((scale, scale))

        for i in range(scale):
            for j in range(scale):
                nx = x - half_scale + i
                ny = y - half_scale + j
                heights[i, j] = self.get_terrain_height(nx, ny)

        dz_dx, dz_dy = np.gradient(heights, self.x_spacing, self.y_spacing)
        slope_x = dz_dx[half_scale, half_scale]
        slope_y = dz_dy[half_scale, half_scale]

        slope_magnitude = np.hypot(slope_x, slope_y)

        # Create terrain surface normal vector
        normal = np.array([-slope_x, -slope_y, 1.0])
        normal /= np.linalg.norm(normal)

        return slope_magnitude, normal


    def is_flat(self,x,y):
        _, normal=self.get_slope_value(x,y, scale=20)
        up=np.array([0,0,1], dtype=float)

        # get axis of rotation
        axis=np.cross(up,normal)

        if np.linalg.norm(axis)<1e-2: # if small is flat
            return True
        else:
            return False


    def set_robot_position(self, x, y):
        z = self.get_terrain_height(x, y)
        pos_field = self.robot_node.getField("translation")
        pos_field.setSFVec3f([x * self.x_spacing, y * self.y_spacing, z + 0.5])

        _, normal = self.get_slope_value(x, y, scale=15)
        up=np.array([0, 0, 1], dtype=float)
        axis=np.cross(up, normal)
        axis/=np.linalg.norm(axis)

        angle=np.arccos(np.clip(np.dot(up,normal), -1.0, 1.0))

        rot_field = self.robot_node.getField("rotation")
        rot_field.setSFRotation([axis[0], axis[1], axis[2], angle])

        #self.robot.step(self.timestep)



    def set_goal_position(self):
        x_max = self.x_dim * self.x_spacing
        y_max = self.y_dim * self.y_spacing
        #z_max = self.z_dim * self.z_spacing
        self.bounds = {'x': (0.0, x_max), 'y': (0.0, y_max)}

        #self.robot.step(self.timestep)

        # Set goal position
        goal_x = random.uniform(*self.bounds['x'])
        goal_y = random.uniform(*self.bounds['y'])
        #goal_z = self.get_terrain_height(goal_x, goal_y) +0.8
        #self.goal = [goal_x, goal_y, goal_z]
        self.goal = [goal_x, goal_y]

        print(f"Goal position: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        return goal_x, goal_y


    '''
    def randomize_positions2(self):
        x_max = self.x_dim * self.x_spacing
        y_max = self.y_dim * self.y_spacing
        z_max = self.z_dim * self.z_spacing
        self.bounds = {'x': (0.0, x_max), 'y': (0.0, y_max), 'z': (0.0, z_max)}

        for attempt in range(100):
            start_x = random.uniform(*self.bounds['x'])
            start_y = random.uniform(*self.bounds['y'])

            # Convert to height field grid indices
            grid_x = int(start_x / self.x_spacing)
            grid_y = int(start_y / self.y_spacing)

            slope, _ = self.get_slope_value(grid_x, grid_y, scale=10)

            if slope <= 0.1:
                start_z = self.get_terrain_height(start_x, start_y) +0.8
                self.translation_field.setSFVec3f([start_x, start_y, start_z])
                print(f"Found safe start {attempt}: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f}), slope: {slope:.3f}")
                break
        else:
            print("Failed to find safe starting point. Using last random position.")
            start_z = self.get_terrain_height(start_x, start_y) +0.8
            self.translation_field.setSFVec3f([start_x, start_y, start_z])

        self.robot.step(self.timestep)

        # Set goal position (no slope check, but you can add it similarly if needed)
        goal_x = random.uniform(*self.bounds['x'])
        goal_y = random.uniform(*self.bounds['y'])
        goal_z = self.get_terrain_height(goal_x, goal_y) +0.8
        self.goal = [goal_x, goal_y, goal_z]

        print(f"Goal position: ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})")


    def randomize_positions(self):
        x_max = self.x_dim * self.x_spacing
        y_max = self.y_dim * self.y_spacing
        z_max = self.z_dim * self.z_spacing
        self.bounds = {'x': (0.0, x_max), 'y': (0.0, y_max), 'z': (0.0, z_max)}

        start_x = random.uniform(*self.bounds['x'])
        start_y = random.uniform(*self.bounds['y'])
        print(f"Terreno x,y {self.get_terrain_height(start_x, start_y)}")
        start_z = self.get_terrain_height(start_x, start_y) + 0.05  # small offset above ground
        self.translation_field.setSFVec3f([start_x, start_y, start_z])
        print(f"Robot Z {start_z}")
        print(f"Starting position: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")

        self.robot.step(self.timestep)

        self.goal = [
            random.uniform(*self.bounds['x']),
            random.uniform(*self.bounds['y']),
            0.0]

        self.goal[2] = self.get_terrain_height(self.goal[0], self.goal[1]) + 0.05

        print(f"Goal position:  ({self.goal[0]:.2f}, {self.goal[1]:.2f}, {self.goal[2]:.2f})")
    '''


    def distance_to_goal(self, pos):
        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        return math.sqrt(dx**2 + dy**2)

    def angle_to_goal(self, pos):
        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        return math.atan2(dy, dx)  # atan2(dx, dy) -> facing +Z

    def get_heading(self):
        # Yaw from IMU (orientation)
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        return yaw  # radians



    def back_up(self, duration_steps=10):
        self.set_motor_velocity(-2.0, -2.0)
        #for _ in range(duration_steps):
         #   if self.robot.step(self.timestep) == -1:
          #      break

    def rotate_safely(self, direction, duration_steps=10):
        # direction: -1 for left, +1 for right
        self.set_motor_velocity(direction, -direction)
        #for _ in range(duration_steps):
         #   if self.robot.step(self.timestep) == -1:
          #      break



    def is_robot_stuck(self, pos1, pos2):
        distance = math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
        if distance < 0.02:
            return True
        return False

    '''
    def get_greedy_motor_commands(self, current_pos, current_heading, goal_pos):

        distance = self.distance_to_goal(current_pos)
        angle = self.angle_to_goal(current_pos)  # relative to goal
        pitch, roll, _ = self.imu.getRollPitchYaw()

        left_speed = 0.0
        right_speed = 0.0

        # stability
        pitch_threshold = 0.25
        roll_threshold = 0.25
        if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
            # instable, try to stable without considering the goal
            if roll > 0:  # Tilted right, so rotate left
                return -1.0, 1.0
            else:  # Tilted left, so rotate right
                return 1.0, -1.0

        # to the goal if not instable
        if distance > 0.2:  # not arrived
            if angle > 0.1:  # turn left
                left_speed = 1.0
                right_speed = 2.0
            elif angle < -0.1:  # turn right
                left_speed = 2.0
                right_speed = 1.0
            else:  # facing goal, go forward
                left_speed = 2.0
                right_speed = 2.0
        else:  # arrived
            left_speed = 0.0
            right_speed = 0.0

        max_speed = self.max_speed
        left_speed = np.clip(left_speed, -max_speed, max_speed)
        right_speed = np.clip(right_speed, -max_speed, max_speed)

        return left_speed, right_speed
    '''


    def run(self):
        start = (random.randint(0, self.x_dim - 25), random.randint(0, self.y_dim - 25))
        while not self.is_flat(x=start[0], y=start[1]):
            start = (random.randint(0, self.x_dim - 25), random.randint(0, self.y_dim - 25))
            print("start not flat")
        goal = self.set_goal_position()
        #goal = (random.randint(0, self.x_dim - 25), random.randint(0, self.y_dim - 10))

        print(f"Iniciando execução... Posição inicial: {start}")
        self.set_robot_position(start[0] * self.x_spacing, start[1] * self.y_spacing)
        self.total_distance =0
        self.last_pos = self.gps.getValues()
        self.speeds_history = []

        dist= self.distance_to_goal(start)
        #dist=math.hypot(goal[0] - start[0], goal[1] - start[1])
        print(f"Distância total: {dist:.2f}")

        contador = 0
        while self.robot.step(self.timestep) != -1:
            pos = self.gps.getValues()
            heading = self.get_heading()
            distance = self.distance_to_goal(pos)

            if self.last_pos is not None:
                # Calcula a distância entre o ponto atual e o ponto anterior (last_pos)
                segment_distance = math.sqrt(
                    (pos[0] - self.last_pos[0]) ** 2 +
                    (pos[1] - self.last_pos[1]) ** 2)

                self.total_distance += segment_distance
            self.last_pos = pos


            if self.timestep > 0 and segment_distance > 0:
                instant_speed = segment_distance / (self.timestep / 1000.0)  # self.timestep é em ms, converter para s
                self.speeds_history.append(instant_speed)


            if contador == 0:
                last_pos = pos
                print(f"atual:({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), map's Z: {self.get_terrain_height(pos[0], pos[1]):.2f}")

            if contador<5000:
                contador +=1
            else:
                contador = 0
                if self.is_robot_stuck(pos, last_pos):
                    print(f"Robot is stuck")
                    break


            avg_speed = 0.0
            if self.speeds_history:
                avg_speed = np.mean(self.speeds_history)


            if distance < 0.2:
                print("Success! Reached goal")
                print(f"Distância total percorrida: {self.total_distance:.2f}")
                print(f"Velocidade média: {avg_speed:.2f}")
                print(f"atual:({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                print(f"atual:({self.goal[0]:.2f}, {self.goal[1]:.2f})")

                self.set_motor_velocity(0, 0)
                #self.left_motor.setVelocity(0)
                #self.right_motor.setVelocity(0)
                break

            desired_angle = self.angle_to_goal(pos)
            angle_diff = desired_angle - heading
            angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

            #if contador == 300:
             #   print(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}), Goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
              #  print(f"Heading: {math.degrees(heading):.1f}°, Goal angle: {math.degrees(desired_angle):.1f}°, Angle diff: {math.degrees(angle_diff):.1f}°")
               # contador = 0

            '''
            if self.is_robot_stuck():
                self.force_escape()
                continue
            '''


            if abs(angle_diff) > 0.01745: #rad correspondente a 1º
                # Turn in place
                turn_speed = 1.0 if angle_diff > 0 else -1.0

                self.set_motor_velocity(-turn_speed, turn_speed)
                #print(f"Rodar: {turn_speed}")
                #self.left_motor.setVelocity(turn_speed)
                #self.right_motor.setVelocity(-turn_speed)
            else:
                # Move forward
                self.set_motor_velocity(25.0, 25.0)
                #print(f"Andar")
                #self.left_motor.setVelocity(3.0)
                #self.right_motor.setVelocity(3.0)

            # Pitch is the up/down tilt of a robot's body
            # Roll is the measure of how much the robot tilts side to side (left or right)
            pitch_threshold = 0.25
            roll_threshold = 0.25
            pitch, roll, _ = self.imu.getRollPitchYaw()
            if abs(pitch) > pitch_threshold or abs(roll) > roll_threshold:
              #  print("Reversing due to instability")
                self.set_motor_velocity(0, 0)
                #self.robot.step(self.timestep * 2)
                #self.set_motor_velocity(-2.0, -2.0)
                #self.robot.step(self.timestep * 10)
                self.back_up()

                # Decide safest rotation direction
                if roll > 0:
                    # Tilted right, so rotate left
                    #print("Tilting right — rotating left to stabilize.")
                    #self.set_motor_velocity(-1.0, 1.0)
                    self.rotate_safely(-1)
                else:
                    # Tilted left, so rotate right
                    #print("Tilting left — rotating right to stabilize.")
                    #self.set_motor_velocity(1.0, -1.0)
                    self.rotate_safely(1)

                #self.robot.step(self.timestep * 10)
                self.set_motor_velocity(25.0, 25.0)

if __name__ == "__main__":
    print("A iniciar a execução")
    MooseNavigator().run()