import heapq
import math
import random
import time
import csv
import matplotlib.pyplot as plt
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from controller import Supervisor

class MooseSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        # Configurações
        self.timestep = int(self.getBasicTimeStep())
        self.battery_max_energy = 100.0
        self.battery_energy = self.battery_max_energy
        self.recharge_threshold = -0.05

        # Robot e terreno
        self.robot = self.getFromDef('MOOSEROBOT')
        self.moose_node = self.robot.getField("children").getMFNode(0)  # Assume que Moose é o primeiro filho

        self.elevation_field = self.getFromDef("EG").getField("height")
        self.elevation_grid = self.getFromDef('EG')
        self.height_field = self.elevation_grid.getField('height')

        self.x_dim = self.elevation_grid.getField('xDimension').getSFInt32()
        self.y_dim = self.elevation_grid.getField('yDimension').getSFInt32()
        self.x_spacing = self.elevation_grid.getField('xSpacing').getSFFloat()
        self.y_spacing = self.elevation_grid.getField('ySpacing').getSFFloat()

        self.height_values = [self.height_field.getMFFloat(i) for i in range(self.height_field.getCount())]
        self.elevation_map = self.get_elevation_map()  # <-- Agora pode chamar

        sigma = 1  # ou qualquer valor apropriado
        self.smoothed_elevation_map = gaussian_filter(self.elevation_map, sigma=sigma)

        # Motores
        self.left_motors = [
            self.getDevice("left_motor_1"),
            self.getDevice("left_motor_2"),
            self.getDevice("left_motor_3"),
            self.getDevice("left_motor_4")
        ]
        self.right_motors = [
            self.getDevice("right_motor_1"),
            self.getDevice("right_motor_2"),
            self.getDevice("right_motor_3"),
            self.getDevice("right_motor_4")
        ]

        for motor in self.left_motors + self.right_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Sensores
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)

        self.altimeter = self.getDevice("altimeter")
        self.altimeter.enable(self.timestep)

        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        self.lidar = self.getDevice("lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Log e bateria
        self.battery_levels = []
        self.image_folder = "battery_plots"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def get_elevation_map(self):
        count = self.height_field.getCount()
        height_values = [self.height_field.getMFFloat(i) for i in range(count)]
        elevation_map = [
            height_values[i * self.x_dim:(i + 1) * self.x_dim]
            for i in range(self.y_dim)
        ]
        return elevation_map

    def get_slope_value(self, x, y, scale=10):
        """This function gives us the gradient of the region of scale//2 around the point (x,y)"""
        heights_in_range=np.zeros((scale, scale))
        for i in range(scale):
            for j in range(scale):
                nx=x-scale//2+i
                ny=y-scale//2+j
                heights_in_range[i,j]=self.height_at(nx,ny)

        dz_x, dz_y = np.gradient(heights_in_range)
        slope_x= dz_x[scale//2,scale//2]
        slope_y= dz_y[scale//2,scale//2]

        slope_magnitude=np.hypot(slope_x**2,slope_y**2)

        normal=np.array([-slope_x, -slope_y, 1.0])
        normal/=np.linalg.norm(normal)

        return slope_magnitude, normal




    def set_robot_position(self, x, y):
        z = self.height_at(x, y)
        pos_field = self.robot.getField("translation")
        pos_field.setSFVec3f([x * self.x_spacing, y * self.y_spacing, z + 1.5])  # Ajuste do z + altura segura

        # calcula o gradiente do terrono em volta e retorna o vetor normal
        _, normal = self.get_slope_value(x, y)
        # direcao que queremos alinhar com o vetor normal e o eixo z do robot
        up=np.array([0, 0, 1], dtype=float)
        # calcular os eixos em que a rotacao vao ter de acontecer para alinhar o eixo z do robot com o vetor normal
        axis=np.cross(up, normal)
        axis/=np.linalg.norm(axis)

        # calcular o angulo entre o up e o vetor normal
        angle=np.arccos(np.clip(np.dot(up,normal), -1.0, 1.0))

        rot_field = self.robot.getField("rotation")
        rot_field.setSFRotation([axis[0], axis[1], axis[2], angle])

        self.step(self.timestep)

    def height_at(self, x, y):
        x = int(x)
        y = int(y)
        x = max(0, min(x, self.x_dim - 1))  # Garantir que x esteja dentro do grid
        y = max(0, min(y, self.y_dim - 1))  # Garantir que y esteja dentro do grid
        index = y * self.x_dim + x
        return self.height_values[index]

    def heuristic(self, a, b):
        #print(f"Calculando heurística entre {a} e {b}")
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def neighbors(self, x, y):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]

        valid = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.x_dim and 0 <= ny < self.y_dim:
                valid.append((nx, ny))
        return valid

    def cost(self, current, neighbor):
        h1 = self.height_at(*current)
        h2 = self.height_at(*neighbor)
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        distance = math.hypot(dx, dy)

        if distance == 0:
            return float('inf')

        slope_deg = math.degrees(math.atan2(abs(h2 - h1), distance))

        return distance + abs(h2 - h1)  # custo com leve penalização

    def a_star(self, start, goal):
        if start == goal:
            return [start]

        print(f"Executando A* de {start} para {goal}")
        visited = set()
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                break

            for next in self.neighbors(*current):
                new_cost = cost_so_far[current] + self.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, goal)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        # Exibindo caminho final
        path = []
        node = goal
        while node:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path

    def grid_to_world(self, x, y):
        # Limitar as coordenadas para que o robô não saia do grid
        x = max(0, min(x, self.x_dim - 1))
        y = max(0, min(y, self.y_dim - 1))

        wx = x * self.x_spacing
        wy = y * self.y_spacing
        wz = self.height_at(x, y) + 0.5
        return [wx, wy, wz]

    def update_battery(self, prev_alt, curr_alt, speed):
        consumption = 0
        recovery = 0
        # Se o robô está subindo, a bateria é consumida mais rápido
        if curr_alt > prev_alt:
            consumption = 0.05 * (speed ** 2) * 1.5  # Mais consumo em subida
        elif curr_alt < prev_alt:
            # Se o robô está descendo, ele recupera energia (regeneração)
            recovery = 0.05 * (speed ** 2) * 0.8  # Recupera energia na descida
        else:
            recovery = 0  # Não há mudança de altura, nenhuma recuperação

        # Atualiza o nível da bateria
        self.battery_energy = max(0, min(self.battery_energy - consumption + recovery, self.battery_max_energy))

    def set_wheel_speeds(self, left_speed, right_speed):
        #print(f"Left motor speed: {left_speed}")
        #print(f"Right motor speed: {right_speed}")

        for motor in self.left_motors:
            motor.setVelocity(left_speed)
        for motor in self.right_motors:
            motor.setVelocity(right_speed)

    def go_to(self, target, d_tolerance=0.3):
        ang_tolerance=0.25 # se o robot estiver alinhado o sufeciente pode andar mais rapido
        turning=False
        going_forward=False

        while self.step(self.timestep) != -1:
            MAX_SPEED = 26  # Velocidade máxima do robô
            turned=False
            pos = self.gps.getValues()
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            distance = math.hypot(dx, dy)

            if distance < d_tolerance:
                #self.set_wheel_speeds(0.0, 0.0)
                print("Robot got to {}".format(target))
                return True

            desired_angle = math.atan2(dy, dx)
            yaw = self.imu.getRollPitchYaw()[2]
            beta = desired_angle - yaw
            beta = (beta + math.pi) % (2 * math.pi) - math.pi

            # nova velociade maxima dependendo do quao alinhado o robot esta com o proximo target do path
            if(abs(beta) < ang_tolerance or turned):
                #MAX_SPEED*=1
                forward_speed=MAX_SPEED
                correction = 2.0 * beta
                left_speed = forward_speed - correction
                right_speed = forward_speed + correction

                if not going_forward:
                    print(f"Velocidade na reta: {forward_speed}")
                going_forward = True
                turning = False
            # turning
            else:
                #MAX_SPEED*=0.2
                turn_speed=5*beta
                left_speed=-turn_speed
                right_speed=turn_speed
                turned = True




            left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

            self.set_wheel_speeds(left_speed, right_speed)

    def move_robot_and_log(self, path, euclidean_dist):
        print("Iniciando movimento...")
        translation_field = self.robot.getField('translation')
        total_distance = 0.0  # agora é atualizado
        altitudes = []

        start_time = time.time()
        last_pos = self.grid_to_world(*path[0])
        last_alt = self.altimeter.getValue()
        altitudes.append(last_alt)

        for i, (x, y) in enumerate(path):
            target_pos = self.grid_to_world(x, y)
            print(f"Movendo para: {target_pos}")

            success = self.go_to(target_pos)

            if not success:
                print("Abortando caminho atual. Tentando nova rota...")
                # Recalcula o caminho atual a partir da posição GPS real até o destino final
                current_grid = (
                    int(self.gps.getValues()[0] / self.x_spacing),
                    int(self.gps.getValues()[1] / self.y_spacing)
                )
                new_path = self.a_star(current_grid, path[-1])
                if len(new_path) > 1 and new_path[0] != new_path[-1]:
                    self.move_robot_and_log(new_path, self.heuristic(current_grid, path[-1]))
                else:
                    print("Nenhuma rota alternativa viável encontrada.")
                return  # Interrompe execução atual

            # Atualiza bateria com base em altitude e velocidade
            curr_alt = self.altimeter.getValue()
            speed = euclidean_dist / self.timestep
            self.update_battery(last_alt, curr_alt, speed)

            # Atualiza distâncias
            distance_traveled = math.hypot(target_pos[0] - last_pos[0], target_pos[1] - last_pos[1])
            total_distance += distance_traveled

            last_pos = target_pos
            last_alt = curr_alt
            altitudes.append(curr_alt)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Distância total percorrida: {total_distance} metros")
        print(f"Tempo total: {total_time} segundos")

        self.battery_levels.append(self.battery_energy)

        return total_distance, total_time

    def plot_battery(self):
        plt.plot(self.battery_levels)
        plt.title("Nível da bateria ao longo do tempo")
        plt.xlabel("Passos")
        plt.ylabel("Nível de Bateria (%)")
        plt.savefig(f"{self.image_folder}/battery_plot_{int(time.time())}.png")
        plt.close()

    def get_current_battery(self):
        return self.battery_energy

    def wait_until_still(self, duration=1.0):
        """Espera até que o robô esteja parado por um certo tempo (em segundos)."""
        stable_time = 0.0
        last_pos = self.gps.getValues()


        while stable_time < duration:
            if self.step(self.timestep) == -1:
                return
            current_pos = self.gps.getValues()
            movement = math.sqrt(sum((current_pos[i] - last_pos[i]) ** 2 for i in range(3)))
            last_pos = current_pos

            if movement < 0.001:  # Tolerância para considerar "parado"
                stable_time += self.timestep / 1000.0
            else:
                stable_time = 0.0  # Reinicia o tempo de estabilidade se houver movimento

    def run(self):
        start = (random.randint(0, self.x_dim - 25), random.randint(0, self.y_dim - 10))
        goal = (random.randint(0, self.x_dim - 25), random.randint(0, self.y_dim - 10))

        print(f"Iniciando execução... Posição inicial: {start}, Meta: {goal}")
        self.set_robot_position(start[0] * self.x_spacing, start[1] * self.y_spacing)
        path = self.a_star(start, goal)
        print(f"Caminho calculado: {path}")

        if len(path) < 2:
            print("Caminho não encontrado!")
            return

        self.wait_until_still()  # Aguarda o robô estar completamente parado
        euclidean_dist = self.heuristic(start, goal)
        print(f"Distância euclidiana entre início e meta: {euclidean_dist:.2f} m")
        self.move_robot_and_log(path, euclidean_dist)
        print("Execução finalizada com sucesso.")

# Esta implementação faz parte do código do robô Moose no Webots para navegação baseada no A*, com consumo de bateria e movimentos registrados.

if __name__ == "__main__":
    print("Início da execução...")
    MooseSupervisor().run()
