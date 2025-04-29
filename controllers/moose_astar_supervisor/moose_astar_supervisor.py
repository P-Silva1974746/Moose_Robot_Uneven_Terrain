import heapq
import math
import random
import time
import csv
import matplotlib.pyplot as plt
import os
from controller import Supervisor
class MooseSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        # Prints de debug para inicialização
        print("Inicializando MooseSupervisor...")

        # Configurações
        self.timestep = int(self.getBasicTimeStep())
        self.battery_max_energy = 100.0
        self.battery_energy = self.battery_max_energy
        self.recharge_threshold = -0.05

        # Robot e terreno
        self.robot = self.getFromDef('MOOSEROBOT')
        self.moose_node = self.robot.getField("children").getMFNode(0)  # Assume que Moose é o primeiro filho
        self.elevation_grid = self.getFromDef('EG')
        self.height_field = self.elevation_grid.getField('height')
        self.height_values = [self.height_field.getMFFloat(i) for i in range(self.height_field.getCount())]
        self.x_dim = self.elevation_grid.getField('xDimension').getSFInt32()
        self.y_dim = self.elevation_grid.getField('yDimension').getSFInt32()
        self.x_spacing = self.elevation_grid.getField('xSpacing').getSFFloat()
        self.y_spacing = self.elevation_grid.getField('ySpacing').getSFFloat()
        print(f"Dimensões do grid: {self.x_dim} x {self.y_dim}")
        print(f"Espaçamento entre células: {self.x_spacing}, {self.y_spacing}")

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

        # Terreno
        self.elevation_grid = self.getFromDef('EG')
        self.height_field = self.elevation_grid.getField('height')
        self.height_values = [self.height_field.getMFFloat(i) for i in range(self.height_field.getCount())]
        self.x_dim = self.elevation_grid.getField('xDimension').getSFInt32()
        self.y_dim = self.elevation_grid.getField('yDimension').getSFInt32()
        self.x_spacing = self.elevation_grid.getField('xSpacing').getSFFloat()
        self.y_spacing = self.elevation_grid.getField('ySpacing').getSFFloat()

        # Log e bateria
        self.battery_levels = []
        self.image_folder = "battery_plots"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def height_at(self, x, y):
        index = y * self.x_dim + x
        print(f"Consultando altura para (x={x}, y={y}), índice {index}")
        return self.height_values[index]

    def heuristic(self, a, b):
        print(f"Calculando heurística entre {a} e {b}")
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def neighbors(self, x, y):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.x_dim and 0 <= ny < self.y_dim:
                valid.append((nx, ny))
        return valid

    def cost(self, current, neighbor):
        h1 = self.height_at(*current)
        h2 = self.height_at(*neighbor)
        dh = abs(h2 - h1)

        # Considera o impacto do terreno (subida/descida) no custo
        terrain_cost = math.hypot(current[0] - neighbor[0], current[1] - neighbor[1])
        return terrain_cost + dh * 2  # Aumenta o custo se a diferença de altura for grande

    def a_star(self, start, goal):
        print(f"Executando A* de {start} para {goal}")
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

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
        for motor in self.left_motors:
            motor.setVelocity(left_speed)
        for motor in self.right_motors:
            motor.setVelocity(right_speed)

    def go_to(self, target, tolerance=0.3):
        MAX_SPEED = min(motor.getMaxVelocity() for motor in self.left_motors + self.right_motors)
        print("Max velocity:", MAX_SPEED)
        TURN_COEFFICIENT = 2.0

        while self.step(self.timestep) != -1:
            pos = self.gps.getValues()
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            distance = math.hypot(dx, dy)

            if distance < tolerance:
                self.set_wheel_speeds(0.0, 0.0)
                break

            desired_angle = math.atan2(dy, dx)
            yaw = self.imu.getRollPitchYaw()[2]

            beta = desired_angle - yaw
            beta = (beta + math.pi) % (2 * math.pi) - math.pi

            correction = TURN_COEFFICIENT * beta
            left_speed = MAX_SPEED - correction
            right_speed = MAX_SPEED + correction

            # Saturar velocidades
            left_speed = max(-MAX_SPEED, min(MAX_SPEED, left_speed))
            right_speed = max(-MAX_SPEED, min(MAX_SPEED, right_speed))

            self.set_wheel_speeds(left_speed, right_speed)

    def move_robot_and_log(self, path, euclidean_dist):
        print("Iniciando movimento...")
        translation_field = self.robot.getField('translation')
        total_distance = 0
        altitudes = []

        start_time = time.time()
        last_pos = self.grid_to_world(*path[0])
        last_alt = self.altimeter.getValue()
        altitudes.append(last_alt)

        for x, y in path:
            target_pos = self.grid_to_world(x, y)
            print(f"Movendo para: {target_pos}")

            self.go_to(target_pos)

            if self.step(self.timestep) == -1:
                return

            current_pos = self.gps.getValues()
            current_alt = self.altimeter.getValue()
            print(f"Posição atual: {current_pos}, Altitude: {current_alt}")

            distance = math.sqrt(sum((current_pos[i] - last_pos[i]) ** 2 for i in range(2)))
            total_distance += distance

            speed = distance / (self.timestep / 1000 * 10)
            self.update_battery(last_alt, current_alt, speed)

            altitudes.append(current_alt)
            last_pos = current_pos
            last_alt = current_alt

            self.battery_levels.append(self.battery_energy)

        end_time = time.time()
        duration = end_time - start_time
        avg_speed = total_distance / duration
        altitude_diff = max(altitudes) - min(altitudes)

        print(f"Execução concluída. Distância total: {total_distance:.2f} m, Tempo total: {duration:.2f} s")
        self.log_data({
            'time_sec': round(duration, 2),
            'distance_m': round(total_distance, 2),
            'avg_speed_mps': round(avg_speed, 2),
            'battery_remaining': round(self.battery_energy, 2),
            'altitude_diff_m': round(altitude_diff, 2),
            'euclidean_dist_m': round(euclidean_dist, 2),
            'path_len_cells': len(path)
        })
        self.plot_battery()

    def log_data(self, data, file_name="performance_log.csv"):
        # Salva apenas a cada execução ou a cada N iterações
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)

    def plot_battery(self, plot_interval=10):
        # Salva o gráfico apenas a cada 10 passos (ajustável)
        if len(self.battery_levels) % plot_interval == 0:
            plt.figure()
            plt.plot(self.battery_levels, label="Nível de Bateria")
            plt.xlabel("Iteração")
            plt.ylabel("Bateria (%)")
            plt.title("Curva de Consumo/Recarga da Bateria")
            plt.grid(True)

            # Salva o gráfico como imagem na pasta 'battery_plots'
            image_path = os.path.join(self.image_folder, f'battery_curve_{int(time.time())}.png')
            plt.savefig(image_path)
            plt.close()  # Fechar a figura após salvar para evitar sobrecarga de memória

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
        start = (random.randint(0, self.x_dim - 1), random.randint(0, self.y_dim - 1))
        goal = (random.randint(0, self.x_dim - 1), random.randint(0, self.y_dim - 1))

        print(f"Iniciando execução... Posição inicial: {start}, Meta: {goal}")
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

if __name__ == "__main__":
    print("Início da execução...")
    MooseSupervisor().run()
