import heapq
import math
import random
import time
import csv
import matplotlib.pyplot as plt
import os
from controller import Supervisor

#Comfirmar battery e A* heuristic
class MooseSupervisor(Supervisor):
    def __init__(self):
        super().__init__()

        # Configurações
        self.timestep = int(self.getBasicTimeStep())
        self.battery_max_energy = 100.0
        self.battery_energy = self.battery_max_energy
        self.recharge_threshold = -0.05  # Diferença de altitude para recarregar

        # Robot e terreno
        self.robot = self.getFromDef('MOOSEROBOT')
        self.elevation_grid = self.getFromDef('EG')
        self.height_field = self.elevation_grid.getField('height')
        self.height_values = [self.height_field.getMFFloat(i) for i in range(self.height_field.getCount())]
        self.x_dim = self.elevation_grid.getField('xDimension').getSFInt32()
        self.y_dim = self.elevation_grid.getField('yDimension').getSFInt32()
        self.x_spacing = self.elevation_grid.getField('xSpacing').getSFFloat()
        self.y_spacing = self.elevation_grid.getField('ySpacing').getSFFloat()

        # Sensores
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)

        self.altimeter = self.getDevice("altimeter")
        self.altimeter.enable(self.timestep)

        # Para plotar o gráfico da bateria
        self.battery_levels = []  # Armazenar os valores da bateria ao longo do tempo

        # Criar pasta para armazenar as imagens
        self.image_folder = "battery_plots"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

    def height_at(self, x, y):
        index = y * self.x_dim + x
        return self.height_values[index]

    def heuristic(self, a, b):
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
        return math.hypot(current[0] - neighbor[0], current[1] - neighbor[1]) + dh

    def a_star(self, start, goal):
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
        # Se está descendo, recarrega
        if curr_alt - prev_alt < self.recharge_threshold:
            self.battery_energy = min(self.battery_energy + 0.2, self.battery_max_energy)
        else:
            # Consome com base na velocidade
            consumption = 0.05 * (speed ** 2)
            self.battery_energy = max(self.battery_energy - consumption, 0)

    def move_robot_and_log(self, path):
        translation_field = self.robot.getField('translation')
        total_distance = 0
        altitudes = []

        start_time = time.time()
        last_pos = self.grid_to_world(*path[0])
        last_alt = self.altimeter.getValue()
        altitudes.append(last_alt)

        for x, y in path:
            pos = self.grid_to_world(x, y)  # Atualizado para garantir que o robô fique no grid
            translation_field.setSFVec3f(pos)

            for _ in range(10):
                if self.step(self.timestep) == -1:
                    return

            current_pos = self.gps.getValues()
            current_alt = self.altimeter.getValue()

            distance = math.sqrt(sum((current_pos[i] - last_pos[i]) ** 2 for i in range(2)))
            total_distance += distance

            speed = distance / (self.timestep / 1000 * 10)
            self.update_battery(last_alt, current_alt, speed)

            altitudes.append(current_alt)
            last_pos = current_pos
            last_alt = current_alt

            # Salvar o nível de bateria a cada iteração
            self.battery_levels.append(self.battery_energy)

        end_time = time.time()
        duration = end_time - start_time
        avg_speed = total_distance / duration
        altitude_diff = max(altitudes) - min(altitudes)

        self.log_data({
            'time_sec': round(duration, 2),
            'distance_m': round(total_distance, 2),
            'avg_speed_mps': round(avg_speed, 2),
            'battery_remaining': round(self.battery_energy, 2),
            'altitude_diff_m': round(altitude_diff, 2)
        })

        # Plotar e salvar o gráfico da bateria
        self.plot_battery()

    def log_data(self, data, file_name="performance_log.csv"):
        with open(file_name, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
        print("Dados logados:", data)

    def plot_battery(self):
        plt.figure()
        plt.plot(self.battery_levels, label="Nível de Bateria")
        plt.xlabel("Iteração")
        plt.ylabel("Bateria (%)")
        plt.title("Curva de Consumo/Recarga da Bateria")
        plt.grid(True)

        # Salvar o gráfico como imagem na pasta 'battery_plots'
        image_path = os.path.join(self.image_folder, f'battery_curve_{int(time.time())}.png')
        plt.savefig(image_path)
        plt.close()  # Fechar a figura após salvar para evitar sobrecarga de memória

    def run(self):
        # Definir 'start' e 'goal' uma única vez
        start = (random.randint(0, self.x_dim - 1), random.randint(0, self.y_dim - 1))
        goal = (random.randint(0, self.x_dim - 1), random.randint(0, self.y_dim - 1))

        print(f"Start: {start}, Goal: {goal}")
        path = self.a_star(start, goal)

        if len(path) < 2:
            print("Caminho não encontrado!")
            return

        self.move_robot_and_log(path)
        print("Execução finalizada com sucesso.")


if __name__ == "__main__":
    print("Início da execução...")
    MooseSupervisor().run()
