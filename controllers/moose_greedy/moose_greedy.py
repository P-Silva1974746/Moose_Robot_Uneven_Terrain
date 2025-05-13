import heapq
import math
import random
import time
from controller import Supervisor

class MooseGreedySupervisor(Supervisor):
    def __init__(self):
        super().__init__()

        # Setup
        self.timestep = int(self.getBasicTimeStep())
        self.robot = self.getFromDef("MOOSEROBOT")

        self.left_motors = [self.getDevice(f"left_motor_{i}") for i in range(1, 5)]
        self.right_motors = [self.getDevice(f"right_motor_{i}") for i in range(1, 5)]

        for motor in self.left_motors + self.right_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # sensores
        self.gps = self.getDevice("gps")
        self.gps.enable(self.timestep)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.timestep)

        # Mapa de elevação (grid)
        self.elevation_grid = self.getFromDef("EG")
        self.height_field = self.elevation_grid.getField("height")
        self.height_values = [self.height_field.getMFFloat(i) for i in range(self.height_field.getCount())]
        self.x_dim = self.elevation_grid.getField("xDimension").getSFInt32()
        self.y_dim = self.elevation_grid.getField("yDimension").getSFInt32()
        self.x_spacing = self.elevation_grid.getField("xSpacing").getSFFloat()
        self.y_spacing = self.elevation_grid.getField("ySpacing").getSFFloat()

    def height_at(self, x, y):
        index = y * self.x_dim + x
        print(f"Altura para (x={x}, y={y}), índice {index}")
        return self.height_values[index]

    def heuristic(self, a, b):
        print(f"Cálculo da heurística entre {a} e {b}")
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def neighbors(self, x, y):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        valid = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.x_dim and 0 <= ny < self.y_dim:
                valid.append((nx, ny))
        return valid

    def greedy_search(self, start, goal):
        print(f"Greedy search de {start} para {goal}")
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start, goal), start))
        came_from = {start: None}
        visited = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                break

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self.neighbors(*current):
                if neighbor not in visited:
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (self.heuristic(neighbor, goal), neighbor))

        # Reconstruir caminho
        path = []
        node = goal
        while node:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path

    def grid_to_world(self, x, y):
        x = max(0, min(x, self.x_dim - 1))
        y = max(0, min(y, self.y_dim - 1))
        wx = x * self.x_spacing
        wy = y * self.y_spacing
        wz = self.height_at(x, y) + 0.5
        return [wx, wy, wz]


    def generate_random_points(self, min_distance=5):
        while True:
            start = (random.randint(0, self.x_dim - 1), random.randint(0, self.y_dim - 1))
            goal = (random.randint(0, self.x_dim - 1), random.randint(0, self.y_dim - 1))
            if self.heuristic(start, goal) >= min_distance:
                return start, goal


    def set_wheel_speeds(self, left_speed, right_speed):
        for motor in self.left_motors:
            motor.setVelocity(left_speed)
        for motor in self.right_motors:
            motor.setVelocity(right_speed)


    def go_to(self, target, tolerance=0.3):
        MAX_SPEED = min(m.getMaxVelocity() for m in self.left_motors + self.right_motors)
        print("Max vel.:", MAX_SPEED)
        TURN_COEFF = 2.0

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

            correction = TURN_COEFF * beta
            left_speed = MAX_SPEED - correction
            right_speed = MAX_SPEED + correction

            self.set_wheel_speeds(
                max(-MAX_SPEED, min(MAX_SPEED, left_speed)),
                max(-MAX_SPEED, min(MAX_SPEED, right_speed)))


    def move_robot(self, path):
        print("A mover o robo...")
        total_distance = 0 #metros
        start_time = time.time() #segundos
        last_pos = self.grid_to_world(*path[0])

        for x, y in path:
            target = self.grid_to_world(x, y)
            print(f"Mover para: {target}")
            self.go_to(target)

            if self.step(self.timestep) == -1:
                return

            current_pos = self.gps.getValues()
            print(f"Posição atual: {current_pos}")

            distance = math.sqrt(sum((current_pos[i] - last_pos[i]) ** 2 for i in range(2)))
            total_distance += distance

        end_time = time.time()
        duration = end_time - start_time
        avg_speed = total_distance / duration

        print(f"Execução concluída. Distância: {total_distance:.2f}, Tempo: {duration:.2f}")

##def função para salvar dados da execução

    def wait_until_still(self, duration=1.0):
        stable_time = 0.0
        last_pos = self.gps.getValues()
        while stable_time < duration:
            if self.step(self.timestep) == -1:
                return
            current_pos = self.gps.getValues()
            movement = math.sqrt(sum((current_pos[i] - last_pos[i]) ** 2 for i in range(3)))
            last_pos = current_pos

            if movement < 0.001:
                stable_time += self.timestep / 1000.0
            else:
                stable_time = 0.0

    def run(self):
        start, goal = self.generate_random_points()
        print(f"Ponto A: {start}")
        print(f"Ponto B: {goal}")

        print(f"A iniciar execução")
        path = self.greedy_search(start, goal)
        print(f"Caminho: {path}")


        if len(path) < 2:
            print("Caminho não encontrado")
            return

        self.wait_until_still()
        print(f"Caminho encontrado com {len(path)} passos")
        self.move_robot(path)
        print("Greedy terminado com sucesso")


if __name__ == "__main__":
    print("A iniciar execução")
    MooseGreedySupervisor().run()