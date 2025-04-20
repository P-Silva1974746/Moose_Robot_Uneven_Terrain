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

    def greedy_search(self, start, goal):
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

    #def move_along_path(self, path):
     #   translation_field = self.robot.getField("translation")
      #  for x, y in path:
       #     pos = self.grid_to_world(x, y)
        #    translation_field.setSFVec3f(pos)
         #   for _ in range(10):  # Esperar para visualização
          #      if self.step(self.timestep) == -1:
           #         return

    def move_along_path(self, path):
        from math import atan2, degrees

        left_motor = self.getDevice("left_wheel_motor")   #webots: adicionar motores com estes nomes
        right_motor = self.getDevice("right_wheel_motor")
        left_motor.setPosition(float("inf"))
        right_motor.setPosition(float("inf"))
        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

        gps = self.getDevice("gps")
        compass = self.getDevice("compass")
        gps.enable(self.timestep)
        compass.enable(self.timestep)

        def get_bearing(): #ângulo em relação ao ponto de referência. direção
            c = compass.getValues()
            rad = atan2(c[0], c[2])  #ângulo
            bearing = (rad - 1.5708) * 180.0 / math.pi  #para graus
            return (bearing + 360) % 360  #entre 0 e 360º

        def angle_to_target(curr_pos, target_pos):
            dx = target_pos[0] - curr_pos[0]
            dz = target_pos[2] - curr_pos[2]
            angle = degrees(atan2(dx, dz))
            return (angle + 360) % 360

        for x, y in path:
            target = self.grid_to_world(x, y)
            while self.step(self.timestep) != -1:
                pos = gps.getValues()
                bearing = get_bearing()
                target_angle = angle_to_target(pos, target)

                angle_diff = target_angle - bearing
                #entre -180 e 180
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360

                #alinhar
                if abs(angle_diff) > 5:
                    if angle_diff > 0: #esquerda
                        left_motor.setVelocity(-1.0)
                        right_motor.setVelocity(1.0)
                    else: #direita
                        left_motor.setVelocity(1.0)
                        right_motor.setVelocity(-1.0)
                else: #frente
                    left_motor.setVelocity(2.0)
                    right_motor.setVelocity(2.0)

                    #parar
                    dx = target[0] - pos[0]
                    dz = target[2] - pos[2]
                    distance = math.sqrt(dx**2 + dz**2)
                    if distance < 0.1:
                        break

        left_motor.setVelocity(0.0)
        right_motor.setVelocity(0.0)

    def run(self):
        start, goal = self.generate_random_points()
        print(f"Ponto A: {start}")
        print(f"Ponto B: {goal}")

        path = self.greedy_search(start, goal)

        if len(path) < 2:
            print("Caminho não encontrado")
            return

        self.move_along_path(path)
        print("Chegada ao destino")

if __name__ == "__main__":
    print("Iniciando execução...")
    MooseGreedySupervisor().run()