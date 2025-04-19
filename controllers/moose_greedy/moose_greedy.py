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

    def move_along_path(self, path):
        translation_field = self.robot.getField("translation")
        for x, y in path:
            pos = self.grid_to_world(x, y)
            translation_field.setSFVec3f(pos)
            for _ in range(10):  # Esperar para visualização
                if self.step(self.timestep) == -1:
                    return

    def run(self):
        start, goal = self.generate_random_points()
        print(f"Ponto A: {start}")
        print(f"Ponto B: {goal}")

        path = self.greedy_search(start, goal)

        if len(path) < 2:
            print("Caminho não encontrado.")
            return

        self.move_along_path(path)
        print("Chegada ao destino com sucesso!")

if __name__ == "__main__":
    print("Iniciando execução...")
    MooseGreedySupervisor().run()
