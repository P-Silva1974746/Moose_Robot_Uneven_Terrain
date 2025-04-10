import random
from controller import Supervisor

class MooseSupervisor(Supervisor):
    def __init__(self):
        super().__init__()

        self.robot = self.getFromDef('MOOSEROBOT')
        if self.robot is None:
            print("Robot node not found!")
            exit(1)

        self.elevation_grid = self.getFromDef('EG')
        if self.elevation_grid is None:
            print("ElevationGrid node not found!")
            exit(1)

        # Acedendo ao campo 'height' corretamente
        self.height_field = self.elevation_grid.getField('height')
        if self.height_field is None:
            print("Campo 'height' não encontrado no ElevationGrid.")
            exit(1)

        # Obter os valores de altura como uma lista
        self.height_values = []
        num_values = self.height_field.getCount()  # Número total de valores de altura
        for i in range(num_values):
            self.height_values.append(self.height_field.getMFFloat(i))

        # Acessando outras propriedades
        self.x_dimension = self.elevation_grid.getField('xDimension').getSFInt32()
        self.y_dimension = self.elevation_grid.getField('yDimension').getSFInt32()
        self.x_spacing = self.elevation_grid.getField('xSpacing').getSFFloat()
        self.y_spacing = self.elevation_grid.getField('ySpacing').getSFFloat()

    def run(self):
        timestep = int(self.getBasicTimeStep())


        # Gerar coordenadas aleatórias dentro dos limites da grade
        x_index = random.randint(0, self.x_dimension - 1)
        y_index = random.randint(0, self.y_dimension - 1)

        # Calcular as coordenadas reais
        x = self.x_spacing * x_index
        y = self.y_spacing * y_index

        # Acessar o valor de altura para a célula selecionada
        index = x_index + y_index * self.x_dimension
        z = self.height_values[index]+5 # +1 garante que o robo não está demasiado perto do chão

        # Definir a nova posição do robô
        translation_field = self.robot.getField('translation')
        translation_field.setSFVec3f([x, y, z])

        while self.step(timestep) != -1:
            pass

if __name__ == "__main__":
    supervisor = MooseSupervisor()
    supervisor.run()