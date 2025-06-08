import random

import numpy as np
import matplotlib.pyplot as plt
import re
from perlin_noise import PerlinNoise


def create_perlin_map(map_w=200, map_h=200, path_file="./worlds/moose_demo.wbt", path_new_file="worlds/moose_perlin.wbt"):
    with open(file=path_file, mode='r') as f:
        data= f.read()

    # find the height elevation map
    match = re.search(r'height\s*\[\s*(.*?)\s]',data, re.DOTALL)

    if match:
        # extract string of numbers
        heights_strings=match.group(1)

        #convert the values
        heights=np.array([float(s.strip()) for s in heights_strings.split(',') if s.strip()])

        #---------------------CODE TO ALTER THE ELEVATION MAP------------------------#
        noise_scale=0.020
        perlin_noise = PerlinNoise(dimension=2,octaves=1,unbias=False)
        #modifed_heights=heights/3
        modified_heights=[]

        max_dist=np.hypot(map_w/2,map_h/2)

        # get random point to create the spawn zone
        x = random.randint(0,map_w-1)
        y = random.randint(0,map_h-1)
        size_spawn=10
        spawn_heights=[]


        # 200 is the size of our elevation grid this could be done with code if we read it from the .wbt file directly
        for j in range (map_h):
            for i in range (map_w):
                dist_x= map_w/2 -i
                dist_y= map_h/2 -j

                #the 0.8 in the dist_y makes it less important meaning the elevation in the map is not a perfect circle
                dist= np.hypot(dist_x,0.8*dist_y)

                # To make a circular map uncomment this line and add the noise instead of assigning it
                #new_height=30-45*dist/max_dist
                new_height= perlin_noise((i+random.random())*noise_scale,(j+random.random())*noise_scale)*5
                if (x-size_spawn)<=i<(x+size_spawn) and (y-size_spawn)<=j<(y+size_spawn):
                    #print(f"The point {(i,j)} is part of the spawn")
                    spawn_heights.append(new_height)

                modified_heights.append(new_height)

        spawn_heights = np.array(spawn_heights)
        spawn_height= np.median(spawn_heights)

        for spawn_i  in range(max(0, x-size_spawn), min(x+size_spawn,map_w-1)):
            for spawn_j in range(max(0, y-size_spawn) , min(y+size_spawn, map_h-1)):
                modified_heights[spawn_i + spawn_j*map_w]=spawn_height
        #---------------------CODE TO ALTER THE ELEVATION MAP------------------------#

        formated_heights=',\n '.join(f"{height:.6f}" for height in modified_heights)

        def replacer(match):
            return f"{match.group(1)}{formated_heights}{match.group(3)}"

        #replace the data
        updated_data=re.sub(r'(height\s*\[\s*)(.*?)(\s*\])',
                            replacer,
                            data,
                            flags=re.DOTALL
        )

        # setting the robot translation to the spawn area
        pos_x = float (x)
        pos_y = float(y)
        pos_z = float(spawn_height + 0.5)

        # Replace MOOSEROBOT translation
        updated_data = re.sub(
            r'(DEF\s+MOOSEROBOT\s+Robot\s*{(?:[^{}]*\{[^{}]*\}[^{}]*)*?[^}]*?)translation\s+[-\d.eE+]+\s+[-\d.eE+]+\s+[-\d.eE+]',
            lambda match: f'{match.group(1)}translation {pos_x:.6f} {pos_y:.6f} {pos_z:.6f}',
            updated_data,
            flags=re.DOTALL
        )

        with open(path_new_file,mode='w') as f:
            f.write(updated_data)

        return x,y, spawn_height


if __name__ == "__main__":
    create_perlin_map(path_new_file="worlds/moose_perlin_teste.wbt")
