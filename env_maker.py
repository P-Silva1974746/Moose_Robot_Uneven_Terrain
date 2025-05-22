import random

import numpy as np
import matplotlib.pyplot as plt
import re
from perlin_noise import PerlinNoise

def replacer(match):
    return f"{match.group(1)}{formated_heights}{match.group(3)}"

PATH_FILE="./worlds/moose_demo.wbt"
with open(file=PATH_FILE,mode='r') as f:
    data= f.read()

MAP_W=200
MAP_H=200


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

    max_dist=np.hypot(MAP_W/2,MAP_H/2)

    # 200 is the size of our elevation grid this could be done with code if we read it from the .wbt file directly
    for j in range (MAP_H):
        for i in range (MAP_H):
            dist_x= MAP_W/2 -i
            dist_y= MAP_H/2 -j

            #the 0.8 in the dist_y makes it less important meaning the elevation in the map is not a perfect circle
            dist= np.hypot(dist_x,0.8*dist_y)
            new_height=30-45*dist/max_dist
            new_height+= perlin_noise((i+random.random())*noise_scale,(j+random.random())*noise_scale)*5
            #new_height=0

            modified_heights.append(new_height)
    #---------------------CODE TO ALTER THE ELEVATION MAP------------------------#

    formated_heights=',\n '.join(f"{height:.6f}" for height in modified_heights)

    #replace the data
    updated_data=re.sub(r'(height\s*\[\s*)(.*?)(\s*\])',
                        replacer,
                        data,
                        flags=re.DOTALL
    )

    PATH_NEW_FILE="worlds/moose_perlin_circle.wbt"
    with open(PATH_NEW_FILE,mode='w') as f:
        f.write(updated_data)


