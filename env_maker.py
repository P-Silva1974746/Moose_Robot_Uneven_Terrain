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


# find the height elevation map
match = re.search(r'height\s*\[\s*(.*?)\s]',data, re.DOTALL)

if match:
    # extract string of numbers
    heights_strings=match.group(1)

    #convert the values
    heights=np.array([float(s.strip()) for s in heights_strings.split(',') if s.strip()])

    #---------------------CODE TO ALTER THE ELEVATION MAP------------------------#
    noise_scale=0.025
    perlin_noise = PerlinNoise(dimension=2,octaves=1,unbias=False)
    #modifed_heights=heights/3
    modified_heights=[]

    # 200 is the size of our elevation grid this could be done with code if we read it from the .wbt file directly
    for j in range (200):
        for i in range (200):
            new_height= perlin_noise((i+random.random())*noise_scale,(j+random.random())*noise_scale)*20
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

    PATH_NEW_FILE="worlds/moose_perlin.wbt"
    with open(PATH_NEW_FILE,mode='w') as f:
        f.write(updated_data)


