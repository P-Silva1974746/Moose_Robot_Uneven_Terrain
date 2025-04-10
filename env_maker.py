import numpy as np
import matplotlib.pyplot as plt
import re

PATH_FILE="worlds/moose_demo.wbt"
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

    modifed_heights=heights/3

    #---------------------CODE TO ALTER THE ELEVATION MAP------------------------#

    formated_heights=',\n '.join(f"{height:.6f}" for height in modifed_heights)

    #replace the data
    updated_data=re.sub(r'(height\s*\[\s*)(.*?)(\s*\])',
                        rf'\1{formated_heights}\3',
                        data,
                        flags=re.DOTALL
    )
    
    with open(PATH_FILE,mode='w') as f:
        f.write(updated_data)


