import numpy as np
import matplotlib.pyplot as plt
import re

PATH_FILE="worlds/moose_perlin.wbt"
with open(file=PATH_FILE,mode='r') as f:
    data= f.read()


match = re.search(r'height\s*\[\s*(.*?)\s]',data, re.DOTALL)

if match:
    # extract string of numbers
    heights_strings=match.group(1)

    heights=np.array([float(s.strip()) for s in heights_strings.split(',') if s.strip()])

    n_entries=len(heights)
    print()
    print(f"Number of entries in the heights: {n_entries}")

    mean=np.mean(heights)
    median=np.median(heights)
    std=np.std(heights)
    min_val=np.min(heights)
    max_value=np.max(heights)

    print()
    print("Statistics:")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std}")
    print(f"Min height: {min_val}")
    print(f"Max height: {max_value}")

    plt.figure(figsize=(8,6))
    plt.hist(heights,bins=30, edgecolor='black', alpha=0.75)
    plt.title("Distribuition of Heights")
    plt.xlabel("Height")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Elevation map
    # dimenstion of our world which is a square
    dim=int(np.sqrt(n_entries))
    if dim*dim!=n_entries:
        raise ValueError(f"The world given is not square, n_entries: {n_entries} is not equal to dim^2: {dim*dim}. Please set xDimension and yDimension manually.")

    height_map= np.zeros((dim,dim))
    for j in range(dim):
        for i in range(dim):
            #index of the height arrays in .wbt files according to the elevation grid documentation of webots
            index= i+j*dim
            height_map[j,i]=heights[index]

    plt.figure(figsize=(8,6))
    cmap=plt.get_cmap('terrain')
    img = plt.imshow(height_map, cmap=cmap, origin='lower')
    plt.colorbar(img, label='Height')
    plt.title("Height Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
else:
    print(f"No data found in the file: {PATH_FILE}")
