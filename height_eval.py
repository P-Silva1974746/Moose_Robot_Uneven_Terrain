import numpy as np
import matplotlib.pyplot as plt
import re

PATH_FILE="worlds/moose_demo.wbt"
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
else:
    print(f"No data found in the file: {PATH_FILE}")
