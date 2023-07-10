"""This file generates POSCAR structures 
for octahedral rotation phases 
with varied rotation angles"""
import numpy as np
import math

# Read the POSCAR file
with open('POSCAR', 'r') as file:
    lines = file.readlines()

# Extract fractional coordinates from the POSCAR file
coordinates = []
read_coordinates = False
for line in lines:
    if read_coordinates:
        coords = [float(coord) for coord in line.split()]
        coordinates.extend(coords)
    if line.strip() == "Direct":
        read_coordinates = True

# Convert the coordinates to a numpy array
coordinates = np.array(coordinates).reshape(-1, 3)
coordinates = coordinates[:40,:]

# Print the resulting array
print(coordinates)

#define the rotations and convert the angles to translation vectors for oct. rots.
angles = np.arange(0.01,0.16,0.01)

translations = []
for angle in angles:
    translations.append(math.tan(angle)*0.25)

#construct arrays with distortions for a0a0c+
r1 = np.zeros()