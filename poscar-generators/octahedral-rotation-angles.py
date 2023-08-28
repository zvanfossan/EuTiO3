"""This file generates POSCAR structures 
for octahedral rotation phases 
with varied rotation angles"""
import numpy as np
import math
from pymatgen.core.structure import IStructure

# Read the a0a0a0 POSCAR file
with open('a0a0a0', 'r') as file:
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
a0a0a0_coordinates = np.array(coordinates).reshape(-1, 3)
a0a0a0_coordinates = a0a0a0_coordinates[:40,:]

# Read the a0a0c- POSCAR file
with open('a0a0c-', 'r') as file:
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
a0a0cplus_coordinates = np.array(coordinates).reshape(-1, 3)
a0a0cplus_coordinates = a0a0cplus_coordinates[:40,:]

#define the rotations and convert the angles to translation vectors for oct. rots.
angles = np.arange(0.01,0.16,0.01)
print("Poscars created with following angles: ",angles)
translations = []
for angle in angles:
    translations.append(math.tan(angle)*0.25)
#construct basis matrix for a0a0c+

a0a0cplus_basis = a0a0cplus_coordinates - a0a0a0_coordinates

for i in range(len(a0a0cplus_basis)):
    for j in range(len(a0a0cplus_basis[0])):
        if a0a0cplus_basis[i][j] != 0:
            a0a0cplus_basis[i][j] = a0a0cplus_basis[i][j] / abs(a0a0cplus_basis[i][j])


for i in range(len(translations)):
    poscar = translations[i]*a0a0cplus_basis + a0a0a0_coordinates
    structure = IStructure(lattice= ((7.8,0,0),(0,7.8,0),(0,0,7.8)),coords=poscar,species=(['Eu','Eu','Eu','Eu','Eu','Eu','Eu','Eu',
                                                                                           'Ti','Ti','Ti','Ti','Ti','Ti','Ti','Ti',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O']))
    structure.to(fmt = 'poscar', filename = f"{angles[i]:.3f}")