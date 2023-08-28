
""" This file generates POSCARs
with varied degrees of GM3+ (0,a) strain """

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

#define strains
xstrain = np.arange(0.99,1.01,0.001)
ystrain = list(reversed(xstrain))
for i in range(len(ystrain)):
    ystrain[i] = float(f'{ystrain[i]:.6f}')

for i in range(len(xstrain)):
    poscar = a0a0a0_coordinates
    structure = IStructure(lattice= ((xstrain[i]*7.8,0,0),(0,ystrain[i]*7.8,0),(0,0,7.8)),coords=poscar,species=(['Eu','Eu','Eu','Eu','Eu','Eu','Eu','Eu',
                                                                                           'Ti','Ti','Ti','Ti','Ti','Ti','Ti','Ti',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O']))
    structure.to(fmt = 'poscar', filename = f"{xstrain[i]:.3f}")