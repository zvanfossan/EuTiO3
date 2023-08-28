
""" This file generates POSCARs
with varied degrees of GM3+ (a,0) strain """

import numpy as np
import math
from pymatgen.core.structure import IStructure

# Read the a0a0a0 POSCAR file
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
a0a0a0_coordinates = np.array(coordinates).reshape(-1, 3)
a0a0a0_coordinates = a0a0a0_coordinates[:40,:]

#define strains
zstrain = np.arange(0.99,1,0.002)
xystrain = list(reversed(np.arange(1,1.0055,0.001)))
for i in range(len(xystrain)):
    xystrain[i] = float(f'{xystrain[i]:.6f}')

for i in range(len(zstrain)):
    poscar = a0a0a0_coordinates
    structure = IStructure(lattice= ((xystrain[i]*7.8,0,0),(0,xystrain[i]*7.8,0),(0,0,zstrain[i]*7.8)),coords=poscar,species=(['Eu','Eu','Eu','Eu','Eu','Eu','Eu','Eu',
                                                                                           'Ti','Ti','Ti','Ti','Ti','Ti','Ti','Ti',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O']))
    structure.to(fmt = 'poscar', filename = f"{zstrain[i]:.3f}")