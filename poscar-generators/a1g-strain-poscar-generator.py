""" This file generates POSCARs
with varied degrees of A1g strain """

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
    if line.strip() == "direct":
        read_coordinates = True

# Convert the coordinates to a numpy array
a0a0a0_coordinates = np.array(coordinates).reshape(-1, 3)
a0a0a0_coordinates = a0a0a0_coordinates[:40,:]

#define strains
strain = np.arange(1,1.01,0.002)
print(strain)
for i in range(len(strain)):
    poscar = a0a0a0_coordinates
    structure = IStructure(lattice= ((strain[i]*7.8,0,0),(0,strain[i]*7.8,0),(0,0,strain[i]*7.8)),coords=poscar,species=(['Eu','Eu','Eu','Eu','Eu','Eu','Eu','Eu',
                                                                                           'Ti','Ti','Ti','Ti','Ti','Ti','Ti','Ti',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O']))
    structure.to(fmt = 'poscar', filename = f"{strain[i]:.3f}")