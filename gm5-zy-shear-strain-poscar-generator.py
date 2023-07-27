
""" This file generates POSCARs
with varied degrees of GM5+ (a,0,0) strain """

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

#define angle of shear strain
strain_angle = np.arange(0,0.2501,0.01)


for i in range(len(strain_angle)):
    poscar = a0a0a0_coordinates
    structure = IStructure(lattice= ((7.8,0,0),(0,7.8,0),(0,7.8*math.sin(strain_angle[i]),7.8*math.cos(strain_angle[i]))),coords=poscar,species=(['Eu','Eu','Eu','Eu','Eu','Eu','Eu','Eu',
                                                                                           'Ti','Ti','Ti','Ti','Ti','Ti','Ti','Ti',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O',
                                                                                           'O','O','O','O','O','O','O','O']))
    structure.to(fmt = 'poscar', filename = f"{strain_angle[i]:.3f}")