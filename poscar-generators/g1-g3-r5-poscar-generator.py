"""This file generates POSCAR structures 
for octahedral rotation phases 
with varied rotation angles"""
import os, shutil, math
import numpy as np
from pymatgen.core.structure import IStructure

def poscar_generator(parent_directory):
    # Read the a0a0a0 POSCAR file
    with open('poscars/gm3_-a0_gm3_0a_r5-_gm1_poscars/a0a0a0','r') as file:
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
    with open('poscars/gm3_-a0_gm3_0a_r5-_gm1_poscars/a0a0c-', 'r') as file:
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
#construct basis matrix for a0a0c+/-

    a0a0cplus_basis = a0a0cplus_coordinates - a0a0a0_coordinates

    for i in range(len(a0a0cplus_basis)):
        for j in range(len(a0a0cplus_basis[0])):
            if a0a0cplus_basis[i][j] != 0:
                a0a0cplus_basis[i][j] = a0a0cplus_basis[i][j] / abs(a0a0cplus_basis[i][j])

    for directory in os.listdir(parent_directory):
        directory_path = os.path.join(parent_directory, directory)

        if os.path.isdir(directory_path):
            for subdirectory in os.listdir(directory_path):
                subdirectory_path = os.path.join(parent_directory, directory, subdirectory)

                if os.path.isdir(subdirectory_path):
                    for subsubdirectory in os.listdir(subdirectory_path):
                        subsubdirectory_path = os.path.join(parent_directory, directory, 
                                                            subdirectory, subsubdirectory)

                        if os.path.isdir(subsubdirectory_path):
                            for threesubdirectory in os.listdir(subsubdirectory_path):
                                threesubdirectory_path = os.path.join(parent_directory, directory,
                                                                      subdirectory, subsubdirectory,
                                                                      threesubdirectory)
                                
                                if os.path.isdir(threesubdirectory_path):
                                    plane_strain_z = float(directory)
                                    plane_strain_xy = 1 + (1 - float(directory))/2
                                    x_strain = float(subdirectory)
                                    y_strain = 2 - float(subdirectory)
                                    a1g = float(threesubdirectory)
                                    translation = math.tan(float(subsubdirectory))*0.25
                                             
                                    poscar = translation*a0a0cplus_basis + a0a0a0_coordinates
                                    structure = IStructure(lattice= ((a1g*plane_strain_xy*x_strain*7.8,0,0),
                                                                         (0,a1g*plane_strain_xy*y_strain*7.8,0),
                                                                         (0,0,a1g*plane_strain_z*7.8)),
                                                           coords=poscar,
                                                           species=(['Eu','Eu','Eu','Eu','Eu','Eu','Eu','Eu',
                                                                    'Ti','Ti','Ti','Ti','Ti','Ti','Ti','Ti',
                                                                    'O','O','O','O','O','O','O','O',
                                                                    'O','O','O','O','O','O','O','O',
                                                                    'O','O','O','O','O','O','O','O']))
                                    structure.to(fmt = 'poscar', filename = 'POSCAR')
                                    destination_folder = os.path.join(threesubdirectory_path)
                                    os.path.basename('POSCAR')
                                    shutil.copy('POSCAR', destination_folder)

parent_directory = 'poscars/gm3_-a0_gm3_0a_r5-_gm1_poscars'
poscar_generator(parent_directory)

 