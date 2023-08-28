
import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def read_json(fjson):
        with open(fjson) as f:
            return json.load(f)

dict = read_json('data/a0a0a0_a1g_strain_free_energy_for_different_magnetic_configurations_and_strains/a1g-exchange-data-processed.json')

strain = dict['strains']
j1xy = dict['exchanges']['j1xy']
j1z = dict['exchanges']['j1z']
j2xy = dict['exchanges']['j2xy']
j2z = dict['exchanges']['j2z']
j3 = dict['exchanges']['j3']

j1xyerror = dict['error']['j1xyerror']
j1zerror = dict['error']['j1zerror']
j2xyerror = dict['error']['j2xyerror']
j2zerror = dict['error']['j2zerror']
j3error = dict['error']['j3error']


plt.errorbar(strain, j1xy, yerr=j1xyerror, capsize=2, linewidth = 2)
plt.errorbar(strain, j1z, yerr=j1zerror, capsize=2, linewidth = 2)
plt.errorbar(strain, j2xy, yerr=j2xyerror, capsize=2, linewidth = 2)
plt.errorbar(strain, j2z, yerr=j2zerror, capsize=2, linewidth = 2)
plt.errorbar(strain, j3, yerr=j3error, capsize=2, linewidth = 2)
#plt.vlines(x = 5.77,ymin=-1e-4,ymax=1e-4,color = 'black', ls='--')
plt.ylim(-1e-5,3e-6)
plt.legend(['J1 x-y','J1 z','J2 x-y','J2 z', 'J3','relaxed structure angle'], fontsize = 12)
plt.xlabel('Strain %')
plt.ylabel('Exchange value')
plt.show()