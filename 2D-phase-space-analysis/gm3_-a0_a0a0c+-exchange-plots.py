import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def read_json(fjson):
        with open(fjson, 'r') as f:
            return json.load(f)
dict = read_json('data/strain-irrep-coupling/gm3_-a0_a0a0c+exchange-values(1).json')

j1xy_matrix = np.zeros((6,6))
j1z_matrix = np.zeros((6,6))
j2xy_matrix = np.zeros((6,6))
j2z_matrix = np.zeros((6,6))
j3_matrix = np.zeros((6,6))
scores = np.zeros((6,6))

for i, angle in enumerate(dict):
      for j, strain in enumerate(dict[angle]):
           j1xy_matrix[i][j] = dict[angle][strain]['values']['J1xy']
           j1z_matrix[i][j] = dict[angle][strain]['values']['J1z']
           j2xy_matrix[i][j] = dict[angle][strain]['values']['J2xy']
           j2z_matrix[i][j] = dict[angle][strain]['values']['J2z']
           j3_matrix[i][j] = dict[angle][strain]['values']['J3']
           scores[i][j] = dict[angle][strain]['R-square']
           if scores[i][j] <= 0.999:
                print('outliers present')

fig, ax = plt.subplots(3,2, figsize= (1,1), constrained_layout = True)

psm = ax[0][0].pcolormesh(j1xy_matrix, cmap='viridis', rasterized=True, vmin=-1e-5, vmax=1e-5)
ax[0][0].set_title('J1-xy')
ax[0][0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][0].set_yticklabels(np.arange(0,0.16,0.03))
ax[0][0].set_xticklabels([0.99,0.992,0.994,0.996,0.998,1])
ax[0][0].set_xlabel("Strain")
ax[0][0].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[0][0])

psm = ax[0][1].pcolormesh(j1z_matrix, cmap='viridis', rasterized=True, vmin=-1e-5, vmax=1e-5)
ax[0][1].set_title('J1-z')
ax[0][1].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][1].set_yticklabels(np.arange(0,0.16,0.03))
ax[0][1].set_xticklabels([0.99,0.992,0.994,0.996,0.998,1])
ax[0][1].set_xlabel("Strain")
ax[0][1].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[0][1])

psm = ax[1][0].pcolormesh(j2xy_matrix, cmap='viridis', rasterized=True, vmin=-8.5e-6, vmax=-6e-6)
ax[1][0].set_title('J2-xy')
ax[1][0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][0].set_yticklabels(np.arange(0,0.16,0.03))
ax[1][0].set_xticklabels([0.99,0.992,0.994,0.996,0.998,1])
ax[1][0].set_xlabel("Strain")
ax[1][0].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[1][0])

psm = ax[1][1].pcolormesh(j2z_matrix, cmap='viridis', rasterized=True, vmin=-8.5e-6, vmax=-6e-6)
ax[1][1].set_title('J2-z')
ax[1][1].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][1].set_yticklabels(np.arange(0,0.16,0.03))
ax[1][1].set_xticklabels([0.99,0.992,0.994,0.996,0.998,1])
ax[1][1].set_xlabel("Strain")
ax[1][1].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[1][1])

psm = ax[2][0].pcolormesh(j3_matrix, cmap='viridis', rasterized=True)#, vmin=-10e-6, vmax=13e-6)
ax[2][0].set_title('J3')
ax[2][0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[2][0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[2][0].set_yticklabels(np.arange(0,0.16,0.03))
ax[2][0].set_xticklabels([0.99,0.992,0.994,0.996,0.998,1])
ax[2][0].set_xlabel("Strain")
ax[2][0].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[2][0])

for ax in ax.flat:
      if not bool(ax.has_data()):
            fig.delaxes(ax)
            
plt.show()