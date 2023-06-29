"""Calculating exchange constants for EuTiO3"""
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EuTiO3-exchange-counting-a0a0c-.csv')
data_1 = data.drop(labels=[1], axis=0) #choose which rows to drop

exchange_coefficients = data_1[['J100 coeff double', 'J110 coeff double',
                                'J111 coeff double','J200 coeff double']].to_numpy()
free_energy = data_1['free energy'].to_numpy()
configurations = data_1['Configuration'].to_numpy()

model = LinearRegression().fit(exchange_coefficients, free_energy)
params = model.coef_
intercept = model.intercept_
estimate = model.predict(exchange_coefficients)

min_energy_index = np.where(free_energy == min(free_energy))[0][0]
free_energy_offset = np.zeros(shape=len(free_energy))
estimate_offset = np.zeros(shape=len(free_energy))
for i in range(len(free_energy_offset)):
    free_energy_offset[i] = (free_energy[i] - min(free_energy))*10**3     #free energies with respect to minimum value
    estimate_offset[i] = (estimate[i] - estimate[min_energy_index])*10**3

print('\n------------------------------------------\n')
print("Model Parameters:", params)
print("\nParamagnetic Energy:", model.intercept_)
print('\nMinimum energy configuration:', configurations[min_energy_index])
print('\n------------------------------------------\n')

line_x = np.array([-1000,1000])
line_y = np.array([-1000,1000])
plt.plot(free_energy_offset, estimate_offset, linestyle = 'none', marker = 'o')
plt.plot(line_x,line_y,linestyle = 'dashed', linewidth = '1', color = 'black')
plt.ticklabel_format(style='sci', useOffset=False)
plt.xlim(-1,max(free_energy_offset)+3)
plt.ylim(-1,max(free_energy_offset)+3)
plt.xlabel("DFT calculated energy (meV)")
plt.ylabel("Linear regression prediction (meV)")
#plt.show()