"""Calculating exchange constants for EuTiO3"""
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EuTiO3-exchange-counting-5.6.csv')
data_1 = data.drop(labels=[11], axis=0) #chose which rows to drop
exchange_coefficients = data_1[['J100 coeff double', 'J110 coeff double', 
                                'J111 coeff double', 'e0 coeff']].to_numpy()
free_energy = data_1['free energy'].to_numpy()
free_energy_offset = np.zeros(shape=(len(free_energy)))
estimate_offset = np.zeros(shape=(len(free_energy)))

model = LinearRegression(fit_intercept=False).fit(exchange_coefficients, free_energy)
params = model.coef_
intercept = model.intercept_
estimate = model.predict(exchange_coefficients)

line_x = np.array([-1000,1000])
line_y = np.array([-1000,1000])

for i in range(len(free_energy_offset)):
    free_energy_offset[i] = (free_energy[i] - free_energy[len(free_energy)-2])*10**3 #choose which row to base free energies on
    estimate_offset[i] = (estimate[i] - estimate[len(estimate)-2])*10**3

plt.plot(free_energy_offset, estimate_offset, linestyle = 'none', marker = 'o')
plt.plot(line_x,line_y,linestyle = 'dashed', linewidth = '1', color = 'black')
plt.ticklabel_format(style='sci', useOffset=False)
plt.xlim(-1,20)
plt.ylim(-1,20)
plt.xlabel("DFT calculated energy (meV)")
plt.ylabel("Linear regression prediction (meV)")
plt.show()


print('\n------------------------------------------\n')
print("Model Parameters:", params)
print('\n------------------------------------------')
