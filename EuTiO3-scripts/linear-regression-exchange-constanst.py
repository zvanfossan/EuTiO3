from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

data = pd.read_csv('EuTiO3-exchange-counting.csv')
exchange_coefficients = data[['J100 coeff', 'J110 coeff', 'J111 coeff']].to_numpy()
free_energy = data[['free energy']].to_numpy()

model = LinearRegression().fit(exchange_coefficients, free_energy)
params = model.coef_

print(params)