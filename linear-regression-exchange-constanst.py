from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('EuTiO3-exchange-counting.csv')
data_1 = data.drop(labels=[11], axis=0)
exchange_coefficients = data_1[['J100 coeff double', 'J110 coeff double', 'J111 coeff double', 'e0 coeff']].to_numpy()
free_energy = data_1['free energy'].to_numpy()

model = LinearRegression(fit_intercept=False).fit(exchange_coefficients, free_energy)
params = model.coef_
intercept = model.intercept_
estimate = model.predict(exchange_coefficients)

line_x = np.array([-1000,0])
line_y = np.array([-1000,0])

plt.plot(free_energy, estimate, linestyle = 'none', marker = 'o')
plt.plot(line_x,line_y,linestyle = 'dashed', linewidth = '1', color = 'black')
plt.ticklabel_format(style='sci', useOffset=False)
plt.xlim(-398.29,-398.27)
plt.ylim(-398.29,-398.27)
plt.xlabel("DFT calculated energy")
plt.ylabel("Linear regression prediction")
plt.show()

print(params)
print(data_1)