import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def read_json(fjson):
        with open(fjson, 'r') as f:
            return json.load(f)
dict = read_json('exchange-data.json')

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


X = np.zeros((36,8))
y_j1xy = np.zeros((36,1))
y_j1z = np.zeros((36,1))
y_j2xy = np.zeros((36,1))
y_j2z = np.zeros((36,1))
y_j3 = np.zeros((36,1))

for i, angle in enumerate(dict):
      for j, strain in enumerate(dict[angle]):
            X[6*i + j][0] = float(strain)-1
            X[6*i + j][1] = float(angle)**2
            X[6*i + j][2] = (float(strain)-1)**2
            X[6*i + j][3] = (float(strain)-1)**3
            X[6*i + j][4] = (float(strain)-1)*float(angle)**2
            X[6*i + j][5] = (float(strain)-1)**4
            X[6*i + j][6] = float(angle)**4
            X[6*i + j][7] = (float(angle)**2)*((float(strain)-1)**2)
            #X[6*i + j][8] = (float(strain)-1)**5
            #X[6*i + j][9] = (float(angle)**2)*((float(strain)-1)**3)
            #X[6*i + j][10] = (float(angle)**4)*(float(strain)-1)

            y_j1xy[6*i + j][0] = dict[angle][strain]['values']['J1xy']
            y_j1z[6*i + j][0] = dict[angle][strain]['values']['J1z']
            y_j2xy[6*i + j][0] = dict[angle][strain]['values']['J2xy']
            y_j2z[6*i + j][0] = dict[angle][strain]['values']['J2z']
            y_j3[6*i + j][0] = dict[angle][strain]['values']['J3']

model = LinearRegression()
fit = model.fit(X,y_j1xy)
model_parameters = model.coef_
intercept = model.intercept_
estimate = model.predict(X)
score = model.score(X,y_j1xy)

plt.plot(X[0:6,0]*100,estimate[0:6])
plt.plot(X[0:6,0]*100,y_j1xy[0:6])
plt.legend(['estimate','actual'])
plt.ylabel('Exchange Energy (eV)')
plt.xlabel('Strain %')
plt.title('J1xy', fontsize = 20)
plt.show()

print("\n----------------------\n")
print("J1xy")
print("Params:", model_parameters)
print("Score:", score)
print("Initial J1xy:", intercept)
print("\n----------------------\n")

fit = model.fit(X,y_j1z)
model_parameters = model.coef_
intercept = model.intercept_
estimate = model.predict(X)
score = model.score(X,y_j1z)

plt.plot(X[0:6,0]*100,estimate[0:6])
plt.plot(X[0:6,0]*100,y_j1z[0:6])
plt.legend(['estimate','actual'])
plt.ylabel('Exchange Energy (eV)')
plt.xlabel('Strain %')
plt.title('J1z', fontsize = 20)
plt.show()

print("J1z")
print("Params:", model_parameters)
print("Score:", score)
print("Initial J1z:", intercept)
print("\n----------------------\n")

fit = model.fit(X,y_j2xy)
model_parameters = model.coef_
intercept = model.intercept_
estimate = model.predict(X)
score = model.score(X,y_j2xy)

plt.plot(X[0:6,0]*100,estimate[0:6])
plt.plot(X[0:6,0]*100,y_j2xy[0:6])
plt.legend(['estimate','actual'])
plt.ylabel('Exchange Energy (eV)')
plt.xlabel('Strain %')
plt.title('J2xy', fontsize = 20)
plt.show()

print("J2xy")
print("Params:", model_parameters)
print("Score:", score)
print("Initial J2xy:", intercept)
print("\n----------------------\n")

fit = model.fit(X,y_j2z)
model_parameters = model.coef_
intercept = model.intercept_
estimate = model.predict(X)
score = model.score(X,y_j2z)

plt.plot(X[0:6,0]*100,estimate[0:6])
plt.plot(X[0:6,0]*100,y_j2z[0:6])
plt.legend(['estimate','actual'])
plt.ylabel('Exchange Energy (eV)')
plt.xlabel('Strain %')
plt.title('J2z', fontsize = 20)
plt.show()

print("J2z")
print("Params:", model_parameters)
print("Score:", score)
print("Initial J2z:", intercept)
print("\n----------------------\n")

fit = model.fit(X,y_j3)
model_parameters = model.coef_
intercept = model.intercept_
estimate = model.predict(X)
score = model.score(X,y_j3)

plt.plot(X[0:6,0]*100,estimate[0:6])
plt.plot(X[0:6,0]*100,y_j3[0:6])
plt.legend(['estimate','actual'])
plt.ylabel('Exchange Energy (eV)')
plt.xlabel('Strain %')
plt.title('J3', fontsize = 20)
plt.show()

print("J3")
print("Params:", model_parameters)
print("Score:", score)
print("Initial J3:", intercept)
print("\n----------------------\n")

fig, ax = plt.subplots(3,2, figsize= (1,1), constrained_layout = True)

psm = ax[0][0].pcolormesh(j1xy_matrix, cmap='viridis', rasterized=True, vmin=-10e-6, vmax=13e-6)
ax[0][0].set_title('J1-xy')
ax[0][0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][0].set_yticklabels(np.arange(0,0.16,0.03))
ax[0][0].set_xticklabels(np.arange(1,1.01,0.002))
ax[0][0].set_xlabel("Strain")
ax[0][0].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[0][0])

psm = ax[0][1].pcolormesh(j1z_matrix, cmap='viridis', rasterized=True, vmin=-10e-6, vmax=13e-6)
ax[0][1].set_title('J1-z')
ax[0][1].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[0][1].set_yticklabels(np.arange(0,0.16,0.03))
ax[0][1].set_xticklabels(np.arange(1,1.01,0.002))
ax[0][1].set_xlabel("Strain")
ax[0][1].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[0][1])

psm = ax[1][0].pcolormesh(j2xy_matrix, cmap='viridis', rasterized=True, vmin=-8.5e-6, vmax=-5.5e-6)
ax[1][0].set_title('J2-xy')
ax[1][0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][0].set_yticklabels(np.arange(0,0.16,0.03))
ax[1][0].set_xticklabels(np.arange(1,1.01,0.002))
ax[1][0].set_xlabel("Strain")
ax[1][0].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[1][0])

psm = ax[1][1].pcolormesh(j2z_matrix, cmap='viridis', rasterized=True, vmin=-8.5e-6, vmax=-5.5e-6)
ax[1][1].set_title('J2-z')
ax[1][1].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][1].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[1][1].set_yticklabels(np.arange(0,0.16,0.03))
ax[1][1].set_xticklabels(np.arange(1,1.01,0.002))
ax[1][1].set_xlabel("Strain")
ax[1][1].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[1][1])

psm = ax[2][0].pcolormesh(j3_matrix, cmap='viridis', rasterized=True)#, vmin=-10e-6, vmax=13e-6)
ax[2][0].set_title('J3')
ax[2][0].set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[2][0].set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
ax[2][0].set_yticklabels(np.arange(0,0.16,0.03))
ax[2][0].set_xticklabels(np.arange(1,1.01,0.002))
ax[2][0].set_xlabel("Strain")
ax[2][0].set_ylabel("Rotation Angle (radians)")
fig.colorbar(psm, ax=ax[2][0])

for ax in ax.flat:
      if not bool(ax.has_data()):
            fig.delaxes(ax)
            
plt.show()

