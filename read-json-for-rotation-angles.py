"""Import a json file of a dictionary containing keys that correspond to
the rotation angles for a specific space group as well as the different 
magnetic configurations to which free energy values are attached.

This script will convert the json to an array from which each column
will be used for a multiple linear regression.

Additionally, nearest-neighbors counting data will be indexed so that
multiple linear regression can be performed for each octahedral rotation angle.

Make sure to operate this file from the EuTiO3 directory in the research-github"""

import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def read_json(fjson):
        with open(fjson) as f:
            return json.load(f)

dict = read_json('data/a0a0c+_magnetic_configurations_free_energy/free-energy-data(1).json')
df = pd.DataFrame.from_dict(dict, orient='columns')

nearest_neigbors_df = pd.read_excel('data/compiled-nearest-neighbor-counting-for-octahedral-rotations.xlsx')
#cubic_df = pd.read_excel('data/EuTiO3-exchange-counting-cubic.xlsx') #add column from separate file for cubic phase free energies

def convert_data_frame_values_to_floats_and_orient_rows(df):
      for column in df:
            for i in range(len(df[column])):
                  df[column][i] = df[column][i]['free energy']
      desired_row_order = ['1_atom','2_atom_100','2_atom_110','2_atom_111','3_atom_100_110',
                       '3_atom_100_111','3_atom_101_110','4_atom_1','4_atom_2','4_atom_3',
                       '4_atom_4','4_atom_5','4_atom_6','ferro']
      df = df.reindex(desired_row_order)
      return df
df = convert_data_frame_values_to_floats_and_orient_rows(df)

desired_column_order = sorted([float(col) for col in df.columns])
desired_column_order[0] = format(desired_column_order[0], '.2f')
desired_column_order = [str(value) for value in desired_column_order]
df = df[desired_column_order]

df = df.drop(labels=['4_atom_5'], axis=0) #drop specific configurations from dataset
nearest_neigbors_df = nearest_neigbors_df.drop(labels=[11],axis=0) #drop specific configurations from dataset
print(df)
print(nearest_neigbors_df.head())

#now perform multiple linear regression using the structured datasets
j1xy = []
j1z = []
j2xy = []
j2z = []
j3 = []
j1xyerror = []
j1zerror = []
j2xyerror = []
j2zerror = []
j3error = []
angle = []
paramag_energy = []
r_square = []
model = LinearRegression()

for rotation_angle in df.columns.tolist():
      y = df[rotation_angle].to_numpy()
      X = nearest_neigbors_df[['J1xy (tetragonal)','J1z (tetragonal)','J2xy (tetragonal)', 'J2z (tetragonal)', 
                               'J3 (tetragonal)']].to_numpy()
      fit = model.fit(X,y)
      model_parameters = model.coef_
      paramagnetic_energy = model.intercept_
      estimate = model.predict(X)
      score = model.score(X,y)

      #find error of coefficients
      def get_standard_error(actual_values, estimated_values, X):
            n = len(actual_values)
            residuals = actual_values - estimated_values
            mse = np.sum(residuals ** 2) / (n - X.shape[1] -1)
            variance_covariance_matrix = mse * np.linalg.inv(np.dot(np.transpose(X), X))
            standard_error = np.sqrt(np.diagonal(variance_covariance_matrix))
            return standard_error
      
      standard_error = get_standard_error(y,estimate,X)

      print('\n------------------------------------------\n')
      print("Rotation angle:", rotation_angle)
      print("\nModel Parameters:", model_parameters)
      print("\nParamagnetic Energy:", paramagnetic_energy)
      print("\nR-square: ", score)
      print("\nerror: ", standard_error)

      j1xy.append(model_parameters[0])
      j1z.append(model_parameters[1])
      j2xy.append(model_parameters[2])
      j2z.append(model_parameters[3])
      j3.append(model_parameters[4])
      j1xyerror.append(standard_error[0])
      j1zerror.append(standard_error[1])
      j2xyerror.append(standard_error[2])
      j2zerror.append(standard_error[3])
      j3error.append(standard_error[4])
      angle.append(rotation_angle)
      paramag_energy.append(paramagnetic_energy)
      r_square.append(score)

      #parity plot data formatting 
      minimum_energy_index = np.where(y == min(y))[0][0]
      free_energy_offset = np.zeros(shape=len(y))
      estimate_offset = np.zeros(shape=len(y))
      for i in range(len(free_energy_offset)):
            free_energy_offset[i] = (y[i] - y[minimum_energy_index])*10**3
            estimate_offset[i] = (estimate[i] - estimate[minimum_energy_index])*10**3

      line_x = np.array([-1000,1000])
      line_y = np.array([-1000,1000])
      #plt.plot(free_energy_offset, estimate_offset, linestyle = 'none', marker = 'o')
      #plt.plot(line_x,line_y,linestyle = 'dashed', linewidth = '1', color = 'black')
      #plt.ticklabel_format(style='sci', useOffset=False)
      #plt.xlim(-5,max(free_energy_offset)+8)
      #plt.ylim(-5,max(free_energy_offset)+8)
      #plt.xlabel("DFT calculated energy (meV)")
      #plt.ylabel("Linear regression prediction (meV)")
      #plt.show()

plt.errorbar(angle, j1xy, yerr=j1xyerror, capsize=3, linewidth = 3)
plt.errorbar(angle, j1z, yerr=j1zerror, capsize=3, linewidth = 3)
plt.errorbar(angle,j2xy, yerr=j2xyerror, capsize=3, linewidth = 3)
plt.errorbar(angle,j2z, yerr=j2zerror, capsize=3, linewidth = 3)
plt.errorbar(angle,j3, yerr=j3error, capsize=3, linewidth = 3)
#plt.vlines(x = 5.77,ymin=-1e-4,ymax=1e-4,color = 'black', ls='--')
#plt.ylim(-1e-5,3e-5)
plt.legend(['J1 x-y','J1 z','J2 x-y','J2 z', 'J3','relaxed structure angle'], fontsize = 12)
plt.xlabel('Octahedral rotation angle (radians)')
plt.ylabel('Exchange value')
plt.show()

print(j1zerror)

exchange_value_data = {}
exchange_value_data['angles'] = angle
exchange_value_data['J1xy'] = j1xy
exchange_value_data['J1xy error'] = j1xyerror
exchange_value_data['J1z'] = j1z
exchange_value_data['J1z error'] = j1zerror
exchange_value_data['J2xy'] = j2xy
exchange_value_data['J2xy error'] = j2xyerror
exchange_value_data['J2z'] = j2z
exchange_value_data['J2z error'] = j2zerror
exchange_value_data['J3'] = j3
exchange_value_data['J3 error'] = j3error
exchange_value_data['score'] = r_square
exchange_value_data['Paramagnetic Energy'] = paramag_energy

#DATA_DIR = './'
#fjson = os.path.join(DATA_DIR, "exchange-data.json")

#def write_json(d,fjson):
#        with open(fjson, "w") as f:
#                json.dump(d,f)
#        return d

#write_json(exchange_value_data, fjson)






