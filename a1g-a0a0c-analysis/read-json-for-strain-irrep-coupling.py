""" Read json file of free energies from varied a1g strain
and a0a0c- rotation angle"""

import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def read_json(fjson):
        with open(fjson) as f:
            return json.load(f)
dict = read_json('data/strain-irrep-coupling/a1g-a0a0c--rotationangle-free-energies/free-energy-data.json')

df = pd.DataFrame.from_dict(dict, orient='columns')
nearest_neigbors_df = pd.read_excel('data/compiled-nearest-neighbor-counting-for-octahedral-rotations.xlsx')

nearest_neigbors_df = nearest_neigbors_df.drop(labels=[11],axis=0) #drop specific configurations from dataset
X = nearest_neigbors_df[['J1xy (tetragonal)','J1z (tetragonal)','J2xy (tetragonal)', 'J2z (tetragonal)', 
                               'J3 (tetragonal)']].to_numpy()
model = LinearRegression()

def get_standard_error(actual_values, estimated_values, X):
                    n = len(actual_values)
                    residuals = actual_values - estimated_values
                    mse = np.sum(residuals ** 2) / (n - X.shape[1] -1)
                    variance_covariance_matrix = mse * np.linalg.inv(np.dot(np.transpose(X), X))
                    standard_error = np.sqrt(np.diagonal(variance_covariance_matrix))
                    return standard_error

desired_column_order = sorted([float(col) for col in df.columns])
desired_column_order[0] = format(desired_column_order[0], '.0f')
desired_column_order = [str(value) for value in desired_column_order]
df = df[desired_column_order]
df.sort_index(inplace=True)

print(df.head())

exchange_data = {}
for column in df:
      exchange_data[column] = {}
      for i, item, in enumerate(df[column]):
            strain = df.index[i]
            exchange_data[column][strain] = {}
            energies = pd.DataFrame.from_dict(df[column][i], orient='index')
            desired_row_order = ['1_atom','2_atom_100','2_atom_110','2_atom_111','3_atom_100_110',
                    '3_atom_100_111','3_atom_101_110','4_atom_1','4_atom_2','4_atom_3',
                    '4_atom_4','4_atom_5','4_atom_6','ferro']
            energies = energies.reindex(desired_row_order)
            energies = energies.drop(labels=['4_atom_5'],axis=0) #drop specific configurations from dataset

            y = energies['free energy'].values

            fit = model.fit(X,y)
            model_parameters = model.coef_
            model_parameters = model_parameters.reshape(-1)
            paramagnetic_energy = model.intercept_
            estimate = model.predict(X)
            score = model.score(X,y)
            #find error of coefficients
            standard_error = get_standard_error(y,estimate,X)

            print(model_parameters[0])

            exchange_data[column][strain]['values'] = {}
            exchange_data[column][strain]['values']['J1xy'] = model_parameters[0]
            exchange_data[column][strain]['values']['J1z'] = model_parameters[1]
            exchange_data[column][strain]['values']['J2xy'] = model_parameters[2]
            exchange_data[column][strain]['values']['J2z'] = model_parameters[3]
            exchange_data[column][strain]['values']['J3'] = model_parameters[4]
            exchange_data[column][strain]['values']['paramagnetic-energy'] = paramagnetic_energy

            exchange_data[column][strain]['error'] = {}
            exchange_data[column][strain]['error']['J1xy'] = standard_error[0]
            exchange_data[column][strain]['error']['J1z'] = standard_error[1]
            exchange_data[column][strain]['error']['J2xy'] = standard_error[2]
            exchange_data[column][strain]['error']['J2z'] = standard_error[3]
            exchange_data[column][strain]['error']['J3'] = standard_error[4]

            exchange_data[column][strain]['R-square'] = score

#DATA_DIR = './'
#fjson = os.path.join(DATA_DIR, "exchange-data.json")

#def write_json(d,fjson):
#        with open(fjson, "w") as f:
#                json.dump(d,f)
#        return d

#write_json(exchange_data, fjson)
        




            

                