import os, json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

X = np.zeros((36,5))
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
            #X[6*i + j][5] = (float(strain)-1)**4
            #X[6*i + j][6] = float(angle)**4
            #X[6*i + j][7] = (float(angle)**2)*((float(strain)-1)**2)
            #X[6*i + j][8] = (float(strain)-1)**5
            #X[6*i + j][9] = (float(angle)**2)*((float(strain)-1)**3)
            #X[6*i + j][10] = (float(angle)**4)*(float(strain)-1)

            y_j1xy[6*i + j][0] = dict[angle][strain]['values']['J1xy']
            y_j1z[6*i + j][0] = dict[angle][strain]['values']['J1z']
            y_j2xy[6*i + j][0] = dict[angle][strain]['values']['J2xy']
            y_j2z[6*i + j][0] = dict[angle][strain]['values']['J2z']
            y_j3[6*i + j][0] = dict[angle][strain]['values']['J3']

X = MinMaxScaler().fit_transform(X)

def plot_3d_components(
    data_array,
    feature_labels,
    title,
    col_gap=1,
    row_gap=1,
    bar_width=0.5,
    cmap_name="viridis_r"
):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(gs[0], projection="3d")

    num_rows = len(data_array)
    print(data_array)
    print('--------------------')
    num_columns = max(len(row) for row in data_array)

    x_positions = np.arange(num_columns) + 1
    z_positions = np.arange(num_rows) + 1
    x_center_positions = x_positions + (num_columns - 1) * col_gap / 2
    z_center_positions = z_positions + (num_rows - 1) * row_gap / 2

    cmap = plt.get_cmap(cmap_name)

    for z, row in enumerate(data_array):
        for x, value in enumerate(row):
            color = cmap(z / (num_rows - 1))

            ax1.bar3d(
                x_center_positions[x],
                z_center_positions[z],
                0,
                bar_width,
                bar_width,
                abs(value),
                color=color,
                shade=True,
                alpha=0.75,
                edgecolor="black",
            )

            if value < 1e-10:
                 ax1.bar3d(
                x_center_positions[x],
                z_center_positions[z],
                0,
                bar_width,
                bar_width,
                abs(value),
                color='white',
                shade=True,
                alpha=0.75,
                edgecolor="none",
            )
    ax1.set_xlabel("Features", labelpad=20)
    ax1.set_ylabel("Polynomial Order", labelpad=20)
    ax1.set_zlabel("Magnitude", labelpad=20)

    # Customize ticks and labels
    ax1.set_xticks(x_center_positions + bar_width - col_gap / 2)
    ax1.set_xticklabels(feature_labels, rotation=0)
    ax1.set_yticks(z_center_positions + bar_width - row_gap / 2)
    ax1.set_yticklabels(range(1,4), va="center")
    ax1.set_zticks(np.arange(0, 1, 0.1))
    ax1.set_zlim(0, 1)
    ax1.tick_params(axis="x", pad=10)
    ax1.tick_params(axis="z", pad=10)
    ax1.tick_params(axis="y", pad=10)
    ax1.set_title(title, pad=30)
    ax1.view_init(elev=45, azim=-60)

    plt.show()
    return None

j1xy_param_values = []
j1z_param_values = []
j2xy_param_values = []
j2z_param_values = []
j3_param_values = []
list_of_lists = [j1xy_param_values,j1z_param_values,j2xy_param_values,j2z_param_values,j3_param_values]
list_of_targets = [y_j1xy, y_j1z, y_j2xy, y_j2z, y_j3]
number_of_included_coefficients = [1,3,5]
model = LinearRegression()

for k, lists in enumerate(list_of_lists):

    for i,item in enumerate(number_of_included_coefficients):
        fit = model.fit(X[:,:item],list_of_targets[k])
        model_parameters = model.coef_
        intercept = model.intercept_
        estimate = model.predict(X[:,:item])
        score = model.score(X[:,:item],list_of_targets[k])
        list_of_lists[k].append(list(model_parameters[0]))

        estimate_matrix = np.zeros((6,6))
        for i in range(6):
            estimate_matrix[i,:] = estimate[6*i:6*i+6,0]
        if k == 4:
            plt.plot([0,0.2,0.4,0.6,0.8,1],list_of_targets[k][:6])
            plt.plot([0,0.2,0.4,0.6,0.8,1],estimate_matrix[0,0:6])
            plt.legend(['actual','estimate'])
            plt.title('J3', fontsize = 20)
            plt.show()

            fig, ax = plt.subplots(1,1, figsize= (1,1), constrained_layout = True)
            psm = ax.pcolormesh(estimate_matrix, cmap='viridis', rasterized=True)#, vmin=-10e-6, vmax=13e-6)
            ax.set_title('J3')
            ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
            ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
            ax.set_yticklabels(np.arange(0,0.16,0.03))
            ax.set_xticklabels(np.arange(1,1.01,0.002))
            ax.set_xlabel("Strain")
            ax.set_ylabel("Rotation Angle (radians)")
            fig.colorbar(psm, ax=ax)
            plt.show()
        
        for i,item in enumerate(list_of_lists[k]):
            while len(list_of_lists[k][i]) < 5:
                list_of_lists[k][i].append(0)
            for j, thing in enumerate(list_of_lists[k][i]):
                list_of_lists[k][i][j] = abs(list_of_lists[k][i][j])
            max_num = max(list_of_lists[k][i])
            for j, thing, in enumerate(list_of_lists[k][i]):
                list_of_lists[k][i][j] = list_of_lists[k][i][j]/max_num
    if k > 10:
        plot_3d_components(
        list_of_lists[k],
        ['$\epsilon$','$\phi^2$','$\epsilon^2$','$\epsilon^3$',
        '$\epsilon$$\phi^2$'],
        title = "J($\Gamma_1^+$, $R_5^-$) Polynomial Coefficients")
