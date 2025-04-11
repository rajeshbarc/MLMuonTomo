import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import uproot
import scienceplots

plt.style.use(['science','ieee','no-latex'])


# Load data
file = uproot.open("MLData.root")
tree = file["mlData"]
df = tree.arrays(library="pd")

# Select features and target
features = [
    'x1', 'y1', 'z1', 'x2', 'y2', 'z2',
    'x3', 'y3', 'z3', 'x4', 'y4', 'z4',
    'pathLength', 'angleIncoming', 'angleOutgoing', 'deviation'
]
target = 'g4Momentum'

# Filter data
dataFilter = df[features + [target]].dropna()
filtered_data = dataFilter[
    (dataFilter['deviation'] > 0e-3) &
    (dataFilter['g4Momentum'] != 0)
    ]


# Feature engineering
def engineeredFeatures(df):
    df = df.copy()
    # Coordinate differences
    df['dx_12'] = df['x2'] - df['x1']
    df['dy_12'] = df['y2'] - df['y1']
    df['dz_12'] = df['z2'] - df['z1']
    df['dx_34'] = df['x4'] - df['x3']
    df['dy_34'] = df['y4'] - df['y3']
    df['dz_34'] = df['z4'] - df['z3']

    # Distances
    #df['dist_12'] = np.sqrt(df['dx_12'] ** 2 + df['dy_12'] ** 2 + df['dz_12'] ** 2)
    #df['dist_34'] = np.sqrt(df['dx_34'] ** 2 + df['dy_34'] ** 2 + df['dz_34'] ** 2)
    df['delta_x'] = (df['x4'] - df['x3']) - (df['x2'] - df['x1'])
    df['delta_y'] = (df['y4'] - df['y3']) - (df['y2'] - df['y1'])
    df['delta_z'] = (df['z4'] - df['z3']) - (df['z2'] - df['z1'])
    df['delta_r'] = np.sqrt(df['delta_x'] ** 2 + df['delta_y'] ** 2 + df['delta_z'] ** 2)
    df['delta_r_norm'] = df['delta_r'] / (df['pathLength'] + 1e-6)

    # Trajectory angles
    df['theta_12'] = np.arctan2(np.sqrt(df['dx_12'] ** 2 + df['dy_12'] ** 2), df['dz_12'])
    df['theta_34'] = np.arctan2(np.sqrt(df['dx_34'] ** 2 + df['dy_34'] ** 2), df['dz_34'])

    # Scattering features
    df['cos_deviation'] = np.cos(df['deviation'])
    df['sin_deviation'] = np.sin(df['deviation'])
    #df['inv_deviation'] = 1 / (df['deviation'] + 1e-6)
    df['inv_sin_dev_half'] = 1 / (np.sin(df['deviation'] / 2) + 1e-6)
    #df['sqrt_deviation'] = np.sqrt(df['deviation'])

    # Angular transformations
    df['cos_incoming'] = np.cos(df['angleIncoming'])
    df['cos_outgoing'] = np.cos(df['angleOutgoing'])
    df['angle_diff'] = df['angleOutgoing'] - df['angleIncoming']
    df['log_momentum'] = np.log(df['g4Momentum'])
    df['cos2_deviation'] = np.cos(df['deviation'])**2
    df['sin2_deviation'] = np.sin(df['deviation'])**2
    df['angle_ratio'] = df['angle_diff'] / (df['deviation'] + 1e-6)
    # Scattering point (assume target between detectors 2 and 3)
    #df['scatter_x'] = (df['x2'] + df['x3']) / 2
    #df['scatter_y'] = (df['y2'] + df['y3']) / 2
    #df['scatter_z'] = (df['z2'] + df['z3']) / 2
    return df


filtered_data = engineeredFeatures(filtered_data)

# Updated feature list
features = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4',
    'pathLength', 'angleIncoming', 'angleOutgoing', 'deviation']

engineeredFeat =[ 'deviation','theta_12', 'theta_34', 'cos_deviation', 'sin_deviation','cos_incoming', 'cos_outgoing', 'angle_diff', 'cos2_deviation',
    'sin2_deviation','delta_r','delta_r_norm','angle_ratio','log_momentum']

# Correlation analysis of Input data
correlation_matrix = filtered_data[features + [target]].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', vmin=-1, vmax=1, annot_kws={"size": 2})
plt.title('Correlation Matrix of Input Data', pad=0.5, fontsize=4)
plt.xticks(rotation=90, fontsize=4)
plt.yticks(rotation=0, fontsize=4)
plt.tick_params(axis='both', which='minor', bottom=False, left=False, top=False, right=False)
plt.tick_params(axis='both', which='major', bottom=False, left=False, top=False, right=False)
plt.tight_layout()
plt.show()
#print("Correlation with g4Momentum:\n", correlation_matrix['g4Momentum'].sort_values(ascending=False))
# Correlation analysis with Engineered data
correlation_matrix = filtered_data[engineeredFeat + [target]].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', vmin=-1, vmax=1, annot_kws={"size": 2})
plt.title('Correlation Matrix of Engineered Data', pad=0.5, fontsize=4)
plt.xticks(rotation=90, fontsize=4)
plt.yticks(rotation=0, fontsize=4)
plt.tick_params(axis='both', which='minor', bottom=False, left=False, top=False, right=False)
plt.tick_params(axis='both', which='major', bottom=False, left=False, top=False, right=False)
plt.tight_layout()
plt.show()
