import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

## Open Root file
train_file = uproot.open("trainMay.root")
test_file = uproot.open("testMay.root")

## Read required Trees
train_tTree = train_file["groundTruthPoCA"]
test_tTree = test_file["groundTruthPoCA"]

## Drop Na
train_df = pd.DataFrame(train_tTree.arrays(library="pd")).dropna()
test_df = pd.DataFrame(test_tTree.arrays(library="pd")).dropna()

## Filtering Train dataset
train_df = train_df[(train_df['angleDev'] > 10e-3) & (train_df['pocaX'] > -500) & (train_df['pocaY'] > -500) & (train_df['pocaZ'] > -500)]

##Segregating Features and labels
muonFeatures = ['inX', 'inY', 'inZ', 'dInX', 'dInY', 'dInZ', 'outX', 'outY', 'outZ', 'dOutX', 'dOutY', 'dOutZ', 'angleDev']
pocaLabel = ['pocaX','pocaZ']
print(f'Features: {muonFeatures} and Labels: {pocaLabel}')


# Prepare data
X_train = np.column_stack([train_df[i] for i in muonFeatures])
y_train = np.column_stack([train_df[i] for i in pocaLabel])
X_test = np.column_stack([test_df[i] for i in muonFeatures])
y_test = np.column_stack([test_df[i] for i in pocaLabel])


# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8,subsample=0.8,reg_lambda=0.5,reg_alpha=0.1, gamma=1,colsample_bytree=0.8, random_state=42)
xgb_model.fit(X_train, y_train)


# Predict and evaluate
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"XGBoost RMSE: {rmse:.2f} mm")
print(f"XGBoost R² Score: {r2:.3f}")
r2_axes = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(2)]
print(f"XGBoost R² pocaX: {r2_axes[0]:.3f}")
print(f"XGBoost R² pocaY: {r2_axes[1]:.3f}")
rmse_per_axis = np.sqrt(((y_test - y_pred) ** 2).mean(axis=0))
print(f"RMSE pocaX: {rmse_per_axis[0]:.2f} mm")
print(f"RMSE pocaY: {rmse_per_axis[1]:.2f} mm")
print(f"STD Deviation of error in X: {np.std(y_pred[:,0]-y_test[:,0]):.2f} mm")
print(f"STD Deviation of error in Z: {np.std(y_pred[:,1]-y_test[:,1]):.2f} mm")

angleDev = test_df['angleDev']
valid_mask = angleDev >= 0.09
predX = y_pred[:, 0][valid_mask]
predZ = y_pred[:, 1][valid_mask]
angleDev = angleDev[valid_mask]

from scipy.stats import binned_statistic_2d


# Bin data and compute mean of z in each bin
stat, xedges, yedges, binnumber = binned_statistic_2d(
    predX, predZ, angleDev, statistic='std', bins=100
)

# Compute count per bin
count, _, _, _ = binned_statistic_2d(
    predX, predZ, angleDev, statistic='count', bins=[xedges, yedges]
)

# Compute threshold
mean_count = np.mean(count)
threshold = mean_count

# Mask bins with too few points
stat_masked = np.ma.masked_where(count < threshold, stat)

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, stat_masked.T, shading='auto', cmap='inferno_r')
plt.colorbar(label='Standard Deviation value')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D histogram with color mapped to Z')
plt.show()
