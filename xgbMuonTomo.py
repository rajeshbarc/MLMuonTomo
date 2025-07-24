import pandas as pd
import uproot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import binned_statistic_dd
import joblib
from sklearn.multioutput import MultiOutputRegressor

# Load ROOT data
imgFile = uproot.open("trainLead.root")
imgTree = imgFile['groundTruthPoCA']
imgDF = imgTree.arrays(library='np')

# Rename 'pocaX' to 'objectX'
for i in ('X','Y','Z'):
    imgDF['object'+i] = imgDF['poca'+i]
    del imgDF['poca'+i]

print(imgDF.keys())
imgPD = pd.DataFrame(imgDF)

# Load ROOT data
imgFile1 = uproot.open("trainFe.root")
imgTree1 = imgFile1['groundTruthPoCA']
imgDF1 = imgTree1.arrays(library='np')

# Rename 'pocaX' to 'objectX'
for i in ('X','Y','Z'):
    imgDF1['object'+i] = imgDF1['poca'+i]
    del imgDF1['poca'+i]
imgPD1 = pd.DataFrame(imgDF1)
train_df = pd.concat([imgPD, imgPD1], ignore_index=True)


train_df = train_df[(train_df['angleDev'] > 0.1e-3) & (train_df['objectX'] > -500) & (train_df['objectY'] > -500) & (train_df['objectZ'] > -500) & (train_df['objectX'] < 500) & (train_df['objectY'] < 500) & (train_df['objectZ'] < 500)]
muonFeatures = ['inX', 'inZ', 'outX', 'outZ', 'dInX','dInZ','dOutX','dOutZ']
muonLabel =['objectX','objectZ']


from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Assuming train_df is already prepared as per your preprocessing

# Prepare feature matrix X and target matrix y
X = train_df[muonFeatures]
y = train_df[muonLabel]
# Initialize and train XGBoost models for objectX and objectZ separately
base_model = XGBRegressor(objective='reg:squarederror', random_state=42)

multi_model = MultiOutputRegressor(base_model)
weights = train_df['angleDev'].values
multi_model.fit(X, y, sample_weight=weights)

# Make predictions
y_pred = multi_model.predict(X)
from sklearn.metrics import r2_score, mean_squared_error

r2_x = r2_score(y['objectX'], y_pred[:, 0])
rmse_x = np.sqrt(mean_squared_error(y['objectX'], y_pred[:, 0]))

r2_z = r2_score(y['objectZ'], y_pred[:, 1])
rmse_z = np.sqrt(mean_squared_error(y['objectZ'], y_pred[:, 1]))

print("Metrics for objectX:")
print(f"R² Score: {r2_x:.4f}")
print(f"RMSE: {rmse_x:.4f}")
print("\nMetrics for objectZ:")
print(f"R² Score: {r2_z:.4f}")
print(f"RMSE: {rmse_z:.4f}")
# Feature importance for objectX model
feature_importance_x = multi_model.estimators_[0].feature_importances_
plt.bar(muonFeatures, feature_importance_x)
plt.title('Feature Importance for objectX')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance for objectZ model
feature_importance_z = multi_model.estimators_[1].feature_importances_
plt.bar(muonFeatures, feature_importance_z)
plt.title('Feature Importance for objectZ')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Export the trained models to files
joblib.dump(multi_model, 'xgb_model.joblib')
print("Models exported")
