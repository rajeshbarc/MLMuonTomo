import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from skimage.metrics import structural_similarity as ssim

# Load ROOT files
train_file = uproot.open("train.root")
test_file = uproot.open("test3.root")
train_tTree = train_file["groundTruthPoCA"]
test_tTree = test_file["groundTruthPoCA"]

# Load data into DataFrames
train_df = pd.DataFrame(train_tTree.arrays(library="pd")).dropna()
test_df = pd.DataFrame(test_tTree.arrays(library="pd")).dropna()

# Filter data
train_df = train_df[(train_df['angleDev'] > 0e-3) & (train_df['pocaX'] > -500) & (train_df['pocaZ'] > -500)]
#test_df = test_df[(test_df['angleDev'] > 0e-3) & (test_df['pocaX'] > -500) & (test_df['pocaZ'] > -500)]

# Feature engineering
def compute_scattering_angle(df):
    dot_product = (df['dInX'] * df['dOutX'] + df['dInY'] * df['dOutY'] + df['dInZ'] * df['dOutZ'])
    return np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi

train_df['scattering_angle'] = compute_scattering_angle(train_df)
test_df['scattering_angle'] = compute_scattering_angle(test_df)
train_df['dispX'] = train_df['outX'] - train_df['inX']
train_df['dispY'] = train_df['outY'] - train_df['inY']
train_df['dispZ'] = train_df['outZ'] - train_df['inZ']
test_df['dispX'] = test_df['outX'] - test_df['inX']
test_df['dispY'] = test_df['outY'] - test_df['inY']
test_df['dispZ'] = test_df['outZ'] - test_df['inZ']

# Define features and labels
muonFeatures = list(train_tTree.keys()[0:12])+ ['angleDev', 'scattering_angle', 'dispX', 'dispY', 'dispZ']
pocaLabel = ['pocaX', 'pocaZ']
print(f'Features: {muonFeatures} and Labels: {pocaLabel}')

# Prepare data
X_train = np.column_stack([train_df[i] for i in muonFeatures])
y_train = np.column_stack([train_df[i] for i in pocaLabel])
X_test = np.column_stack([test_df[i] for i in muonFeatures])
y_test = np.column_stack([test_df[i] for i in pocaLabel])

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
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
print(f"XGBoost R² pocaZ: {r2_axes[1]:.3f}")
rmse_per_axis = np.sqrt(((y_test - y_pred) ** 2).mean(axis=0))
print(f"RMSE pocaX: {rmse_per_axis[0]:.2f} mm")
print(f"RMSE pocaZ: {rmse_per_axis[1]:.2f} mm")
print(f"STD Deviation of error in X: {np.std(y_pred[:,0]-y_test[:,0]):.2f} mm")
print(f"STD Deviation of error in Z: {np.std(y_pred[:,1]-y_test[:,1]):.2f} mm")

# SSIM for 2D reconstruction
true_hist, xedges, yedges = np.histogram2d(y_test[:, 0], y_test[:, 1], bins=50, range=[[-500, 500], [-500, 500]])
pred_hist, _, _ = np.histogram2d(y_pred[:, 0], y_pred[:, 1], bins=50, range=[[-500, 500], [-500, 500]])
ssim_score = ssim(true_hist, pred_hist, data_range=pred_hist.max() - pred_hist.min())
print(f"SSIM Score: {ssim_score:.3f}")

# Cross-validation
for i, label in enumerate(['pocaX', 'pocaZ']):
    cv_scores = cross_val_score(xgb_model, X_train, y_train[:, i], cv=5, scoring='r2', n_jobs=-1)
    print(f"Cross-Validated R² Scores for {label}: {cv_scores}")
    print(f"Mean R² CV Score for {label}: {cv_scores.mean():.3f}")

# Visualizations
plt.figure(figsize=(6, 5))
plt.scatter(y_test[:, 0], y_pred[:, 0], label='pocaX')
plt.xlabel('True pocaX')
plt.ylabel('Predicted pocaX')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.scatter(y_test[:, 1], y_pred[:, 1], label='pocaZ')
plt.xlabel('True pocaZ')
plt.ylabel('Predicted pocaZ')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.hist(y_pred[:, 0] - y_test[:, 0], bins=50, label='pocaX Error')
plt.hist(y_pred[:, 1] - y_test[:, 1], bins=50, label='pocaZ Error', alpha=0.5)
plt.xlabel('Prediction Error (mm)')
plt.ylabel('Count')
plt.legend()
plt.show()

plt.figure(figsize=(6, 5))
plt.hist2d(y_pred[:, 0], y_pred[:, 1], bins=50, range=[[-500, 500], [-500, 500]], cmap='Reds')
plt.xlabel('pocaX')
plt.ylabel('pocaZ')
plt.title('2D Reconstruction')
plt.colorbar(label='Count')
plt.show()

plt.figure(figsize=(6, 5))
plt.hexbin(y_pred[:, 0], y_pred[:, 1], C=test_df['angleDev'], reduce_C_function=np.std,
           gridsize=50, extent=[-500, 500, -500, 500], cmap='plasma')
plt.colorbar(label='Std. Dev Scattering Angle')
plt.xlabel('pocaX')
plt.ylabel('pocaZ')
plt.title('2D Reconstruction with Scattering Angle')
plt.show()

# Save predictions to ROOT file
rootFrame = pd.DataFrame({"x": y_pred[:, 0], "z": y_pred[:, 1], "angleDev":np.array(test_df['angleDev'])})
data_dict = {col: rootFrame[col].to_numpy() for col in rootFrame.columns}
with uproot.recreate("image_xgboost_test3.root") as file:
    file["tree"] = data_dict

importances = xgb_model.feature_importances_
feature_importance = pd.DataFrame({'Feature': muonFeatures, 'Importance': importances})
print(feature_importance.sort_values(by='Importance', ascending=False))
# Drop features with importance < 0.05
important_features = feature_importance[feature_importance['Importance'] > 0.05]['Feature'].tolist()
X_train = np.column_stack([train_df[i] for i in important_features])
X_test = np.column_stack([test_df[i] for i in important_features])
