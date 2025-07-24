import uproot
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error



# Step 1: Load test data from ROOT
imgFile = uproot.open("test3.root")
imgTree = imgFile['groundTruthPoCA']
imgDF = imgTree.arrays(library='np')

# Rename 'pocaX', 'pocaY', 'pocaZ' to 'objectX', etc.
for i in ('X', 'Y', 'Z'):
    imgDF['object' + i] = imgDF['poca' + i]
    del imgDF['poca' + i]

# Convert to DataFrame
test_df = pd.DataFrame(imgDF)
#test_df = test_df[test_df['angleDev']>30e-3]

# Step 2: Define features and labels
muonFeatures = ['inX', 'inZ', 'outX', 'outZ', 'dInX','dInZ','dOutX','dOutZ']
muonLabel = ['objectX', 'objectZ']

# Drop NaNs or infinite values (optional but recommended)
test_df = test_df.replace([np.inf, -np.inf], np.nan).dropna(subset=muonFeatures + muonLabel + ['angleDev'])

# Step 3: Prepare test features
X_test = test_df[muonFeatures].values
angleDev = test_df['angleDev'].values

# Optional: Normalize if you trained with normalized features
# If you used StandardScaler in training, reload and apply the same scaler here
# scaler = joblib.load("scaler.joblib")
# X_test = scaler.transform(X_test)

# Step 4: Load trained model
model = joblib.load("xgb_model.joblib")

# Step 5: Predict objectX and objectZ
y_pred = model.predict(X_test)
predX = y_pred[:, 0]
predZ = y_pred[:, 1]

# Step 6: Plot hexbin with std(angleDev) as color
plt.figure(figsize=(10, 8))
hb = plt.hexbin(predX, predZ, C=angleDev, gridsize=100, reduce_C_function=np.std, cmap='inferno_r')
plt.colorbar(hb, label='Std Dev of angleDev')
plt.xlabel('Predicted objectX')
plt.ylabel('Predicted objectZ')
plt.title('Hexbin Plot of Predicted (objectX, objectZ) with angleDev Std Dev')
plt.tight_layout()
plt.show()

print("Evaluating Performance")
from scipy.stats import binned_statistic_2d
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

trueX = test_df['objectX'].values
trueZ = test_df['objectZ'].values
pocaX = test_df['pX'].values
pocaZ = test_df['pZ'].values

# Define image space grid
bins = 100
x_edges = np.linspace(-500, 500, bins)
z_edges = np.linspace(-500, 500, bins)

# Function to create 2D histograms/images from point cloud
def make_image(x, z, values):
    stat, _, _, _ = binned_statistic_2d(x, z, values, statistic='std', bins=[x_edges, z_edges])
    stat = np.nan_to_num(stat)  # replace NaNs with 0
    return stat

# Build image grids
img_pred = make_image(predX, predZ, angleDev)
img_true = make_image(trueX, trueZ, angleDev)
img_poca = make_image(pocaX, pocaZ, angleDev)

# SSIM & PSNR between predicted and true
ssim_pred_true = ssim(img_true, img_pred, data_range=img_true.max() - img_true.min())
psnr_pred_true = psnr(img_true, img_pred, data_range=img_true.max() - img_true.min())

# SSIM & PSNR between poca and true
ssim_poca_true = ssim(img_true, img_poca, data_range=img_true.max() - img_true.min())
psnr_poca_true = psnr(img_true, img_poca, data_range=img_true.max() - img_true.min())

# Print metrics
print("\n🧠 Image Similarity Metrics (angleDev-weighted heatmaps):")
print(f"SSIM (Predicted vs True): {ssim_pred_true:.4f}")
print(f"PSNR (Predicted vs True): {psnr_pred_true:.2f} dB")
print(f"SSIM (PoCA vs True):      {ssim_poca_true:.4f}")
print(f"PSNR (PoCA vs True):      {psnr_poca_true:.2f} dB")

# Optional: Show images
titles = ['True', 'Predicted', 'PoCA']
images = [img_true, img_pred, img_poca]

plt.figure(figsize=(18, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], extent=[-500, 500, -500, 500], origin='lower', cmap='inferno')
    plt.title(f"{titles[i]} Heatmap")
    plt.xlabel("objectX")
    plt.ylabel("objectZ")
    plt.colorbar(label="Mean angleDev")
plt.tight_layout()
plt.show()


