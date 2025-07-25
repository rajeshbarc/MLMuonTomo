import uproot
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# Step 1: Load test data from ROOT
imgFile = uproot.open("Event1e7_3Fe_2U.root")
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
#test_df = test_df.sample(n=10**5, random_state=42)
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
hb = plt.hexbin(predX, predZ, C=angleDev, gridsize=50, reduce_C_function=np.std, cmap='inferno_r')
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
bins = 200
x_edges = np.linspace(-500, 500, bins)
z_edges = np.linspace(-500, 500, bins)

# Function to create 2D histograms/images from point cloud
def make_image(x, z, values):
    stat, _, _, _ = binned_statistic_2d(x, z, values, statistic='std', bins=[x_edges, z_edges])
    stat = np.nan_to_num(stat)  # replace NaNs with 0
    return stat

'''
# Function to create 2D histograms/images from point cloud with thresholding
def make_image_with_threshold(x, z, values, x_edges, z_edges, percentile=10):
    stat, _, _, _ = binned_statistic_2d(x, z, values, statistic='std', bins=[x_edges, z_edges])
    stat = np.nan_to_num(stat, nan=0.0)  # temporarily replace NaNs with 0 for percentile calc
    threshold = np.percentile(stat[stat > 0], percentile)  # ignore 0s to compute percentile
    stat_masked = np.where(stat < threshold, 0, stat)  # mask values below threshold
    return stat_masked
'''
def make_image_with_threshold(x, z, values, x_edges, z_edges, percentile=25):
    # Compute std of angleDev in each bin
    std_stat, _, _, _ = binned_statistic_2d(x, z, values, statistic='std', bins=[x_edges, z_edges])
    std_stat = np.nan_to_num(std_stat, nan=0.0)

    # Compute count in each bin
    count_stat, _, _, _ = binned_statistic_2d(x, z, values, statistic='count', bins=[x_edges, z_edges])
    count_stat = np.nan_to_num(count_stat, nan=0.0)

    # Thresholds
    std_thresh = np.percentile(std_stat[std_stat > 0], percentile)
    count_thresh = np.percentile(count_stat[count_stat > 0], percentile)
    # Apply combined mask
    mask = (std_stat >= std_thresh) & (count_stat >= count_thresh)
    masked_image = np.where(mask, std_stat, 0)
    return masked_image



# Build image grids
img_pred = make_image_with_threshold(predX, predZ, angleDev,x_edges, z_edges)
img_true = make_image(trueX, trueZ, angleDev)
img_poca = make_image_with_threshold(pocaX, pocaZ, angleDev,x_edges, z_edges)

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
    plt.imshow(images[i], extent=[-500, 500, -500, 500], origin='lower', cmap='inferno_r')
    plt.title(f"{titles[i]} Heatmap")
    plt.xlabel("objectX")
    plt.ylabel("objectZ")
    plt.colorbar(label="Std. Deviation angleDev")
plt.tight_layout()
plt.show()




'''
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.ndimage import center_of_mass
from scipy.stats import entropy
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings("ignore")  # Suppress SSIM/PSNR divide-by-zero for empty masks

# Flatten all images
flat_true = img_true.flatten()
flat_pred = img_pred.flatten()
flat_poca = img_poca.flatten()

mse_pred = mean_squared_error(flat_true, flat_pred)
rmse_pred = np.sqrt(mse_pred)
mae_pred = mean_absolute_error(flat_true, flat_pred)

mse_poca = mean_squared_error(flat_true, flat_poca)
rmse_poca = np.sqrt(mse_poca)
mae_poca = mean_absolute_error(flat_true, flat_poca)
def normalized_cross_correlation(img1, img2):
    img1_mean = img1 - np.mean(img1)
    img2_mean = img2 - np.mean(img2)
    numerator = np.sum(img1_mean * img2_mean)
    denominator = np.sqrt(np.sum(img1_mean**2) * np.sum(img2_mean**2))
    return numerator / denominator if denominator != 0 else 0.0

ncc_pred = normalized_cross_correlation(img_true, img_pred)
ncc_poca = normalized_cross_correlation(img_true, img_poca)
com_true = center_of_mass(img_true)
com_pred = center_of_mass(img_pred)
com_poca = center_of_mass(img_poca)

def dist(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

com_shift_pred = dist(com_true, com_pred)
com_shift_poca = dist(com_true, com_poca)
def compute_fwhm(profile, x_axis):
    max_val = np.max(profile)
    half_max = max_val / 2.0
    interp_func = interp1d(x_axis, profile - half_max, kind='linear', bounds_error=False, fill_value='extrapolate')
    zero_crossings = np.where(np.diff(np.sign(profile - half_max)))[0]
    if len(zero_crossings) >= 2:
        x1 = x_axis[zero_crossings[0]]
        x2 = x_axis[zero_crossings[-1]]
        return abs(x2 - x1)
    return np.nan

center_row = img_true.shape[1] // 2
x_axis = np.linspace(-500, 500, img_true.shape[0])

fwhm_true = compute_fwhm(img_true[:, center_row], x_axis)
fwhm_pred = compute_fwhm(img_pred[:, center_row], x_axis)
fwhm_poca = compute_fwhm(img_poca[:, center_row], x_axis)
def image_entropy(img):
    hist, _ = np.histogram(img, bins=100, range=(img.min(), img.max()), density=True)
    hist = hist[hist > 0]  # Remove zero bins
    return entropy(hist)

entropy_true = image_entropy(img_true)
entropy_pred = image_entropy(img_pred)
entropy_poca = image_entropy(img_poca)
print("\n📈 Extended Evaluation Metrics:")
print(f"RMSE (Predicted vs True): {rmse_pred:.4f}")
print(f"MAE  (Predicted vs True): {mae_pred:.4f}")
print(f"NCC  (Predicted vs True): {ncc_pred:.4f}")
print(f"COM Shift (Predicted):    {com_shift_pred:.2f} px")
print(f"FWHM (Predicted):         {fwhm_pred:.2f} units")
print(f"Entropy (Predicted):      {entropy_pred:.4f}")

print(f"\nRMSE (PoCA vs True):      {rmse_poca:.4f}")
print(f"MAE  (PoCA vs True):      {mae_poca:.4f}")
print(f"NCC  (PoCA vs True):      {ncc_poca:.4f}")
print(f"COM Shift (PoCA):         {com_shift_poca:.2f} px")
print(f"FWHM (PoCA):              {fwhm_poca:.2f} units")
print(f"Entropy (PoCA):           {entropy_poca:.4f}")

print(f"\nFWHM (True):              {fwhm_true:.2f} units")
print(f"Entropy (True):           {entropy_true:.4f}")


# Define masks for ROI (non-zero values) and background
threshold = np.percentile(img_true[img_true > 0], 50)
mask_true = img_true > threshold  # ROI where histogram has non-zero values
mask_pred = img_pred > threshold
mask_poca = img_poca > threshold


# Metric 2: Contrast
def compute_contrast(hist, mask):
    roi = hist[mask]
    background = hist[~mask & ~np.isnan(hist)]
    if len(roi) > 0 and len(background) > 0:
        return np.mean(roi) / np.mean(background)
    return np.nan

# Metric 3: Uniformity
def compute_uniformity(hist, mask):
    background = hist[~mask & ~np.isnan(hist)]
    if len(background) > 0:
        return np.std(background) / np.mean(background)
    return np.nan

# Compute metrics for True, Predicted, and PoCA images
contrast_true = compute_contrast(img_true, mask_true)
contrast_pred = compute_contrast(img_pred, mask_pred)
contrast_poca = compute_contrast(img_poca, mask_poca)

uniformity_true = compute_uniformity(img_true, mask_true)
uniformity_pred = compute_uniformity(img_pred, mask_pred)
uniformity_poca = compute_uniformity(img_poca, mask_poca)

# Print contrast and uniformity metrics
print("\n🧠 Contrast and Uniformity Metrics:")
print(f"Contrast (True):     {contrast_true:.4f}")
print(f"Contrast (Predicted): {contrast_pred:.4f}")
print(f"Contrast (PoCA):     {contrast_poca:.4f}")
print(f"Uniformity (True):   {uniformity_true:.4f}")
print(f"Uniformity (Predicted): {uniformity_pred:.4f}")
print(f"Uniformity (PoCA):   {uniformity_poca:.4f}")
'''



import torch
import piq
# Convert images to PyTorch tensors
img_pred_tensor = torch.tensor(img_pred, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
img_true_tensor = torch.tensor(img_true, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
img_poca_tensor = torch.tensor(img_poca, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Normalize tensors to [0, 1] (important for MS-SSIM)
def normalize_tensor(img):
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)

img_pred_tensor = normalize_tensor(img_pred_tensor)
img_true_tensor = normalize_tensor(img_true_tensor)
img_poca_tensor = normalize_tensor(img_poca_tensor)

# Compute MS-SSIM
msssim_pred_true = piq.multi_scale_ssim(img_pred_tensor, img_true_tensor, data_range=1.0).item()
msssim_poca_true = piq.multi_scale_ssim(img_poca_tensor, img_true_tensor, data_range=1.0).item()
print(f"MS-SSIM (Predicted vs True): {msssim_pred_true:.4f}")
print(f"MS-SSIM (PoCA vs True):      {msssim_poca_true:.4f}")
