import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error

#plt.style.use(['science', 'ieee', 'no-latex'])


# Define Gaussian function for FWHM
def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# Modified FWHM computation to handle NaNs
def compute_fwhm(hist, bins, axis=0):
    # Find peak in histogram, ignoring NaNs
    hist_clean = np.where(np.isnan(hist), 0, hist)  # Replace NaNs with 0 for peak finding
    peak_idx = np.argmax(hist_clean)
    peak_x, peak_y = np.unravel_index(peak_idx, hist.shape)

    # Extract profile along specified axis
    if axis == 0:  # X-axis
        profile = hist[:, peak_y]
        x_vals = (bins[:-1] + bins[1:]) / 2
    else:  # Z-axis
        profile = hist[peak_x, :]
        x_vals = (bins[:-1] + bins[1:]) / 2

    # Remove NaNs from profile and corresponding x_vals
    valid = ~np.isnan(profile)
    profile_clean = profile[valid]
    x_vals_clean = x_vals[valid]

    # Check if profile has enough valid points for fitting
    if len(profile_clean) < 4 or np.all(profile_clean == 0):
        return np.nan  # Return NaN if insufficient data

    # Fit Gaussian
    try:
        popt, _ = curve_fit(gaussian, x_vals_clean, profile_clean,
                            p0=[np.max(profile_clean), x_vals_clean[np.argmax(profile_clean)], 10],
                            bounds=([0, np.min(x_vals_clean), 0], [np.inf, np.max(x_vals_clean), np.inf]))
        fwhm = 2.355 * abs(popt[2])  # FWHM = 2.355 * sigma
    except (RuntimeError, ValueError):
        fwhm = np.nan
    return fwhm



# Load ROOT files
test_file = uproot.open("testMay.root")
processed_file = uproot.open("image_testMay.root")

test_tTree = test_file["groundTruthPoCA"]
processed_tTree = processed_file["tree"]

# Convert TTrees to DataFrames
test_df = pd.DataFrame(test_tTree.arrays(library="pd")).dropna()
processed_df = pd.DataFrame(processed_tTree.arrays(library="pd")).dropna()

# Filter data 
test_df = test_df[(test_df['angleDev'] > 30e-3)]
processed_df = processed_df[(processed_df['angleDev'] > 30e-3)]

# Extract coordinates and angleDev
x_test, z_test, angle_test = test_df['pocaX'], test_df['pocaZ'], test_df['angleDev']
x_proc, z_proc, angle_proc = processed_df['pX'], processed_df['pZ'], processed_df['angleDev']

# Histogram parameters
bins = 100
x_bins = np.linspace(-500, 500, bins + 1)
z_bins = np.linspace(-500, 500, bins + 1)


# Compute histograms for standard deviation
def compute_std_histogram(x, z, angle_dev, x_bins, z_bins):
    counts, x_edges, z_edges = np.histogram2d(x, z, bins=[x_bins, z_bins])
    sum_weights, _, _ = np.histogram2d(x, z, bins=[x_bins, z_bins], weights=angle_dev)
    sum_weights_squared, _, _ = np.histogram2d(x, z, bins=[x_bins, z_bins], weights=angle_dev ** 2)
    valid = counts > 0
    mean = np.zeros_like(counts, dtype=float)
    mean[valid] = sum_weights[valid] / counts[valid]
    mean_squared = np.zeros_like(counts, dtype=float)
    mean_squared[valid] = sum_weights_squared[valid] / counts[valid]
    std = np.sqrt(np.abs(mean_squared - mean ** 2))
    std[~valid] = np.nan
    return std


std_test = compute_std_histogram(x_test, z_test, angle_test, x_bins, z_bins)
std_proc = compute_std_histogram(x_proc, z_proc, angle_proc, x_bins, z_bins)

# Apply mask 
threshold = np.nanmedian(std_test) * 0.5
mask_test = (std_test > threshold)
mask_proc =  (std_proc > threshold)
#mask = (std_test > 0.01) & (std_proc > 0.01)
std_test_masked = np.where(mask_test, std_test, np.nan)
std_proc_masked = np.where(mask_proc, std_proc, np.nan)

# Metric 1: Spatial Resolution (FWHM)
fwhm_x_test = compute_fwhm(std_test_masked, x_bins, axis=0)
fwhm_z_test = compute_fwhm(std_test_masked, z_bins, axis=1)
fwhm_x_proc = compute_fwhm(std_proc_masked, x_bins, axis=0)
fwhm_z_proc = compute_fwhm(std_proc_masked, z_bins, axis=1)


# Metric 2: Contrast
def compute_contrast(hist, mask):
    roi = hist[mask]
    background = hist[~mask & ~np.isnan(hist)]
    if len(roi) > 0 and len(background) > 0:
        return np.mean(roi) / np.mean(background)
    return np.nan


contrast_test = compute_contrast(std_test, mask_test)
contrast_proc = compute_contrast(std_proc, mask_proc)


# Metric 3: Uniformity
def compute_uniformity(hist, mask):
    background = hist[~mask & ~np.isnan(hist)]
    if len(background) > 0:
        return np.std(background) / np.mean(background)
    return np.nan


uniformity_test = compute_uniformity(std_test, mask_test)
uniformity_proc = compute_uniformity(std_proc, mask_proc)

# Metric 4: SSIM
# Normalize histograms to avoid SSIM issues with NaNs
std_test_norm = np.where(np.isnan(std_test), 0, std_test)
std_proc_norm = np.where(np.isnan(std_proc), 0, std_proc)
ssim_value, _ = ssim(std_test_norm, std_proc_norm, data_range=np.max(std_test_norm) - np.min(std_test_norm), full=True)

# Metric 5: MSE
valid_pixels = ~np.isnan(std_test) & ~np.isnan(std_proc)
mse_value = mean_squared_error(std_test[valid_pixels], std_proc[valid_pixels])

# Print results
print(f"Metrics Comparison:")
print(f"Spatial Resolution (FWHM X): Original = {fwhm_x_test:.2f}, Processed = {fwhm_x_proc:.2f}")
print(f"Spatial Resolution (FWHM Z): Original = {fwhm_z_test:.2f}, Processed = {fwhm_z_proc:.2f}")
print(f"Contrast: Original = {contrast_test:.2f}, Processed = {contrast_proc:.2f}")
print(f"Uniformity: Original = {uniformity_test:.2f}, Processed = {uniformity_proc:.2f}")
print(f"SSIM: {ssim_value:.2f}")
print(f"MSE: {mse_value:.6f}")


def compute_psnr(ref, img):
    mse = np.nanmean((ref - img) ** 2)
    if mse == 0:
        return np.inf
    max_pixel = np.nanmax(ref)
    return 20 * np.log10(max_pixel / np.sqrt(mse))

psnr_value = compute_psnr(std_test, std_proc)
print(f"PSNR: {psnr_value:.2f} dB")


# Visualize both images
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(std_test_masked.T, origin='lower', extent=[-500, 500, -500, 500], cmap='inferno_r')
plt.title('Original Image (Std of angleDev)')
plt.xlabel('X')
plt.ylabel('Z')
plt.colorbar(label='Std. of Scattering Angle')
plt.subplot(1, 2, 2)
plt.imshow(std_proc_masked.T, origin='lower', extent=[-500, 500, -500, 500], cmap='inferno_r')
plt.title('Processed Image (Std of angleDev)')
plt.xlabel('X')
plt.ylabel('Z')
plt.colorbar(label='Std. of Scattering Angle')
plt.tight_layout()
plt.show()
