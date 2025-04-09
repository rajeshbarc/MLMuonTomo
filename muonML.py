import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load data from ROOT file
file = uproot.open("MLData.root")  # Replace with your ROOT file path
tree = file["mlData"]  # Replace with your TTree name if different
df = tree.arrays(library="pd")

# Filter data
dataFilter = df[['deviation', 'g4Momentum']].dropna()
filtered_data = dataFilter[(dataFilter['deviation'] > 90e-3) & (dataFilter['g4Momentum'] !=0) & (dataFilter['g4Momentum'] <=9000)]  # Assuming deviation in radians

# Prepare features and target
X = filtered_data[['deviation']]  # Feature (scattering angle)
'''
X = np.column_stack([
    filtered_data['deviation'],
    np.cos(filtered_data['deviation']),
    np.sin(filtered_data['deviation'])
])
'''
y = filtered_data['g4Momentum']   # Target (momentum in MeV)
print(f"Mean of g4Momentum: {np.mean(y)} MeV")
print(f"Std Deviation of g4Momentum{np.std(y)} Mev")
print(f"Minimum of g4Momentum: {np.min(y)} and Maximum of g4Momentum: {np.max(y)} Mev")



# --- Random Forest Model ---
def rfLog():
    y_log = np.log1p(filtered_data['g4Momentum'])  # log(1 + x) for safety

    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_log.values.reshape(-1, 1))  # y or y_log can be changed here

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    rf_model.fit(X_train, y_train.ravel())  # ravel() flattens y_train

    y_pred_rf_scaled = rf_model.predict(X_test)
    y_pred_rf_orig = scaler_y.inverse_transform(y_pred_rf_scaled.reshape(-1, 1))
    y_pred_orig = np.expm1(y_pred_rf_orig)
    y_test_orig = np.expm1(scaler_y.inverse_transform(y_test.reshape(-1, 1)))
    r2_rf = r2_score( y_test_orig, y_pred_orig)
    mse_rf = mean_squared_error(y_pred_orig, y_test_orig)
    std_rf = np.std(y_pred_orig- y_test_orig)

    print("\n#### Random Forest Model: #####")
    print(f"R^2 Score: {r2_rf}")
    print(f"Mean Squared Error: {mse_rf} MeV²")
    print(f"Standard Deviation of Residuals: {std_rf} MeV")
    plt.hist(y_pred_orig-y_test_orig, bins=200, alpha=0.7, color='orange')
    plt.show()

def polyLog():
    y_log = np.log1p(filtered_data['g4Momentum'])  # log(1 + x) for safety

    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_log.values.reshape(-1, 1))  # y or y_log can be changed here

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    poly = PolynomialFeatures(degree=5)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly_scaled = poly_model.predict(X_test_poly)
    y_pred_poly_orig = scaler_y.inverse_transform(y_pred_poly_scaled)
    y_pred_orig = np.expm1(y_pred_poly_orig)
    y_test_orig = np.expm1(scaler_y.inverse_transform(y_test.reshape(-1, 1)))
    r2_pf = r2_score( y_test_orig, y_pred_orig)
    mse_pf = mean_squared_error(y_pred_orig, y_test_orig)
    std_pf = np.std(y_pred_orig- y_test_orig)
    print("\n#### Polynomial Features Log Model: #####")
    print(f"R^2 Score: {r2_pf}")
    print(f"Mean Squared Error: {mse_pf} MeV²")
    print(f"Standard Deviation of Residuals: {std_pf} MeV")
    plt.hist(y_pred_orig-y_test_orig, bins=200, alpha=0.7, color='orange')
    plt.show()

def polyNatLog():
    y_log = np.log(filtered_data['g4Momentum'])  # log(1 + x) for safety
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_log.values.reshape(-1, 1))  # y or y_log can be changed here

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    poly = PolynomialFeatures(degree=7, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly_scaled = poly_model.predict(X_test_poly)
    y_pred_poly_orig = scaler_y.inverse_transform(y_pred_poly_scaled)
    y_pred_orig = np.exp(y_pred_poly_orig)
    y_test_orig = np.exp(scaler_y.inverse_transform(y_test.reshape(-1, 1)))
    r2_pf = r2_score( y_test_orig, y_pred_orig)
    mse_pf = mean_squared_error(y_pred_orig, y_test_orig)
    std_pf = np.std(y_pred_orig- y_test_orig)
    print("\n#### Polynomial Features Nat Log Model: #####")
    print(f"R^2 Score: {r2_pf}")
    print(f"Mean Squared Error: {mse_pf} MeV²")
    print(f"Standard Deviation of Residuals: {std_pf} MeV")
    plt.hist(y_pred_orig-y_test_orig, bins=200, alpha=0.7, color='orange')
    plt.show()


def polyInv():
    y_Inv = 1 / np.log(filtered_data['g4Momentum'])
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    #y_scaled = y_Inv
    y_scaled = scaler_y.fit_transform(y_Inv.values.reshape(-1, 1))  # y or y_log can be changed here

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    poly = PolynomialFeatures(degree=5)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly_scaled = poly_model.predict(X_test_poly)
    y_pred_poly_orig = scaler_y.inverse_transform(y_pred_poly_scaled)
    y_pred_orig = np.exp(1 / y_pred_poly_orig)
    y_test_orig = np.exp(1 / (scaler_y.inverse_transform(y_test.reshape(-1, 1))))
    #y_test_orig = np.exp(1 / (scaler_y.inverse_transform(y_test.reshape(-1, 1)))) - 1 - 0.001
    r2_pf = r2_score( y_test_orig, y_pred_orig)
    mse_pf = mean_squared_error(y_pred_orig, y_test_orig)
    std_pf = np.std(y_pred_orig- y_test_orig)
    print("\n#### Polynomial Features Inverse Model: #####")
    print(f"R^2 Score: {r2_pf}")
    print(f"Mean Squared Error: {mse_pf} MeV²")
    print(f"Standard Deviation of Residuals: {std_pf} MeV")
    plt.hist(y_pred_orig-y_test_orig, bins=200, alpha=0.7, color='orange')
    plt.show()

def xgbLog():
    import xgboost as xgb
    y_log = np.log(filtered_data['g4Momentum'])  # log(1 + x) for safety
    # Scale the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y_log.values.reshape(-1, 1))  # y or y_log can be changed here

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=True
    )
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    # Predict
    y_pred_xgb_scaled = model.predict(X_test)
    y_pred_xgb_orig = scaler_y.inverse_transform(y_pred_xgb_scaled.reshape(-1, 1))
    y_pred_orig = np.exp(y_pred_xgb_orig)
    y_test_orig = np.exp((scaler_y.inverse_transform(y_test.reshape(-1, 1))))
    r2_pf = r2_score(y_test_orig, y_pred_orig)
    mse_pf = mean_squared_error(y_pred_orig, y_test_orig)
    std_pf = np.std(y_pred_orig - y_test_orig)
    print("\n#### XGB Log Model: #####")
    print(f"R^2 Score: {r2_pf}")
    print(f"Mean Squared Error: {mse_pf} MeV²")
    print(f"Standard Deviation of Residuals: {std_pf} MeV")
    plt.hist(y_pred_orig - y_test_orig, bins=200, alpha=0.7, color='orange')
    plt.show()


if __name__=="__main__":
    rfLog()     #Random Forest Log Output
    polyLog()   # Polynomial Log Output
    polyNatLog() # Polynomial Nat Log Output
    polyInv()   #Polynomial inverse Log
    xgbLog()    #xgBoost Log
