# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:13:15 2025

@author: CGSP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Importing Model Method
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor
from sklearn.base import BaseEstimator, RegressorMixin
class ELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=20, random_state=42):
        self.n_hidden = n_hidden
        self.random_state = random_state
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.W = np.random.randn(X.shape[1], self.n_hidden)
        self.b = np.random.randn(self.n_hidden)
        H = np.tanh(X @ self.W + self.b)
        self.beta = np.linalg.pinv(H) @ y
        return self
    def predict(self, X):
        H = np.tanh(X @ self.W + self.b)
        return H @ self.beta
    

'''
pandas : 2.3.2
numpy : 1.26.4
scikit-learn : 1.6.1
matplotlib : 3.9.2
lightgbm : 4.6.0
xgboost : 3.0.4
catboost : 1.2.8
tabpfn : 2.1.3
'''


# Reading data
file_path = "Code-upload/AIsp_origindata.xlsx"
df = pd.read_excel(file_path, sheet_name=0)  

# Convert categorical variables to numerical
df['Irrigation'] = df['Irrigation'].map({'Irrigated': 1, 'Rainfed': 0}).astype(int)
df['Tillage'] = df['Tillage'].map({'Con-tillage': 1, 'No-tillage': 0}).astype(int)

# Divide the canopy into 3 subvariables
df['canopy_compact'] = (df['Canopy'] == 'compact').astype(int)
df['canopy_smart']   = (df['Canopy'] == 'smart').astype(int)
df['canopy_other']   = (~df['Canopy'].isin(['compact', 'smart'])).astype(int)

# Remove unnecessary columns
df = df.drop(columns=['Canopy', 'Yopt']) 
# The relationship between Yopt and Dopt can be utilized in subsequent analyses.

# Separate independent and dependent variables
y = df["OPD"].values
X = df.drop(columns=["OPD"]).values

# Standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Define method list
models = {
    "LightGBM": LGBMRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
    "SVR": SVR(),
    "RF": RandomForestRegressor(random_state=42),
    "MLP": MLPRegressor(max_iter=1000, random_state=42),
    "DT": DecisionTreeRegressor(random_state=42),
    "ELM": ELMRegressor(),
    "Bay": BayesianRidge(),
    "GBR": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "TabPFN": TabPFNRegressor(device="cuda")  # If GPU is unavailable, change device="cpu".
}

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

metrics = {name: {"MAE": [], "RMSE": [], "R2": []} for name in models.keys()}

for name, model in models.items():
    print(f"Method: {name}")
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Anti-standardization
        y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
        rmse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
        r2 = r2_score(y_test_rescaled, y_pred_rescaled)

        metrics[name]["MAE"].append(mae)
        metrics[name]["RMSE"].append(rmse)
        metrics[name]["R2"].append(r2)


fold_metrics_dfs = {}
for name in models.keys():
    fold_metrics_dfs[name] = pd.DataFrame(metrics[name])

# Save all fold metrics to Excel
with pd.ExcelWriter("Code-upload/Models_fold_metrics.xlsx") as writer:
    for name, df_metrics in fold_metrics_dfs.items():
        df_metrics.to_excel(writer, sheet_name=name, index=False)
        

# result
results_df = pd.DataFrame({
    model: [np.mean(m["MAE"]), np.mean(m["RMSE"]), np.mean(m["R2"])]
    for model, m in metrics.items()
}, index=["MAE", "RMSE", "R2"]).T

print("\n=== result ===")
print(results_df)

results_df.to_excel('Code-upload/AIsp_method_results.xlsx')
