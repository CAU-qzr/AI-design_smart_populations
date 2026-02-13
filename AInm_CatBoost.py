# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 18:46:20 2025

@author: CAU-QZR
"""

import random
import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from catboost import CatBoostRegressor
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')


'''
NumPy version: 1.26.4
Pandas version: 2.3.1
PyTorch version: 2.8.0+cpu
Matplotlib version: 3.9.2
Scikit-learn version: 1.6.1
SciPy version: 1.16.1
CatBoost version: 1.2.8
SHAP version: 0.48.0
Joblib version: 1.5.1
'''

plt.rcParams['font.family'] = 'SimHei'  
plt.rcParams['axes.unicode_minus'] = False

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

print("PyTorch Currently using the device:", "GPU" if torch.cuda.is_available() else "CPU")

# Create the results save directory
results_dir = "Code-upload/CatBoost_Results"
os.makedirs(results_dir, exist_ok=True)


# --------------------------
# Data Loading and Preprocessing
# --------------------------
def load_and_preprocess_data():
    
    file_path = "Code-upload/AInm_origindata.xlsx"
    df = pd.read_excel(file_path, sheet_name=0)

    # Round to two decimal places
    df.iloc[:, 4:11] = df.iloc[:, 4:11].round(2)

    # Convert categorical variables to numerical
    df['Irrigation'] = df['Irrigation'].map({'Irrigated': 1, 'Rainfed': 0}).astype(int)
    df['Tillage'] = df['Tillage'].map({'Con-tillage': 1, 'No-tillage': 0}).astype(int)

    # Separation feature and target
    y = df["Nopt"].values
    X = df.drop(columns=["Nopt"]).values
    feature_names = df.drop(columns=["Nopt"]).columns.tolist()
    
    return X, y, feature_names, df

X, y, feature_names, df = load_and_preprocess_data()



# --------------------------
# 10-fold cross-validation
# --------------------------
def run_repeated_cv(X, y, n_splits=10, n_repeats=1):
    
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    
    all_y_true = []
    all_y_pred = []
    fold_results = []
    models = []
    
    # Create an empty array for SHAP analysis
    n_samples, n_features = X.shape
    all_shap_values = np.zeros((len(X), len(feature_names))) 
    all_shap_interaction_values = np.zeros((n_samples, n_features, n_features)) 
    shap_counts = np.zeros(len(X)) 
    
    total_folds = n_splits * n_repeats
    fold_count = 0
    
    for train_idx, test_idx in rkf.split(X):
        fold_count += 1
        print(f" Training fold {fold_count}/{total_folds}... ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train the CatBoost model
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='RMSE',
            random_state=SEED,
            verbose=100,  
            task_type='GPU' if torch.cuda.is_available() else 'CPU'
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        models.append(model)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        corr, _ = pearsonr(y_test, y_pred)
        
        fold_results.append({
            'fold': fold_count,
            'mse': mse,
            'r2': r2,
            'corr': corr
        })
        
        print(f" Fold {fold_count} - MSE: {mse:.4f}, R²: {r2:.4f}, Correlation coefficient: {corr:.4f} ")
        
        print(f" Calculating SHAP values for the {fold_count} fold... ")
        try:
            # Create SHAP Explainer
            explainer = shap.TreeExplainer(model)
            
            shap_values_fold = explainer.shap_values(X_test)
            
            # Calculate SHAP interaction values
            if len(test_idx) < 200:  # Calculate interaction values only when the test set is small to avoid memory issues
                shap_interaction_fold = explainer.shap_interaction_values(X_test)
                
                for i, idx in enumerate(test_idx):
                    all_shap_interaction_values[idx] += shap_interaction_fold[i]
            
            for i, idx in enumerate(test_idx):
                all_shap_values[idx] += shap_values_fold[i]
                shap_counts[idx] += 1
                
        except Exception as e:
            print(f" Fold {fold_count} SHAP computation failed: {e} ")
            continue
    
    return (np.array(all_y_true), np.array(all_y_pred), 
            fold_results, models, all_shap_values, all_shap_interaction_values)

# Run repeated cross-validation
all_y_true, all_y_pred, fold_results, models, all_shap_values, all_shap_interaction_values = run_repeated_cv(X, y)

# Calculate the overall evaluation indicators
overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_r2 = r2_score(all_y_true, all_y_pred)
overall_corr, _ = pearsonr(all_y_true, all_y_pred)
num_n = len(all_y_true)

# Print overall evaluation results
print("\n" + "="*50)
print("10-fold cross-validation overall results:")
print(f"RMSE: {overall_rmse:.4f}")
print(f"R²: {overall_r2:.4f}")
print(f"Correlation Coefficient: {overall_corr:.4f}")
print("="*50)

# Save cross-validation results
cv_results_df = pd.DataFrame({
    'True_Values': all_y_true,
    'Predicted_Values': all_y_pred
})
cv_results_df.to_excel(f'{results_dir}/CatBoost_CV_Scatter.xlsx', index=False)


plt.figure(figsize=(10, 8))

max_val = max(all_y_true.max(), all_y_pred.max())-30
min_val = min(all_y_true.min(), all_y_pred.min())-30
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, color = 'grey')

plt.scatter(all_y_true, all_y_pred, alpha=0.3, s=100, c='green', edgecolors='black', linewidth=0.5)

plt.xlabel('Actual Nopt (kg ha$^{-1}$)', fontsize=24)
plt.ylabel('Predicted Nopt (kg ha$^{-1}$)', fontsize=24)
plt.title('CatBoost-10-Fold Cross Validation', fontsize=24)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

textstr = f'RMSE = {overall_rmse:.2f}\nR$^{2}$ = {overall_r2:.2f}\nCorr = {overall_corr:.2f}\nNum = {num_n}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(f'{results_dir}/CatBoost_CV_Scatter.pdf', dpi=300, bbox_inches='tight')
plt.show()


# --------------------------
# Retrain the final model
# --------------------------
def train_final_model(X, y):
    """Train the final model using all the data."""
    
    final_model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        loss_function='RMSE',
        random_state=SEED,
        verbose=100, 
        task_type='GPU' if torch.cuda.is_available() else 'CPU'
    )
    final_model.fit(X, y)
    
    # save model
    final_model.save_model(f'{results_dir}/Catboost_Nopt.cbm')
    print(f" Model saved '{results_dir}/Catboost_Nopt.cbm ")
    
    return final_model

final_model = train_final_model(X, y)

feature_importance = final_model.get_feature_importance()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

importance_df = importance_df.sort_values(by="Importance", ascending=False)
print(importance_df)



# --------------------------
# SHAP Visualization
# --------------------------
def plot_shap_summary(shap_values, feature_names):
    
    # Calculate Feature Importance (Average Absolute SHAP Value)
    importance = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    colors = plt.cm.Greens(np.linspace(0.1, 0.9, len(importance_df)))
    
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors, alpha=0.8)
    plt.xlabel('Mean(|SHAP|)', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/SHAP_Feature_Importance.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    importance_df.to_excel(f'{results_dir}/CatBoost_Feature_Importance.xlsx', index=False)
    
    return importance_df

importance_df = plot_shap_summary(all_shap_values, feature_names)


# Save SHAP values
shap_df = pd.DataFrame(all_shap_values, columns=feature_names)
shap_df.to_excel(f'{results_dir}/All_Samples_SHAP_Values.xlsx', index=False)


# Save detailed results of 10-fold cross-validation
fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_excel(f'{results_dir}/Fold_Detailed_Results.xlsx', index=False)