# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 16:28:19 2025

@author: CGSP
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
from tabpfn import TabPFNRegressor
from tabpfn_extensions import TunedTabPFNRegressor
from tabpfn_extensions import interpretability
import joblib
import warnings
warnings.filterwarnings('ignore')


'''
numpy: 1.26.4
pandas: 2.3.2
torch: 2.5.1+cu121
matplotlib: 3.9.2
sklearn: 1.6.1
scipy: 1.16.1
tabpfn: 2.1.3
tabpfn_extensions: 0.1.3
joblib: 1.5.1
'''


plt.rcParams['font.family'] = 'SimHei'  
plt.rcParams['axes.unicode_minus'] = False

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

print("PyTorch Currently using the device:", "GPU" if torch.cuda.is_available() else "CPU")

# Create the results save directory
results_dir = "Code-upload/TabPFN_Results"
os.makedirs(results_dir, exist_ok=True)



# --------------------------
# Data Loading and Preprocessing
# --------------------------
def load_and_preprocess_data():

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
    feature_names = df.drop(columns=["OPD"]).columns.tolist()
    
    return X, y, feature_names, df

X, y, feature_names, df = load_and_preprocess_data()


# Save X and feature_names
np.save(f'{results_dir}/X_data.npy', X)
np.save(f'{results_dir}/feature_names.npy', feature_names)  


    
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
    shap_counts = np.zeros(len(X)) 
    
    total_folds = n_splits * n_repeats
    fold_count = 0
    
    for train_idx, test_idx in rkf.split(X):
        fold_count += 1
        print(f"Training fold {fold_count}/{total_folds}...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
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
        
        print(f"Fold {fold_count} - MSE: {mse:.4f}, R²: {r2:.4f}, Correlation coefficient: {corr:.4f}")
        
        # Compute SHAP values for the current fold's test set
        print(f" Calculating SHAP values for the {fold_count} fold... ")
        try:
            test_sample_size = min(len(X_test), 100) 
            
            shap_exp = interpretability.shap.get_shap_values(
                estimator=model,
                test_x=X_test[:test_sample_size], 
                attribute_names=feature_names,
                algorithm="permutation",
            )
            
            shap_values_array = shap_exp.values
            
            for i, idx in enumerate(test_idx[:test_sample_size]):
                all_shap_values[idx] = shap_values_array[i] 
                shap_counts[idx] = 1 
                    
        except Exception as e:
            print(f" {fold_count} SHAP error: {e}")
            continue
    
    return (np.array(all_y_true), np.array(all_y_pred), 
            fold_results, models, all_shap_values)

# Run repeated cross-validation
all_y_true, all_y_pred, fold_results, models, all_shap_values = run_repeated_cv(X, y)

# Calculate overall assessment metrics
overall_rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
overall_r2 = r2_score(all_y_true, all_y_pred)
overall_corr, _ = pearsonr(all_y_true, all_y_pred)
num_n = len(all_y_true)

# Print overall evaluation results
print("\n" + "="*50)
print("Overall results 10-fold cross-validation repetitions:")
print(f"RMSE: {overall_rmse:.4f}")
print(f"R²: {overall_r2:.4f}")
print(f"Correlation Coefficient: {overall_corr:.4f}")
print("="*50)

cv_results_df = pd.DataFrame({
    'True_Values': all_y_true,
    'Predicted_Values': all_y_pred
})
cv_results_df.to_excel(f'{results_dir}/TabPFN_CV_Results.xlsx', index=False)

# Visualized prediction results
plt.figure(figsize=(10, 8))
plt.scatter(all_y_true, all_y_pred, alpha=0.3, s=100, c='green', edgecolors='black', linewidth=0.5)

max_val = max(all_y_true.max(), all_y_pred.max())
min_val = min(all_y_true.min(), all_y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

plt.xlabel('Actual density (10$^{4}$ pl ha$^{-1}$)', fontsize=24)
plt.ylabel('Predicted density (10$^{4}$ pl ha$^{-1}$)', fontsize=24)
plt.title('TabPFN-10-Fold Cross Validation', fontsize=24)

plt.xlim(min_val-1, max_val+1)
plt.ylim(min_val-1, max_val+1)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

textstr = f'RMSE = {overall_rmse:.2f}\nR$^{2}$ = {overall_r2:.2f}\nCorr = {overall_corr:.2f}\nNum = {num_n}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(f'{results_dir}/TabPFN_CV_Scatter.pdf', dpi=300, bbox_inches='tight')
plt.show()



# --------------------------
# Final Training Model
# --------------------------
def train_final_model(X, y):

    print("The final model is being trained using all available data...")
    final_model = TabPFNRegressor(device='cpu')
    final_model.fit(X, y)
    
    # Save the final model
    joblib.dump(final_model, f'{results_dir}/TabPFN_OPD.pkl')
    print(f"The final model has been saved as '{results_dir}/TabPFN_OPD.pkl'")
    
    return final_model

final_model = train_final_model(X, y)



# --------------------------
#  SHAP Visualization
# --------------------------
def plot_shap_summary(shap_values, feature_names):
    canopy_features = ['canopy_compact', 'canopy_smart', 'canopy_other']
    # canopy_indices = [feature_names.index(feat) for feat in canopy_features if feat in feature_names]
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # SHAP values for merged canopy features
    canopy_importance = 0
    remaining_indices = []
    remaining_names = []
    remaining_importance = []
    
    for i, (name, importance) in enumerate(zip(feature_names, mean_abs_shap)):
        if name in canopy_features:
            canopy_importance += importance
        else:
            remaining_indices.append(i)
            remaining_names.append(name)
            remaining_importance.append(importance)
    
    if canopy_importance > 0:  
        new_feature_names = ['Canopy'] + remaining_names
        new_importance_values = [canopy_importance] + remaining_importance
    else:
        new_feature_names = remaining_names
        new_importance_values = remaining_importance
    
    importance_df = pd.DataFrame({
        'Feature': new_feature_names,
        'Importance': new_importance_values
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
    
    importance_df.to_excel(f'{results_dir}/TabPFN_Feature_Importance.xlsx', index=False)
    
    return importance_df


importance_df = plot_shap_summary(all_shap_values, feature_names)


# Save SHAP values
shap_df = pd.DataFrame(all_shap_values, columns=feature_names)
shap_df.to_excel(f'{results_dir}/All_Samples_SHAP_Values.xlsx', index=False)


# Save detailed results of 10-fold cross-validation
fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_excel(f'{results_dir}/Fold_Detailed_Results.xlsx', index=False)

