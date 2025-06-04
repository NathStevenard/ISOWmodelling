from .utils import load_data
from .utils import resample
from .utils import x_for_models
from .utils import impute_data
from .utils import best_xgb_params
from .utils import split_data
from .utils import fit_imputer

import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import shap

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This module predict the ISOW intensity across the last 800,000 years.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- CAN BE MODIFIED ---
model_linear = LinearRegression()  # Define the linear model
# --- CAN BE MODIFIED ---

def model(start, end, nsim=100):
    """
    This function perform a hybrid model, using linear and nonlinear (XGBoost) models, to produce a predicted
    ISOW stength variability.
    Linear model (info): simple linear regression model (Pedregosa et al., 2011).
    XGBoost model (info): a gradient-boosted tree ensemble model (Chen and Guestrin, 2016). Hyper-parameters are
    optimized at each iteration using Optune (Akiba et al., 2019).
    :param start: int, float - Starting date of the period.
    :param end: int, float - Ending date of the period.
    :param nsim: int - Number of iteration (default = 100)
    :return: Two pd.DataFrames
        - output --> percentiles of the ISOW_model for the desired time period.
        - shap_values_median_df --> associated SHAP values for each forcing used.
    """
    print("\n~~~~~~~~~~~~ The prediction of ISOW strength is starting ~~~~~~~~~~~~")

    # --- CHECK INPUTS ---
    if not isinstance(start, int) or not isinstance(start, int):
        raise TypeError('The starting date must be an integer or a float')
    if not isinstance(end, int) or not isinstance(end, int):
        raise TypeError('The ending date must be an integer or a float')

    if start < 0 or start > 800 or start > end:
        raise ValueError('The starting date must be comprised between 0 and 800 and lower than "end" value.')
    if end < 0 or end > 800:
        raise ValueError('The ending date must be comprised between 0 and 800.')

    # --- PREPARE DATA ---
    # Load the data
    X, X_ext, y_all = load_data()
    X_ext = resample(X_ext, start, end)

    # Prepare output data
    y_pred_all = []
    shap_values_list = []

    # --- START THE PREDICTION ---
    for i in tqdm(range(nsim), desc="Bootstrap Iterations"):

        # ---- PREPROCESSING ----
        # Define a random ISOW intensity dataset (between the 1st to the 99th percentile)
        y = np.array([y_all.iloc[i, np.random.choice(len(y_all.columns))] for i in range(y_all.shape[0])])

        # Select training and validation datasets
        X_train, X_val, y_train, y_val = split_data(X, y)

        # Define linear and XGB datasets
        X_train_lin, X_train_xgb = x_for_models(X_train)
        X_val_lin, X_val_xgb = x_for_models(X_val)
        X_ext_lin, X_ext_xgb = x_for_models(X_ext)

        # Imputation of NaN values
        imputer_lin = fit_imputer(X_train_lin)
        X_train_lin = impute_data(X_train_lin, imputer_lin)
        X_val_lin = impute_data(X_val_lin, imputer_lin)
        X_ext_lin = impute_data(X_ext_lin, imputer_lin)

        imputer_xgb = fit_imputer(X_train_xgb)
        X_train_xgb = impute_data(X_train_xgb, imputer_xgb)
        X_val_xgb = impute_data(X_val_xgb, imputer_xgb)
        X_ext_xgb = impute_data(X_ext_xgb, imputer_xgb)

        # ---- TRAINING: LINEAR MODEL ----
        # Start to train the linear model
        model_linear.fit(X_train_lin, y_train)
        y_train_pred = model_linear.predict(X_train_lin) # To calculate the residues of training
        y_val_pred = model_linear.predict(X_val_lin) # To calculate the residues of validation

        # Calculate residues (to train the next XGB model)
        residuals_train = y_train - y_train_pred
        residuals_val = y_val - y_val_pred

        # ---- TRAINING: XGB-REGRESSOR MODEL ----
        # Train the XGB Regressor model (find best_params for each iteration)
        best_params_xgb = best_xgb_params(X_train_xgb, X_val_xgb, residuals_train, residuals_val)
        model_xgb = XGBRegressor(**best_params_xgb)
        model_xgb.fit(X_train_xgb, residuals_train)

        # ---- PREDICTION ACROSS THE LAST 800,000 YEARS ----
        y_pred_all.append((model_linear.predict(X_ext_lin) + model_xgb.predict(X_ext_xgb)).tolist())

        # ---- EVALUATE FEATURE'S INFLUENCES WITH SHAP ----
        # Calculate the SHAP values for the linear model
        explainer_linear = shap.Explainer(model_linear, X_ext_lin)
        shap_values_linear = explainer_linear(X_ext_lin).values

        # Calculate the SHAP values for the XGB model
        explainer_xgb = shap.Explainer(model_xgb)
        shap_values_xgb = explainer_xgb(X_ext_xgb).values

        # Convert SHAP values to two pd.DataFrames (linear and XGB)
        shap_values_linear_df = pd.DataFrame(shap_values_linear, columns=X_ext_lin.columns, index=X_ext_lin.index)
        shap_values_xgb_df = pd.DataFrame(shap_values_xgb, columns=X_ext_xgb.columns, index=X_ext_xgb.index)

        # Find common columns between linear and XGB models
        common_columns = shap_values_linear_df.columns.intersection(shap_values_xgb_df.columns)

        # If there are common columns, add them
        if not common_columns.empty:
            for col in common_columns:
                shap_values_linear_df[col] += shap_values_xgb_df[col]

            # Drop the common columns from shap_values_xgb_df, since they are now combined
            shap_values_xgb_df = shap_values_xgb_df.drop(columns=common_columns)

        # Combine both SHAP values
        shap_values_combined = pd.concat([shap_values_linear_df, shap_values_xgb_df], axis=1)

        # Append to store the simulation's SHAP values
        shap_values_list.append(shap_values_combined)

        if i+1 == nsim:     # To store the name of the features
            features_names = shap_values_combined.columns.tolist()

    # ---- CALCULATE PERCENTILE OF PREDICTIONS AND AVERAGE SHAP VALUES ----
    # For the prediction (percentile calculation)
    y_pred_all = pd.DataFrame(y_pred_all).T
    percentiles = np.arange(1, 100)
    pct_prediction = np.percentile(y_pred_all, percentiles, axis=1).T
    output = pd.DataFrame(pct_prediction, index=X_ext.index)

    # Define medians of SHAP values for each feature
    shap_values_median = np.median(np.array([df.values for df in shap_values_list]), axis=0)
    shap_values_median_df = pd.DataFrame(shap_values_median, columns=features_names, index=X_ext.index)

    # ---- EXPORT THE MODEL's OUTPUTS ----
    # Define the repository path (and create it if it doesn't exist)
    DIR_OUTPUT = Path('./outputs').resolve()
    os.makedirs(DIR_OUTPUT, exist_ok=True)

    # Export predictions (percentiles)
    output.to_csv(DIR_OUTPUT / "ISOW_modeled.csv", sep='\t', index=True)
    print("ISOW_modeled is saved in: ", DIR_OUTPUT / "ISOW_modeled.csv")
    # Export SHAP values
    shap_values_median_df.to_csv(DIR_OUTPUT / "SHAP_values.csv", sep='\t', index=True)
    print("SHAP values are saved in: ", DIR_OUTPUT / "SHAP_values.csv")

    print("\n~~~~~~~~~~~~~ Prediction of ISOW strength is done ~~~~~~~~~~~~~")
    return output, shap_values_median_df