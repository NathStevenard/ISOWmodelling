from .utils import load_data
from .utils import x_for_models
from .utils import best_xgb_params
from .utils import fit_imputer
from .utils import impute_data
from .plots import plot_kfold_residuals

import os
from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This module run a k-fold cross validation.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- CAN BE MODIFIED ---
model_linear = LinearRegression()  # Define the linear model
# --- CAN BE MODIFIED ---

def k_fold_cross_validation(k=5, save=False, plot=False):
    """
    This function run a cross-validation (k-fold) to evaluate the model via metrics.
    :param k: integer - number of folds. Default = 5 to have 80% to train, 20% to validate.
    :param save: boolean - If you want to save the metrics into a .csv file.
    :param plot: boolean - If you want to plot the residuals for each fold.
    :return: Three datasets:
        - results: models parameters
        - X_splits: X_train used for each fold
        - resid: y_val - prediction for each fold
    """
    # Load the data
    X, _, y = load_data()
    y = pd.Series(y["pct_50"])

    # Define folds
    kf = KFold(n_splits=k, shuffle=True, random_state=0)

    # Prepare necessary lists
    rmse_scores = []
    r2_scores = []
    pearson_corrs = []
    spearman_corrs = []
    trained_models = []
    X_splits = []
    resid = []

    # Run the cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X), start=1):
        print(f"==> Fold {fold}/{k} is running...")

        # ---- PREPROCESSING ----
        # Select training and validation datasets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Define linear and XGB datasets
        X_train_lin, X_train_xgb = x_for_models(X_train)
        X_val_lin, X_val_xgb = x_for_models(X_val)

        # Imputation of NaN values
        imputer_lin = fit_imputer(X_train_lin)
        X_train_lin = impute_data(X_train_lin, imputer_lin)
        X_val_lin = impute_data(X_val_lin, imputer_lin)

        imputer_xgb = fit_imputer(X_train_xgb)
        X_train_xgb = impute_data(X_train_xgb, imputer_xgb)
        X_val_xgb = impute_data(X_val_xgb, imputer_xgb)

        # ---- LINEAR MODEL ----
        # Train the linear model
        model_linear.fit(X_train_lin, y_train)
        y_train_pred = model_linear.predict(X_train_lin)
        y_val_pred = model_linear.predict(X_val_lin)

        # Calculate residuals
        residuals_train = y_train - y_train_pred
        residuals_val = y_val - y_val_pred

        # ---- XBG REGRESSOR ---
        # Find the "best" hyper-parameters for the XGB model
        best_params_xgb = best_xgb_params(
            X_train_xgb,
            X_val_xgb,
            residuals_train,
            residuals_val
        )

        # Train the model (with y = residuals from the linear model)
        model_xgb = XGBRegressor(**best_params_xgb)
        model_xgb.fit(X_train_xgb, residuals_train)
        trained_models.append((model_linear, model_xgb)) # Store trained data

        # Add linear and XGB models to produce the final prediction
        y_val_pred_final = y_val_pred + model_xgb.predict(X_val_xgb)

        # ---- CALCULATE METRICS ----
        rmse = root_mean_squared_error(y_val, y_val_pred_final)
        r2 = r2_score(y_val, y_val_pred_final)
        pearson_corr = np.corrcoef(y_val, y_val_pred_final)[0, 1]
        spearman_corr, _ = spearmanr(y_val, y_val_pred_final)

        # Fill the lists
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr)
        X_splits.append(X_train_xgb)
        resid.append(y_val - y_val_pred_final)

        print(f"Fold {fold} - RMSE: {rmse:.4f}, R²: {r2:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")

    # Summarize the metrics (average performance)
    metrics = {
        "RMSE_mean": np.mean(rmse_scores), "RMSE_std": np.std(rmse_scores),
        "R2_mean": np.mean(r2_scores), "R2_std": np.std(r2_scores),
        "Pearson_mean": np.mean(pearson_corrs), "Pearson_std": np.std(pearson_corrs),
        "Spearman_mean": np.mean(spearman_corrs), "Spearman_std": np.std(spearman_corrs)
    }

    # Print the average performances
    print("\nPerformance summary:")
    print(f"RMSE: {metrics['RMSE_mean']:.4f} ± {metrics['RMSE_std']:.4f}")
    print(f"R²: {metrics['R2_mean']:.4f} ± {metrics['R2_std']:.4f}")
    print(f"Pearson: {metrics['Pearson_mean']:.4f} ± {metrics['Pearson_std']:.4f}")
    print(f"Spearman: {metrics['Spearman_mean']:.4f} ± {metrics['Spearman_std']:.4f}")

    results = {"Trained_models": trained_models}

    # Export the average performances (if desired):
    if save:
        DIR_OUTPUT = Path('./outputs').resolve()
        os.makedirs(DIR_OUTPUT, exist_ok=True)
        # Export the metrics
        output = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        output.to_csv(DIR_OUTPUT / "performances.csv")
        print("\n Metrics are saved in :", DIR_OUTPUT / "performances.csv")

    if plot:
        plot_kfold_residuals(resid)

    print("\n~~~~~~~~~~~~ The K-fold cross correlation is done. ~~~~~~~~~~~~")
    return results, X_splits, resid