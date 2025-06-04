from pathlib import Path
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna
logging.getLogger("optuna").setLevel(logging.WARNING)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
This module merge functions necessary to run the model.
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- DO NOT MODIFY ----------------------------------------
# Definir the dir
DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DIR / 'data'

# Define age scales
age = np.arange(1, 380.1, 0.1)  # Period to use for training
age_ext = np.arange(0, 800.1, 0.1) # Period to use for the prediction dataset
# --- DO NOT MODIFY ---------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# --- CAN BE MODIFIED -------------------------------------
features_lin = ["NPS", "IRD"]
features_xgb = ["NPS", "IRD", "d18O", "CO2", "precession", "obliquity"]
# --- CAN BE MODIFIED -------------------------------------

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_data():
    """
    This function load the data necessary to perform the model.
    :return: pd.Dataframes - "X", "X_ext" and "y".
    """
    X = pd.read_csv(DATA_DIR / "X.csv", delimiter=',', header=0)
    X_ext = pd.read_csv(DATA_DIR / "X_extended.csv", delimiter=',',header=0)
    y = pd.read_csv(DATA_DIR / "y.csv", delimiter=',',header=0)

    X = X.dropna(how='all')
    X_ext = X_ext.dropna(how='all')
    y = y.dropna(how='all')

    X.index = age
    X_ext.index = age_ext
    y.index = age

    return X, X_ext, y

def resample(X_ext, start, end):
    """
    This function return a subsampled dataset, fitting with the period of time you want to predict ISOW strength.
    :param X_ext: pd.DataFrame - Forcing values covering the last 800,000 years.
    :param start: int, float - Starting date of the period.
    :param end: int, float - Ending date of the period.
    :return: pd.DataFrame - the subsampled DataFrame.
    """
    # Find index for start and end
    start_idx = X_ext.index.get_indexer([start], method='nearest')[0]
    end_idx = X_ext.index.get_indexer([end], method='nearest')[0]

    # Return the subsampled dataset
    return X_ext.iloc[start_idx:end_idx + 1]

def x_for_models(X):
    """
    This function return two "X" for each model used (linear and XGB).
    :param X: pd.Dataframe - The "X" or "X_ext" from the function "load_data()"
    :return: pd.Dataframes - "X_lin" (linear model) and "X_xgb" (XGB model)
    """
    X_lin = X[features_lin]
    X_xgb = X[features_xgb]

    return X_lin, X_xgb

def split_data(X, y):
    """
    This function split the datasets to produce a "training" and a "valid" datasets.
    :param X: pd.DataFrame - Input (forcings) used in the model.
    :param y: pd.DataFrame - Input (ISOW_data) used in the model.
    :return: pd.DataFrames - four datasets:
        - X_train: used to train the model.
        - y_train: used to train the model.
        - X_val: used to validate the model.
        - y_val: used to validate the model.
    """
    return train_test_split(
        X,
        y,
        train_size=0.8, # --- CAN BE MODIFIED ---
        shuffle=True,
        random_state=None
    )

def fit_imputer(X_train):
    """
    This function "fit" the value to impute training datasets.
    :param X_train: pd.DataFrame - X_train from the split (split_data function).
    :return: The imputer.
    """
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    return imputer

def impute_data(X, imputer):
    """
    Impute NaN data in a dataset.
    :param X: pd.DataFrame - dataset with NaN values.
    :param imputer: Imputer - fitted with X_train.
    :return: pd.DataFrame - cleaned dataset (without NaN values).
    """
    return pd.DataFrame(
        imputer.transform(X),
        columns=X.columns,
        index=X.index
    )

def best_xgb_params(X_train, X_val, y_train, y_val):
    """
    This function find the "best" hyper-parameters of the XGBoost model to avoid overfitting.
    :param X_train: pd.DataFrame - Forcing used to train the model.
    :param X_val: pd.DataFrame - Forcing used to validate the model.
    :param y_train: pd.DataFrame - ISOW strength data used to train the model.
    :param y_val: pd.DataFrame - ISOW strength data used to validate the model.
    :return: dic - the "best" hyperparameters.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        msq = mean_squared_error(y_val, preds)
        return msq

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    return study.best_trial.params