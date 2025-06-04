# --- IMPORT THE PACKAGES ON YOUR LOCAL ENVIRONMENT ---------------------
import isow

# --- DEFINE THE PERIOD YOU WANT TO SIMULATE ISOW STRENGTH --------------
"""

"""
start = 0
end = 800

# --- K-FOLD CROSS VALIDATION -------------------------------------------
"""
This cross validation allows an estimation of the model's performances. If you don't want to run the cross-
validation, you can simply comment the line (add "#" in front of the line). Optional arguments are described above.
"""
# Params (optional): k=3 (default is 5). --> Number of folds.
# Params (optional): save=True (default is False) --> Save the metrics (average).
# Params (optional): plot=True (default is False) --> Plot the residuals for each fold.
results, X_splits, residuals = isow.k_fold_cross_validation(k=5, save=True, plot=True)

# --- PREDICTION OF ISOW INTENSITY --------------------------------------
"""
This is the function to model ISOW across the last 800,000 years. By default, the number of iterations through the
bootstrap (nsim) is fixed to 100. You can modify it.
"""

# Params (optional): nsim=500 (default 100). --> Number of simulations for the bootstrap process.
isow_modeled, shap_values = isow.model(start=start, end=end, nsim=100)

# --- PLOTS -------------------------------------------------------------
"""
Optional. This is a few functions to vizualize the results.
"""
plot = isow.plots.PlotISOW(start, end)    # Initiate the "plot" class

plot.summary()                            # Time-series of the simulated ISOW vs forcings
plot.shap_summary()                       # Violin plots of SHAP values
plot.shap_bars()                          # Histograms of mean absolute SHAP values
plot.shap_time_series()                   # Time-series of SHAP values
plot.shap_dependance()                    # Dependance plots of the SHAP values

# -----------------------------------------------------------------------