from .utils import load_data
from .utils import resample

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.collections import LineCollection
import pandas as pd
import seaborn as sns
import shap
import numpy as np
import os
import math

"""
This module regroup some functions to visualize the prediction results.

PLEASE DO NOT MODIFY ANYTHING IN THIS MODULE.
"""
# ~~~~~~~~~~~~~~~~ K-FOLD RESIDUALS ~~~~~~~~~~~~~~~~

def plot_kfold_residuals(residuals_list):
    """
    This function plot the residuals (y - y_pred) from the cross-validation for each fold.
    :param residuals_list: list of lists from the cross-validation.
    """
    DIR_FIG = Path('./figures').resolve()
    os.makedirs(DIR_FIG, exist_ok=True)

    # Define the organization of subplots
    num_folds = len(residuals_list)
    cols = min(num_folds, 3)
    rows = (num_folds + cols - 1) // cols

    # Create the figure
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).reshape(-1)

    # Loop over folds
    for i, (residuals, ax) in enumerate(zip(residuals_list, axes)):
        sns.histplot(residuals, bins=30, kde=True, color="royalblue", edgecolor="black", alpha=0.7, ax=ax)
        ax.axvline(0, color='red', linestyle='dashed', label='Zero Residual')
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title(f"Fold {i + 1}")
        ax.legend()

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(DIR_FIG / "residuals_kfold.png", dpi=300)
    plt.show()
    print("\n The figure (k-fold residuals) is saved as :", DIR_FIG / "residuals_kfold.png")

# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------

class PlotISOW():
    def __init__(self, start, end):
        self.DIR_DATA = Path('./outputs').resolve()
        self.DIR_FIG = Path('./figures').resolve()
        os.makedirs(self.DIR_FIG, exist_ok=True)

        # ~~~~~~~~~~~~~~~~ LOAD DATA ~~~~~~~~~~~~~~~~
        _, X_ext, _ = load_data()
        self.X_ext = resample(X_ext, start, end)
        self.age = self.X_ext.index

    # ~~~~~~~~~~~~~~~~ SHADING PLOT ~~~~~~~~~~~~~~~~
    @staticmethod
    def shading_plot(scale, shading_mat, color, ax, lim, label=None):
        num_shades = 49  # Number of band around the median
        hcloud = []

        # Convert the data to the right format
        if isinstance(scale, pd.Series):
            scale = scale.to_numpy()
        if not isinstance(shading_mat, pd.DataFrame):
            shading_mat = pd.DataFrame(shading_mat)
        if len(shading_mat.iloc[0]) == len(scale) and len(shading_mat) != len(scale):
            shading_mat = shading_mat.T

        # Calculate color: transition from the initial color (median) to the white (most external percentile)
        end_color = np.array([1, 1, 1])  # Light gray
        shades = np.linspace(color, end_color, num_shades)

        for ii in range(1, num_shades + 1):
            bot = shading_mat.iloc[:, 99 - ii].values  # Lower percentile
            top = shading_mat.iloc[:, ii].values  # Upper percentile

            # Construct the x and y coordinates for the iith band
            confy = np.concatenate([[bot[0]], [top[0]], top[1:], [bot[-1]], bot[::-1][:-1]])
            confx = np.concatenate([[scale[0]], [scale[0]], scale[1:], [scale[-1]], scale[::-1][:-1]])

            color_band = shades[ii - 1]

            # Plot the iith band
            patches = ax.fill(confx, confy, color=color_band, edgecolor='none', alpha=0.05)
            for patch in patches:
                patch.set_rasterized(True)  # Rasterise each polygon patch
                hcloud.append(patch)

        # Plot the median
        median = shading_mat.iloc[:, 49].values
        ax.plot(scale, median, '-', linewidth=0.7, color=color, label=label)

        # Add legend and details
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylabel(label, fontweight='bold')

    # ~~~~~~~~~~~~~~~~ RESULT OVERVIEW ~~~~~~~~~~~~~~~~
    def summary(self):
        """
        Method to plot the time-series of ISOW_model and forcing used in the model.
        """
        try:
            ISOW_modeled = pd.read_csv(self.DIR_DATA / "ISOW_modeled.csv", delimiter="\t", index_col=0)
        except:
            raise "ISOW_model is not already determined. Please, run isow.model() before trying to plot."

        lim = [min(self.age), max(self.age)]

        fig, axes = plt.subplots(5, 1, figsize=(10, 16), sharex=False)

        # Obliquity and precession
        axes[0].plot(self.age, self.X_ext['obliquity'], linewidth=1, color='black')
        axes[0].set_xlabel('Age (ka)', fontweight='bold')
        axes[0].set_ylabel('Obliquity (°)', fontweight='bold')
        axes[0].xaxis.set_label_position('top')
        axes[0].xaxis.tick_top()
        axes[0].set_xlim(lim[0], lim[1])

        axes2 = axes[0].twinx()
        axes2.plot(self.age, self.X_ext['precession'], linewidth=1, color='darkorange')
        axes2.set_ylabel('Precession ($e \cdot \sin(\omega)$)', color='darkorange', fontweight='bold')
        axes2.tick_params(axis='y', colors='darkorange')
        axes2.invert_yaxis()
        axes2.set_xlim(lim[0], lim[1])

        # d18O data
        axes[1].plot(self.age, self.X_ext['d18O'], linewidth=1, color='dodgerblue')
        axes[1].set_ylabel('Benthic \u03B4¹⁸O U1385', fontweight="bold", color="dodgerblue")
        axes[1].set_xticklabels([])
        axes[1].invert_yaxis()
        axes[1].tick_params(axis='y', colors='dodgerblue')
        axes[1].set_xlim(lim[0], lim[1])

        # CO2 data
        axes[2].plot(self.age, self.X_ext['CO2'], linewidth=1, color='forestgreen')
        axes[2].set_ylabel('CO$_2$ composite (ppmv)', fontweight="bold", color="forestgreen")
        axes[2].set_xticklabels([])
        axes[2].yaxis.set_label_position('right')
        axes[2].yaxis.tick_right()
        axes[2].tick_params(axis='y', colors='forestgreen')
        axes[2].set_xlim(lim[0], lim[1])

        # ISOW strength
        self.shading_plot(self.age, ISOW_modeled, (0, 0, 1), axes[3], [lim[0], lim[1]])
        axes[3].set_ylabel('ISOW$_{data}$', fontweight="bold", color="blue")
        axes[3].set_xticklabels([])
        axes[3].tick_params(axis='y', colors='blue')
        axes[3].yaxis.set_label_position('right')
        axes[3].yaxis.tick_right()

        # ODP-983 data
        axes[4].fill_between(self.age, self.X_ext['NPS'], 0, color='tomato', alpha=0.6)
        axes[4].set_ylabel('ODP-983 NPS (%)', fontweight="bold", color="tomato")
        axes[4].set_xlabel('Age (ka)', fontweight='bold')
        axes[4].invert_yaxis()
        axes[4].set_ylim(100, 0)
        axes[4].tick_params(axis='y', colors='tomato')
        axes[4].set_xlim(lim[0], lim[1])

        axes2 = axes[4].twinx()
        axes2.fill_between(self.age, np.exp(self.X_ext['IRD']), 0, color='cornflowerblue', alpha=0.6)
        axes2.set_ylabel('ODP-983 IRD (#/g)', fontweight="bold", color="cornflowerblue")
        axes2.set_ylim(0, 4000)
        axes2.tick_params(axis='y', colors='cornflowerblue')
        axes2.set_xlim(lim[0], lim[1])

        plt.tight_layout()
        plt.savefig(self.DIR_FIG / "isow_modeled_summary.png", dpi=300)
        plt.show()

    # ~~~~~~~~~~~~~~~~ SHAP PLOTS ~~~~~~~~~~~~~~~~
    def shap_summary(self):
        """
        Method to save a violin plot of SHAP values.
        """
        try:
            shap_values = pd.read_csv(self.DIR_DATA / "SHAP_values.csv", delimiter="\t", index_col=0)
        except:
            raise "SHAP values are not already determined. Please, run isow.model() before trying to plot."
        # --- SHAP SUMMARY PLOT ---

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values.values,
                          self.X_ext.values,
                          feature_names=shap_values.columns.tolist(),
                          show=False)
        plt.savefig(self.DIR_FIG / "SHAP_summary_plot.png", dpi=300)

    def shap_bars(self):
        """
        Method to save a histogram of SHAP mean absolute values.
        """
        try:
            shap_values = pd.read_csv(self.DIR_DATA / "SHAP_values.csv", delimiter="\t", index_col=0)
        except:
            raise "SHAP values are not already determined. Please, run isow.model() before trying to plot."
        # --- SHAP BAR PLOT ---
        plt.figure(figsize=(10, 6))
        shap_exp_median = shap.Explanation(
            values=shap_values.values,
            data=self.X_ext.values,
            feature_names=list(shap_values.columns)
        )
        shap.plots.bar(shap_exp_median, show=False)
        plt.savefig(self.DIR_FIG / "SHAP_bar_plot.png", dpi=300)

    def shap_time_series(self):
        """
        Method to plot SHAP time-series for each forcing used in the model. Color indicates forcing values.
        """
        try:
            shap_values = pd.read_csv(self.DIR_DATA / "SHAP_values.csv", delimiter="\t", index_col=0)
        except:
            raise "SHAP values are not already determined. Please, run isow.model() before trying to plot."
        # --- SHAP TIME-SERIES PLOT ---
        shap_df = shap_values.copy()
        features = shap_values.columns.tolist()

        # Define subplot dimensions
        num_features = len(shap_values.columns)
        cols = 3
        rows = math.ceil(num_features / cols)

        # Create the figure
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3), sharex=True, sharey=True)
        axes = axes.flatten()  # Transformer en liste pour itération facile

        # Internal function to extract "colors"
        def get_colormap(values):
            norm = colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
            cmap = cm.coolwarm
            return cmap, norm

        # Plot each feature
        for i, feature in enumerate(shap_values.columns):
            ax = axes[i]
            feat = features[i]
            shap_val = shap_values[feat].values
            feat_val = self.X_ext[feat].values

            cmap, norm = get_colormap(feat_val)
            color_vals = cmap(norm(feat_val))

            # Line colored by feature value
            points = np.array([self.age, shap_val]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create LineCollection with colors mapped to feature values
            lc = LineCollection(segments, colors=color_vals, linewidths=1)
            ax.add_collection(lc)
            lc.set_rasterized(True)

            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_xlim(0, 800)
            ax.set_ylim(-3, 2)
            ax.set_title(feat, fontsize=10)
            ax.set_ylabel("SHAP")

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Figure details
        plt.xlabel("Time (ka)")
        fig.suptitle("SHAP Time Series on 800 ka (Feature Contributions)", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.DIR_FIG / "SHAP_time-series_plot.png", dpi=300)
        plt.show()

    def shap_dependance(self):
        """
        Method to save dependance plots: scatter plots with SHAP against forcing values.
        """
        try:
            shap_values = pd.read_csv(self.DIR_DATA / "SHAP_values.csv", delimiter="\t", index_col=0)
        except:
            raise "SHAP values are not already determined. Please, run isow.model() before trying to plot."
        # --- SHAP DEPENDANCE PLOT ---
        # Create the figure
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)

        for i, feature in enumerate(shap_values.columns):
            row, col = divmod(i, 3)

            # Take the values and SHAP values of the feature
            shap_values_feature = shap_values[feature]
            feature_values = self.X_ext[feature]

            # Scatter plot
            axes[row, col].scatter(feature_values, shap_values_feature, alpha=0.2, color="blue", s=10)
            axes[row, col].axhline(0, color='lightgrey', linewidth=0.5)

            axes[row, col].set_xlabel(f"{feature} Value")
            axes[row, col].set_ylabel("SHAP Value")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust the figure
        plt.tight_layout()
        plt.savefig(self.DIR_FIG / "SHAP_dependance_plot.png", dpi=300)
        plt.show()

