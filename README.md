# ISOWmodelling: A Hybrid Reconstruction of Deep North Atlantic Circulation Over the Last 800,000 Years.

![Licence MIT](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11+-blue)

---

This repository contains the full modelling framework developed to reconstruct the strength 
of the Iceland-Scotland Overflow Water (ISOW), a key deep branch of the North Atlantic Deep 
Water (NADW), over the last 800,000 years. The approch combines linear regression and 
machine learning (XGBoost) in a Monte Carlo setting to capture both linear and nonlinear 
relationships between climate forcings and ISOW variability.

The model and dataset are described in detail in:

Stevenard, N., Govin, A., Kissel, C., Swingedouw, D., Toucanne, S., & Bouttes, N, in prep.
*800,000 years of deep North Atlantic bistability*.

**DISCLAIMER**

This code is provided under the MIT License. However, **any scientific use of this code, 
including but not limited to publication, citation, or distribution of results derived from 
it, is strictly prohibited without prior written permission from the author(s).**

By accessing or using this repository, you agree to respect this restriction. For any 
inquiries or requests regarding scientific use, please contact the repository owner directly.

Thank you for your understanding.

---

## Quickstart

**Clone the repository:**
```bash
git clone https://github.com/NathStevenard/ISOWmodelling.git
cd ISOWmodelling
```

**Create and activate a virtual environment**
```bash
python -m venv env
source env/bin/activate
```

**Install the package and dependancy**
```bash
pip install .
```

## How to use

In your folder ***ISOWmodelled/***, create a new Python file (or use *test_script.py*). Once created, you can use ***"import isow"*** and run the functions.

```python
import isow

start = 0
end = 0
# Run the cross-validation:
results, X_splits, residuals = isow.k_fold_cross_validation(k=5, save=True, plot=True)

# Run the model for the last 800,000 years
isow_modeled, shap_values = isow.model(start, end, nsim=100)

# Plot the results
plot = isow.plots.PlotISOW(start, end)
plot.summary()  
```
More details are available in the *test_script.py* file.

---
## Input data

All preprocessed time series are located in ***data/***, including:
    
    Obliquity and Precession (Berger & Loutre, 1991)

    Atmospheric CO2 concentration (composite from Legrain et al., 2024)

    Benthic d18O from the IODP site U1385 (Hodell et al., 2023)

    Ice-rafted debris (IRD, in log) and N. pachyderma sinistral (NPS) counts from ODP site 983 
    (Barker et al., 2021)

    ISOW strength stack based on a Monte Carlo process using Principal Component Analysis. 
    This dataset is produced in this study.

The chronology of the ODP site 983 was rescaled to the "hybrid chronology" of site U1385
(Hodell et al., 2023).

## Model overview

The hybrid modeling pipeline works in two steps:

    A linear model predicts ISOW strength using IRD, NPS.

    An XGBoost model predicts the residuals using nonlinear climate drivers (IRD, NPS, CO2, 
    benthic d18O, obliquity, precession).

    Predictions are repeated over 100 Monte Carlo iterations to estimate uncertainties.

    SHAP values are computed to assess variable importance and track their impact through 
    time.

## Outputs

    Performances: metrics related to the cross-validation    

    ISOW_modeled: predicted ISOW strength over the 0-800,000 year period (percentiles)

    SHAP_values: SHAP (Lundberg & Lee, 2017) time series for each features

    Figures: figures associated with the plot functions

---

## Citations

### Related paper
Please, cite the original paper once published (upcoming reference).

### Other citations
If you use SHAP, please cite:

    - Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

If you use the forcing data please, cite:

    - IRD & NPS: Barker, Stephen (2021): Planktic foraminiferal and Ice Rafted Debris (IRD) counts from ODP Site 983 [dataset]. PANGAEA.
    - CO2: Legrain, E., Capron, E., Menviel, L., Wohleber, A., Parrenin, F., Teste, G., ... & Stocker, T. F. (2024). Centennial-scale variations in the carbon cycle enhanced by high obliquity. Nature Geoscience, 17(11), 1154-1161.
    - d18O: Hodell, David A; Crowhurst, Simon J; Lourens, Lucas Joost; Margari, Vasiliki; Nicolson, John; Rolfe, James E; Skinner, Luke C; Thomas, Nicola C; Tzedakis, Polychronis C; Mleneck-Vautravers, Maryline J; Wolff, Eric William (2023): Benthic and planktonic oxygen and carbon isotopes and XRF data at IODP Site U1385 and core MD01-2444 from 0 to 1.5 Ma [dataset bundled publication]. PANGAEA.
    - Obliquity and Precession: Berger, A., & Loutre, M. F. (1991). Insolation values for the climate of the last 10 million years. Quaternary science reviews, 10(4), 297-317.

Please do not forget to cite developers for all Python module you use.

## Contact

For questions or contributions, please contact:
nathan.stevenard@univ-grenoble-alpes.fr or open an issue.
