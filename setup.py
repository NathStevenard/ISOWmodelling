from setuptools import setup, find_packages

setup(
    name='ISOWmodelling',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.25.2',
        'pandas>=2.2.2',
        'scikit-learn>=1.5.1',
        'shap>=0.47.0',
        'xgboost>=2.1.4',
        'optuna>=4.2.1',
        'tqdm>=4.66.5',
        'matplotlib>=3.10.1',
        'seaborn>=0.13.2',
        'scipy>=1.12.0',
        'setuptools>=76.0.0'
    ],
    author='Stevenard Nathan',
    author_email='nathan.stevenard@univ-grenoble-alpes.fr',
    description='Hybrid data-driven model the ISOW strength over the last 800,000 years using climatic records.',
    url='https://github.com/NathStevenard/ISOWmodelling',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11.8',
)