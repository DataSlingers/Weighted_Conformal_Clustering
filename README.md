# Weighted_Conformal_Clustering

This repository contains the code for the weighted conformal clustering method and the reproducible experiments used in the accompanying paper "Weighted Conformal Clustering" (https://arxiv.org/abs/2606.00436).

## Repository Overview

- `environment.yml` defines the Conda environment used for reproducibility. It pins the Python version and package dependencies needed to run the code and notebooks in this repository.
- `weighted_conformal_clustering/` contains the Python package that implements our method.
- `Reproducible_Code/` contains the Jupyter notebooks used to reproduce the figures in the paper.

## Setup

Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate conformal_clustering
```

This environment is intended to provide a reproducible software setup for rerunning the experiments and figures.

## Reproducing the Paper Figures

The notebooks in `Reproducible_Code/` reproduce the figures and experiments from the paper. The `Reproducible_Code/Figures/` and `Reproducible_Code/Results/` directories contain generated outputs associated with these experiments.

## Using the Package

The core implementation lives in `weighted_conformal_clustering/`. You can use this package as the main codebase for running the weighted conformal clustering method and related utilities in your own scripts or notebooks within this repository.

## Contents

- `weighted_conformal_clustering/`: implementation of the weighted conformal clustering method
- `Reproducible_Code/`: notebooks for reproducing paper figures
- `environment.yml`: reproducible Conda environment specification
