# Robust Time Series Causal Discovery for ABM Validation

This repository contains the implementation and evaluation of Robust Cross-Validated (RCV) causal discovery methods for time series data, with applications in Agent-Based Model (ABM) validation.

## Project Overview

This project introduces novel RCV extensions to VAR-LiNGAM and PCMCI causal discovery methods, aiming to improve robustness and accuracy in identifying causal structures in complex time series data. It also presents an enhanced ABM validation framework incorporating these methods.

## Repository Structure

- `data/`: Contains real and synthetic datasets used in experiments.
- `results/`: Stores experimental results and analysis.
- `src/`: Source code for causal discovery methods and RCV extensions.
- `*.ipynb`: Jupyter notebooks for running experiments and analysis.

## Key Components

1. RCV-VAR-LiNGAM and RCV-PCMCI implementations (`src/rcv_varlingam.py`, `src/rcv_pcmci.py`)
2. Synthetic dataset generator (`data/synthetic/generate_synthetic_data.ipynb`)
3. Experimental notebooks for synthetic and fMRI data (`run_experiments_*.ipynb`)
4. Enhanced ABM Validation Framework (`ABM_Validation_Framework.ipynb`)

## Setup and Usage

1. Clone the repository
2. Install required dependencies (list dependencies or include a requirements.txt)
3. Run the Jupyter notebooks to reproduce experiments or use the ABM validation framework

## Results

Experimental results demonstrate the superior performance of RCV methods across various data characteristics and scales, particularly in handling non-linear, non-Gaussian, and non-stationary time series data.

For detailed findings and analysis, refer to the results directory and individual experiment notebooks.