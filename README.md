# RCV: A Robust Cross-Validated Framework for Time Series Causal Discovery

This repository contains the implementation of RCV (Robust Cross-Validated) framework for time series causal discovery, along with its applications and experimental validations.

## Project Overview

The RCV framework extends traditional causal discovery methods by incorporating cross-validation and robustness checks, significantly improving the reliability of causal structure identification in time series data. This project demonstrates the framework's effectiveness through applications with VAR-LiNGAM and PCMCI methods, and provides comprehensive experimental validation.

## Repository Structure

- `data/`: Contains real and synthetic datasets used in experiments.
- `results/`: Stores experimental results and analysis.
- `src/`: Source code for the RCV framework and method implementations.
  - `rcv_framework.py`: Core implementation of the Robust Cross-Validation framework
  - `run_causal_discovery.py`: Implementation of various causal discovery methods and their RCV extensions
  - `causal_matrix_evaluation.py`: Utilities for evaluating causal matrices
  - `models/`: External model implementations (VAR-LiNGAM, PCMCI, TCDF)
- `*.ipynb`: Jupyter notebooks for running experiments and analysis.

## Key Components

1. RCV Framework Implementation (`src/rcv_framework.py`)
   - Generic RCV implementation applicable to various causal discovery methods
   - Grid search functionality for parameter optimization
   - Robust validation and adjustment procedures

2. Causal Discovery Methods (`src/run_causal_discovery.py`)
   - Base implementations: VAR-LiNGAM, PCMCI, TCDF, VAR-LiNGAM Bootstrap
   - RCV extensions: RCV-VAR-LiNGAM, RCV-PCMCI
   - Utility functions for matrix manipulation and evaluation

3. Experimental Components
   - Synthetic dataset generator (`data/synthetic/generate_synthetic_data.ipynb`)
   - Experimental notebooks for synthetic and fMRI data (`run_experiments_*.ipynb`)
   - Application examples and case studies

## Setup and Usage

1. Clone the repository
2. Install required dependencies (list dependencies or include a requirements.txt)
3. Run the Jupyter notebooks to reproduce experiments or use the framework

### Example Usage

```python
from src.run_causal_discovery import run_rcv_varlingam, run_rcv_pcmci

# Run RCV-VARLiNGAM
results = run_rcv_varlingam(data, n_splits=5)

# Run RCV-PCMCI
results = run_rcv_pcmci(data, n_splits=7)