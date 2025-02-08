# RCD: A Robust Consensus-Driven Framework for Time Series Causal Discovery

This repository contains the implementation of RCD (Robust Consensus-Driven) framework for time series causal discovery, along with its applications and experimental validations.

## Project Overview

The RCD framework extends traditional causal discovery methods by incorporating k-fold validation and robustness checks, significantly improving the reliability of causal structure identification in time series data. This project demonstrates the framework's effectiveness through applications with VAR-LiNGAM and PCMCI methods, and provides comprehensive experimental validation.

## Repository Structure

- `data/`: Contains real and synthetic datasets used in experiments.
- `results/`: Stores experimental results and analysis.
- `src/`: Source code for the RCD framework and method implementations.
  - `rcd_framework.py`: Core implementation of the Robust Consensus-Driven framework
  - `run_causal_discovery.py`: Implementation of various causal discovery methods and their RCD extensions
  - `causal_matrix_evaluation.py`: Utilities for evaluating causal matrices
  - `models/`: External model implementations (VAR-LiNGAM, PCMCI, TCDF)
- `*.ipynb`: Jupyter notebooks for running experiments and analysis.

## Key Components

1. RCD Framework Implementation (`src/rcd_framework.py`)
   - Generic RCD implementation applicable to various causal discovery methods
   - Grid search functionality for parameter optimization
   - Robust validation and adjustment procedures

2. Causal Discovery Methods (`src/run_causal_discovery.py`)
   - Base implementations: VAR-LiNGAM, PCMCI, TCDF, VAR-LiNGAM Bootstrap
   - RCD extensions: RCD-VAR-LiNGAM, RCD-PCMCI
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
from src.run_causal_discovery import run_rcd_varlingam, run_rcd_pcmci

# Run RCD-VARLiNGAM
results = run_rcd_varlingam(data, n_splits=5)

# Run RCD-PCMCI
results = run_rcd_pcmci(data, n_splits=7)