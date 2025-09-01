# SOFA Library

**Semillero de Óptica y Fotónica Aplicada** - A modular Python library for optical fiber communication research.

This library provides a comprehensive collection of tools for simulating and analyzing communication systems, specifically designed for the 'Semillero de Óptica y Fotónica Aplicada' (SOFA) research group.

## Features

The SOFA library is organized into specialized modules:

### 🔧 Core Modules

- **`sofa.modulation`** - 16-QAM modulation/demodulation and signal processing
  - Constellation dictionaries and normalization
  - Traditional grid-based demodulation
  - AWGN noise generation and signal synchronization
  - Complex-to-real signal transformations

- **`sofa.ml_demodulation`** - Machine learning-based demodulation
  - K-fold cross-validation framework
  - KNN, SVM, and K-Means demodulation algorithms
  - Hyperparameter optimization with GridSearchCV

- **`sofa.neural_networks`** - Deep learning demodulation
  - Neural network classifier creation
  - Neural network-based demodulation with early stopping

- **`sofa.metrics`** - Performance evaluation
  - Symbol Error Rate (SER) calculation
  - Bit Error Rate (BER) calculation

- **`sofa.data_io`** - Data management and persistence
  - CSV/JSON/HDF5/joblib file operations
  - Automatic backup and versioning system
  - Database loading utilities

- **`sofa.utils`** - Utilities and configuration
  - Logging setup and management
  - Git repository utilities
  - Curve fitting operations

## Installation

### Option 1: Development Installation (recommended)

```bash
git clone <repository-url>
cd sofa-lib
pip install -e .
```

### Option 2: Standard Installation

```bash
git clone <repository-url>
cd sofa-lib
pip install .
```

### Option 3: With Neural Network Support

```bash
pip install -e ".[neural]"  # Includes TensorFlow
```

### Option 4: Development Tools

```bash
pip install -e ".[dev]"     # Includes testing and formatting tools
```

## Usage

### Import Options

**Option 1: Import individual modules (recommended)**
```python
from sofa import modulation, metrics, ml_demodulation

# Use 16-QAM constellation
constellation = modulation.MOD_DICT
result = modulation.demodulate(received_data, constellation)

# Calculate performance metrics
ser = metrics.symbol_error_rate(rx_symbols, tx_symbols)
ber = metrics.bit_error_rate(rx_symbols, tx_symbols)

# Use machine learning demodulation
knn_result = ml_demodulation.demodulate_knn(rx_data, tx_symbols, k=5)
```

**Option 2: Import entire package (backward compatibility)**
```python
import sofa

# All functions available at package level
result = sofa.demodulate(received_data, sofa.MOD_DICT)
ser = sofa.symbol_error_rate(rx_symbols, tx_symbols)
```

**Option 3: Import specific functions**
```python
from sofa.modulation import MOD_DICT, demodulate, add_awgn
from sofa.metrics import symbol_error_rate
from sofa.ml_demodulation import demodulate_knn
```

### Quick Start Example

```python
import numpy as np
from sofa import modulation, metrics

# Generate test constellation
tx_symbols = np.random.randint(0, 16, 1000)
tx_constellation = np.array([modulation.MOD_DICT[sym] for sym in tx_symbols])

# Add noise
rx_constellation = modulation.add_awgn(tx_constellation, snr=20)

# Demodulate
rx_symbols = modulation.demodulate(rx_constellation, modulation.MOD_DICT)

# Calculate performance
ser = metrics.symbol_error_rate(rx_symbols, tx_symbols)
ber = metrics.bit_error_rate(rx_symbols, tx_symbols)

print(f"SER: {ser:.4f}, BER: {ber:.4f}")
```

## Dependencies

- **Core**: `numpy`, `scipy`
- **Machine Learning**: `scikit-learn`
- **Deep Learning**: `tensorflow` (optional)
- **Data Processing**: `polars`, `h5py`, `joblib`

## Module Dependencies

- `modulation`: numpy
- `ml_demodulation`: numpy, scikit-learn, modulation
- `neural_networks`: tensorflow, numpy, modulation, ml_demodulation
- `metrics`: numpy
- `data_io`: polars, h5py, joblib, json, pathlib
- `utils`: scipy, subprocess, pathlib, logging
