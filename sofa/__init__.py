"""
SOFA Library - Signal processing and machine learning for optical fiber communications.

This package provides modular tools for:
- 16-QAM modulation/demodulation and signal processing
- Machine learning-based demodulation methods
- Neural network demodulation
- Performance metrics (SER, BER)
- Data I/O operations
- Utility functions

For backward compatibility, all functions are available at the package level.
"""

import os

# Import all public functions for backward compatibility
from .modulation import (
    MOD_DICT,
    mod_norm,
    demodulate,
    realify,
    calc_noise,
    add_awgn,
    sync_signals,
)

from .ml_demodulation import (
    kfold_cross_validation,
    demodulate_knn,
    demodulate_svm,
    demodulate_kmeans,
    find_best_params,
)

try:
    from .neural_networks import (
        classifier_model,
        demodulate_neural,
    )
except ImportError:
    # TensorFlow not available, skip neural network functions
    pass

from .metrics import (
    symbol_error_rate,
    bit_error_rate,
)

from .data_io import (
    load_16gbaud_db,
    save_json,
    load_json,
    save_hdf5,
    load_hdf5,
    joblib_save,
    joblib_load,
)

from .utils import (
    setup_logger,
    find_root,
    curve_fit,
)

# Initialize logger for backward compatibility
FILENAME = os.path.basename(__file__)[:-3] if __file__ else "sofa"
logger = setup_logger(FILENAME)

# Define public API
__all__ = [
    # Modulation
    "MOD_DICT",
    "mod_norm",
    "demodulate",
    "realify",
    "calc_noise",
    "add_awgn",
    "sync_signals",
    
    # ML Demodulation
    "kfold_cross_validation",
    "demodulate_knn",
    "demodulate_svm",
    "demodulate_kmeans",
    "find_best_params",
    
    # Neural Networks
    "classifier_model",
    "demodulate_neural",
    
    # Metrics
    "symbol_error_rate",
    "bit_error_rate",
    
    # Data I/O
    "load_16gbaud_db",
    "save_json",
    "load_json",
    "save_hdf5",
    "load_hdf5",
    "joblib_save",
    "joblib_load",
    
    # Utils
    "setup_logger",
    "find_root",
    "curve_fit",
    
    # Logger instance
    "logger",
]
