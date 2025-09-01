"""
Utility functions and configuration helpers.

This module contains various utility functions for logging setup,
path management, and curve fitting operations.
"""

import logging
import subprocess
from pathlib import Path

import scipy as sp


def setup_logger(name: str) -> logging.Logger:
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(f"{name}.log"),
        ],
    )
    logger = logging.getLogger(name)
    return logger


def find_root() -> Path:
    """
    Find the root directory of the Git project.

    Returns:
        Path: The absolute path of the Git root directory, or None if not found.
    """
    try:
        # Run 'git rev-parse --show-toplevel' to get the root directory
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Check if the command was successful
        if result.returncode == 0:
            return Path(result.stdout.strip())
        else:
            # Git command failed, print the error message
            logging.getLogger(__name__).error(result.stderr.strip())
            return None

    except Exception as e:
        # Handle exceptions, e.g., subprocess.CalledProcessError
        logging.getLogger(__name__).error(str(e))
        return None


def curve_fit(f, x, y):
    """
    Calculates the optimal parameters given a function and data points to optimize.

    Parameters:
        f: Function to be optimized.
        x: Coordinates on the x-axis.
        y: Coordinates on the y-axis.

    Returns:
        np.ndarray: Optimized parameters to fit the function to the given coordinates.
    """
    popt, _ = sp.optimize.curve_fit(f, x, y)
    return popt
