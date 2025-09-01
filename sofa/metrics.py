"""
Performance evaluation metrics for signal processing.

This module contains functions for calculating various error rates
and performance metrics for communication systems.
"""

import numpy as np


def symbol_error_rate(sym_rx: np.ndarray, sym_tx: np.ndarray) -> float:
    """
    Calculates the symbol error rate (SER).

    Parameters:
        sym_rx: Vector of received symbols.
        sym_tx: Vector of transmitted symbols.

    Returns:
        float: Symbol error rate, the proportion of symbol errors.
    """
    error = sum(rx != tx for rx, tx in zip(sym_rx, sym_tx))
    ser = error / len(sym_tx)
    return ser


def bit_error_rate(sym_rx: np.ndarray, sym_tx: np.ndarray) -> float:
    """
    Calculates the bit error rate (BER).

    Parameters:
        sym_rx: Vector of received symbols.
        sym_tx: Vector of transmitted symbols.

    Returns:
        float: Bit error rate, the proportion of bit errors.
    """
    # Convert symbols to binary strings
    sym_rx_str = "".join([f"{sym:04b}" for sym in sym_rx])
    sym_tx_str = "".join([f"{sym:04b}" for sym in sym_tx])

    error = sum(sym_rx_str[i] != sym_tx_str[i] for i in range(len(sym_rx_str)))
    ber = error / len(sym_rx_str)
    return ber
