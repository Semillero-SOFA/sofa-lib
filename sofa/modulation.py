"""
16-QAM modulation/demodulation and signal processing utilities.

This module contains functions for modulation/demodulation operations,
constellation manipulation, noise generation, and signal synchronization.
"""

import numpy as np

"""
Demodulation dictionary for 16-QAM symbols.

MOD_DICT is a dictionary that maps integer values (0 to 15) to complex numbers representing the 16-QAM constellation points.
The keys in the dictionary correspond to the binary representation of the symbols, while the values represent the complex
coordinates of the symbols in the 16-QAM constellation.
"""
MOD_DICT = {
    0: -3 + 3j,  # 0000
    1: -3 + 1j,  # 0001
    2: -3 - 3j,  # 0010
    3: -3 - 1j,  # 0011
    4: -1 + 3j,  # 0100
    5: -1 + 1j,  # 0101
    6: -1 - 3j,  # 0110
    7: -1 - 1j,  # 0111
    8: +3 + 3j,  # 1000
    9: +3 + 1j,  # 1001
    10: 3 - 3j,  # 1010
    11: 3 - 1j,  # 1011
    12: 1 + 3j,  # 1100
    13: 1 + 1j,  # 1101
    14: 1 - 3j,  # 1110
    15: 1 - 1j,  # 1111
}


def mod_norm(const: np.ndarray, power: float = 1.0) -> float:
    """
    Modify the scale of a given constellation.

    The modified normalization factor is calculated based on the desired power, allowing the scale
    of the constellation to be adjusted. This is useful for mapping the constellation to a specific
    range, such as -1 to 1 or -3 to 3.

    Parameters:
        const (np.ndarray): The input constellation for which the scale is modified.
            It should be a NumPy array of complex numbers representing the constellation points.
        power (float, optional): The desired power that determines the new scale of the constellation.
            A higher power results in a larger scale. Default is 1.0.

    Returns:
        float: The modified normalization factor, which can be used to adjust the scale of the constellation.
    """
    constPow = np.mean([x**2 for x in np.abs(const)])
    scale = np.sqrt(power / constPow)
    return scale


def realify(X: np.ndarray) -> np.ndarray:
    """
    Transforms a vector of complex numbers into a pair of real and imaginary floats.

    Parameters:
        X (np.ndarray): Received constellation.

    Returns:
        np.ndarray: Transformed constellation.
    """
    return np.column_stack((X.real, X.imag))


def demodulate(X_rx: np.ndarray, mod_dict: dict) -> np.ndarray:
    """
    Demodulates using the traditional grid-based method.

    Parameters:
        X_rx (np.ndarray): Received constellation.
        mod_dict (dict): Modulation dictionary.

    Returns:
        np.ndarray: Demodulated constellation.
    """
    demodulated = np.empty(len(X_rx), dtype=int)

    for i, x in enumerate(X_rx):
        # Distance to each centroid
        dist = np.abs(np.array(list(mod_dict.values())) - x)
        # Index of the minimum distance value
        index = np.argmin(dist)
        # Nearest centroid to the symbol
        demodulated[i] = index

    return demodulated


def calc_noise(X: np.ndarray, snr: float) -> np.ndarray:
    """
    Adds noise to a vector to achieve a desired Signal-to-Noise Ratio (SNR).

    Parameters:
        X (np.ndarray): Input vector to which noise is applied.
        snr (float): Desired Signal-to-Noise Ratio (SNR) in dB.

    Returns:
        np.ndarray: Vector with added noise.

    Example:
        >>> X = np.array([1, 2, 3, 4, 5])
        >>> snr = 20
        >>> calc_noise(X, snr)
        array([ 0.82484866,  2.3142245 ,  3.57619233,  3.34866241,  4.87691381])
    """
    X_avg_p = np.mean(np.power(X, 2))
    X_avg_db = 10 * np.log10(X_avg_p)

    noise_avg_db = X_avg_db - snr
    noise_avg_p = np.power(10, noise_avg_db / 10)

    # Setting mean to 0 (loc=0) by default for the normal distribution
    noise = np.random.normal(scale=np.sqrt(noise_avg_p), size=len(X))
    return X + noise


def add_awgn(X: np.ndarray, snr: float) -> np.ndarray:
    """
    Adds additive white Gaussian noise (AWGN) to a constellation.

    The AWGN is added independently to the real and imaginary parts of the complex constellation.

    Parameters:
        snr (float): Signal-to-Noise Ratio (SNR) in dB.
        X (np.ndarray): Original constellation.

    Returns:
        tuple[np.ndarray, np.ndarray]: Constellation with added noise, represented by the real and imaginary parts.
    """
    Xr = np.real(X)
    Xi = np.imag(X)
    return calc_noise(Xr, snr) + 1j * calc_noise(Xi, snr)


def sync_signals(tx: np.ndarray, rx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Synchronizes two signals.

    Parameters:
        tx: Short signal, usually the received signal.
        rx: Long signal, usually the transmitted signal.

    Returns:
        tuple[np.ndarray, np.ndarray]: Synchronized copies of both signals in the same order as the input parameters.
    """
    tx_long = np.concatenate((tx, tx))
    correlation = np.abs(
        np.correlate(
            np.abs(tx_long) - np.mean(np.abs(tx_long)),
            np.abs(rx) - np.mean(np.abs(rx)),
            mode="full",
        )
    )
    delay = np.argmax(correlation) - len(rx) + 1

    sync_signal = tx_long[delay:]
    sync_signal = sync_signal[: len(rx)]

    return sync_signal, rx
