''' Author: Kevin Martinez
    Date created: 2020/09/28
    Purpose: This file contains the functions that are used in the semillero SOFA
    Acknowledgments: This file was created with the help of the semillero SOFA '''

import numpy as np

''' Add noise to the signal
    Parameters: signal, target SNR in dB
    Return: signal with noise'''
def add_noise(signal, target_snr_db):
    X_avg_p = np.mean(signal ** 2)
    X_avg_db = 10 * np.log10(X_avg_p)
    noise_avg_db_r = X_avg_db - target_snr_db
    noise_avg_p_r = 10 ** (noise_avg_db_r / 10)
    mean_noise = 0
    noise_r = np.random.normal(mean_noise, np.sqrt(noise_avg_p_r), len(signal))
    return signal + noise_r

''' Function that calculates the BER of a given constellation
    Parameters: Symbols of the constellation, threshold to demap the symbols
    Return: BER of the constellation'''
def demapper_sym(symbols_I, symbols_Q, threshold=2.0):
    symbol = []
    # Demapper symbols to bits (hard decision)
    for i in range(len(symbols_I)):
        if symbols_I[i] <= -threshold and symbols_Q[i] >= threshold:  # -3+3j
            symbol.append(0)
        elif symbols_I[i] <= -threshold and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold:  # -3+1j
            symbol.append(1)
        elif symbols_I[i] <= -threshold and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold:  # -3-j
            symbol.append(3)
        elif symbols_I[i] <= -threshold and symbols_Q[i] <= -threshold:  # -3-3j
            symbol.append(2)
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] >= threshold:  # -1+3j
            symbol.append(4)
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold:  # -1+j
            symbol.append(5)
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold:  # -1-j
            symbol.append(7)
        elif symbols_I[i] >= -threshold and symbols_I[i] <= 0 and symbols_Q[i] <= -threshold:  # -1-3j
            symbol.append(6)
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] >= threshold:  # 1+3j
            symbol.append(12)
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold:  # 1+j
            symbol.append(13)
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold:  # 1-j
            symbol.append(15)
        elif symbols_I[i] >= 0 and symbols_I[i] <= threshold and symbols_Q[i] <= -threshold:  # 1-3j
            symbol.append(14)
        elif symbols_I[i] >= threshold and symbols_Q[i] >= threshold:  # 3+3j
            symbol.append(8)
        elif symbols_I[i] >= threshold and symbols_Q[i] >= 0 and symbols_Q[i] <= threshold:  # 3+1j
            symbol.append(9)
        elif symbols_I[i] >= threshold and symbols_Q[i] <= 0 and symbols_Q[i] >= -threshold:  # 3-1j
            symbol.append(11)
        elif symbols_I[i] >= threshold and symbols_Q[i] <= -threshold:  # 3-3j
            symbol.append(10)
    return symbol

''' Function that calculates the BER of a given constellation
    Parameters: Symbols of the constellation, threshold to demap the symbols
    Return: BER of the constellation'''
def bit_error_rate(y_true, y_pred):
    true = ''.join([f"{sym:04b}" for sym in y_true])
    pred = ''.join([f"{sym:04b}" for sym in y_pred])
    # Calculate BER (bit error rate) of the constellation (true vs pred)
    ber = sum([1 for i in range(len(true)) if true[i] != pred[i]]) / len(true)
    return ber

''' Function that calculates the SER of a given constellation
    Parameters: Symbols of the constellation, threshold to demap the symbols
    Return: SER of the constellation'''
def symbol_error_rate(y_true, y_pred):
    # Calculate SER (symbol error rate) of the constellation (true vs pred)
    ser = sum(y_true != y_pred) / len(y_true)
    return ser

''' Function that modulate the signal 16-QAM
    Parameters: Signal to modulate
    Return: Modulated signal in symbols'''
def modulate_16QAM(signal):
    mod_dict = {0: -3+3j, 1: -3+1j, 2: -3-3j, 3: -3-1j,
                4: -1+3j, 5: -1+1j, 6: -1-3j, 7: -1-1j,
                8: 3+3j, 9: 3+1j, 10: 3-3j, 11: 3-1j,
                12: 1+3j, 13: 1+1j, 14: 1-3j, 15: 1-1j}
    symbols = []
    for i in range(len(signal)):
        symbols.append(mod_dict[signal[i]])
    return symbols

''' Function that demodulate the signal 16-QAM
    Parameters: Signal to demodulate
    Return: Demodulated signal in symbols'''
def demodulate_16QAM(signal):
    mod_dict = {-3+3j: 0, -3+1j: 1, -3-3j: 2, -3-1j: 3,
                -1+3j: 4, -1+1j: 5, -1-3j: 6, -1-1j: 7,
                3+3j: 8, 3+1j: 9, 3-3j: 10, 3-1j: 11,
                1+3j: 12, 1+1j: 13, 1-3j: 14, 1-1j: 15}
    symbols = []
    for i in range(len(signal)):
        symbols.append(mod_dict[signal[i]])
    return symbols

'''Function that syncronizes the tx signal with the rx signal
    Parameters: tx_signal, rx_signal
    Return: signal_demod in the same length of rx_signal and syncronized'''
def syncronize(tx_signal, rx_signal):
    tx = np.concatenate((tx_signal, tx_signal))
    corr = np.abs(np.correlate(np.abs(tx) - np.mean(np.abs(tx)),
                  np.abs(rx_signal) - np.mean(np.abs(rx_signal)), mode='full'))
    delay = np.argmax(corr) - len(rx_signal) + 1
    signal = tx[delay:]
    signal = signal[:len(rx_signal)]
    signal_demod = demodulate_16QAM(signal)
    return signal_demod

''' Function that calculate a fit curve of a given signal
    Parameters: x_step, x, y, degree of the polynomial
    Return: curve of the signal'''
def polinomial_func(x_step, x, y, deg=2):
    p = np.polyfit(x, y, deg)
    z = np.poly1d(p)
    curve = z(x_step)
    return curve

''' Function that calculate the density of a given point
    Parameters: labels, X, umbral
    Return: labels with the new labels of the noise points'''
def threshold_density(labels, X, umbral=0.5):
    noise_labels = np.where(labels == -1)[0]
    for i in noise_labels:
        point = X[i]
        density = np.sum(np.linalg.norm(X - point, axis=1) <= umbral)
        if density > 1:
            distance_to_points = np.linalg.norm(X - point, axis=1)
            neighbors = np.where(distance_to_points <= umbral)[0]
            neighbors_labels = labels[neighbors]
            neighbors_labels = neighbors_labels[neighbors_labels != -1]
            if len(neighbors_labels) > 0:
                new_labels = np.bincount(neighbors_labels).argmax()
                labels[i] = new_labels
    return labels

''' Function that generate a mesh of a given data
    Parameters: X, alpha
    Return: mesh of the data'''
def generate_mesh(X, model, alpha=0.01):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, alpha),
                         np.arange(y_min, y_max, alpha))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return xx, yy, Z
