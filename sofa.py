""" Funciones comunes para aplicaciones de modulación y
demodulación 16QAM
"""
import numpy as np
import scipy as sp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

"""
Diccionario para modular símbolos 16QAM
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


def mod_norm(const, power: float = 1.0):
    """Calcula el coeficiente de escala para normalizar una señal dada
    una potencia promedio.

    :param const: constelación.
    :param power - float : potencia promedio deseada.
    :return: coeficiente de escala.
    """
    constPow = np.mean([x**2 for x in np.abs(const)])
    scale = np.sqrt(power / constPow)
    return scale


def calc_noise(snr: float, X):
    """Añade ruido a un vector.

    :param snr - float : relación señal a ruido (SNR)
    :param X: vector original
    :return: vector con ruido
    """
    X_avg_p = np.mean(np.power(X, 2))
    X_avg_db = 10 * np.log10(X_avg_p)

    noise_avg_db = X_avg_db - snr
    noise_avg_p = np.power(10, noise_avg_db / 10)
    # Al no poner el parámetro loc se asume media 0
    noise = np.random.normal(scale=np.sqrt(noise_avg_p), size=len(X))
    return X + noise


def add_awgn(snr: float, X):
    """Añade ruido a una constelación.

    :param snr: relación señal a ruido (SNR)
    :param X: constelación original
    :return: constelación con ruido
    """
    Xr = np.real(X)
    Xi = np.imag(X)
    return (calc_noise(snr, Xr), calc_noise(snr, Xi))


def realify(X):
    """Transforma un vector de números complejos en un vector de pares,
    real e imaginario de tipo flotante.

    :param X: constelación recibida
    :return: constelación transformada"""

    return np.array([X.real, X.imag]).T


def demodulate(X_rx, mod_dict):
    """Demodula usando el método tradicional de rejilla.

    :param X_rx: constelación recibida
    :param mod_dict: diccionario de modulación
    :return: constelación demodulada"""
    demodulated = np.empty(len(X_rx), dtype=int)

    for i, x in enumerate(X_rx):
        # Distancia a cada centroide
        dist = np.abs(np.array(list(mod_dict.values())) - x)
        # Índice del valor mínimo de distancia
        index = list(dist).index(np.min(dist))
        index = np.argmin(dist).flatten()
        # Centroide más cercano al símbolo
        demodulated[i] = index
    return demodulated


def demodulate_knn(X_rx, sym_tx, k, train_size=0.4):
    """Demodula usando KNN.

    :param X_rx: constelación recibida
    :param sym_tx: símbolos transmitidos
    :param k: parámetro k del algoritmo KNN
    :param test_size: fracción de datos para entrenamiento
    :return: constelación demodulada
    """
    X = realify(X_rx)
    y = sym_tx

    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size)

    # Número de vecinos
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model.predict(X)


def demodulate_svm(X_rx, sym_tx, C, gamma, train_size=0.4):
    """Demodula usando SVM.

    :param X_rx: constelación recibida
    :param sym_tx: símbolos transmitidos
    :param C: parámetro C del algoritmo SVM
    :param gamma: parámetro gamma del algoritmo SVM
    :param train_size: fracción de datos para entrenamiento
    :return: constelación demodulada
    """
    X = realify(X_rx)
    y = sym_tx

    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size)

    model = SVC(C=C, gamma=gamma)
    model.fit(X_train, y_train)
    return model.predict(X)


def demodulate_kmeans(X_rx, mod_dict):
    """Demodula usando K-Means.

    :param X_rx: constelación recibida
    :param mod_dict: diccionario de modulación
    :return: constelación demodulada, modelo entrenado
    """
    X = realify(X_rx)
    A_mc = [(x.real, x.imag) for x in list(mod_dict.values())]
    model = KMeans(n_clusters=16, n_init=1, init=np.array(A_mc))
    model.fit(X)

    return model


def find_best_params(model, param_grid, X_rx, sym_tx):
    """Encuentra los parámetros que mejor funcionan para unos datos específicos
    :param model: modelo de ML a optimizar
    :param param_grid: diccionario de parámetros del modelo
    :param X: datos de entrada
    :param y: datos de salida validados
    :return: diccionario de parámetros optimizado
    """
    X = realify(X_rx)
    y = sym_tx

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)

    grid = GridSearchCV(model(), param_grid, verbose=0)

    grid.fit(X_train, y_train)

    return grid.best_params_


def symbol_error_rate(sym_rx, sym_tx):
    """Calcula la rata de error de símbolo.

    :param sym_rx: vector de símbolos recibidos
    :param sym_tx: vector de símbolos transmitidos
    :return: rata de error de símbolo, cantidad de símbolos erróneos"""
    # error = 0
    error = sum(rx != tx for rx, tx in zip(sym_rx, sym_tx))
    SER = error / len(sym_tx)
    return SER, error


def bit_error_rate(sym_rx, sym_tx):
    """Calcula la rata de error de bit.

    :param sym_rx: vector de símbolos recibidos
    :param sym_tx: vector de símbolos transmitidos
    :return: rata de error de bit, cantidad de bits erróneos"""
    # Se transforman los símbolos a binario
    sym_rx_str = "".join([f"{sym:04b}" for sym in sym_rx])
    sym_tx_str = "".join([f"{sym:04b}" for sym in sym_tx])

    error = sum(sym_rx_str[i] != sym_tx_str[i] for i in range(len(sym_rx_str)))
    BER = error / len(sym_rx_str)
    return BER, error


def curve_fit(f, x, y):
    """Calcula los parámetros óptimos dada una función y unos puntos a optimizar.

    :param f: función la cuál debe ser optimizada
    :param x: coordenadas en el eje x
    :param y: coordenadas en el eye y
    return: parámetros optimizados para ajustar la función a las coordenadas dadas"""
    popt, _ = sp.optimize.curve_fit(f, x, y)
    return popt


def sync_signals(tx, rx):
    """Sincroniza dos señales.

    :param short_signal: señal corta, usualmente la recibida
    :param long_signal: señal larga, usualmente la transmitida
    :return: una copia de ambas señales sincronizadas, en el mismo orden de entrada de parámetros
    """
    # Se concatena para asegurar de que el array recibido esté contenido dentro
    # del largo.
    tx_long = np.concatenate((tx, tx))
    correlation = np.abs(
        np.correlate(
            np.abs(tx_long) - np.mean(np.abs(tx_long)),
            np.abs(rx) - np.mean(np.abs(rx)),
            mode="full",
        )
    )
    delay = np.argmax(correlation) - len(rx) + 1
    # print(f"El retraso es de {delay} posiciones")

    sync_signal = tx_long[delay:]
    sync_signal = sync_signal[: len(rx)]

    return sync_signal
