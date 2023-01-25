""" Funciones comunes para aplicaciones de modulación y
demodulación 16QAM
"""
import numpy as np
import matplotlib.pyplot as plt

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
    8: 3 + 3j,  # 1000
    9: 3 + 1j,  # 1001
    10: 3 - 3j,  # 1010
    11: 3 - 1j,  # 1011
    12: 1 + 3j,  # 1100
    13: 1 + 1j,  # 1101
    14: 1 - 3j,  # 1110
    15: 1 - 1j,
}  # 1111


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

def demodulate(X_rx, mod_dict):
    """Demodula usando el método tradicional de rejilla.

    :param X_rx: constelación recibida
    :param mod_dict: diccionario de modulación
    :return: constelación demodulada"""
    demodulated = []
    # Diccionario para demodular
    demod_dict = dict(zip(mod_dict.values(), mod_dict.keys()))
    for x in X_rx:
        # Distancia a cada centroide
        dist = np.abs(np.array(list(mod_dict.values())) - x)
        # Índice del valor mínimo de distancia
        index = list(dist).index(np.min(dist))
        # Centroide más cercano al símbolo
        demodulated.append(index)
    return np.array(demodulated)

def demodulate_knn(X_rx, sym_tx, k):
    """Demodula usando KNN.
    
    :param X_rx: constelación recibida
    :param sym_tx: símbolos transmitidos
    :param k: parámetro k del algoritmo KNN
    :return: constelación demodulada
    """
    X = np.array([X_rx.real, X_rx.imag]).T
    y = sym_tx
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    
    # Número de vecinos
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train, y_train)
    return model.predict(X)

def demodulate_svm(X_rx, sym_tx, C, gamma):
    """Demodula usando SVM.
    
    :param X_rx: constelación recibida
    :param sym_tx: símbolos transmitidos
    :param C: parámetro C del algoritmo SVM
    :param gamma: parámetro gamma del algoritmo SVM
    :return: constelación demodulada
    """
    X = np.array([X_rx.real, X_rx.imag]).T
    y = sym_tx
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    model = SVC(C = C, gamma = gamma)
    model.fit(X_train, y_train)
    return model.predict(X)

def demodulate_kmeans(X_rx, mod_dict):
    """Demodula usando K-Means.
    
    :param X_rx: constelación recibida
    :param mod_dict: diccionario de modulación
    :return: constelación demodulada
    """
    X = list(zip(X_rx.real, X_rx.imag))
    A_mc = [(x.real, x.imag) for x in list(mod_dict.values())]
    model = KMeans(n_clusters = 16, n_init = 1, init = np.array(A_mc))
    model.fit(X)

    return model.predict(X)

def symbol_error_rate(sym_rx, sym_tx):
    """Calcula la rata de error de símbolo.

    :param sym_rx: vector de símbolos recibidos
    :param sym_tx: vector de símbolos transmitidos
    :return: rata de error de símbolo, cantidad de símbolos erróneos"""
    error = 0
    for i in range(len(sym_tx)):
        if sym_rx[i] != sym_tx[i]:
            error += 1
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


def sync_signals(short_signal, long_signal):
    """Sincroniza dos señales.
    
    :param short_signal: señal corta, usualmente la recibida
    :param long_signal: señal larga, usualmente la transmitida
    :return: una copia de ambas señales sincronizadas, en el mismo orden de entrada de parámetros
    """
    # Se concatena para asegurar de que el array recibido esté contenido dentro
    # del largo.
    longer_signal = np.concatenate((long_signal, long_signal, long_signal))
    correlation = np.correlate(short_signal, longer_signal, mode="full")
    plt.figure()
    plt.plot(correlation)
    plt.show()
    index = np.argmax(correlation) + 1
    print(index)
    delay = index - len(short_signal)
    print(delay)
    print(f"El retraso es de {delay} posiciones")

    sync_signal = np.roll(longer_signal, delay)
    sync_signal = sync_signal[: len(short_signal)]

    return short_signal, sync_signal
