import json
import os
import warnings
from collections import defaultdict

import h5py
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ImportWarning)
    import tensorflow as tf

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


def kfold_cross_validation(
    X: np.ndarray, y: np.ndarray, n_splits: int, algorithm_func, *args, **kwargs
) -> tuple:
    """
    Performs k-fold cross-validation using the specified algorithm function.

    Parameters:
        X : np.ndarray
            Input data.
        y : np.ndarray
            Target labels.
        n_splits : int
            Number of folds.
        algorithm_func : callable
            Algorithm function to be used for each fold.
        *args : Any
            Variable length arguments to be passed to the algorithm function.
        **kwargs : Any
            Keyword arguments to be passed to the algorithm function.

    Returns:
        tuple
            Results and test data for each fold.
    """
    results = []
    tests = []
    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        result = algorithm_func(X_train, y_train, X_test, *args, **kwargs)
        results.append(result)
        tests.append(y_test)

    return np.array(results), np.array(tests)


def demodulate_knn(
    X_rx: np.ndarray, sym_tx: np.ndarray, k: int, n_splits: int = 5
) -> tuple:
    """
    Demodulates using KNN with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        sym_tx : np.ndarray
            Transmitted symbols.
        k : int
            Parameter k for the KNN algorithm.

    Returns:
        tuple
            Demodulated constellation and test data.
    """

    def algorithm_func(X_train, y_train, X_test, k):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    X = realify(X_rx)
    y = sym_tx

    return kfold_cross_validation(X, y, n_splits, algorithm_func, k=k)


def demodulate_svm(
    X_rx: np.ndarray, sym_tx: np.ndarray, C: float, gamma: float, n_splits: int = 5
) -> tuple:
    """
    Demodulates using SVM with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        sym_tx : np.ndarray
            Transmitted symbols.
        C : float
            Parameter C for the SVM algorithm.
        gamma : float
            Parameter gamma for the SVM algorithm.
        n_splits : int, optional
            Number of folds for k-fold cross-validation. Default is 5.

    Returns:
        tuple
            Demodulated constellation and test data.
    """

    def algorithm_func(X_train, y_train, X_test, C, gamma):
        model = SVC(C=C, gamma=gamma)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    X = realify(X_rx)
    y = sym_tx

    return kfold_cross_validation(X, y, n_splits, algorithm_func, C=C, gamma=gamma)


def demodulate_kmeans(X_rx: np.ndarray, mod_dict: dict, n_splits: int = 5) -> tuple:
    """
    Demodulates using K-Means with k-fold cross-validation.

    Parameters:
        X_rx : np.ndarray
            Received constellation.
        mod_dict : dict
            Modulation dictionary.
        n_splits : int, optional
            Number of folds for cross-validation, by default 5.

    Returns:
        tuple
            Demodulated constellation and test data.
    """

    def algorithm_func(X_train, _, X_test):
        A_mc = np.array([(x.real, x.imag) for x in list(mod_dict.values())])
        model = KMeans(n_clusters=16, n_init=1, init=A_mc)
        model.fit(X_train)
        return model.predict(X_test)

    X = realify(X_rx)
    # Create an empty array as a placeholder for y
    y = np.empty_like(X)

    return kfold_cross_validation(X, y, n_splits, algorithm_func)


def classifier_model(
    layers_props_lst: list, loss_fn: tf.keras.losses.Loss, input_dim: int
) -> tf.keras.models.Sequential:
    """
    Creates a neural network classifier model with specified layers and loss function.

    Parameters:
        layers_props_lst (list): List of layer properties dictionaries.
            Each dictionary should contain the desired properties for each layer, such as the number of units and activation function.
        loss_fn (tf.keras.losses.Loss): Loss function to optimize in the neural network.

    Returns:
        tf.keras.models.Sequential: Compiled model.

    """
    model = tf.keras.Sequential()

    for i, layer_props in enumerate(layers_props_lst):
        if i == 0:
            model.add(tf.keras.layers.Dense(input_dim=input_dim, **layer_props))
        else:
            model.add(tf.keras.layers.Dense(**layer_props))

    model.add(tf.keras.layers.Dense(units=16, activation="softmax"))

    model.compile(loss=loss_fn, optimizer="adam")

    return model


def demodulate_neural(
    X_rx: np.ndarray,
    sym_tx: np.ndarray,
    layer_props_lst: list,
    loss_fn: tf.keras.losses.Loss,
    n_splits: int = 5,
) -> tuple:
    """
    Demodulates using a neural network with k-fold cross-validation.

    Parameters:
        X_rx (np.ndarray): Received constellation.
        sym_tx (np.ndarray): Transmitted symbols.
        layer_props_lst (list): List of layer properties dictionaries for the neural network.
        loss_fn (tf.keras.losses.Loss): Loss function to optimize in the neural network.
        n_splits (int, optional): Number of folds for cross-validation. Default is 5.

    Returns:
        np.ndarray: Demodulated constellation.
    """

    def algorithm_func(X_train, y_train, X_test):
        model = classifier_model(layer_props_lst, loss_fn, X_train.shape[0])
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=300, mode="min", restore_best_weights=True
        )
        model.fit(
            X_train,
            y_train,
            epochs=5000,
            batch_size=64,
            callbacks=[callback],
            verbose=0,
        )
        return model.predict(X_test, verbose=0)

    X = realify(X_rx)
    y = sym_tx

    return kfold_cross_validation(X, y, n_splits, algorithm_func)


def find_best_params(
    model, param_grid: dict, X_rx: np.ndarray, sym_tx: np.ndarray
) -> dict:
    """
    Finds the best parameters for a given model using specific data.

    Parameters:
        model: ML model to optimize.
        param_grid: Dictionary of model parameters.
        X_rx: Input data.
        sym_tx: Validated output data.

    Returns:
        dict: Optimized parameter dictionary.
    """
    X = realify(X_rx)
    y = sym_tx

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3)

    grid = GridSearchCV(model(), param_grid, verbose=0)

    grid.fit(X_train, y_train)

    return grid.best_params_


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


def curve_fit(f, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def __do_backup(filename: str, n_backups: int = 0) -> None:
    """
    Perform backup rotation for a file.

    Parameters:
        filename (str): The name of the file to create or overwrite.
        n_backups (int): The number of backup files to keep.

    Returns:
        None
    """

    # Function to get backup filenames
    def backup_filename(index):
        return f"{filename}.bak{index}"

    # Backup existing files
    for i in range(n_backups, 0, -1):
        src = backup_filename(i - 1) if i - 1 > 0 else filename
        dst = backup_filename(i)
        os.rename(src, dst) if os.path.exists(src) else None


def save_json(data: dict, filename: str, n_backups: int = 3) -> None:
    """
    Save data to a JSON file with backup rotation.

    Parameters:
        data (dict): A dictionary containing datasets to be saved.
        filename (str): The name of the JSON file to create or overwrite.
        n_backups (int): The number of backup files to keep.

    Returns:
        None
    """
    # Do backup rotation before writing
    __do_backup(filename, n_backups)

    try:
        # Save data to the main file
        with open(filename, "w") as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def load_json(filename: str) -> dict:
    """
    Load data from a JSON file.

    Parameters:
        filename (str): The name of the JSON file to load data from.

    Returns:
        defaultdict: A nested defaultdict containing the loaded data.
    """

    def dict_factory():
        return defaultdict(dict_factory)

    loaded_data = defaultdict(dict_factory)

    try:
        # Load data from the file
        with open(filename, "r") as json_file:
            loaded_data = json.load(json_file)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")

    return loaded_data


def save_hdf5(data: dict, filename: str, n_backups: int = 3) -> None:
    """
    Save data to an HDF5 file with backup rotation.

    Parameters:
        data (dict): A dictionary containing datasets to be saved.
        filename (str): The name of the HDF5 file to create or overwrite.
        n_backups (int): The number of backup files to keep.

    Returns:
        None
    """
    # Do backup rotation before writing
    __do_backup(filename, n_backups)

    try:
        # Save data to the main file
        with h5py.File(filename, "w") as f:

            def store_dict(group, data_dict):
                for key, value in data_dict.items():
                    if isinstance(value, (dict, defaultdict)):
                        subgroup = group.create_group(key)
                        store_dict(subgroup, value)
                    elif key == "model":
                        # Save 'model' key as JSON
                        group.create_dataset(key, data=json.dumps(value))
                    elif key in {"loss", "train", "test", "prod"}:
                        # Save each k-fold score in a separate group
                        scores_group = group.create_group(key)
                        for i, vector in enumerate(value, start=1):
                            scores_group.create_dataset(str(i), data=vector)
                    else:
                        # Save other keys as numpy arrays
                        group.create_dataset(key, data=value)

            store_dict(f, data)
    except Exception as e:
        raise RuntimeError(f"Error: {e}")


def load_hdf5(filename: str):
    """
    Load data from an HDF5 file.
    This function recursively loads data from an HDF5 file.

    Parameters:
        filename (str): The name of the HDF5 file to load data from.

    Returns:
        defaultdict: A nested defaultdict containing the loaded data.
    """

    def dict_factory():
        return defaultdict(dict_factory)

    loaded_data = defaultdict(dict_factory)

    with h5py.File(filename, "r") as f:

        def load_dict(group):
            data_dict = defaultdict(dict_factory)
            for key in group.keys():
                if isinstance(group[key], h5py.Group):
                    data_dict[key] = load_dict(group[key])
                elif isinstance(group[key], h5py.Dataset):
                    if key == "model":
                        data_dict[key] = json.loads(group[key][()].decode("utf-8"))
                    else:
                        data_dict[key] = group[key][()]
            return data_dict

        loaded_data = load_dict(f)

    return loaded_data
