#!/usr/bin/env python
# coding: utf-8

# # Inter-channel interference (ICI) estimation using constellation diagrams Gaussian Mixture Models in a 32 GBd system.

# ## Initialization

# ### Libraries

# In[1]:


import os
from collections import defaultdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy as sp
import tensorflow.keras as ker
from joblib import dump, load
from matplotlib.colors import LogNorm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import utils
from tensorflow.keras.callbacks import EarlyStopping

import sofa

# ### Globals

# In[2]:


# Special function to read the known data structure
def read_data(folder_rx):
    data = defaultdict(
        defaultdict(
            defaultdict(defaultdict(defaultdict(
                defaultdict().copy).copy).copy).copy
        ).copy
    )
    for dist_pow in os.listdir(folder_rx):
        if not os.path.isdir(os.path.join(folder_rx, dist_pow)):
            continue
        print(f"reading {dist_pow}")
        for spac in os.listdir(os.path.join(folder_rx, dist_pow)):
            print(f"reading {dist_pow}/{spac}")
            consts = os.listdir(os.path.join(folder_rx, dist_pow, spac))
            for const in consts:
                if const.endswith("xlsx"):
                    continue
                song, orth, osnr, spacing, distance, power = const.split("_")
                # Remove extension
                power = power.split(".")[0]
                spacing = spacing.replace("p", ".")
                osnr = osnr.replace("p", ".")
                mat = sp.io.loadmat(os.path.join(
                    folder_rx, dist_pow, spac, const))
                data[distance][power][spacing][osnr][song][orth] = mat["rconst"][0]
    return data


def split(a, n):
    k, m = divmod(len(a), n)
    return np.array(
        [a[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]
    )


# In[3]:


def calc_once(varname, fn, args):
    """Calculate a variable only once."""
    if varname not in globals() or eval(varname) is None:
        return fn(**args)
    return eval(varname)


def estimation_model(
    layers_props_lst: list, loss_fn: ker.losses.Loss, input_dim: int
) -> ker.models.Sequential:
    """Compile a sequential model for regression purposes."""
    model = ker.Sequential()
    # Hidden layers
    for i, layer_props in enumerate(layers_props_lst):
        if i == 0:
            model.add(ker.layers.Dense(input_dim=input_dim, **layer_props))
        else:
            model.add(ker.layers.Dense(**layer_props))
    # Regressor
    model.add(ker.layers.Dense(units=1, activation="linear"))

    model.compile(loss=loss_fn, optimizer="adam")

    return model


def estimation_crossvalidation(
    X, y, X_prod, y_prod, n_splits, layer_props, loss_fn, callbacks
):
    """Crossvalidation of an estimation network."""
    # Scores dict
    scores = {}
    scores["model"] = []
    scores["loss"] = []
    scores["mae"] = {"train": [], "test": [], "prod": []}
    scores["r2"] = {"train": [], "test": [], "prod": []}
    scores["rmse"] = {"train": [], "test": [], "prod": []}

    # K-fold crossvalidation
    kf = KFold(n_splits=n_splits, shuffle=True)

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Input variables standarizer
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test_kf = sc.transform(X_test)
        X_prod_kf = sc.transform(X_prod)

        model = estimation_model(layer_props, loss_fn, X_train.shape[1])

        # Save test scalar loss
        if callbacks:
            loss = model.fit(
                X_train,
                y_train,
                epochs=5000,
                batch_size=64,
                callbacks=callbacks,
                verbose=0,
            )
        else:
            loss = model.fit(X_train, y_train, epochs=5000,
                             batch_size=64, verbose=0)
        print(f"Needed iterations: {len(loss.history['loss'])}")
        loss = loss.history["loss"]

        # Predict using train values
        predictions_train = model.predict(X_train, verbose=0)
        # Predict using test values
        predictions_test = model.predict(X_test_kf, verbose=0)
        # Predict using production values
        predictions_prod = model.predict(X_prod_kf, verbose=0)

        # Dataframe for better visualization
        train_data_train = pl.DataFrame(
            {"ICI": [y_train], "Predicted ICI": [predictions_train]}
        )
        train_data_test = pl.DataFrame(
            {"ICI": [y_test], "Predicted ICI": [predictions_test]}
        )
        train_data_prod = pl.DataFrame(
            {"ICI": [y_prod], "Predicted ICI": [predictions_prod]}
        )

        # MAE
        mae_score_train = mean_absolute_error(
            *train_data_train["ICI"], *train_data_train["Predicted ICI"]
        )
        mae_score_test = mean_absolute_error(
            *train_data_test["ICI"], *train_data_test["Predicted ICI"]
        )
        mae_score_prod = mean_absolute_error(
            *train_data_prod["ICI"], *train_data_prod["Predicted ICI"]
        )

        # R²
        r2_score_train = r2_score(
            *train_data_train["ICI"], *train_data_train["Predicted ICI"]
        )
        r2_score_test = r2_score(
            *train_data_test["ICI"], *train_data_test["Predicted ICI"]
        )
        r2_score_prod = r2_score(
            *train_data_prod["ICI"], *train_data_prod["Predicted ICI"]
        )

        # RMSE
        rmse_score_train = mean_squared_error(
            *train_data_train["ICI"], *train_data_train["Predicted ICI"], squared=False
        )
        rmse_score_test = mean_squared_error(
            *train_data_test["ICI"], *train_data_test["Predicted ICI"], squared=False
        )
        rmse_score_prod = mean_squared_error(
            *train_data_prod["ICI"], *train_data_prod["Predicted ICI"], squared=False
        )

        # Append to lists
        scores["model"].append(model)
        scores["loss"].append(loss)
        scores["mae"]["train"].append(mae_score_train)
        scores["mae"]["test"].append(mae_score_test)
        scores["mae"]["prod"].append(mae_score_prod)
        scores["r2"]["train"].append(r2_score_train)
        scores["r2"]["test"].append(r2_score_test)
        scores["r2"]["prod"].append(r2_score_prod)
        scores["rmse"]["train"].append(rmse_score_train)
        scores["rmse"]["test"].append(rmse_score_test)
        scores["rmse"]["prod"].append(rmse_score_prod)

    return scores


def test_estimation_model(
    data,
    data_prod,
    n_splits,
    max_neurons,
    activations,
    use_osnr=True,
    loss_fn="mean_absolute_error",
):
    """Test a spectral spacing estimation model with given parameters."""
    n_feat = data.shape[1]
    var_n = n_feat - 1 if use_osnr else n_feat - 2

    # Split variables
    # Variables
    X = np.array(data[:, 0:var_n])
    X_prod = np.array(data_prod[:, 0:var_n])
    # Tags
    y = np.array(data[:, -1])
    y_prod = np.array(data_prod[:, -1])

    # Layer properties
    layer_props = [
        {"units": max_neurons // (2**i), "activation": activation}
        for i, activation in enumerate(activations)
    ]
    print(f"{layer_props}{' + OSNR' if use_osnr else ''}")
    callbacks = [
        EarlyStopping(
            monitor="loss", patience=30, mode="min", restore_best_weights=True
        )
    ]

    return estimation_crossvalidation(
        X, y, X_prod, y_prod, n_splits, layer_props, loss_fn, callbacks
    )


# In[4]:


def plot_constellation_diagram(X, ax):
    ax.scatter(X.real, X.imag, alpha=0.5)
    ax.set_title("Constellation diagram")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")


def calculate_gmm(data, gm_kwargs):
    return GaussianMixture(**gm_kwargs).fit(data)


def calculate_1d_histogram(X, bins):
    hist_y, hist_x = np.histogram(X.real, bins=bins)
    # Remove last bin edge
    hist_x = hist_x[:-1]

    return hist_x, hist_y


def plot_1d_histogram(X, bins=128, ax=None):
    ax.hist(X, bins=bins, density=True, alpha=0.5,
            label="Calculated histogram")


def plot_gmm_1d(gm, limits, ax):
    x = np.linspace(*limits, 1000)

    logprob = gm.score_samples(x.reshape(-1, 1))
    responsibilities = gm.predict_proba(x.reshape(-1, 1))
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    ax.plot(x, pdf_individual, "--", label="Adjusted histogram")


def plot_gmm_2d(gm, limits, ax):
    x = y = np.linspace(*limits)
    X, Y = np.meshgrid(x, y)
    Z = -gm.score_samples(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    ax.contour(
        X,
        Y,
        Z,
        norm=LogNorm(vmin=1.0, vmax=1000.0),
        levels=np.logspace(0, 3, 25),
        cmap="seismic",
    )


def calculate_3d_histogram(X, bins, limits):
    hist, xedges, yedges = np.histogram2d(
        X.real, X.imag, bins=bins, range=[[*limits], [*limits]]
    )
    # Create the meshgrid for the surface plot, excluding the last edge
    x_mesh, y_mesh = np.meshgrid(xedges[:-1], yedges[:-1])
    return hist, x_mesh, y_mesh


def plot_3d_histogram(x_mesh, y_mesh, hist, ax):
    ax.plot_surface(
        x_mesh, y_mesh, hist.T, cmap="seismic", rstride=1, cstride=1, edgecolor="none"
    )
    ax.set_title("3D Histogram")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")


def plot_results(x_values, scores, xlabel, log=False, intx=False):
    plt.figure(figsize=(8, 6), layout="constrained")
    plt.scatter(x_values, scores)
    plt.plot(x_values, scores)
    plt.xlabel(xlabel)
    plt.ylabel("MAE")
    if log:
        plt.xscale("log", base=2)
    if intx:
        plt.xticks(x_values)
    plt.grid(True)
    plt.show()


def joblib_load(file):
    try:
        return load(file)
    except FileNotFoundError:
        print(f"[ERROR]: File {file} not found")
        return None


def joblib_save(var, file):
    dump(var, file)


# ### Load data

# In[5]:


folder_rx = "../../../Databases/32GBd/ICI characterization/"

# Read received data
data = read_data(folder_rx)

# Try to load histograms
file_models_hist = "./results/models32_hist.pkl"
file_models_gmm = "./results/models32_gmm.pkl"

models_hist = joblib_load(file_models_hist)
models_gmm = joblib_load(file_models_gmm)
models_tuple = (
    None
    if models_hist is None or models_gmm is None
    else (models_hist, models_gmm)
)


# ## Calculate histograms

# In[6]:


def get_histograms(data):
    histograms_hist = defaultdict(
        defaultdict(
            defaultdict(defaultdict(defaultdict(
                defaultdict(list).copy).copy).copy).copy
        ).copy
    )
    histograms_gmm = defaultdict(
        defaultdict(
            defaultdict(defaultdict(defaultdict(
                defaultdict(list).copy).copy).copy).copy
        ).copy
    )
    bins = 128
    limits = [-5, 5]

    for distance in data.keys():
        for power in data[distance].keys():
            for spacing in data[distance][power].keys():
                for osnr in data[distance][power][spacing].keys():
                    for song in data[distance][power][spacing][osnr].keys():
                        for orth in data[distance][power][spacing][osnr][song].keys():
                            print(f"Calculating GMM for {
                                  distance}/{power}/{spacing}/{osnr}/{song}/{orth}")
                            X_rx = data[distance][power][spacing][osnr][song][orth]
                            X_chs = split(X_rx, 3)

                            for n, x_ch in enumerate(X_chs):
                                # Calculate 2D GMM
                                input_data = np.vstack(
                                    (x_ch.real, x_ch.imag)).T
                                gm_kwargs = {
                                    "means_init": np.array(
                                        list(product([-3, -1, 1, 3], repeat=2))
                                    ),
                                    "n_components": 16,
                                }
                                gm_2d = calculate_gmm(input_data, gm_kwargs)

                                # Calculate 3D histogram
                                hist, x_mesh, y_mesh = calculate_3d_histogram(
                                    x_ch, bins, limits)

                                # Save 3D histogram
                                histograms_hist[distance][power][spacing][osnr][song][orth].append(
                                    hist)

                                # Calculate I and Q histograms
                                hist_x, hist_y = calculate_1d_histogram(
                                    x_ch.real, bins)
                                input_data = np.repeat(
                                    hist_x, hist_y).reshape(-1, 1)
                                gm_kwargs = {
                                    "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
                                    "n_components": 4,
                                }
                                gm_i = calculate_gmm(input_data, gm_kwargs)

                                # Q
                                hist_x, hist_y = calculate_1d_histogram(
                                    x_ch.imag, bins)
                                input_data = np.repeat(
                                    hist_x, hist_y).reshape(-1, 1)
                                gm_kwargs = {
                                    "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
                                    "n_components": 4,
                                }
                                gm_q = calculate_gmm(input_data, gm_kwargs)

                                # Save gaussians
                                histograms_gmm[distance][power][spacing][osnr][song][orth].append([
                                                                                                  gm_2d, gm_i, gm_q])

    histograms_hist = dict(histograms_hist)
    histograms_gmm = dict(histograms_gmm)
    return histograms_hist, histograms_gmm


models_tuple = calc_once("models_tuple", get_histograms, {"data": data})
models_hist, models_gmm = models_tuple
joblib_save(models_hist, file_models_hist)
joblib_save(models_gmm, file_models_gmm)


# In[7]:


data["0km"]["0dBm"]["37.5GHz"]["23.04dB"]["Song4"]["Y"].shape


# In[8]:


def plot_histograms(data, histograms_gmm, distance, power, spacing, osnr, song, orth):
    def plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth):
        # Extract data
        X_ch = data[distance][power][spacing][osnr][song][orth]

        plt.figure(figsize=(12, 12), layout="tight")

        # Plot constellation diagram
        ax = plt.subplot(2, 2, 1)
        plot_constellation_diagram(X_ch, ax)

        gm_2d = histograms_gmm[distance][power][spacing][osnr][song][orth][0][0]

        # Plot 2D GMM
        plot_gmm_2d(gm_2d, limits, ax)
        ax.grid(True)

        # Calculate 3D histogram
        hist, x_mesh, y_mesh = calculate_3d_histogram(X_ch, bins, limits)

        # Plot 3D histogram
        ax = plt.subplot(2, 2, 2, projection="3d")
        plot_3d_histogram(x_mesh, y_mesh, hist, ax)

        # Plot I and Q histograms separately
        # I
        ax = plt.subplot(2, 2, 3)
        plot_1d_histogram(X_ch.real, bins=bins, ax=ax)

        hist_x, hist_y = calculate_1d_histogram(X_ch.real, bins)
        input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
        gm_kwargs = {
            "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
            "n_components": 4,
        }
        gm_i = calculate_gmm(input_data, gm_kwargs)
        plot_gmm_1d(gm_i, limits, ax)

        ax.set_title("I-Histogram")
        ax.set_xlabel("I")
        ax.grid(True)

        # Q
        ax = plt.subplot(2, 2, 4)
        plot_1d_histogram(X_ch.imag, bins=bins, ax=ax)

        hist_x, hist_y = calculate_1d_histogram(X_ch.imag, bins)
        input_data = np.repeat(hist_x, hist_y).reshape(-1, 1)
        gm_kwargs = {
            "means_init": np.array([-3, -1, 1, 3]).reshape(4, 1),
            "n_components": 4,
        }
        gm_q = calculate_gmm(input_data, gm_kwargs)
        plot_gmm_1d(gm_q, limits, ax)
        ax.set_title("Q-Histogram")
        ax.set_xlabel("Q")
        ax.grid(True)

        plt.suptitle(f"Plots for constellation at {distance} with {power} launch power, {
                     spacing} spectral spacing, {osnr} OSNR, Song {song[-1]}, {orth} component.")

        plt.show()

    bins = 128
    limits = [-5, 5]

    plot(data, histograms_gmm, distance, power, spacing, osnr, song, orth)


def plot_menu(data):
    def select_menu(data, name):
        print(f"Select {name}")
        options = {n: option for n, option in enumerate(data.keys())}
        for n in options.keys():
            print(f"[{n}]: {options[n]}")
        choice = None
        while choice is None:
            choice = options.get(int(input()))
            if choice is None:
                print("Invalid input, try again")
        print(f"Selected {choice}")
        return choice

    # Distance
    distance = select_menu(data, "distance")

    # Power
    power = select_menu(data[distance], "launch power")

    # Spacing
    spacing = select_menu(data[distance][power], "spacing")

    # OSNR
    osnr = select_menu(data[distance][power][spacing], "OSNR")

    # Song
    song = select_menu(data[distance][power][spacing][osnr], "song")

    # Component
    orth = select_menu(data[distance][power][spacing][osnr][song], "component")

    plot_histograms(data, models_gmm, distance,
                    power, spacing, osnr, song, orth)


# ### Plot constellation diagrams, contour for GMM and histograms

# In[9]:


plot_menu(data)


# ## Pre-process data

# In[10]:


# Dataframe with 98 columns
# First 32 for means
# Following 64 for each value of the covariances matrixes
# (repeated values are included)
# Next to last for OSNR value in dB
# Last column for spectral spacing value in GHz

n_features = 82
df_dict = {f"col{n}": [] for n in range(n_features)}
data_list = []

# Iterate over the dictionary and populate the DataFrame
for distance in models_gmm.keys():
    for power in models_gmm[distance].keys():
        for spacing in models_gmm[distance][power].keys():
            for osnr in models_gmm[distance][power][spacing].keys():
                for song in models_gmm[distance][power][spacing][osnr].keys():
                    for orth in models_gmm[distance][power][spacing][osnr][song].keys():
                        for n in range(3):
                            gmm_2d = models_gmm[distance][power][spacing][osnr][song][orth][n][0]
                            means = gmm_2d.means_.flatten()
                            covariances_raw = gmm_2d.covariances_.flatten()
                            # Remove repeated covariances
                            covariances = []
                            for x, covariance in enumerate(covariances_raw):
                                if x % 4 != 1:
                                    covariances.append(covariance)
                            osnr_value = np.array([float(osnr[:-2])])
                            spacing_value = np.array([float(spacing[:-3])])

                            features = np.concatenate(
                                (means, covariances, osnr_value, spacing_value))
                            row_dict = {f"col{n}": feature for n,
                                        feature in enumerate(features)}
                            data_list.append(row_dict)

# Convert the list of dictionaries into a DataFrame
df = pl.DataFrame(data_list)

# Print the DataFrame
df.write_json("./results/gmm32_features.json")


# In[11]:


# Show the original dataframe
df


# In[12]:


# Shuffle the dataframe
df_shuffled = df.sample(n=len(df), shuffle=True, seed=1036681523)
df_shuffled


# In[13]:


# Extract 10% of the data to use later for "production" testing
df_prod = df_shuffled[: int(len(df_shuffled) * 0.1)]
df_prod


# In[14]:


# Use the rest of the data for normal testing
df_new = df_shuffled[int(len(df_shuffled) * 0.1):]
df_new


# ## Hyperparameters evaluation

# The following hyperparameters are going to be combined and evaluated:
# - Maximum number of neurons in the first layer (8, 16, 32, 64, 128, 256, 512, 1024).
# - Number of hidden layers (1, 2, 3).
# - Activation functions (ReLu, tanh, sigmoid).
# - Using or not the OSNR value as an additional feature.
#
# Results will have the following structure:
# ```
# {"xyz": {"n_neurons": {"osnr": results}}}
# ```
# Where `xyz` will be each initial of the activation functions in the model (r for ReLu, t for tanh and s for sigmoid), `n_neurons` will be the maximum number of neurons in the model (corresponding to the first layer), `osnr` will be a string telling if that model used OSNR as input or not (`"osnr"` or `wo_osnr`).
# Finally the results will store the loss history, the serialized model in JSON format in a string and MAE, RMSE and R² values for training, test and production data.

# In[15]:


osnr_lst = ["osnr", "wo_osnr"]
max_neurons = [str(2**n) for n in range(3, 11)]
functs = ["relu", "tanh", "sigmoid"]
layers_n = [1, 2, 3]

combinations = [
    [list(subset) for subset in product(functs, repeat=n)] for n in layers_n
]

hidden_layers = [item for sublist in combinations for item in sublist]


# In[ ]:


results_file = "./results/gmm32_reg_results.h5"
try:
    histograms_reg_results = sofa.load_hdf5(results_file)
except:
    print("Error loading from file, creating a new dictionary")
    histograms_reg_results = defaultdict(
        defaultdict(defaultdict(defaultdict().copy).copy).copy
    )

# Evaluar
for activations in hidden_layers:
    for neurons in max_neurons:
        for osnr in osnr_lst:
            args = {
                "data": df_new,
                "data_prod": df_prod,
                "n_splits": 5,
                "max_neurons": int(neurons),
                "activations": activations,
                "use_osnr": True if osnr == "osnr" else False,
            }
            act_fn_name = "".join([s[0] for s in activations])
            if histograms_reg_results[act_fn_name][neurons][osnr] == defaultdict():
                # Get results
                results = test_estimation_model(**args)
                # Serialize model
                results["model"] = [
                    utils.serialize_keras_object(model) for model in results["model"]
                ]
                # Save serialized model for serialization
                histograms_reg_results[act_fn_name][neurons][osnr] = results
                # Save results with serialized model
                print("Saving results...")
                sofa.save_hdf5(histograms_reg_results, results_file)
                print("Results saved!")
print("Training complete")


# ## Resultados

# In[ ]:


def get_avg_score(results, target_value, target="neurons", metric="mae", score="test"):
    mae_lst = []
    for activations in hidden_layers:
        if target == "layers" and len(activations) != target_value:
            continue
        for neurons in max_neurons:
            if target == "neurons" and neurons != target_value:
                continue
            for osnr in osnr_lst:
                if target == "osnr" and osnr != target_value:
                    continue
                act_fn_name = "".join([s[0] for s in activations])
                mae_lst.append(
                    np.mean(
                        [*results[act_fn_name][neurons]
                            [osnr]["mae"]["test"].values()]
                    )
                )
    return mae_lst


# In[ ]:


gmm_neurons_avg_results = [
    np.mean(
        get_avg_score(
            histograms_reg_results,
            neurons,
            target="neurons",
            metric="mae",
            score="test",
        )
    )
    for neurons in max_neurons
]
x = list(map(int, max_neurons))
plot_results(x, gmm_neurons_avg_results, "Maximum number of neurons", log=True)


# In[ ]:


gmm_layers_avg_results = [
    np.mean(
        get_avg_score(
            histograms_reg_results, layers, target="layers", metric="mae", score="test"
        )
    )
    for layers in range(1, 4)
]
x = range(1, 4)
plot_results(x, gmm_layers_avg_results,
             "Number of layers", log=False, intx=True)


# In[ ]:


gmm_osnr_avg_results = [
    np.mean(
        get_avg_score(
            histograms_reg_results, osnr, target="osnr", metric="mae", score="test"
        )
    )
    for osnr in ["osnr", "wo_osnr"]
]
print("With OSNR  Without OSNR")
print(f"{gmm_osnr_avg_results[0]:.3f}       {gmm_osnr_avg_results[1]:.3f}")


# ## Sort models by score

# In[ ]:


# Find better model by test score
def get_better_models(results, metric="mae", score="test"):
    scores = []
    for activations in hidden_layers:
        for neurons in max_neurons:
            for osnr in osnr_lst:
                act_fn_name = "".join([s[0] for s in activations])
                coll = results[act_fn_name][neurons][osnr][metric][score].values()
                if isinstance(coll, defaultdict):
                    continue
                score_value = np.mean([*coll])
                scores.append((score_value, [act_fn_name, neurons, osnr]))
    scores.sort(key=lambda x: x[0])
    return pl.dataframe.DataFrame(scores)


# In[ ]:


better_models_df = get_better_models(
    histograms_reg_results, metric="mae", score="test")
better_models_df.head(10)


# In[ ]:


better_models_df.tail(10)