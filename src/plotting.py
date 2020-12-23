import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from os import path


def plot_bo(directory, t, acquisition_stats, D):
    """
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    # lambda evaluations
    data = pd.DataFrame()
    # remove the last entry
    data["x"] = D["x"].flatten()[:len(D["x"])-1]
    data["y"] = D["y"][:len(D["y"])-1]

    # acquistion function lines
    acq_values = [acq["u"] for acq in acquisition_stats]
    mus = [acq["mu"] for acq in acquisition_stats]
    mus_sigmas_p = [acq["mu"] + acq["sigma"] for acq in acquisition_stats]
    mus_sigmas_n = [acq["mu"] - acq["sigma"] for acq in acquisition_stats]
    x_values = [acq["x"] for acq in acquisition_stats]

    g = sns.FacetGrid(data, height=6)
    g = g.map(plt.scatter, "x", "y", edgecolor="w")
    plt.plot(x_values, acq_values, color="red")
    # plot the last entry as red point
    plt.plot(D["x"][len(D["x"])-1], D["y"][len(D["x"])-1], "or")
    plt.plot(x_values, mus, color="blue")
    plt.plot(x_values, mus_sigmas_p, color="blue", linestyle="dashed")
    plt.plot(x_values, mus_sigmas_n, color="blue", linestyle="dashed")
    plt.savefig(path.join(directory, "bo_step_{}.png".format(t)))
