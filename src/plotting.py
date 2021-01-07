import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from os import path


def plot_bo(directory, t, acquisition_stats, D):
    """
    Function used for creating plots in each bayesian optimization iteration.
    The last entry in `D` is plotted with its own color and not yet considered
    for the acquisition function (it represents the current "choice").
    The mean values are the continuous blue line.
    The standard deviations are the dashed blue lines wrapping the mean line.
    The acquistion function is the continuous red line.
    Depending on cost function values, the plot scales may be off - plotting 
    the acquisition function and the GP predictions in separate plots may help.

    Args:
        directory (str):
            Directory where the plots should be stored.
        t (int):
            Current iteraton count of bayesian optimization.
        acquisition_stats (
            list of {"x": float, "mu": float, "sigma": float, "u": float}
        ):
            List of dictionaries holding the evaluations of the acquistion
            function of the current bayesian optimization iteration.
            This saves us from re-evaluating the acquisition function for
            logging purposes.
        D ({"x": ndarray, "y": ndarray}):
            Dataset dictionary of all evaluated cost function values so far.
            
    Returns:
        None
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
