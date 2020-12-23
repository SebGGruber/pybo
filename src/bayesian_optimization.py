import numpy as np
import warnings
from tqdm import tqdm


def bayesian_optimization(
    search_space,
    cost_function,
    acquisition_function,
    predictive_model,
    max_evaluations,
    logging_function
):
    """
    """
    
    # randomly pick the first two lambdas
    start_lambdas = np.random.choice(search_space, 2)
    start_cs = [cost_function(lam) for lam in start_lambdas]
    c_inc = np.max(start_cs)
    # "dataset" of evaluated lambdas
    D = {
        "x": np.array(start_lambdas),
        "y": np.array(start_cs)
    }

    # progress bar displayed in the console output
    with tqdm(total=max_evaluations, disable=False) as progress_bar:

        for t in range(max_evaluations):

            # ignoring irrelevant warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # reshape to single feature, multi sample array
                predictive_model.fit(D["x"].reshape(-1, 1), D["y"])

            def acq(x):
                return acquisition_function(predictive_model, x, c_inc)

            # list with entries of 
            # {u: function value, mu: mean, sigma: standard deviation, x: x}
            acquisition_stats = [acq(x) for x in search_space]
            us = [acq["u"] for acq in acquisition_stats]
            # maximum of the acquisition function w.r.t. the search space
            best_lambda = search_space[np.argmax(us)]

            c_inc = cost_function(best_lambda)

            # reshape to single feature, multi sample array
            D["x"] = np.append(D["x"], best_lambda)
            D["y"] = np.append(D["y"], c_inc)

            # console and file logging
            logging_function(t, acquisition_stats, D)
            postfix = {"lambda": best_lambda, "c_inc": c_inc}
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)

    lambda_hat = D["x"][np.argmax(D["y"])]

    return lambda_hat
