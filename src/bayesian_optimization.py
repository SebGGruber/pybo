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
    Function plugging its arguments into the bayesian optimization framework.
    Before the algorithm is started, two randomly drawn configurations will
    be evaluated. These do not count for the `max_evaluations` number.

    Args:
        search_space (ndarray):
            The search space used to optimize over
            with bayesian optimization. Should have the form of a XD numpy
            array where `X` is the search space dimension.
        cost_function (ndarray -> float):
            Cost function evaluating the
            optimization objective for a given configuration.
            The higher the better.
        acquisition_function (
            (`predictive_model`, ndarray, float) -> float
        ):
            Acquisition function for a `predictive_model` on a specific
            element of the search space (second argument) and with the current
            highest cost function value (third argument).
        predictive_model (sklearn.gaussian_process.GaussianProcessRegressor):
            Gaussian process model fitting the observed cost function and used
            for defining the acquisition function.
        max_evaluations (int):
            The maximum amount of iterations of the bayesian optimization
            framework. Excludes the randomly drawn pre-runs.
        logging_function(
            (int, list of {"x", "mu", "sigma", "u"}, {"x", "y"}) -> None
        ):
            Function used to write logs of some kind (currently only plots).
            First argument is the iteration, second the current values of
            the acquisition function, third argument the dataset `D` of the
            all the know cost function evaluations.


    Returns:
        ndarray: Best configuration found in the search_space.
    """
    
    print("Evaluating two randomly picked configurations...")
    # randomly pick the first two lambdas
    start_lambdas = np.random.choice(search_space, 2)
    start_cs = [cost_function(lam) for lam in start_lambdas]
    c_inc = np.max(start_cs)
    # "dataset" of evaluated lambdas
    D = {
        "x": np.array(start_lambdas),
        "y": np.array(start_cs)
    }
    print("Random evaluations done. Starting bayesian optimization...")

    # progress bar displayed in the console output
    with tqdm(total=max_evaluations, disable=False) as progress_bar:

        # start bayesian optimization iterations with iterator `t`
        for t in range(max_evaluations):

            # ignoring irrelevant warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # reshape to single feature, multi sample array
                predictive_model.fit(D["x"].reshape(-1, 1), D["y"])

            def acq(x):
                """Helper function for returning the acquisition value."""
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

    # return the entry in `D` with the highest cost function value
    lambda_hat = D["x"][np.argmax(D["y"])]

    return lambda_hat
