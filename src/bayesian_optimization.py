import numpy as np
import warnings
from tqdm import tqdm


def bayesian_optimization(
    search_space,
    cost_function,
    acquisition_function,
    predictive_model,
    max_evaluations
):
    """
    """
    
    # pick the first lambda by random
    start_lambda = np.random.choice(search_space, 1)
    c_inc = cost_function(start_lambda)
    D = {"x": [start_lambda], "y": [c_inc]}

    # progress bar displayed in the console output
    with tqdm(total=max_evaluations, disable=False) as progress_bar:

        for _ in range(max_evaluations):

            predictive_model.fit(D["x"], D["y"])

            def acq(x):
                return acquisition_function(predictive_model, x, c_inc)

            best_index = np.argmax([acq(x) for x in search_space])
            best_lambda = search_space[best_index]

            c_inc = cost_function(best_lambda)

            D["x"].append(best_lambda)
            D["y"].append(c_inc)

            postfix = {"lambda": best_lambda, "c_inc": c_inc}
            progress_bar.set_postfix(postfix)
            progress_bar.update(1)

    lambda_hat = D["x"][np.argmax(D["x"])]

    return lambda_hat
