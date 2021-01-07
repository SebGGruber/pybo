import warnings
from scipy.stats import norm


def EI(gpr, x, c_inc):
    """
    Expected Improvement used as acquisition function.
    
    Args:
        gpr (sklearn.gaussian_process.GaussianProcessRegressor):
            GP regressor used for fitting the cost function
        x (ndarray):
            Value of the search space to evaluate.
        c_inc (float):
            Current best/highest cost function value.

    Returns:
        {"u": float, "mu": float, "sigma": float, "x": ndarray}
    """

    # ignoring irrelevant warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # reshape to single sample array
        mu, sigma = gpr.predict(x.reshape(1, -1), return_std=True)

    if sigma > 0:
        Z = (mu - c_inc) / sigma
        u = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
    else:
        u = 0

    # return all values for plotting
    return {
        "u": float(u),
        "mu": float(mu),
        "sigma": float(sigma),
        "x": x
    }

