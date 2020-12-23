import warnings
from scipy.stats import norm

def EI(gpr, x, c_inc):
    """
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sigma = gpr.predict(x, return_std=True)

    Z = (mu - c_inc) / sigma

    if sigma > 0:
        u = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
    else:
        u = 0

    return u

