import numpy
from numpy import sin, cos, exp, pi
from numpy.random import randn

def gradient_step(xk, df, sigma):
    """
    Take a single step in the gradient descent algorithm.

    Parameters
    ----------
    xk : float
    df : function
        The derivative of the function `f` we are trying to minimize.
    sigma : float
        In the interval (0,1). Affects how small or large the step can be.

    Returns
    -------
    xkp1 : float
        The next iterate.
    """
    xkp1 = xk - sigma*df(xk)
    return xkp1

def gradient_descent(f, df, x, sigma=0.5, epsilon=1e-8):
    """
    Perform gradient descent to find the a minima x* of f.

    A local minima, x*, is such that `f(x*) <= f(x)` for all
    `x` near `x*`.

    Parameters
    ----------
    f : function
    df : function
        The function to minimize and its derivative.
    x : float
        A guess for the minimum.
    sigma : float
        (Optional) In the interval (0,1). Affects how small or large the step
        can be.
    epsilon : float
        The tolerance / accuracy of the algorithm.

    Returns
    -------
    xkp1 : float
        A local minima of `f` accurate to `epilson`.

    """
    if (sigma <= 0) or (1 <= sigma):
        raise ValueError('Invalid scaling factor.')

    # randomly perturb x for robustness
    if abs(df(x)) < 1e-14:
        x = x + randn(1)/100

    xkp1 = x
    xk = x + 1
    while (abs(xkp1 - xk) > epsilon):
        xk = xkp1
        xkp1 = gradient_step(xk, df, sigma)
    return xkp1

