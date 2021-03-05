import numpy as np

def k_squared_exp(x_i, x_j, hp_ampl=1**2, hp_l=1.5):
    """ Squared Exponential Kernel Function

    Args:
        hp_ampl = hyperparameter; amplitude
        hp_l = hyperparameter; length scale
        x_i = first argument to kernel, k(xi, xj)
        x_j = second argument to kernel, k(xi, xj)

    Returns:
        Computed value from kernel.
    """
    return (hp_ampl) * np.exp(-0.5*(x_i-x_j)**2/hp_l**2)

def dk_squared_exp_l(x_i, x_j, hp_ampl=1**2, hp_l=1.5):
    """ Squared Exponential Kernel Function

    Args:
        hp_ampl = hyperparameter; amplitude
        hp_l = hyperparameter; length scale
        x_i = first argument to kernel, k(xi, xj)
        x_j = second argument to kernel, k(xi, xj)

    Returns:
        Computed value from kernel.
    """
    return (hp_ampl) * np.exp(-0.5*(x_i-x_j)**2/hp_l**2) * ((x_i-x_j)**2 / hp_l**3)

def dk_squared_exp_a(x_i, x_j, hp_ampl=1**2, hp_l=1.5):
    """ Squared Exponential Kernel Function

    Args:
        hp_ampl = hyperparameter; amplitude
        hp_l = hyperparameter; length scale
        x_i = first argument to kernel, k(xi, xj)
        x_j = second argument to kernel, k(xi, xj)

    Returns:
        Computed value from kernel.
    """
    return (2*hp_ampl) * np.exp(-0.5*(x_i-x_j)**2/hp_l**2)