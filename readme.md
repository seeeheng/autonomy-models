Implementing some ML techniques learnt from a CMU Autonomy Course.

1. Gaussian Process Regression
Utilizes Gaussians, as well as techniques from Baye's rule, to learn Gaussians over functions through Gaussian Process Regression. Sweeps a kernel function over a range of values to learn the covariance matrix and mean of a dependent Gaussian based on the initial value. Able to subsequently make probabilistic predictions based on training data given to it.

- Able to learn function small datasets
- Allows for stochastic representation of model