import numpy as np
from spdist.scipy import multivariate_beta_gaussian


def test_covariance():
    mean = np.zeros(2)
    scale = np.array([[1, .2],
                      [.2, .3]])

    for alpha in (1.25, 1.5, 2, 2.5):
        X = multivariate_beta_gaussian.rvs(
                mean=mean,
                scale=scale,
                size=10000,
                alpha=alpha,
                random_state=0)

        empirical = np.cov(X.T)
        theoretical = mbg.variance(mean=mean, scale=scale, alpha=alpha)
        assert np.allclose(empirical, theoretical)
