import numpy as np
from spcdist.scipy import multivariate_beta_gaussian


def test_covariance():
    mean = np.zeros(2)
    scale = np.array([[1, .2],
                      [.2, .3]])

    for alpha in (1.25, 1.5, 2, 2.5):
        mbg = multivariate_beta_gaussian(mean=mean, scale=scale, alpha=alpha)
        X = mbg.rvs(size=10000, random_state=0)
        empirical = np.cov(X.T)
        theoretical = mbg.variance()
        assert np.allclose(empirical, theoretical, atol=1e-2, rtol=1e-2)
