import numpy as np
from spcdist.scipy import multivariate_beta_gaussian

def test_low_rank():
    n = 1000
    x = multivariate_beta_gaussian.rvs(mean=1, scale=.5, alpha=2, size=n)

    mean = np.array([1, 42])
    scale = np.array([.5, 0])

    X = multivariate_beta_gaussian.rvs(mean=mean, scale=scale, alpha=2,
                                       allow_singular=True,
                                       size=n)

    assert np.all(X[:, 1] == 42)
    assert np.allclose(np.var(x), np.var(X[:, 0]), atol=1e-2, rtol=1e-2)
