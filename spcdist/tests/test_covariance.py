import numpy as np
from spcdist.scipy import multivariate_beta_gaussian

import torch
from spcdist.torch import MultivariateBetaGaussian


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


def test_beta_gaussial_full():
    mean = torch.zeros(2)
    scale = torch.tensor([[1, .2],
                          [.2, .3]])
    wbg = MultivariateBetaGaussian(loc=mean, scale=scale, alpha=2)
    wbg_numpy = multivariate_beta_gaussian(mean=mean.numpy(), scale=scale.numpy(), alpha=2)
    assert np.allclose(wbg_numpy.tsallis_entropy(), wbg.tsallis_entropy.numpy())

    X = wbg_numpy.rvs(size=10000, random_state=0)
    assert np.allclose(wbg_numpy.pdf(X), wbg.pdf(torch.tensor(X).to(torch.float32)).numpy(), atol=1e-2, rtol=1e-2)