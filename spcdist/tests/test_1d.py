from spcdist.scipy import multivariate_beta_gaussian
from spcdist.scipy_1d import (EntmaxGaussian1D,
                              Gaussian1D,
                              SparsemaxGaussian1D,
                              BiweightGaussian1D,
                              TriweightGaussian1D)

import numpy as np

import pytest

_cases = [
    (Gaussian1D, 1),
    (SparsemaxGaussian1D, 2),
    (BiweightGaussian1D, 3 / 2),
    (TriweightGaussian1D, 4 / 3),
]


@pytest.mark.parametrize("Cls,alpha", _cases)
@pytest.mark.parametrize("mu", [0, 0.1])
@pytest.mark.parametrize("sigma_sq", [1, 1.3])
def test_variance_1d(Cls, alpha, mu, sigma_sq):

    if alpha == 1:
        pytest.skip("1-d alpha=1 not implemented")

    dist_1d = Cls(mu, sigma_sq)
    dist_gen = EntmaxGaussian1D(alpha, mu, sigma_sq)

    variance_expected = dist_gen.variance()
    variance_obtained = dist_1d.variance()
    assert np.allclose(variance_expected, variance_obtained)


@pytest.mark.parametrize("Cls,alpha", _cases)
@pytest.mark.parametrize("mu", [0, 0.1])
@pytest.mark.parametrize("sigma_sq", [1, 1.3])
def test_pdf_1d(Cls, alpha, mu, sigma_sq):

    if alpha == 1:
        pytest.skip("1-d alpha=1 not implemented")

    dist_1d = Cls(mu, sigma_sq)
    dist_gen = EntmaxGaussian1D(alpha, mu, sigma_sq)

    t = np.linspace(-5, 5, 1000)
    obtained = dist_1d.pdf(t)
    expected = dist_gen.pdf(t)
    assert np.allclose(expected, obtained)


@pytest.mark.parametrize("_,alpha", _cases)
@pytest.mark.parametrize("mu", [0, 0.1])
@pytest.mark.parametrize("sigma_sq", [1, 1.3])
@pytest.mark.skip("Not implemented.")
def test_entropy(_, alpha, mu, sigma_sq):
    dist_gen_1d = EntmaxGaussian1D(alpha, mu, sigma_sq)
    dist_gen_mv = multivariate_beta_gaussian(mu, sigma_sq, alpha)
    entr_expected = dist_gen_1d.tsallis_entropy()
    entr_obtained = dist_gen_mv.tsallis_entropy()
    assert np.allclose(entr_expected, entr_obtained)


@pytest.mark.parametrize("Cls,alpha", _cases)
@pytest.mark.parametrize("mu", [0, 0.1])
@pytest.mark.parametrize("sigma_sq", [1, 1.3])
def test_variance_mv(Cls, alpha, mu, sigma_sq):
    dist_1d = Cls(mu, sigma_sq)
    dist_gen = multivariate_beta_gaussian(mu, sigma_sq, alpha)
    variance_expected = dist_gen.variance()
    variance_obtained = dist_1d.variance()
    assert np.allclose(variance_expected, variance_obtained)


@pytest.mark.parametrize("Cls,alpha", _cases)
@pytest.mark.parametrize("mu", [0, 0.1])
@pytest.mark.parametrize("sigma_sq", [1, 1.3])
def test_pdf_mv(Cls, alpha, mu, sigma_sq):

    t = np.linspace(-5, 5, 1000)
    t_col = t[:, np.newaxis]

    dist_1d = Cls(mu, sigma_sq)
    dist_gen = multivariate_beta_gaussian(mu, sigma_sq, alpha)

    obtained = dist_1d.pdf(t)
    expected = dist_gen.pdf(t_col)
    assert np.allclose(expected, obtained)
