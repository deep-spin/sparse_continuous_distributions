import pytest
import torch

from functools import partial

from spcdist.attention.continuous_entmax import (ContinuousEntmax,
                                                 ContinuousSoftmax,
                                                 ContinuousSparsemax,
                                                 ContinuousBiweight,
                                                 ContinuousTriweight)

from spcdist.attention.basis_function import GaussianBasisFunctions


# generate basis functions
nb_waves = 8
max_seq_len = 10
degrees = torch.arange(0, 4).float()
pos = torch.arange(nb_waves // 2).float()
omegas = max_seq_len * 1.0 / (10000. ** (2 * pos / nb_waves))
mus, sigmas = torch.meshgrid(torch.linspace(0, 1, nb_waves // 2), torch.Tensor([0.1, 0.5]))
mus = mus.flatten()
sigmas = sigmas.flatten()

basis_functions = [GaussianBasisFunctions(mus, sigmas)]
basis_function_ids = [repr(f) for f in basis_functions]

continuous_attn_classes = [
    ContinuousSoftmax,
    ContinuousSparsemax,
    ContinuousBiweight,
    ContinuousTriweight,
    partial(ContinuousEntmax, alpha=1.2)
]


@pytest.mark.parametrize('Cls', continuous_attn_classes)
@pytest.mark.parametrize('basis_function', basis_functions, ids=basis_function_ids)
def test_softmax(Cls, basis_function):
    psi = [basis_function]
    fmax = Cls(psi=psi)

    # directly generate fixed canonical parameters
    torch.manual_seed(42)

    for _ in range(10):
        theta = torch.randn(5, 2, dtype=torch.double)
        theta[:, 1] = -(theta[:, 1] ** 2)
        theta[:, 1] -= .1  # To be safe and not become positive under perturbation.
        theta.requires_grad_()
        res = torch.autograd.gradcheck(fmax, theta, eps=1e-4, rtol=1e-2, atol=1e-2)
        assert res
