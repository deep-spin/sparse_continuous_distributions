import pytest

import torch
from continuous_sparsemax import ContinuousSparsemax
from continuous_biweight import ContinuousBiweight
from continuous_triweight import ContinuousTriweight
from continuous_entmax import ContinuousEntmax
from basis_function import GaussianBasisFunctions

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

##
if False:
    psi = basis_functions
    sparsemax = ContinuousSparsemax(psi=psi)
    theta = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
    theta[:, 1] = -(theta[:, 1] ** 2)
    theta[:, 1] -= .1  # To be safe and not become positive under perturbation.
    print(sparsemax(theta))
    biweight = ContinuousBiweight(psi=psi)
    print(biweight(theta))
##

@pytest.mark.parametrize('basis_function', basis_functions, ids=basis_function_ids)
def test_sparsemax(basis_function):
    psi = [basis_function]
    sparsemax = ContinuousSparsemax(psi=psi)
    
    # directly generate fixed canonical parameters
    torch.manual_seed(42)

    for _ in range(10):
        theta = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
        theta[:, 1] = -(theta[:, 1] ** 2)
        theta[:, 1] -= .1  # To be safe and not become positive under perturbation.
        res = torch.autograd.gradcheck(sparsemax, theta, eps=1e-4)
        assert res


@pytest.mark.parametrize('basis_function', basis_functions, ids=basis_function_ids)
def test_biweight(basis_function):
    psi = [basis_function]
    biweight = ContinuousBiweight(psi=psi)

    # directly generate fixed canonical parameters
    torch.manual_seed(42)

    for _ in range(10):
        theta = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
        theta[:, 1] = -(theta[:, 1] ** 2)
        theta[:, 1] -= .1  # To be safe and not become positive under perturbation.
        res = torch.autograd.gradcheck(biweight, theta, eps=1e-4)
        assert res


@pytest.mark.parametrize('basis_function', basis_functions, ids=basis_function_ids)
def test_triweight(basis_function):
    psi = [basis_function]
    triweight = ContinuousTriweight(psi=psi)

    # directly generate fixed canonical parameters
    torch.manual_seed(42)

    for _ in range(10):
        theta = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
        theta[:, 1] = -(theta[:, 1] ** 2)
        theta[:, 1] -= .1  # To be safe and not become positive under perturbation.
        res = torch.autograd.gradcheck(triweight, theta, eps=1e-4)
        assert res


@pytest.mark.parametrize('basis_function', basis_functions, ids=basis_function_ids)
def test_entmax(basis_function):
    psi = [basis_function]
    entmax = ContinuousEntmax(psi=psi, alpha=1.2)

    # directly generate fixed canonical parameters
    torch.manual_seed(42)

    for _ in range(10):
        theta = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
        theta[:, 1] = -(theta[:, 1] ** 2)
        theta[:, 1] -= .1  # To be safe and not become positive under perturbation.
        res = torch.autograd.gradcheck(entmax, theta, eps=1e-4)
        assert res
