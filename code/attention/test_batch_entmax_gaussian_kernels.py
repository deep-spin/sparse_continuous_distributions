from batch_entmax_gaussian import (EntmaxGaussian1D, SparsemaxGaussian1D,
                                   BiweightGaussian1D, TriweightGaussian1D)
from basis_function import GaussianBasisFunctions
from batch_entmax_gaussian_kernels import (EntmaxGaussian1DKernel,
                                           SparsemaxGaussian1DKernel,
                                           BiweightGaussian1DKernel,
                                           TriweightGaussian1DKernel)
import torch

if __name__ == '__main__':
    mu = torch.randn(3)
    sigma_sq = torch.rand(3)
    mu_j = torch.randn(2)
    sigma_sq_j = torch.rand(2)

    sparsemax = SparsemaxGaussian1D(mu, sigma_sq)
    biweight = BiweightGaussian1D(mu, sigma_sq)
    triweight = TriweightGaussian1D(mu, sigma_sq)

    psi = GaussianBasisFunctions(mu_j, torch.sqrt(sigma_sq_j))

    print("Numerical solution:")
    for entmax in [sparsemax, biweight, triweight]:
        alpha = entmax._alpha
        entmax_kernel = EntmaxGaussian1DKernel(alpha=alpha,
                                               mu=mu,
                                               sigma_sq=sigma_sq)
        att = entmax_kernel.attention(psi)
        att_grad = entmax_kernel.attention_gradient(psi)
        print(att)
        print(att_grad)

    print("Analytical solution:")
    sparsemax_kernel = SparsemaxGaussian1DKernel(mu, sigma_sq)
    biweight_kernel = BiweightGaussian1DKernel(mu, sigma_sq)
    triweight_kernel = TriweightGaussian1DKernel(mu, sigma_sq)
    for kernel in [sparsemax_kernel, biweight_kernel, triweight_kernel]:
        att = kernel.attention(psi)
        att_grad = kernel.attention_gradient(psi)
        print(att)
        print(att_grad)

