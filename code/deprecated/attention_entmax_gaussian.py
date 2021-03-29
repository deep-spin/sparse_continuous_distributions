from entmax_gaussian import (EntmaxGaussian1D, Gaussian1D, SparsemaxGaussian1D,
                             BiweightGaussian1D, TriweightGaussian1D)
from gaussian_basis_functions import GaussianBasisFunctions
from entmax_gaussian_kernels import (EntmaxGaussian1DKernel,
                                     SparsemaxGaussian1DKernel,
                                     BiweightGaussian1DKernel,
                                     TriweightGaussian1DKernel)
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
        

if __name__ == '__main__':
    mu = -0.2
    sigma_sq = 1.1
    mu_j = 0.5
    sigma_sq_j = 1.5
    gaussian = Gaussian1D(mu_j, sigma_sq_j)
    sparsemax = SparsemaxGaussian1D(mu, sigma_sq)
    biweight = BiweightGaussian1D(mu, sigma_sq)
    triweight = TriweightGaussian1D(mu, sigma_sq)

    psi = GaussianBasisFunctions(mu_j, np.sqrt(sigma_sq_j))

    for entmax in [sparsemax, biweight, triweight]:
        entmax_times_rbf = lambda t: entmax.pdf(t) * gaussian.pdf(t)
        att, err = integrate.quad(entmax_times_rbf,
                                  entmax._mu - entmax.support_size()/2,
                                  entmax._mu + entmax.support_size()/2)
        print(att)
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

