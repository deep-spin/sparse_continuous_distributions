from entmax_gaussian import EntmaxGaussian
from entmax_gaussian import (EntmaxGaussian1D, Gaussian1D, SparsemaxGaussian1D,
                             BiweightGaussian1D, TriweightGaussian1D)
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mu = 0
    sigma_sq = 1

    t = np.linspace(-5, 5, 100000)

    # Use specific classes for special values of alpha.
    gaussian = Gaussian1D(mu, sigma_sq)
    sparsemax = SparsemaxGaussian1D(mu, sigma_sq)
    biweight = BiweightGaussian1D(mu, sigma_sq)
    triweight = TriweightGaussian1D(mu, sigma_sq)
    plt.plot(t, sparsemax.pdf(t), label='2')
    plt.plot(t, biweight.pdf(t), label='1.5')
    plt.plot(t, triweight.pdf(t), label='1.333')
    plt.plot(t, gaussian.pdf(t), label='1')
    for entmax in [sparsemax, biweight, triweight]:
        y, err = integrate.quad(entmax.pdf, -5, 5)
        print(y)
    plt.legend()
    plt.show()

    # Use general class which handles arbitrary alpha.
    for alpha in [2, 3/2, 4/3]:
        entmax = EntmaxGaussian1D(alpha, mu, sigma_sq)
        plt.plot(t, entmax.pdf(t), label='%f' % alpha)
        y, err = integrate.quad(entmax.pdf, -5, 5)
        print(y)
    plt.legend()
    plt.show()

    # Use general class for multivariate entmax Gaussian.
    for alpha in [2, 3/2, 4/3]:
        entmax = EntmaxGaussian(alpha,
                                np.array([mu]),
                                np.array([[sigma_sq]]))
        plt.plot(t, entmax.pdf(t[None, :]), label='%f' % alpha)
        y, err = integrate.quad(entmax.pdf, -5, 5)
        print(y)
        print(entmax.mean())
        print(entmax.variance())
        print(entmax.tsallis_entropy())
    plt.legend()
    plt.show()
