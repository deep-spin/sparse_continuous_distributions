import numpy as np
import matplotlib.pyplot as plt

from spcdist.scipy_1d import EntmaxGaussian1D

if __name__ == "__main__":


    ts = np.linspace(-3, 3, num=1000).reshape(-1, 1)

    for alpha in (2, 1.5, 0.9):
        mbg = EntmaxGaussian1D(mu=0, sigma_sq=1, alpha=alpha)
        plt.plot(ts, mbg.pdf(ts), label=fr"$\alpha={alpha}$")
    plt.legend()
    plt.show()
