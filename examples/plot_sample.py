import numpy as np
import matplotlib.pyplot as plt

from spcdist.scipy import multivariate_beta_gaussian

if __name__ == "__main__":

    plt.figure()

    # check in 2d
    mean = np.zeros(2)

    scale = np.array([[1, .2],
                      [.2, .5]])

    for alpha in (1.1, 1.5, 2):
        mbg = multivariate_beta_gaussian(mean=mean, scale=scale, alpha=alpha)
        X = mbg.rvs(size=3000, random_state=42)
        plt.scatter(X[:, 0], X[:, 1], alpha=.5, marker='.',
                    label=rf'$\alpha={alpha}$')

    plt.legend()
    plt.show()



