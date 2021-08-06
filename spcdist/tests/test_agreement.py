import numpy as np
import torch

from spcdist.scipy import multivariate_beta_gaussian
from spcdist.torch import MultivariateBetaGaussianDiag

def main():

    mean = np.array([10, 11.1])
    scale_diag = np.array([0.5, 1.5])
    alpha = 3

    mbg = multivariate_beta_gaussian(mean=mean,
                                     scale=np.diag(scale_diag),
                                     alpha=alpha)

    print("entropy", mbg.tsallis_entropy())
    print("radius", mbg.radius)
    print("tau", mbg.tau)

    mbg_t = MultivariateBetaGaussianDiag(torch.from_numpy(mean),
                                         torch.from_numpy(scale_diag),
                                         alpha)

    print("entropy", mbg_t.tsallis_entropy)
    print("radius", torch.exp(mbg_t.log_radius))
    print("tau", mbg_t._tau)







if __name__ == '__main__':
    main()

