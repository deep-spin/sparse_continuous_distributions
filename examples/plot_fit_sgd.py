import numpy as np
import torch
import matplotlib.pyplot as plt

from spcdist.torch import (MultivariateBetaGaussian,
                           MultivariateBetaGaussianDiag)

def main():
    torch.manual_seed(42)
    torch.set_default_dtype(torch.double)

    n_samples = 100
    d = 2

    true_loc = torch.tensor([0.0, 0.0])
    true_sigma = torch.tensor([.5, 2.0])

    loc = torch.randn(d, requires_grad=True)
    scale_diag = torch.randn(d, requires_grad=True)

    alpha = torch.tensor(1.25, requires_grad=True)

    parameters = [loc, scale_diag]

    opt = torch.optim.Adam(parameters, lr=1.0)

    n_iter = 10000

    for it in range(n_iter):
        opt.zero_grad()

        scale = scale_diag ** 2
        mbg = MultivariateBetaGaussianDiag(loc, scale, alpha=alpha)
        # get a sample
        X = true_loc + torch.randn((n_samples, d)) * true_sigma

        loss = mbg.cross_fy(X).mean()
        if it % 1000 == 0:
            print(loss.item())
        loss.backward()
        opt.step()

    print("true", true_loc, true_sigma)
    print("learned", loc, scale_diag)

    plt.figure()

    n_samples = 10000

    # true sample
    X = true_loc + torch.randn((n_samples, d)) * true_sigma
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(),
                marker='.', alpha=.2,
                label="true")

    Z = mbg.rsample(sample_shape=(n_samples,)).detach()
    plt.scatter(Z[:, 0].numpy(), Z[:, 1].numpy(),
                marker='.', alpha=.2,
                label="pred")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

