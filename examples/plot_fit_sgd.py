import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.distributions import MultivariateNormal
from spcdist.torch import (MultivariateBetaGaussian,
                           MultivariateBetaGaussianDiag)


def meshplot(f, ax, n_samples=100):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x = np.linspace(xmin, xmax, n_samples)
    y = np.linspace(ymin, ymax, n_samples)
    mesh_x, mesh_y = np.meshgrid(x, y)

    X = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    z = f(torch.from_numpy(X)).detach()
    z = z.numpy().reshape(*mesh_x.shape)
    ax.contourf(mesh_x, mesh_y, z)


def _scatter(ax, X, title):
    X = X.detach().numpy()
    ax.scatter(X[:, 0],
               X[:, 1],
               alpha=.5,
               marker='.',
               color='C1',
               zorder=10,)
    ax.set_title(title)


def main():

    torch.set_default_dtype(torch.double)

    n_iter = 1000
    n_samples = 1000
    batch_size = 32
    alpha = 2
    dim = 2

    fig, axes = plt.subplots(1, 3, figsize=(10, 4),
                             sharex=True,
                             sharey=True,
                             constrained_layout=True)
    ax_true, ax_pred_before, ax_pred_after = axes

    true_loc = torch.randn(dim)
    true_cov = torch.tensor([[1, .2],
                             [.2, .5]])
    mvn = MultivariateNormal(true_loc, true_cov)

    loc = torch.randn(dim, requires_grad=True)
    scale_diag = torch.randn(dim, requires_grad=True)
    mbg = MultivariateBetaGaussianDiag(loc, scale_diag ** 2, alpha=alpha)

    _scatter(ax_true, mvn.rsample((n_samples,)), title="Multivariate normal")
    _scatter(ax_pred_before, mbg.rsample((n_samples,)),
             title="Before fitting")

    gauss_pdf = lambda x: torch.exp(mvn.log_prob(x))
    meshplot(gauss_pdf, ax_true)
    meshplot(mbg.pdf, ax_pred_before)

    parameters = [loc, scale_diag]
    opt = torch.optim.Adam(parameters, lr=1.0)

    for it in range(n_iter):
        opt.zero_grad()

        scale = scale_diag ** 2
        mbg = MultivariateBetaGaussianDiag(loc, scale, alpha=alpha)
        # get a sample
        X = mvn.rsample((batch_size,))

        loss = mbg.cross_fy(X).mean()
        loss.backward()
        opt.step()

    _scatter(ax_pred_after, mbg.rsample(sample_shape=(n_samples,)),
             title="After fitting")
    meshplot(mbg.pdf, ax_pred_after)

    plt.show()


if __name__ == '__main__':
    main()

