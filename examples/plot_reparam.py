"""demonstrate reparametrization trick"""

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from spcdist.torch import MultivariateBetaGaussianDiag

import matplotlib.pyplot as plt


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


def _scatter(ax, X):
    X = X.detach().numpy()
    ax.scatter(X[:, 0],
               X[:, 1],
               alpha=.5,
               marker='.',
               color='C1',
               zorder=10,)


if __name__ == "__main__":

    torch.set_default_dtype(torch.double)
    n_iter = 1000
    batch_size = 32
    n_samples = 300
    alpha = 3.0
    dim = 2

    fig, axes = plt.subplots(1, 3, figsize=(10, 4),
                             sharex=True,
                             sharey=True,
                             constrained_layout=True)
    ax_true, ax_pred_before, ax_pred_after = axes

    # true gaussian:
    true_loc = torch.randn(dim)
    true_cov = torch.tensor([[1, .2],
                             [.2, .5]])

    mvn = MultivariateNormal(true_loc, true_cov)
    X_true = mvn.rsample(sample_shape=(n_samples,))
    _scatter(ax_true, X_true)

    # initial predicted distribution
    loc = torch.randn(dim, requires_grad=True)
    scale_base = torch.randn(dim, requires_grad=True)

    mvbg = MultivariateBetaGaussianDiag(loc, scale_base ** 2,
                                        alpha=alpha)

    X_pred = mvbg.rsample(sample_shape=(n_samples,)).detach()
    _scatter(ax_pred_before, X_pred)

    gauss_pdf = lambda x: torch.exp(mvn.log_prob(x))
    meshplot(gauss_pdf, ax_true)
    meshplot(mvbg.pdf, ax_pred_before)

    # fit
    parameters = [loc, scale_base]
    opt = torch.optim.Adam(parameters, lr=1.0)
    for _ in range(n_iter):
        opt.zero_grad()
        mvbg = MultivariateBetaGaussianDiag(loc, scale_base ** 2,
                                            alpha=alpha)
        X_spl = mvbg.rsample(sample_shape=(batch_size,))
        obj = (mvbg.log_prob(X_spl) -mvn.log_prob(X_spl)).mean()
        obj.backward()
        opt.step()
        print(obj.item())

    X_spl = mvbg.rsample(sample_shape=(n_samples,))
    _scatter(ax_pred_after, X_spl)

    meshplot(mvbg.pdf, ax_pred_after)

    ax_true.set_title("True target (Gaussian)")
    ax_pred_before.set_title("Predicted (before training")
    ax_pred_after.set_title("Predicted (after training")

    plt.show()

