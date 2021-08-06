import numpy as np
import matplotlib.pyplot as plt

import torch
from spcdist.torch import MultivariateBetaGaussian


def plot_contour(mbg, ax, n_samples=500, label=False, supp=False, **kwargs):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x = np.linspace(xmin, xmax, n_samples)
    y = np.linspace(ymin, ymax, n_samples)

    mesh_x, mesh_y = np.meshgrid(x, y)

    X = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])
    z = mbg.pdf(torch.from_numpy(X)).detach()
    z = z.numpy().reshape(*mesh_x.shape)

    if supp:  # shade the support
        ax.contourf(mesh_x, mesh_y, z,
                    levels=[0, np.finfo(np.double).eps], colors='#ccc')

    # draw the contour lines
    CS = ax.contour(mesh_x, mesh_y, z, **kwargs)

    if label:  # label the contour lines
        ax.clabel(CS, fmt="%.2f")


if __name__ == "__main__":

    dim = 2
    n_samples = 300
    rng = np.random.default_rng(5)
    torch.set_default_dtype(torch.double)

    # location and scale,
    # chosen so the contour plot doesn't require many decimals.
    params = {
        'spherical': (np.zeros(dim), .4 * np.eye(dim)),
        'elliptical': (np.zeros(dim), .4 * np.array([[1.5, 1], [1, 1.2]]))
    }

    # contour plot levels
    levels = [0.01, 0.03, 0.1, 0.3]

    configurations = [
        dict(name="Gaussian", alpha="1.01"),
        dict(name="Triweight", alpha="4/3"),
        dict(name="Biweight", alpha="3/2"),
        dict(name="Epanechnikov", alpha="2"),
    ]

    fig, axes = plt.subplots(len(params), len(configurations),
                           figsize=(8.2, 4),
                           constrained_layout=True,
                           sharex='row', sharey='row')

    for i, cfg in enumerate(configurations):
        for j, shape in enumerate(["spherical", "elliptical"]):
            loc, scale = params[shape]
            alpha_ = eval(cfg['alpha'], {}, {})

            loc = torch.from_numpy(loc)
            scale = torch.from_numpy(scale)

            mbg = MultivariateBetaGaussian(loc, scale, alpha=alpha_)

            # sample from the beta-Gaussian
            X = mbg.rsample((n_samples,))

            # plot samples and contours
            ax = axes[j, i]
            ax.set_title(r"{name} ($\alpha={alpha}$)".format(**cfg))
            ax.set_aspect('equal', share=True)
            ax.scatter(X[:, 0], X[:, 1], marker='.', s=1)
            label = i == j == 0 # only show label on first subplot
            plot_contour(mbg, ax, colors='k', label=label, supp=True,
                         levels=levels)

    plt.show()
