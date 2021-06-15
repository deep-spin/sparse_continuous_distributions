import numpy as np
import matplotlib.pyplot as plt

from spcdist.scipy import multivariate_beta_gaussian, scale_from_cov

levels = [0.01, 0.03, 0.1, 0.3]


def plot_contour(mbg, ax, n_samples=500, label=False, supp=False, **kwargs):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    x = np.linspace(xmin, xmax, n_samples)
    y = np.linspace(ymin, ymax, n_samples)
    X, Y = np.meshgrid(x, y)
    Z = mbg.pdf(np.column_stack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)

    if supp:  # shade the support
        ax.contourf(X, Y, Z, levels=[0, np.finfo(np.double).eps], colors='#ccc')

    # draw the contour lines
    CS = ax.contour(X, Y, Z, levels=levels, **kwargs)

    if label:  # label the contour lines
        ax.clabel(CS)


if __name__ == "__main__":

    dim = 2
    n_samples = 300
    rng = np.random.default_rng(5)

    params = {
        'spherical': (np.zeros(dim), np.eye(dim)),
        'elliptical': (np.zeros(dim), np.array([[1.5, 1], [1, 1.2]]))
    }

    configurations = [
        dict(name="Gaussian", alpha=1),
        dict(name="Triweight", alpha=4/3),
        dict(name="Biweight", alpha=3/2),
        dict(name="Epanechnikov", alpha=2),
    ]

    fig, axes = plt.subplots(len(params), len(configurations),
                           figsize=(8.2, 4),
                           constrained_layout=True,
                           sharex='row', sharey='row')

    for i, cfg in enumerate(configurations):
        for j, shape in enumerate(["spherical", "elliptical"]):
            loc, scale = params[shape]
            mbg = multivariate_beta_gaussian(loc, scale, alpha=cfg['alpha'])
            X = mbg.rvs(n_samples, random_state=rng)

            loc_fit = np.mean(X, axis=0)
            cov_fit = np.cov(X, rowvar=False)
            scale_fit = scale_from_cov(cfg['alpha'], cov_fit)
            mbg_fit = multivariate_beta_gaussian(loc_fit, scale_fit, alpha=cfg['alpha'])

            ax = axes[j, i]
            ax.set_title(r"{name} ($\alpha={alpha:.1f}$)".format(**cfg))
            ax.set_aspect('equal', share=True)
            ax.scatter(X[:, 0], X[:, 1], marker='.', s=1)
            plot_contour(mbg, ax, colors='k', label=i == j == 0, supp=True)
            plot_contour(mbg_fit, ax, colors='C1', alpha=.5, linestyles="--")

    plt.show()
