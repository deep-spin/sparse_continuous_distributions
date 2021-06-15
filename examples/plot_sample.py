import numpy as np
import matplotlib.pyplot as plt

from spcdist.scipy import multivariate_beta_gaussian, scale_from_cov


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
        ax.clabel(CS, fmt="%.2f")


if __name__ == "__main__":

    dim = 2
    n_samples = 300
    rng = np.random.default_rng(5)

    # location and scale,
    # chosen so the contour plot doesn't require many decimals.
    params = {
        'spherical': (np.zeros(dim), .4 * np.eye(dim)),
        'elliptical': (np.zeros(dim), .4 * np.array([[1.5, 1], [1, 1.2]]))
    }

    # contour plot levels
    levels = [0.01, 0.03, 0.1, 0.3]

    configurations = [
        dict(name="Gaussian", alpha=1),
        dict(name="Triweight", alpha=4 / 3),
        dict(name="Biweight", alpha=3 / 2),
        dict(name="Epanechnikov", alpha=2),
    ]

    fig, axes = plt.subplots(len(params), len(configurations),
                           figsize=(8.2, 4),
                           constrained_layout=True,
                           sharex='row', sharey='row')

    for i, cfg in enumerate(configurations):
        for j, shape in enumerate(["spherical", "elliptical"]):
            loc, scale = params[shape]
            mbg = multivariate_beta_gaussian(loc, scale,
                                             alpha=cfg['alpha'])

            # sample from the beta-Gaussian
            X = mbg.rvs(n_samples, random_state=rng)

            # fit a new beta-Gaussian to the sample
            loc_fit = np.mean(X, axis=0)
            cov_fit = np.cov(X, rowvar=False)
            scale_fit = scale_from_cov(cfg['alpha'], cov_fit)
            mbg_fit = multivariate_beta_gaussian(loc_fit, scale_fit,
                                                 alpha=cfg['alpha'])

            # plot samples and contours
            ax = axes[j, i]
            ax.set_title(r"{name} ($\alpha={alpha:.1f}$)".format(**cfg))
            ax.set_aspect('equal', share=True)
            ax.scatter(X[:, 0], X[:, 1], marker='.', s=1)
            label = i == j == 0 # only show label on first subplot
            plot_contour(mbg, ax, colors='k', label=label, supp=True,
                         levels=levels)
            plot_contour(mbg_fit, ax, colors='C1', alpha=.5, levels=levels,
                         linestyles="--")

    plt.show()
