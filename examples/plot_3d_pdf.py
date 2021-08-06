import numpy as np
import matplotlib.pyplot as plt

from spcdist.scipy import multivariate_beta_gaussian

def plot_3d(mbg, ax, n_samples=150):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = np.linspace(xmin, xmax, n_samples)
    y = np.linspace(ymin, ymax, n_samples)
    X, Y = np.meshgrid(x, y)
    Z = mbg.pdf(np.column_stack([X.ravel(), Y.ravel()]))
    Z = Z.reshape(X.shape)
    # ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5,
                    cmap='autumn', edgecolor='white')

def main():
    dim = 2
    loc = np.zeros(dim)
    scale = .4 * np.array([[1.5, 1], [1, 1.2]])

    fig = plt.figure(figsize=(7, 7)) # , constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')

    configurations = [
        dict(name="Gaussian", alpha="1", ax=ax1),
        dict(name="Triweight", alpha="4/3", ax=ax2),
        dict(name="Biweight", alpha="3/2", ax=ax3),
        dict(name="Truncated Paraboloid", alpha="2", ax=ax4),
    ]

    for i, cfg in enumerate(configurations):
        alpha_ = eval(cfg['alpha'], {}, {})
        mbg = multivariate_beta_gaussian(loc, scale, alpha_)

        ax = cfg['ax']
        ax.set_title(r"{name} ($\alpha={alpha}$)".format(**cfg),
                     fontsize=16)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('$t_1$', fontsize=16)
        ax.set_ylabel('$t_2$', fontsize=16)
        ax.grid(False)
        plot_3d(mbg, ax)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

