from entmax_gaussian import EntmaxGaussian
import numpy as np
import matplotlib.pyplot as plt


def plot_contour(ax, entmax, color='k'):
    sample = entmax.sample(1000) # 100000
    xmin = sample[:, 0].min() - 0.1
    xmax = sample[:, 0].max() + 0.1
    ymin = sample[:, 1].min() - 0.1
    ymax = sample[:, 1].max() + 0.1
    delta = 0.001
    x = np.arange(xmin, xmax, delta)
    y = np.arange(ymin, ymax, delta)
    X, Y = np.meshgrid(x, y)
    Z = entmax.pdf(np.concatenate([X.flatten()[None, :], Y.flatten()[None, :]]))
    Z = Z.reshape(X.shape)
    CS = ax.contour(X, Y, Z, colors=color)
    ax.clabel(CS)
    ax.axis('equal')


np.random.seed(1)
mu_ell = np.random.randn(2)
A = np.random.randn(2, 2)
Sigma_ell = A.dot(A.T)

fig, ax = plt.subplots(2, 4)

names = ['Gaussian', 'Triweight', 'Biweight', 'TP']

for i, alpha in enumerate([1, 4/3, 3/2, 2]):
    for j, spherical in enumerate([True, False]):
        if spherical:
            mu = np.zeros(2)
            Sigma = np.eye(2)
        else:
            mu = mu_ell
            Sigma = Sigma_ell

        entmax = EntmaxGaussian(alpha, mu, Sigma)
        sample = entmax.sample(100)  # 1000
        #print(sample)
        print(entmax.mean())
        print(entmax.variance())
        print(entmax._Sigma_from_variance(entmax.variance()))  # should equal Sigma.

        mu_fit = np.mean(sample, axis=0)
        variance_fit = np.cov(sample, rowvar=False)
        Sigma_fit = entmax._Sigma_from_variance(variance_fit)
        print(mu_fit)
        print(variance_fit)
        print(Sigma_fit)

        entmax_fit = EntmaxGaussian(alpha, mu_fit, Sigma_fit)

        ax[j, i].set_title("%s ($\\alpha=%.2d$)" % (names[i], alpha))
        ax[j, i].plot(sample[:, 0], sample[:, 1], 'b.')

        plot_contour(ax[j, i], entmax, color='k')
        plot_contour(ax[j, i], entmax_fit, color='r')

plt.show()
