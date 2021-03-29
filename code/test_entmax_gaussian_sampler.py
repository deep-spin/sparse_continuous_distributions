from entmax_gaussian import EntmaxGaussian
import numpy as np
import matplotlib.pyplot as plt

alpha = 4/3 #2
mu = np.random.randn(2) # np.zeros(2)
A = np.random.randn(2, 2)
Sigma = A.dot(A.T)
#Sigma = np.eye(2)
entmax = EntmaxGaussian(alpha,
                        mu,
                        Sigma)
sample = entmax.sample(1000)
print(sample)
print(entmax.mean())
print(entmax.variance())
print(entmax._Sigma_from_variance(entmax.variance()))  # should equal Sigma.

mu_fit = np.mean(sample, axis=0)
variance_fit = np.cov(sample, rowvar=False)
Sigma_fit = entmax._Sigma_from_variance(variance_fit)
print(mu_fit)
print(variance_fit)
print(Sigma_fit)

entmax_fit = EntmaxGaussian(alpha,
                            mu_fit,
                            Sigma_fit)

plt.plot(sample[:, 0], sample[:, 1], 'b.')

def plot_contour(entmax, color='k'):
    sample = entmax.sample(100000)
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
    CS = plt.contour(X, Y, Z, colors=color)
    plt.clabel(CS)
    plt.axis('equal')

#fig, ax = plt.subplots()
#CS = ax.contour(X, Y, Z)
#ax.clabel(CS, inline=True, fontsize=10)
#ax.set_title('Simplest default with labels')

plot_contour(entmax, color='k')
plot_contour(entmax_fit, color='r')

plt.show()
