from scipy.special import gamma
from scipy.linalg import sqrtm
import numpy as np

from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen

from utils import _PSD, _process_parameters


_LOG_2PI = np.log(2 * np.pi)


def _radius(n, alpha):
    """Return radius R for a given dimension n and alpha."""

    a_m1 = alpha - 1
    a_ratio = alpha / a_m1

    return ((gamma(n / 2 + a_ratio) / (gamma(a_ratio) * np.pi ** (n / 2))) *
            (2 / a_m1) ** (1 / a_m1)) ** (a_m1 / (2 + a_m1 * n))


def scale_from_cov(alpha, cov):
    """Compute scale parameter Sigma, given full-rank covariance matrix cov."""

    # XXX: can support low-rank with a single factorization.

    if alpha == 1:
        return cov

    n = cov.shape[0]

    radius = _radius(n, alpha)
    scale_tilde = ((n + 2 * alpha / (alpha - 1)) / radius ** 2) * cov
    det_st = np.linalg.det(scale_tilde)
    scale = (det_st ** ((alpha - 1) / 2)) * scale_tilde
    return Sigma


class multivariate_beta_gaussian_gen(multi_rv_generic):
    r"""A multivariate beta-Gaussian random variable.

    The `mean` keyword specifies the mean.
    The `loc` keyword specifies the Sigma matrix (uniquely defines the covariance.

    Methods
    -------
    ``pdf(x, mean=None, scale=1, alpha-2, allow_singular=False)``
        Probability density function.
    ``rvs(mean=None, scale=1, alpha=2, size=1, allow_singular=False, random_state=None)``
        Draw random samples from a multivariate beta-Gaussian distribution.
    ``variance(mean=None, scale=1, alpha=2, allow_singular=False)``
        Compute the covariance matrix given the scale matrix.
    ``tsallis_entropy(mean=None, scale=1, alpha=2, allow_singular=False)``
        Compute the Tsallis entropy of the multivariate beta-Gaussian.
    """

    def __init__(self, seed=None):
        super().__init__(seed)

    def __call__(self, mean=None, scale=1, alpha=2, allow_singular=False, seed=None):
        return multivariate_beta_gaussian_frozen(mean, scale, alpha, allow_singular, seed)

    def _process_parameters(self, dim, mean, scale, alpha):
        """
        Infer dimensionality from mean or covariance matrix, ensure that
        mean and covariance are full vector resp. matrix.
        """
        return _process_parameters(dim, mean, scale, alpha)

    def _tau(self, alpha, log_det, rank):
        """Return the threshold tau in the density expression."""

        if alpha == 1:  # Gaussian
            return -0.5 * (rank * _LOG_2PI + log_det)

        else:  # Tsallis
            a_m1 = alpha - 1
            radius = _radius(rank, alpha)
            return -(radius ** 2) / 2 * np.exp(-log_det / (rank + (2 / a_m1)))

    def _pdf(self, x, mean, prec_U, log_det, rank, alpha, radius):
        dev = x - mean
        neg_maha = -0.5 * np.sum(np.square(np.dot(dev, prec_U)), axis=-1)

        logpdf = neg_maha - self._tau(alpha, log_det, rank)  # Tsallis log
        # XXX could return a (log_beta)pdf at this point

        if alpha > 1:
            a_m1 = alpha - 1
            pdf = np.maximum(a_m1 * logpdf, 0) ** (1 / a_m1)
        else:
            pdf = np.exp(logpdf)

        return pdf

    def pdf(self, x, mean=None, scale=1, alpha=2, allow_singular=False):
        dim, mean, scale, alpha = self._process_parameters(None, mean, scale, alpha)
        psd = _PSD(scale, allow_singular=allow_singular)
        radius = _radius(psd.rank, alpha)
        return self._pdf(x, mean, psd.U, psd.log_pdet, psd.rank, alpha, radius)

    def _rvs(self, mean, scale_sqrt, rank, log_det, alpha, radius, size, random_state):

        a_m1 = alpha - 1

        # Sample uniformly from sphere.
        if np.isscalar(size):
            size = (size,)

        u = random_state.randn(*(size + (rank,)))
        u /= np.linalg.norm(u, axis=-1)[..., np.newaxis]

        # Sample radius.
        # ratio = r^2 / radius^2, so r = radius * sqrt(ratio).
        ratio = random_state.beta(rank / 2, alpha / a_m1, size=size)
        r = radius * np.sqrt(ratio)
        z = r[:, np.newaxis] * u

        Uz = z @ scale_sqrt.T
        Uz *= np.exp(-log_det / (2 * rank + 4 / a_m1))

        return mean + Uz

    def rvs(self, mean=None, scale=1, alpha=2, size=1, allow_singular=False, random_state=None):

        dim, mean, scale, alpha = self._process_parameters(None, mean, scale, alpha)
        random_state = self._get_random_state(random_state)

        if alpha == 1:
            out = random_state.multivariate_normal(mean, scale, size)
        else:
            psd = _PSD(scale, allow_singular=allow_singular)
            radius = _radius(psd.rank, alpha)
            out = self._rvs(mean,
                            psd.L,
                            psd.rank,
                            psd.log_pdet,
                            alpha,
                            radius,
                            size,
                            random_state)

        return out

    def _variance(self, scale, alpha, log_det, rank):
        if alpha == 1:
            return scale
        else:
            a_ratio = alpha / (alpha - 1)
            tau = self._tau(alpha, log_det, rank)
            return ((-2 * tau) / (rank + 2 * a_ratio)) * scale

    def variance(self, mean=None, scale=1, alpha=2, allow_singular=False):
        """Compute the covariance given the scale matrix. (mean is ignored.)"""
        dim, mean, scale, alpha = self._process_parameters(None, mean, scale, alpha)
        psd = _PSD(scale, allow_singular=allow_singular)
        return self._variance(scale, alpha, psd.log_pdet, psd.rank)

    def _tsallis_entropy(self, alpha, rank, tau):
        a_m1 = alpha - 1
        return (1 / (alpha * a_m1)) + ((2 * tau) / (2 * alpha + rank * a_m1))

    def tsallis_entropy(self, mean=None, scale=1, alpha=2, allow_singular=False):
        """Compute Tsallis alpha-entropy. (mean is ignored.)"""
        dim, mean, scale, alpha = self._process_parameters(None, mean, scale, alpha)
        psd = _PSD(scale, allow_singular=allow_singular)
        tau = self._tau(alpha, psd.log_pdet, psd.rank)
        return self._tsallis_entropy(alpha, psd.rank, tau)


multivariate_beta_gaussian = multivariate_beta_gaussian_gen()


class multivariate_beta_gaussian_frozen(multi_rv_frozen):
    def __init__(self, mean=None, scale=1, alpha=2, allow_singular=False, seed=None):
        self._dist = multivariate_beta_gaussian_gen(seed)
        self.dim, self.mean, self.scale, self.alpha = \
            self._dist._process_parameters(None, mean, scale, alpha)
        self.scale_info = _PSD(self.scale, allow_singular=allow_singular)
        self.radius = _radius(self.scale_info.rank, alpha)
        self.tau = self._dist._tau(alpha, self.scale_info.log_pdet, self.scale_info.rank)

    def pdf(self, x):
        out = self._dist._pdf(x,
                              self.mean,
                              self.scale_info.U,
                              self.scale_info.log_pdet,
                              self.scale_info.rank,
                              self.alpha,
                              self.radius)
        return out
        # return _squeeze_output(out)

    def rvs(self, size=1, random_state=None):
        random_state = self._dist._get_random_state(random_state)
        if self.alpha == 1:
            return random_state.multivariate_normal(self.mean, self.scale, size)
        else:
            return self._dist._rvs(self.mean,
                                   self.scale_info.L,
                                   self.scale_info.rank,
                                   self.scale_info.log_pdet,
                                   self.alpha,
                                   self.radius,
                                   size,
                                   random_state)

    def tsallis_entropy(self):
        return self._dist._tsallis_entropy(self.alpha, self.scale_info.rank,
                self.tau)


class EntmaxGaussian(object):
    def __init__(self, alpha, mu, Sigma):
        """Create multivariate (2-alpha)-Gaussian with parameter alpha.
        The density is:
            p(x) = [(alpha - 1) * (-tau - f(x))]_+ ** (1 / (alpha - 1)),
        where f(x) is a quadratic score function:
            f(x) = .5 * (x - mu).T @ Sigma_inv @ (x - mu).
        """
        self._alpha = alpha
        self._n = np.size(mu)
        self._R = _radius(self._n, alpha) if alpha != 1 else np.inf
        self._mu = mu
        self._Sigma = Sigma
        self._Sigma_inv = np.linalg.inv(Sigma)
        self._Sigma_inv_sqrt = sqrtm(self._Sigma_inv)

    def _tau(self):
        """Return the threshold tau in the density expression."""
        return -(self._R**2)/2 * np.linalg.det(self._Sigma) ** (
            -1 / (self._n + 2/(self._alpha - 1)))

    def _Sigma_from_variance(self, variance):
        if self._alpha == 1:
            return np.array(variance)
        Sigma_tilde = ((self._n + 2*self._alpha/(self._alpha-1)) / self._R**2
                       ) * variance
        Sigma = (np.linalg.det(Sigma_tilde) ** ((self._alpha-1)/2)
                 ) * Sigma_tilde
        return Sigma

    def mean(self):
        return self._mu

    def variance(self):
        if self._alpha == 1:
            return np.array(self._Sigma)
        return ((-2*self._tau())/(self._n + 2*self._alpha/(self._alpha-1)) *
                self._Sigma)

    def pdf(self, x):
        """Return the probability density function value for `x`."""
        y = self._Sigma_inv_sqrt.dot(x - self._mu[:, None])
        negenergy = -.5 * np.sum(y**2, 0)
        if self._alpha == 1:
            return np.exp(negenergy) / (
                (2*np.pi)**(self._n/2) * np.linalg.det(self._Sigma)**(1/2))
        else:
            return np.maximum(0, (self._alpha-1)*(
                negenergy - self._tau()))**(1/(self._alpha-1))

    def sample(self, m):
        """Generate a random sample of size `m`."""
        if self._alpha == 1:
            z = np.random.randn(m, self._n)
            A = sqrtm(self._Sigma)
            return (self._mu[:, None] + A.dot(z.T)).T

        # Sample uniformly from sphere.
        u = np.random.randn(m, self._n)
        u /= np.linalg.norm(u, axis=1)[:, np.newaxis]

        # Sample radius.
        # ratio = r^2 / R^2, so r = R * sqrt(ratio).
        ratio = np.random.beta(self._n / 2,
                               self._alpha / (self._alpha - 1),
                               size=m)
        r = self._R * np.sqrt(ratio)
        z = r[:, np.newaxis] * u
        A = (np.linalg.det(self._Sigma) ** (-1/(2*self._n + 4/(self._alpha-1)))
             ) * sqrtm(self._Sigma)
        return (self._mu[:, None] + A.dot(z.T)).T

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        return (1/(self._alpha*(self._alpha-1)) -
                (-2*self._tau())/(2*self._alpha + self._n * (self._alpha-1)))


class EntmaxGaussian1D(object):
    def __init__(self, alpha, mu=0, sigma_sq=None, support_size=None):
        """Create 1D (2-alpha)-Gaussian with parameter alpha.
        The density is
            p(x) = [(alpha-1)*(-tau - .5*(x-mu)**2/sigma_sq)]_+**(1/(alpha-1)).
        If sigma_sq == None, it can be paremetrized by the support_size instead
        (convenient for uniform distributions, where alpha=inf).
        """
        self._alpha = alpha
        self._R = _radius(1, alpha) if alpha != 1 else np.inf
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        """Return the threshold tau in the density expression."""
        return (-(self._R**2)/2 * self._sigma_sq **
                (-(self._alpha-1) / (self._alpha+1)))

    def _compute_a(self):
        """Return the value a = |x-mu| where the density vanishes."""
        return np.sqrt(-2 * self._tau * self._sigma_sq)

    def _sigma_sq_from_a(self, a):
        return (a / self._R) ** (self._alpha+1)

    def _sigma_sq_from_variance(self, variance):
        return ((1 + 2*self._alpha/(self._alpha-1)) / (self._R**2)
                * variance) ** ((self._alpha + 1)/2)

    def mean(self):
        return self._mu

    def variance(self):
        if self._alpha == np.inf:
            return self._a**2 / 3
        else:
            # Equivalently (without tau):
            # return ((self._R**2) / (1 + 2*self._alpha/(self._alpha-1)) *
            #        self._sigma_sq ** (2/(self._alpha + 1)))
            return ((-2*self._tau)/(1 + 2*self._alpha/(self._alpha-1)) *
                    self._sigma_sq)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        """Return the probability density function value for `x`."""
        if self._alpha == np.inf:
            p = np.zeros_like(x)
            mask = (x >= self._mu - self._a) & (x <= self._mu + self._a)
            p[mask] = 1/self.support_size()
            return p
        else:
            return np.maximum(0, (self._alpha-1)*(
                -self._tau - .5*(x-self._mu)**2/self._sigma_sq)
                              )**(1/(self._alpha-1))

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class Gaussian1D(object):
    def __init__(self, mu=0, sigma_sq=1):
        """Create 1D beta-Gaussian with alpha=1 (Gaussian)."""
        self._alpha = 1
        self._mu = mu
        self._sigma_sq = sigma_sq

    def mean(self):
        return self._mu

    def variance(self):
        return self._sigma_sq

    def pdf(self, x):
        return (1/np.sqrt(2*np.pi*self._sigma_sq) *
                np.exp(-.5*(x-self._mu)**2/self._sigma_sq))

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class SparsemaxGaussian1D(object):
    def __init__(self, mu=0, sigma_sq=None, support_size=None):
        """Create 1D beta-Gaussian with alpha=2 (sparsemax)."""
        self._alpha = 2
        self._R = (3/2)**(1/3)
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        return -.5*((3/2)**2/self._sigma_sq)**(1/3)

    def _compute_a(self):
        return ((3/2)*self._sigma_sq)**(1/3)

    def _sigma_sq_from_a(self, a):
        return (2/3) * a**3

    def _sigma_sq_from_variance(self, variance):
        return 2/3 * (5*variance)**(3/2)

    def mean(self):
        return self._mu

    def variance(self):
        return 1/5 * ((3/2) * self._sigma_sq)**(2/3)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        return np.maximum(0, -self._tau - .5*(x-self._mu)**2/self._sigma_sq)

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class BiweightGaussian1D(object):
    def __init__(self, mu=0, sigma_sq=None, support_size=None):
        """Create 1D beta-Gaussian with alpha=1.5 (biweight)."""
        self._alpha = 1.5
        self._R = _radius(1, self._alpha)  # 15**(1/5)
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        return -.5*(15**2/self._sigma_sq)**(1/5)

    def _compute_a(self):
        return (15*self._sigma_sq**2)**(1/5)

    def _sigma_sq_from_a(self, a):
        return (a / self._R) ** (self._alpha+1)

    def _sigma_sq_from_variance(self, variance):
        return (1/15)**(1/2) * (7*variance)**(5/4)

    def mean(self):
        return self._mu

    def variance(self):
        return ((-2*self._tau)/(1 + 2*self._alpha/(self._alpha-1)) *
                self._sigma_sq)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        return np.maximum(
            0, .5*(-self._tau - .5*(x-self._mu)**2/self._sigma_sq))**2

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class TriweightGaussian1D(object):
    def __init__(self, mu=0, sigma_sq=None, support_size=None):
        """Create 1D beta-Gaussian with alpha=4/3 (triweight)."""
        self._alpha = 4/3
        self._R = _radius(1, self._alpha)  # (945/4)**(1/7)
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        return -.5*((945/4)**2/self._sigma_sq)**(1/7)

    def _compute_a(self):
        return ((945/4)*self._sigma_sq**3)**(1/7)

    def _sigma_sq_from_a(self, a):
        return (a / self._R) ** (self._alpha+1)

    def _sigma_sq_from_variance(self, variance):
        return (4/945)**(1/3) * (9*variance)**(7/6)

    def mean(self):
        return self._mu

    def variance(self):
        return ((-2*self._tau)/(1 + 2*self._alpha/(self._alpha-1)) *
                self._sigma_sq)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        return np.maximum(
            0, (1/3)*(-self._tau - .5*(x-self._mu)**2/self._sigma_sq))**3

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


def check_cov():
    mbg = multivariate_beta_gaussian_gen()
    mean = np.zeros(2)
    scale = np.array([[1, .2],
                      [.2, .3]])

    for alpha in (1.25, 1.5, 2, 2.5):
        X = mbg.rvs(mean=mean, scale=scale, size=10000, alpha=alpha, random_state=0)
        empirical = np.cov(X.T)
        theoretical = mbg.variance(mean=mean, scale=scale, alpha=alpha)
        print(np.linalg.norm(empirical - theoretical))


def main():

    import matplotlib.pyplot as plt

    # alias
    mbg = multivariate_beta_gaussian

    # check in 1d with default parameters:
    for alpha in (1.5, 2, 2.5):
        X = mbg.rvs(alpha=alpha, size=10000)
        plt.hist(X, alpha=.2, label=alpha, density=True)
    plt.legend()
    plt.show()

    # check low rank:
    plt.figure()
    X = mbg.rvs(mean=1, scale=.5, alpha=2, size=100000)
    plt.hist(X, alpha=.2, label="full rank", density=True)

    mean = np.array([1, 42])
    scale = np.array([.5, 0])
    X = mbg.rvs(mean=mean, scale=scale, alpha=2, allow_singular=True,
            size=100000)

    assert np.all(X[:, 1] == 42)
    plt.hist(X[:, 0], alpha=.2, label="low rank", density=True)
    plt.legend()
    plt.show()

    # check in 2d
    mean = np.zeros(2)
    scale = np.array([[1, .2],
                      [.2, .3]])

    _, (ax1, ax2) = plt.subplots(1, 2)

    for alpha in (1.1, 1.5, 2):
        eg = EntmaxGaussian(alpha=alpha, mu=mean, Sigma=scale)

        X = mbg.rvs(mean=mean, scale=scale, size=10000, alpha=alpha)
        ax1.scatter(X[:, 0], X[:, 1], alpha=.2, marker='.')

        print(eg.pdf(X[:5].T))
        print(mbg.pdf(X[:5], mean=mean, scale=scale, alpha=alpha))

        # X = np.random.multivariate_normal(mean, scale, 10000)
        X = eg.sample(10000)
        ax2.scatter(X[:, 0], X[:, 1], alpha=.2, marker='.')

        print("entropy:", mbg.tsallis_entropy(mean, scale, alpha))

    plt.show()

    # more elegant: with frozen object
    plt.figure()

    for alpha in (1.1, 1.5, 2):
        mbg = multivariate_beta_gaussian(mean=mean, scale=scale, alpha=alpha)
        X = mbg.rvs(size=10000)
        plt.scatter(X[:, 0], X[:, 1], alpha=.2, marker='.')
        print(mbg.pdf(X[:5]))
        print("entropy", mbg.tsallis_entropy())
    plt.show()



if __name__ == '__main__':
    check_cov()
    main()

