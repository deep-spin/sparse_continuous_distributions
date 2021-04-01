from scipy.special import gamma
from scipy.linalg import sqrtm
import numpy as np
import math


def _radius(n, alpha):
    """Return radius R for a given dimension n and alpha."""
    return ((gamma(n/2 + alpha/(alpha-1)) /
             (gamma(alpha/(alpha-1)) * np.pi**(n/2))) *
            (2 / (alpha-1)) ** (1/(alpha-1))) ** ((alpha-1)/(2 + (alpha-1)*n))


class EntmaxGaussian(object):
    def __init__(self, alpha, mu, Sigma):
        """Create multivariate (2-alpha)-Gaussian with parameter alpha.
        The density is:
            p(x) = [(alpha-1)*(-tau - f(x)]_+ ** (1/(alpha-1)),
        where f(x) is a quadratic score function:
            f(x) = .5*(x-mu).T.dot(Sigma_inv).dot(x-mu).
        """
        self._alpha = alpha
        self._n = np.size(mu)
        self._R = _radius(self._n, alpha) if alpha != 1 else math.inf
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
        self._R = _radius(1, alpha) if alpha != 1 else math.inf
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
