from scipy import integrate
from scipy.special import erf
import numpy as np


def Phi(x):
    return .5*(1 + erf(x/np.sqrt(2)))

def phi(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-.5*x**2)


class GaussianBasisFunctions(object):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)."""

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)

    def _phi(self, t):
        return 1.0 / np.sqrt(2 * np.pi) * np.exp(-0.5 * t ** 2)

    def _Phi(self, t):
        return 0.5 * (1 + erf(t / np.sqrt(2)))

    def evaluate(self, t):
        return self._phi((t - self.mu) / self.sigma) / self.sigma

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return self._Phi((b - self.mu) / self.sigma) - self._Phi(
            (a - self.mu) / self.sigma
        )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return self.mu * (
            self._Phi((b - self.mu) / self.sigma)
            - self._Phi((a - self.mu) / self.sigma)
        ) - self.sigma * (
            self._phi((b - self.mu) / self.sigma)
            - self._phi((a - self.mu) / self.sigma)
        )

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (
            (self.mu ** 2 + self.sigma ** 2)
            * (
                self._Phi((b - self.mu) / self.sigma)
                - self._Phi((a - self.mu) / self.sigma)
            )
            - (
                self.sigma
                * (b + self.mu)
                * self._phi((b - self.mu) / self.sigma)
            )
            + (
                self.sigma
                * (a + self.mu)
                * self._phi((a - self.mu) / self.sigma)
            )
        )

    def integrate_t3_times_psi(self, a, b):
        """Compute integral int_a^b (t**3) * psi(t)."""
        u = (a - self.mu) / self.sigma
        v = (b - self.mu) / self.sigma
        c0 = self.mu**3
        c1 = 3 * self.sigma * self.mu**2
        c2 = 3 * self.sigma**2 * self.mu
        c3 = self.sigma**3
        return ((c0 + c2) * (self._Phi(v) - self._Phi(u)) -
                ((c1 + v*c2 + (2 + v**2)*c3) * self._phi(v) -
                 (c1 + u*c2 + (2 + u**2)*c3) * self._phi(u)))

    def integrate_t4_times_psi(self, a, b):
        """Compute integral int_a^b (t**4) * psi(t)."""
        u = (a - self.mu) / self.sigma
        v = (b - self.mu) / self.sigma
        c0 = self.mu**4
        c1 = 4 * self.sigma * self.mu**3
        c2 = 6 * self.sigma**2 * self.mu**2
        c3 = 4 * self.sigma**3 * self.mu
        c4 = self.sigma**4
        return ((c0 + c2 + 3*c4) * (self._Phi(v) - self._Phi(u)) -
                ((c1 + v*c2 + (2 + v**2)*c3 + (3*v + v**3)*c4) * self._phi(v) -
                 (c1 + u*c2 + (2 + u**2)*c3 + (3*u + u**3)*c4) * self._phi(u)))

    def integrate_t5_times_psi(self, a, b):
        """Compute integral int_a^b (t**5) * psi(t)."""
        u = (a - self.mu) / self.sigma
        v = (b - self.mu) / self.sigma
        c0 = self.mu**5
        c1 = 5 * self.sigma * self.mu**4
        c2 = 10 * self.sigma**2 * self.mu**3
        c3 = 10 * self.sigma**3 * self.mu**2
        c4 = 5 * self.sigma**4 * self.mu
        c5 = self.sigma**5
        return ((c0 + c2 + 3*c4) * (self._Phi(v) - self._Phi(u)) -
                ((c1 + v*c2 + (2 + v**2)*c3 + (3*v + v**3)*c4 +
                  (8 + 4*v**2 + v**4)*c5) * self._phi(v) -
                 (c1 + u*c2 + (2 + u**2)*c3 + (3*u + u**3)*c4 +
                  (8 + 4*u**2 + u**4)*c5) * self._phi(u)))

    def integrate_t6_times_psi(self, a, b):
        """Compute integral int_a^b (t**5) * psi(t)."""
        u = (a - self.mu) / self.sigma
        v = (b - self.mu) / self.sigma
        c0 = self.mu**6
        c1 = 6 * self.sigma * self.mu**5
        c2 = 15 * self.sigma**2 * self.mu**4
        c3 = 20 * self.sigma**3 * self.mu**3
        c4 = 15 * self.sigma**4 * self.mu**2
        c5 = 6 * self.sigma**5 * self.mu        
        c6 = self.sigma**6
        return ((c0 + c2 + 3*c4 + 15*c6) * (self._Phi(v) - self._Phi(u)) -
                ((c1 + v*c2 + (2 + v**2)*c3 + (3*v + v**3)*c4 +
                  (8 + 4*v**2 + v**4)*c5 + (15*v + 5*v**3 + v**5)*c6) *
                 self._phi(v) -
                 (c1 + u*c2 + (2 + u**2)*c3 + (3*u + u**3)*c4 +
                  (8 + 4*u**2 + u**4)*c5 + (15*u + 5*u**3 + u**5)*c6) *
                 self._phi(u)))

    def integrate_tk_times_psi_numeric(self, a, b, k):
        """Compute integral int_a^b (t**k) * psi(t) numerically."""
        # https://en.wikipedia.org/wiki/List_of_integrals_of_Gaussian_functions
        f = lambda t: self.evaluate(t) * t**k
        y, err = integrate.quad(f, a, b)
        return y
