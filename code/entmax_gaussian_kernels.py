from entmax_gaussian import (EntmaxGaussian1D, SparsemaxGaussian1D,
                             BiweightGaussian1D, TriweightGaussian1D)
from scipy import integrate
import numpy as np


class Uniform1DKernel(object):
    def __init__(self, mu=0, support_size=1):
        self._mu = mu
        self._a = support_size/2
        self._density_value = 1 / (2*self._a)

    def expectation_t(self):
        """Return E_{t~uniform}[t]."""
        return self._mu

    def expectation_t2(self):
        """Return E_{t~uniform}[t**2]."""
        return (1/3)*self._a**2 + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~uniform}[psi(t)]."""
        return (self._density_value *
                psi.integrate_psi(self._mu - self._a, self._mu + self._a))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~uniform}[t * psi(t)]."""
        return (self._density_value *
                psi.integrate_t_times_psi(
                    self._mu - self._a, self._mu + self._a))

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~uniform}[t**2 * psi(t)]."""
        return (self._density_value *
                psi.integrate_t2_times_psi(
                    self._mu - self._a, self._mu + self._a))


class EntmaxGaussian1DKernel(object):
    def __init__(self, alpha=2, mu=0, sigma_sq=None, support_size=None,
                 use_escort=True):
        self._alpha = alpha
        self._mu = mu
        self._entmax = EntmaxGaussian1D(alpha=alpha, mu=mu, sigma_sq=sigma_sq,
                                        support_size=support_size)
        self._sigma_sq = self._entmax._sigma_sq
        self._a = self._entmax.support_size() / 2
        self._escort = self._escort_kernel() if use_escort else None

    def _escort_kernel(self):
        """Return the (2-alpha)-escort distribution, which is a gamma-entmax
        with gamma = 1/(2-alpha)."""
        alpha_escort = 1/(2-self._alpha) if self._alpha != 2 else np.inf
        return EntmaxGaussian1DKernel(
            alpha=alpha_escort,
            mu=self._mu,
            support_size=self._entmax.support_size(),
            use_escort=False)

    def _escort_normalizer(self):
        """Compute the escort normalizer ||p||_beta^beta with beta = 2-alpha."""
        return (self._alpha - 1) * self._escort._unnormalized_mass(
            self._entmax._tau)

    def _unnormalized_mass(self, tau):
        """Integrate ((alpha-1)(-tau - f(t)))_+**(1/(alpha-1)) for
        arbitrary tau."""
        # Numeric integration.
        f = lambda t: (((self._alpha-1)*(-tau - t**2/(2*self._sigma_sq)))**
                       (1/(self._alpha-1)))
        mass, err = integrate.quad(f, -self._a, self._a)
        return mass

    def attention(self, psi):
        return self.expectation_psi(psi)

    def attention_gradient(self, psi):
        escort_normalizer = self._escort_normalizer()
        left = self._escort.expectation_t_times_psi(psi)
        right = self._escort.expectation_t() * self._escort.expectation_psi(psi)
        g1 = escort_normalizer * (left - right)
        left = self._escort.expectation_t2_times_psi(psi)
        right = (self._escort.expectation_t2() *
                 self._escort.expectation_psi(psi))
        g2 = escort_normalizer * (left - right)
        return g1, g2

    def expectation_t(self):
        """Return E_{t~entmax}[t]."""
        return self._mu

    def expectation_t2(self):
        """Return E_{t~entmax}[t**2]."""
        return self._entmax.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~entmax}[psi(t)]."""
        entmax_times_psi = lambda t: self._entmax.pdf(t) * psi.evaluate(t)
        y, err = integrate.quad(entmax_times_psi,
                                self._mu - self._a,
                                self._mu + self._a)
        return y

    def expectation_t_times_psi(self, psi):
        """Return E_{t~sparsemax}[t * psi(t)]."""
        entmax_t_times_psi = lambda t: self._entmax.pdf(t) * t * psi.evaluate(t)
        y, err = integrate.quad(entmax_t_times_psi,
                                self._mu - self._a,
                                self._mu + self._a)
        return y

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~sparsemax}[t**2 * psi(t)]."""
        entmax_t2_times_psi = (lambda t: self._entmax.pdf(t) * t**2 *
                               psi.evaluate(t))
        y, err = integrate.quad(entmax_t2_times_psi,
                                self._mu - self._a,
                                self._mu + self._a)
        return y


class SparsemaxGaussian1DKernel(EntmaxGaussian1DKernel):
    def __init__(self, mu=0, sigma_sq=None, support_size=None,
                 use_escort=True):
        super().__init__(alpha=2, mu=mu, sigma_sq=sigma_sq,
                         support_size=support_size, use_escort=False)
        self._sparsemax = SparsemaxGaussian1D(mu=mu, sigma_sq=sigma_sq,
                                              support_size=support_size)
        self._a = self._sparsemax.support_size() / 2
        self._escort = self._escort_kernel() if use_escort else None

    def _escort_kernel(self):
        """Return the (2-alpha)-escort distribution, which is a uniform
        with the same support."""
        return Uniform1DKernel(mu=self._mu, support_size=2*self._a)

    def _escort_normalizer(self):
        return 2 * self._a

    def _unnormalized_mass(self, tau):
        """Integrate ((alpha-1)(-tau - f(t)))_+**(1/(alpha-1)) for
        arbitrary tau."""
        return -2 * tau * self._a - self._a**3/(3*self._sigma_sq)

    def expectation_t2(self):
        """Return E_{t~sparsemax}[t**2]."""
        # TODO: This can be made general code by renaming self._sparsemax.
        return self._sparsemax.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~sparsemax}[psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c0 = -self._sparsemax._tau - self._mu**2 / (2*self._sigma_sq)
        c1 = self._mu / self._sigma_sq
        c2 = -1 / (2*self._sigma_sq)
        return (c0 * psi.integrate_psi(u, v) +
                c1 * psi.integrate_t_times_psi(u, v) +
                c2 * psi.integrate_t2_times_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~sparsemax}[t * psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c1 = -self._sparsemax._tau - self._mu**2 / (2*self._sigma_sq)
        c2 = self._mu / self._sigma_sq
        c3 = -1 / (2*self._sigma_sq)
        return (c1 * psi.integrate_t_times_psi(u, v) +
                c2 * psi.integrate_t2_times_psi(u, v) +
                c3 * psi.integrate_t3_times_psi(u, v))

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~sparsemax}[t**2 * psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c2 = -self._sparsemax._tau - self._mu**2 / (2*self._sigma_sq)
        c3 = self._mu / self._sigma_sq
        c4 = -1 / (2*self._sigma_sq)
        return (c2 * psi.integrate_t2_times_psi(u, v) +
                c3 * psi.integrate_t3_times_psi(u, v) +
                c4 * psi.integrate_t4_times_psi(u, v))


class BiweightGaussian1DKernel(EntmaxGaussian1DKernel):
    def __init__(self, mu=0, sigma_sq=None, support_size=None,
                 use_escort=True):
        super().__init__(alpha=1.5, mu=mu, sigma_sq=sigma_sq,
                         support_size=support_size, use_escort=False)
        self._biweight = BiweightGaussian1D(mu=mu, sigma_sq=sigma_sq,
                                            support_size=support_size)
        self._a = self._biweight.support_size() / 2
        self._escort = self._escort_kernel() if use_escort else None

    def _escort_kernel(self):
        """Return the (2-alpha)-escort distribution, which is a
        sparsemax Gaussian with the same support."""
        return SparsemaxGaussian1DKernel(
            mu=self._mu,
            support_size=self._biweight.support_size(),
            use_escort=False)

    def _escort_normalizer(self):
        # TODO: This can be made general code by renaming self._biweight.
        return (self._alpha - 1) * self._escort._unnormalized_mass(
            self._biweight._tau)

    def _unnormalized_mass(self, tau):
        """Integrate ((alpha-1)(-tau - f(t)))_+**(1/(alpha-1)) for
        arbitrary tau."""
        return (1/2 * tau**2 * self._a +
                (tau / (6 * self._sigma_sq)) * self._a ** 3 +
                (1 / (40 * self._sigma_sq**2)) * self._a ** 5)

    def expectation_t2(self):
        """Return E_{t~biweight}[t**2]."""
        # TODO: This can be made general code by renaming self._biweight.
        return self._biweight.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~biweight}[psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c = (self._alpha-1)**2
        tau = self._biweight._tau
        c0 = c * (tau * (tau + self._mu**2 / self._sigma_sq) +
                  self._mu**4 / (4*self._sigma_sq**2))
        c1 = -c * (tau * 2*self._mu / self._sigma_sq +
                   self._mu**3 / self._sigma_sq**2)
        c2 = c * (tau / self._sigma_sq + 3*self._mu**2 / (2*self._sigma_sq**2))
        c3 = -c * self._mu / self._sigma_sq**2
        c4 = c / (4*self._sigma_sq**2)
        return (c0 * psi.integrate_psi(u, v) +
                c1 * psi.integrate_t_times_psi(u, v) +
                c2 * psi.integrate_t2_times_psi(u, v) +
                c3 * psi.integrate_t3_times_psi(u, v) +
                c4 * psi.integrate_t4_times_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~biweight}[t * psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c = (self._alpha-1)**2
        tau = self._biweight._tau
        c1 = c * (tau * (tau + self._mu**2 / self._sigma_sq) +
                  self._mu**4 / (4*self._sigma_sq**2))
        c2 = -c * (tau * 2*self._mu / self._sigma_sq +
                   self._mu**3 / self._sigma_sq**2)
        c3 = c * (tau / self._sigma_sq + 3*self._mu**2 / (2*self._sigma_sq**2))
        c4 = -c * self._mu / self._sigma_sq**2
        c5 = c / (4*self._sigma_sq**2)
        return (c1 * psi.integrate_t_times_psi(u, v) +
                c2 * psi.integrate_t2_times_psi(u, v) +
                c3 * psi.integrate_t3_times_psi(u, v) +
                c4 * psi.integrate_t4_times_psi(u, v) +
                c5 * psi.integrate_t5_times_psi(u, v))

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~biweighted}[t**2 * psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c = (self._alpha-1)**2
        tau = self._biweight._tau
        c2 = c * (tau * (tau + self._mu**2 / self._sigma_sq) +
                  self._mu**4 / (4*self._sigma_sq**2))
        c3 = -c * (tau * 2*self._mu / self._sigma_sq +
                   self._mu**3 / self._sigma_sq**2)
        c4 = c * (tau / self._sigma_sq + 3*self._mu**2 / (2*self._sigma_sq**2))
        c5 = -c * self._mu / self._sigma_sq**2
        c6 = c / (4*self._sigma_sq**2)
        return (c2 * psi.integrate_t2_times_psi(u, v) +
                c3 * psi.integrate_t3_times_psi(u, v) +
                c4 * psi.integrate_t4_times_psi(u, v) +
                c5 * psi.integrate_t5_times_psi(u, v) +
                c6 * psi.integrate_t6_times_psi(u, v))


class TriweightGaussian1DKernel(EntmaxGaussian1DKernel):
    def __init__(self, mu=0, sigma_sq=None, support_size=None,
                 use_escort=True):
        super().__init__(alpha=4/3, mu=mu, sigma_sq=sigma_sq,
                         support_size=support_size, use_escort=False)
        self._triweight = TriweightGaussian1D(mu=mu, sigma_sq=sigma_sq,
                                              support_size=support_size)
        self._a = self._triweight.support_size() / 2
        self._escort = self._escort_kernel() if use_escort else None

    def _escort_kernel(self):
        """Return the (2-alpha)-escort distribution, which is a
        sparsemax Gaussian with the same support."""
        return BiweightGaussian1DKernel(
            mu=self._mu,
            support_size=self._triweight.support_size(),
            use_escort=False)

    def _escort_normalizer(self):
        # TODO: This can be made general code by renaming self._triweight.
        return (self._alpha - 1) * self._escort._unnormalized_mass(
            self._triweight._tau)

    def _unnormalized_mass(self, tau):
        """Integrate ((alpha-1)(-tau - f(t)))_+**(1/(alpha-1)) for
        arbitrary tau."""
        raise NotImplementedError

    def expectation_t2(self):
        """Return E_{t~biweight}[t**2]."""
        # TODO: This can be made general code by renaming self._biweight.
        return self._triweight.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~triweight}[psi(t)]."""
        u = self._mu - self._a
        v = self._mu + self._a
        c = -(self._alpha-1)**3
        tau = self._triweight._tau
        c0 = c * (tau**3 + 3/2 * tau**2 * self._mu**2 / self._sigma_sq +
                  3/4 * tau * self._mu**4 / self._sigma_sq**2 +
                  1/8 * self._mu**6 / self._sigma_sq**3)
        c1 = -c * (3 * tau**2 * self._mu / self._sigma_sq +
                   3 * tau * self._mu**3 / self._sigma_sq**2 +
                   3/4 * self._mu**5 / self._sigma_sq**3)
        c2 = c * (3/2 * tau**2 / self._sigma_sq +
                  9/2 * tau * self._mu**2 / self._sigma_sq**2 +
                  15/8 * self._mu**4 / self._sigma_sq**3)
        c3 = -c * (3 * tau * self._mu / self._sigma_sq**2 +
                   5/2 * self._mu**3 / self._sigma_sq**3)
        c4 = c * (3/4 * tau / self._sigma_sq**2 +
                  15/8 * self._mu**2 / self._sigma_sq**3)
        c5 = -c * (3/4 * self._mu / self._sigma_sq**3)
        c6 = c * (1/8 * 1 / self._sigma_sq**3)
        return (c0 * psi.integrate_psi(u, v) +
                c1 * psi.integrate_t_times_psi(u, v) +
                c2 * psi.integrate_t2_times_psi(u, v) +
                c3 * psi.integrate_t3_times_psi(u, v) +
                c4 * psi.integrate_t4_times_psi(u, v) +
                c5 * psi.integrate_t5_times_psi(u, v) +
                c6 * psi.integrate_t6_times_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~biweight}[t * psi(t)]."""
        raise NotImplementedError

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~biweighted}[t**2 * psi(t)]."""
        raise NotImplementedError
