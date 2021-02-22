from batch_entmax_gaussian import (EntmaxGaussian1D, SparsemaxGaussian1D,
                                   BiweightGaussian1D, TriweightGaussian1D)
import torch
import math


def _batch_linspace(start, end, steps):
    """Linspace with a batch dimension.
    start and end are 1D tensors, steps is a scalar.
    Returns a 2D tensor dim(steps) x dim(start)."""
    t = torch.linspace(0, 1, steps).unsqueeze(1)
    return (1-t)*start.unsqueeze(0) + t*end.unsqueeze(0)


class Uniform1DKernel(object):
    def __init__(self, mu, support_size):
        self._mu = mu
        self._a = support_size/2
        self._density_value = 1 / (2*self._a)

    def expectation_t(self):
        """Return E_{t~uniform}[t]."""
        # dim(mu)
        return self._mu

    def expectation_t2(self):
        """Return E_{t~uniform}[t**2]."""
        # dim(mu)
        return (1/3)*self._a**2 + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~uniform}[psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        return (self._density_value.unsqueeze(1) * psi.integrate_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~uniform}[t * psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        return (self._density_value.unsqueeze(1) *
                psi.integrate_t_times_psi(u, v))

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~uniform}[t**2 * psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        return (self._density_value.unsqueeze(1) *
                psi.integrate_t2_times_psi(u, v))


class EntmaxGaussian1DKernel(object):
    def __init__(self, alpha, mu, sigma_sq=None, support_size=None,
                 use_escort=True):
        self._alpha = alpha
        self._mu = mu
        self._entmax = EntmaxGaussian1D(alpha=alpha, mu=mu, sigma_sq=sigma_sq,
                                        support_size=support_size)
        self._sigma_sq = self._entmax._sigma_sq
        self._a = self._entmax.support_size() / 2
        self._escort = self._escort_kernel() if use_escort else None
        self._num_samples = 1000  # Number of samples for numeric integrals.

    def _escort_kernel(self):
        """Return the (2-alpha)-escort distribution, which is a gamma-entmax
        with gamma = 1/(2-alpha)."""
        alpha_escort = 1/(2-self._alpha) if self._alpha != 2 else math.inf
        return EntmaxGaussian1DKernel(
            alpha=alpha_escort,
            mu=self._mu,
            support_size=self._entmax.support_size(),
            use_escort=False)
    
    def _escort_normalizer(self):
        """Compute the escort normalizer ||p||_beta^beta with beta = 2-alpha."""
        alpha_escort = self._escort._alpha
        # M x dim(mu)
        t = _batch_linspace(-self._a, self._a, self._num_samples)
        # M x dim(mu)
        sigma_sq = self._sigma_sq.unsqueeze(0)
        tau = self._entmax._tau.unsqueeze(0)
        # M x dim(mu)
        f = ((self._alpha-1)*(-tau - t**2/(2*sigma_sq)))**(1/(alpha_escort-1))
        mass = torch.trapz(f, t, dim=0)
        return mass

    def mean(self):
        return self._entmax.mean()
    
    def variance(self):
        return self._entmax.variance()

    def sigma_sq_from_support_size(self, support_size):
        return self._entmax.sigma_sq_from_a(support_size/2)

    def sigma_sq_from_variance(self, variance):
        return self._entmax._sigma_sq_from_variance(variance)

    def attention(self, psi):
        return self.expectation_psi(psi)  # dim(mu) x dim(psi)

    def attention_gradient(self, psi):
        escort_normalizer = self._escort_normalizer()  # dim(mu) 
        left = self._escort.expectation_t_times_psi(psi)  # dim(mu) x dim(psi)
        # (dim(mu)) * (dim(mu) x dim(psi)) -> dim(mu) x dim(psi)
        right = (self._escort.expectation_t().unsqueeze(1) *
                 self._escort.expectation_psi(psi))
        # (dim(mu)) * (dim(mu) x dim(psi)) -> dim(mu) x dim(psi)
        g1 = escort_normalizer.unsqueeze(1) * (left - right)
        left = self._escort.expectation_t2_times_psi(psi)  # dim(mu) x dim(psi)
        # (dim(mu)) * (dim(mu) x dim(psi)) -> dim(mu) x dim(psi)
        right = (self._escort.expectation_t2().unsqueeze(1) *
                 self._escort.expectation_psi(psi))
        # (dim(mu)) * (dim(mu) x dim(psi)) -> dim(mu) x dim(psi)
        g2 = escort_normalizer.unsqueeze(1) * (left - right)
        return g1, g2

    def expectation_t(self):
        """Return E_{t~entmax}[t]."""
        return self._mu

    def expectation_t2(self):
        """Return E_{t~entmax}[t**2]."""
        return self._entmax.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~entmax}[psi(t)].
        The output is a tensor with dimension dim(mu) x dim(psi)."""
        # M x dim(mu)
        t = _batch_linspace(self._mu - self._a, self._mu + self._a,
                            self._num_samples)
        # M x dim(mu)
        kernel_values = self._entmax.pdf(t)
        # M x dim(mu) x dim(psi)
        values = psi.evaluate(t.view(-1, 1)).view(t.shape[0], t.shape[1], -1)
        # M x dim(mu) x dim(psi)
        values = kernel_values.unsqueeze(2) * values
        # dim(mu) x dim(psi)
        total = torch.trapz(values, t.unsqueeze(-1), dim=0)
        return total

    def expectation_t_times_psi(self, psi):
        """Return E_{t~sparsemax}[t * psi(t)].
        The output is a tensor with dimension dim(mu) x dim(psi)."""
        # M x dim(mu)
        t = _batch_linspace(self._mu - self._a, self._mu + self._a,
                            self._num_samples)
        # M x dim(mu)
        kernel_values = t * self._entmax.pdf(t)
        # M x dim(mu) x dim(psi)
        values = psi.evaluate(t.view(-1, 1)).view(t.shape[0], t.shape[1], -1)
        # M x dim(mu) x dim(psi)
        values = kernel_values.unsqueeze(2) * values
        # dim(mu) x dim(psi)
        total = torch.trapz(values, t.unsqueeze(-1), dim=0)
        return total

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~sparsemax}[t**2 * psi(t)].
        The output is a tensor with dimension dim(mu) x dim(psi)."""
        # M x dim(mu)
        t = _batch_linspace(self._mu - self._a, self._mu + self._a,
                            self._num_samples)
        # M x dim(mu)
        kernel_values = t**2 * self._entmax.pdf(t)
        # M x dim(mu) x dim(psi)
        values = psi.evaluate(t.view(-1, 1)).view(t.shape[0], t.shape[1], -1)
        # M x dim(mu) x dim(psi)
        values = kernel_values.unsqueeze(2) * values
        # dim(mu) x dim(psi)
        total = torch.trapz(values, t.unsqueeze(-1), dim=0)
        return total


class SparsemaxGaussian1DKernel(EntmaxGaussian1DKernel):
    def __init__(self, mu, sigma_sq=None, support_size=None,
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

    def sigma_sq_from_support_size(self, support_size):
        # TODO: This can be made general code by renaming self._sparsemax.
        return self._sparsemax.sigma_sq_from_a(support_size/2)

    def sigma_sq_from_variance(self, variance):
        # TODO: This can be made general code by renaming self._sparsemax.
        return self._sparsemax._sigma_sq_from_variance(variance)

    def expectation_t2(self):
        """Return E_{t~sparsemax}[t**2]."""
        # TODO: This can be made general code by renaming self._sparsemax.
        return self._sparsemax.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~sparsemax}[psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        # c0, c1, c2 are dim(mu)
        c0 = -self._sparsemax._tau - self._mu**2 / (2*self._sigma_sq)
        c1 = self._mu / self._sigma_sq
        c2 = -1 / (2*self._sigma_sq)
        return (c0.unsqueeze(1) * psi.integrate_psi(u, v) +
                c1.unsqueeze(1) * psi.integrate_t_times_psi(u, v) +
                c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~sparsemax}[t * psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        # c1, c2, c3 are dim(mu)
        c1 = -self._sparsemax._tau - self._mu**2 / (2*self._sigma_sq)
        c2 = self._mu / self._sigma_sq
        c3 = -1 / (2*self._sigma_sq)
        return (c1.unsqueeze(1) * psi.integrate_t_times_psi(u, v) +
                c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v) +
                c3.unsqueeze(1) * psi.integrate_t3_times_psi(u, v))

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~sparsemax}[t**2 * psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        # c2, c3, c4 are dim(mu)
        c2 = -self._sparsemax._tau - self._mu**2 / (2*self._sigma_sq)
        c3 = self._mu / self._sigma_sq
        c4 = -1 / (2*self._sigma_sq)
        return (c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v) +
                c3.unsqueeze(1) * psi.integrate_t3_times_psi(u, v) +
                c4.unsqueeze(1) * psi.integrate_t4_times_psi(u, v))


class BiweightGaussian1DKernel(EntmaxGaussian1DKernel):
    def __init__(self, mu, sigma_sq=None, support_size=None,
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
        #return (self._alpha - 1) * self._escort._unnormalized_mass(
        #    self._biweight._tau)
        return -self._biweight._tau * self._a - self._a**3/(6*self._sigma_sq)

    def sigma_sq_from_support_size(self, support_size):
        # TODO: This can be made general code by renaming self._biweight.
        return self._biweight.sigma_sq_from_a(support_size/2)

    def sigma_sq_from_variance(self, variance):
        # TODO: This can be made general code by renaming self._biweight.
        return self._biweight._sigma_sq_from_variance(variance)

    def expectation_t2(self):
        """Return E_{t~biweight}[t**2]."""
        # TODO: This can be made general code by renaming self._biweight.
        return self._biweight.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~biweight}[psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        # c0, c1, c2, c3, c4 are dim(mu)
        c = (self._alpha-1)**2
        tau = self._biweight._tau
        c0 = c * (tau * (tau + self._mu**2 / self._sigma_sq) +
                  self._mu**4 / (4*self._sigma_sq**2))
        c1 = -c * (tau * 2*self._mu / self._sigma_sq +
                   self._mu**3 / self._sigma_sq**2)
        c2 = c * (tau / self._sigma_sq + 3*self._mu**2 / (2*self._sigma_sq**2))
        c3 = -c * self._mu / self._sigma_sq**2
        c4 = c / (4*self._sigma_sq**2)
        return (c0.unsqueeze(1) * psi.integrate_psi(u, v) +
                c1.unsqueeze(1) * psi.integrate_t_times_psi(u, v) +
                c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v) +
                c3.unsqueeze(1) * psi.integrate_t3_times_psi(u, v) +
                c4.unsqueeze(1) * psi.integrate_t4_times_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~biweight}[t * psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        c = (self._alpha-1)**2
        tau = self._biweight._tau
        # c1, c2, c3, c4, c5 are dim(mu)
        c1 = c * (tau * (tau + self._mu**2 / self._sigma_sq) +
                  self._mu**4 / (4*self._sigma_sq**2))
        c2 = -c * (tau * 2*self._mu / self._sigma_sq +
                   self._mu**3 / self._sigma_sq**2)
        c3 = c * (tau / self._sigma_sq + 3*self._mu**2 / (2*self._sigma_sq**2))
        c4 = -c * self._mu / self._sigma_sq**2
        c5 = c / (4*self._sigma_sq**2)
        return (c1.unsqueeze(1) * psi.integrate_t_times_psi(u, v) +
                c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v) +
                c3.unsqueeze(1) * psi.integrate_t3_times_psi(u, v) +
                c4.unsqueeze(1) * psi.integrate_t4_times_psi(u, v) +
                c5.unsqueeze(1) * psi.integrate_t5_times_psi(u, v))

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~biweighted}[t**2 * psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        c = (self._alpha-1)**2
        tau = self._biweight._tau
        # c2, c3, c4, c5, c6 are dim(mu)
        c2 = c * (tau * (tau + self._mu**2 / self._sigma_sq) +
                  self._mu**4 / (4*self._sigma_sq**2))
        c3 = -c * (tau * 2*self._mu / self._sigma_sq +
                   self._mu**3 / self._sigma_sq**2)
        c4 = c * (tau / self._sigma_sq + 3*self._mu**2 / (2*self._sigma_sq**2))
        c5 = -c * self._mu / self._sigma_sq**2
        c6 = c / (4*self._sigma_sq**2)
        return (c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v) +
                c3.unsqueeze(1) * psi.integrate_t3_times_psi(u, v) +
                c4.unsqueeze(1) * psi.integrate_t4_times_psi(u, v) +
                c5.unsqueeze(1) * psi.integrate_t5_times_psi(u, v) +
                c6.unsqueeze(1) * psi.integrate_t6_times_psi(u, v))


class TriweightGaussian1DKernel(EntmaxGaussian1DKernel):
    def __init__(self, mu, sigma_sq=None, support_size=None,
                 use_escort=True):
        super().__init__(alpha=4/3, mu=mu, sigma_sq=sigma_sq,
                         support_size=support_size, use_escort=False)
        self._triweight = TriweightGaussian1D(mu=mu, sigma_sq=sigma_sq,
                                              support_size=support_size)
        self._a = self._triweight.support_size() / 2
        self._escort = self._escort_kernel() if use_escort else None

    def _escort_kernel(self):
        """Return the (2-alpha)-escort distribution, which is a
        biweight Gaussian with the same support."""
        return BiweightGaussian1DKernel(
            mu=self._mu,
            support_size=self._triweight.support_size(),
            use_escort=False)

    def _escort_normalizer(self):
        # TODO: This can be made general code by renaming self._triweight.
        #return (self._alpha - 1) * self._escort._unnormalized_mass(
        #    self._triweight._tau)
        tau = self._triweight._tau
        return 1/9 * (2 * tau**2 * self._a +
                      (2 * tau / (3 * self._sigma_sq)) * self._a ** 3 +
                      (1 / (10 * self._sigma_sq**2)) * self._a ** 5)

    def sigma_sq_from_support_size(self, support_size):
        # TODO: This can be made general code by renaming self._triweight.
        return self._triweight.sigma_sq_from_a(support_size/2)

    def sigma_sq_from_variance(self, variance):
        # TODO: This can be made general code by renaming self._triweight.
        return self._triweight._sigma_sq_from_variance(variance)

    def expectation_t2(self):
        """Return E_{t~biweight}[t**2]."""
        # TODO: This can be made general code by renaming self._biweight.
        return self._triweight.variance() + self._mu**2

    def expectation_psi(self, psi):
        """Return E_{t~triweight}[psi(t)]."""
        # dim(mu) x dim(psi)
        u = (self._mu - self._a).unsqueeze(1)
        v = (self._mu + self._a).unsqueeze(1)
        c = -(self._alpha-1)**3
        tau = self._triweight._tau
        # c0, c1, c2, c3, c4, c5, c6 are dim(mu)
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
        return (c0.unsqueeze(1) * psi.integrate_psi(u, v) +
                c1.unsqueeze(1) * psi.integrate_t_times_psi(u, v) +
                c2.unsqueeze(1) * psi.integrate_t2_times_psi(u, v) +
                c3.unsqueeze(1) * psi.integrate_t3_times_psi(u, v) +
                c4.unsqueeze(1) * psi.integrate_t4_times_psi(u, v) +
                c5.unsqueeze(1) * psi.integrate_t5_times_psi(u, v) +
                c6.unsqueeze(1) * psi.integrate_t6_times_psi(u, v))

    def expectation_t_times_psi(self, psi):
        """Return E_{t~biweight}[t * psi(t)]."""
        raise NotImplementedError

    def expectation_t2_times_psi(self, psi):
        """Return E_{t~biweighted}[t**2 * psi(t)]."""
        raise NotImplementedError
