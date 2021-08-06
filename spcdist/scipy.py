"""scipy implementation of beta-Gaussians (compatible to scipy.stats)"""

from scipy.special import gamma
from scipy.linalg import sqrtm
import numpy as np

from scipy.stats._multivariate import multi_rv_generic, multi_rv_frozen
from scipy.stats._multivariate import multivariate_normal

from .utils import _PSD, _process_parameters


_LOG_2PI = np.log(2 * np.pi)


def _radius(n, alpha):
    """Return radius R for a given dimension n and alpha."""

    if alpha == 1:
        return np.inf

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
    return scale


class multivariate_beta_gaussian_gen(multi_rv_generic):
    r"""A multivariate beta-Gaussian random variable.

    The `mean` keyword specifies the mean.
    The `scale` keyword specifies the Sigma matrix (uniquely defines the
    covariance).

    Currently does not support $alpha<1$.

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
        a_m1 = alpha - 1
        pdf = np.maximum(a_m1 * logpdf, 0) ** (1 / a_m1)
        return pdf

    def pdf(self, x, mean=None, scale=1, alpha=2, allow_singular=False):
        if alpha == 1:
            return multivariate_normal(mean, scale).pdf(x)
        dim, mean, scale, alpha = self._process_parameters(None, mean, scale, alpha)
        psd = _PSD(scale, allow_singular=allow_singular)
        radius = _radius(psd.rank, alpha)
        return self._pdf(x, mean, psd.U, psd.log_pdet, psd.rank, alpha, radius)

    def _rvs(self, mean, scale_sqrt, rank, log_det, alpha, radius, size, random_state):

        a_m1 = alpha - 1

        # Sample uniformly from sphere.
        if np.isscalar(size):
            size = (size,)

        u = random_state.standard_normal((size + (rank,)))
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
        if self.alpha == 1:
            return multivariate_normal(self.mean, self.scale).pdf(x)
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
