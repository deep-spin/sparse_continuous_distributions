"""pytorch implementation of beta-Gaussians and reparametrized sampling"""

import numpy as np
from scipy.special import gamma

import math
import numbers

import torch
from torch.distributions import constraints
from torch.distributions import Beta
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property


LOG_2 = np.log(2)
LOG_PI = np.log(np.pi)


class _RealMatrix(constraints._Real):
    event_dim = 2


# reimplement scipy's cutoff for eigenvalues
def _eigvalsh_to_eps(spectra, cond=None, rcond=None):
    spectra = spectra.detach()
    if rcond is not None:
        cond = rcond
    if cond in [None, -1]:
        t = str(spectra.dtype)[-2:]
        factor = {'32': 1E3, '64': 1E6}
        cond = factor[t] * torch.finfo(spectra.dtype).eps

    return cond * torch.max(torch.abs(spectra), dim=-1).values


class FactorizedScale(object):
    def __init__(self, scales, cond=None, rcond=None, upper=True):
        """Factorized representation of a batch of scale PSD matrices."""

        self._zero = scales.new_zeros(size=(1,))

        # scales: [B x D x D]

        # s, u = torch.symeig(scales, eigenvectors=True, upper=True)
        s, u = torch.linalg.eigh(scales)

        eps = _eigvalsh_to_eps(s, cond, rcond)

        if torch.any(torch.min(s, dim=-1).values < -eps):
            raise ValueError('scale is not positive semidefinite')

        # probably could use searchsorted
        self.s_mask = s > eps.unsqueeze(dim=-1)

        self.u = u
        self.s = torch.where(self.s_mask, s, self._zero)
        self.s_inv = torch.where(self.s_mask, 1 / s, self._zero)

    @lazy_property
    def rank(self):
        return self.s_mask.sum(dim=-1)

    @lazy_property
    def trace(self):
        return self.s.sum(dim=-1)

    @lazy_property
    def trace_inv(self):
        return self.s_inv.sum(dim=-1)

    @lazy_property
    def log_det(self):
        log_s = torch.where(self.s_mask, torch.log(self.s), self._zero)
        return torch.sum(log_s, dim=-1)

    @lazy_property
    def log_det_inv(self):
        log_s_inv = torch.where(self.s_mask, torch.log(self.s_inv), self._zero)
        return torch.sum(log_s_inv, dim=-1)

    @lazy_property
    def L(self):
        return self.u @ torch.diag_embed(torch.sqrt(self.s))

    @lazy_property
    def L_inv(self):
        return self.u @ torch.diag_embed(torch.sqrt(self.s_inv))


class DiagScale(FactorizedScale):
    def __init__(self, scales, cond=None, rcond=None, upper=True):
        """Compact representation of a batch of diagonal scale matrices."""

        self._zero = scales.new_zeros(size=(1,))

        eps = _eigvalsh_to_eps(scales, cond, rcond)

        if torch.any(torch.min(scales, dim=-1).values < -eps):
            raise ValueError('scale is not positive semidefinite')

        # probably could use searchsorted
        self.s_mask = scales > eps.unsqueeze(dim=-1)

        self.u = torch.eye(scales.shape[-1])
        self.s = torch.where(self.s_mask, scales, self._zero)
        self.s_inv = torch.where(self.s_mask, 1 / scales, self._zero)


class MultivariateBetaGaussian(Distribution):
    arg_constraints = {'loc': constraints.real_vector,
                       'scale': _RealMatrix(),
                       'alpha': constraints.greater_than(1)}
    support = constraints.real_vector
    has_rsample = True

    def __init__(self, loc, scale=None, alpha=2, validate_args=None):
        """Batched multivariate beta-Gaussian random variable.

        The r.v. is parametrized in terms of a location (mean), scale
        (proportional to covariance) matrix, and scalar alpha>1.

        The pdf takes the form

        p(x) = [(alpha - 1) * -.5 (x-u)' inv(Sigma) (x-u) - tau]_+ ** (alpha - 1)

        where (u, Sigma) are the location and scale parameters.

        Parameters
        ---------- loc: tensor, shape (broadcastable to) (*batch_dims, D)
            mean of the the distribution.

        scale: tensor, shape (broadcastable to) (*batch_dims, D, D)
            scale parameter Sigma of the distribution.

        alpha: scalar or tensor broadcastable to (*batch_dims)
            The exponent parameter of the distribution.
            For alpha -> 1, the distribution converges to a Gaussian.
            For alpha = 2, the distribution is a Truncated Paraboloid
                (n-d generalization of the Epanechnikov kernel.)
            For alpha -> infty, the distribution converges to a
            uniform on an ellipsoid.
        """

        if isinstance(alpha, numbers.Number):
            alpha = loc.new_tensor(alpha)

        # dimensions must be compatible to:
        # mean: [B x D]
        # scale: [B x D x D]
        # alpha: [B x 1]

        batch_shape = torch.broadcast_shapes(scale.shape[:-2],
                                             loc.shape[:-1],
                                             alpha.shape)

        event_shape = loc.shape[-1:]

        loc = loc.expand(batch_shape + (-1,))
        scale = scale.expand(batch_shape + (-1, -1))
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.scale = scale
        self.alpha = alpha

        self._fact_scale = FactorizedScale(scale)

        super().__init__(batch_shape, event_shape, validate_args)

    def _naive_radius(self):
        assert len(self._batch_shape) == 1

        for i in range(self._batch_shape[0]):

            a = self.alpha[i].item()
            a_m1 = a - 1
            a_ratio = a / a_m1

            n = self._fact_scale.rank[i].item()

            print(((gamma(n / 2 + a_ratio) / (gamma(a_ratio) * np.pi ** (n / 2))) *
                   (2 / a_m1) ** (1 / a_m1)) ** (a_m1 / (2 + a_m1 * n)))

    @lazy_property
    def log_radius(self):
        """Logarithm of the max-radius R of the distribution."""

        alpha = self.alpha
        alpha_m1 = alpha - 1
        alpha_ratio = alpha / alpha_m1

        n = self._fact_scale.rank
        half_n = n / 2

        lg_n_a = torch.lgamma(half_n + alpha_ratio)
        lg_a = torch.lgamma(alpha_ratio)

        log_first = lg_n_a - lg_a - half_n * LOG_PI
        log_second = (LOG_2 - torch.log(alpha_m1)) / alpha_m1
        log_inner = log_first + log_second
        log_radius = (alpha_m1 / (2 + alpha_m1 * n)) * log_inner
        return log_radius

    @lazy_property
    def _tau(self):
        n = self._fact_scale.rank
        c = n + (2 / (self.alpha - 1))
        scaled_log_det = self._fact_scale.log_det / c
        return -torch.exp(2 * self.log_radius - LOG_2 - scaled_log_det)

    @lazy_property
    def tsallis_entropy(self):
        """The Tsallis entropy -Omega_alpha of the distribution"""
        n = self._fact_scale.rank
        alpha_m1 = self.alpha - 1
        alpha_term = 1 / (self.alpha * alpha_m1)
        denom = 2 * self.alpha + n * alpha_m1
        tau_term = 2 * self._tau / denom
        return alpha_term + tau_term

    def _mahalanobis(self, x, broadcast_batch=False):

        # x: shape [B', D] -- possibly different B', right?
        # loc: shape [B, D].

        # 1. Mahalanobis term

        d = x.shape[-1]
        loc_batch_shape = self.batch_shape

        if broadcast_batch:
            # assume loc is [B, D] and manually insert ones
            # to make it [B, 1,...1, D]

            x_batch_shape = x.shape[:-1]
            x = x.reshape(x_batch_shape
                          + tuple(1 for _ in self.batch_shape)
                          + (d,))

        # these must be broadcastable to each other and end in d.

        # [B', B, D]
        diff = x - self.loc

        # right now with B=[], now, this yields [B', D]

        Li = self._fact_scale.L_inv
        diff = diff.unsqueeze(dim=-1)
        diff_scaled = (Li.transpose(-2, -1) @ diff).squeeze(dim=-1)
        maha = diff_scaled.square().sum(dim=-1) / 2
        return maha

    def pdf(self, x, broadcast_batch=False):
        """Probability of an broadcastable observation x (could be zero)"""
        f = -self._tau - self._mahalanobis(x, broadcast_batch)
        return torch.clip((self.alpha - 1) * f, min=0) ** (1 / (self.alpha - 1))

    def log_prob(self, x, broadcast_batch=False):
        """Log-probability of an broadcastable observation x (could be -inf)"""
        return torch.log(self.pdf(x, broadcast_batch))

    def cross_fy(self, x, broadcast_batch=False):
        """The cross-Omega Fenchel-Young loss w.r.t. a Dirac observation x"""
        n_a_m1 = self._fact_scale.rank * (self.alpha - 1)

        c = torch.log(n_a_m1) - torch.log(2 * self.alpha + n_a_m1) - LOG_2

        return (self._mahalanobis(x, broadcast_batch)
                + self.tsallis_entropy
                + torch.exp(2 * self.log_radius + c))

    def rsample(self, sample_shape):
        """Draw samples from the distribution."""
        shape = self._extended_shape(sample_shape)
        # print(shape) if called with (5,) gives (5,2,3)

        radius = torch.exp(self.log_radius)
        radius = radius.expand(sample_shape + radius.shape)

        mask = self._fact_scale.s_mask.expand(shape)

        # project U onto the correct sphere)
        U = torch.randn(shape)
        U = torch.where(mask, U, U.new_zeros(1))
        norm = U.norm(dim=-1).unsqueeze(dim=-1)
        U /= norm

        n = self._fact_scale.rank
        half_n = n / 2
        alpha_m1 = self.alpha - 1
        alpha_ratio = self.alpha / alpha_m1

        ratio_dist = Beta(half_n, alpha_ratio).expand(shape[:-1])
        ratio = ratio_dist.rsample()
        r = radius * torch.sqrt(ratio)

        Z = r.unsqueeze(dim=-1) * U
        Z = Z.unsqueeze(dim=-1)

        L = self._fact_scale.L

        # z @ Lt = (L @ Zt).t

        LZ = (L @ Z).squeeze(dim=-1)

        c = torch.exp(-self._fact_scale.log_det / (2 * n + 4 / alpha_m1))
        c = c.expand(sample_shape + c.shape).unsqueeze(-1)

        return self.loc + c * LZ


class MultivariateBetaGaussianDiag(MultivariateBetaGaussian):
    arg_constraints = {'loc': constraints.real_vector,
                       'scale': constraints.greater_than(0),
                       'alpha': constraints.greater_than(1)}

    def __init__(self, loc, scale=None, alpha=2, validate_args=None):
        """Batched multivariate beta-Gaussian random variable w/ diagonal scale.

        The r.v. is parametrized in terms of a location (mean), diagonal scale
        (proportional to covariance) matrix, and scalar alpha>1.

        The pdf takes the form

        p(x) = [(alpha - 1) * -.5 (x-u)' inv(Sigma) (x-u) - tau]_+ ** (alpha - 1)

        where (u, Sigma) are the location and scale parameters.

        Parameters
        ----------
        loc: tensor, shape (broadcastable to) (*batch_dims, D)
            mean of the the distribution.

        scale: tensor, shape (broadcastable to) (*batch_dims, D)
            diagonal of the scale parameter Sigma.

        alpha: scalar or tensor broadcastable to (*batch_dims)
            The exponent parameter of the distribution.
            For alpha -> 1, the distribution converges to a Gaussian.
            For alpha = 2, the distribution is a Truncated Paraboloid
                (n-d generalization of the Epanechnikov kernel.)
            For alpha -> infty, the distribution converges to a
            uniform on an ellipsoid.
        """

        if isinstance(alpha, numbers.Number):
            alpha = loc.new_tensor(alpha)

        batch_shape = torch.broadcast_shapes(scale.shape[:-1],
                                             loc.shape[:-1],
                                             alpha.shape)

        event_shape = loc.shape[-1:]

        loc = loc.expand(batch_shape + (-1,))
        scale = scale.expand(batch_shape + (-1,))
        alpha = alpha.expand(batch_shape)

        self.loc = loc
        self.scale = scale
        self.alpha = alpha

        self._fact_scale = DiagScale(scale)

        Distribution.__init__(self, batch_shape, event_shape, validate_args)
