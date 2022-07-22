import torch
import math


class BasisFunctions(object):
    def __init__(self):
        pass

    def __len__(self):
        """Number of basis functions."""
        raise NotImplementedError

    def to(self, device):
        """Move to device."""
        raise NotImplementedError

    def evaluate(self, t):
        raise NotImplementedError

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        raise NotImplementedError

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        raise NotImplementedError

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError

    def integrate_t3_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError

    def integrate_t4_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError

    def integrate_t5_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError

    def integrate_t6_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        raise NotImplementedError


class PowerBasisFunctions(BasisFunctions):
    """Function phi(t) = t**degree."""

    def __init__(self, degree):
        self.degree = degree.unsqueeze(0)

    def __len__(self):
        """Number of basis functions."""
        return self.degree.size(1)

    def to(self, device):
        self.degree = self.degree.to(device)
        return self

    def evaluate(self, t):
        return t ** self.degree

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        return (b ** (self.degree + 1) - a ** (self.degree + 1)) / (
            self.degree + 1
        )

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        return (b ** (self.degree + 2) - a ** (self.degree + 2)) / (
            self.degree + 2
        )

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        return (b ** (self.degree + 3) - a ** (self.degree + 3)) / (
            self.degree + 3
        )

    def __repr__(self):
        return f"PowerBasisFunction(degree={self.degree})"


class SineBasisFunctions(BasisFunctions):
    """Function phi(t) = sin(omega*t)."""

    def __init__(self, omega):
        self.omega = omega.unsqueeze(0)

    def __repr__(self):
        return f"SineBasisFunction(omega={self.omega})"

    def __len__(self):
        """Number of basis functions."""
        return self.omega.size(1)

    def to(self, device):
        self.omega = self.omega.to(device)
        return self

    def evaluate(self, t):
        return torch.sin(self.omega * t)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        # The antiderivative of sin(omega*t) is -cos(omega*t)/omega.
        return (
            -torch.cos(self.omega * b) + torch.cos(self.omega * a)
        ) / self.omega

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        # The antiderivative of t*sin(omega*t) is
        # (sin(omega*t) - omega*t*cos(omega*t)) / omega**2.
        return (
            torch.sin(self.omega * b)
            - self.omega * b * torch.cos(self.omega * b)
            - torch.sin(self.omega * a)
            + self.omega * a * torch.cos(self.omega * a)
        ) / (self.omega ** 2)

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        # The antiderivative of (t**2)*sin(omega*t) is
        # ((2-(t**2)*(omega**2))*cos(omega*t) + 2*omega*t*sin(omega*t)) / omega**3.  # noqa
        return (
            (2 - (b ** 2) * (self.omega ** 2)) * torch.cos(self.omega * b)
            + 2 * self.omega * b * torch.sin(self.omega * b)
            - (2 - (a ** 2) * (self.omega ** 2)) * torch.cos(self.omega * a)
            - 2 * self.omega * a * torch.sin(self.omega * a)
        ) / (self.omega ** 3)


class CosineBasisFunctions(BasisFunctions):
    """Function phi(t) = cos(omega*t)."""

    def __init__(self, omega):
        self.omega = omega.unsqueeze(0)

    def __repr__(self):
        return f"CosineBasisFunction(omega={self.omega})"

    def __len__(self):
        """Number of basis functions."""
        return self.omega.size(1)

    def to(self, device):
        self.omega = self.omega.to(device)
        return self

    def evaluate(self, t):
        return torch.cos(self.omega * t)

    def integrate_psi(self, a, b):
        """Compute integral int_a^b psi(t)."""
        # The antiderivative of cos(omega*t) is sin(omega*t)/omega.
        return (
            torch.sin(self.omega * b) - torch.sin(self.omega * a)
        ) / self.omega

    def integrate_t_times_psi(self, a, b):
        """Compute integral int_a^b t * psi(t)."""
        # The antiderivative of t*cos(omega*t) is
        # (cos(omega*t) + omega*t*sin(omega*t)) / omega**2.
        return (
            torch.cos(self.omega * b)
            + self.omega * b * torch.sin(self.omega * b)
            - torch.cos(self.omega * a)
            - self.omega * a * torch.sin(self.omega * a)
        ) / (self.omega ** 2)

    def integrate_t2_times_psi(self, a, b):
        """Compute integral int_a^b (t**2) * psi(t)."""
        # The antiderivative of (t**2)*cos(omega*t) is
        # (((t**2)*(omega**2)-2)*cos(omega*t) + 2*omega*t*sin(omega*t)) / omega**3.  # noqa
        return (
            ((b ** 2) * (self.omega ** 2) - 2) * torch.sin(self.omega * b)
            + 2 * self.omega * b * torch.cos(self.omega * b)
            - ((a ** 2) * (self.omega ** 2) - 2) * torch.sin(self.omega * a)
            - 2 * self.omega * a * torch.cos(self.omega * a)
        ) / (self.omega ** 3)


class GaussianBasisFunctions(BasisFunctions):
    """Function phi(t) = Gaussian(t; mu, sigma_sq)."""

    def __init__(self, mu, sigma):
        self.mu = mu.unsqueeze(0)
        self.sigma = sigma.unsqueeze(0)

    def __repr__(self):
        return f"GaussianBasisFunction(mu={self.mu}, sigma={self.sigma})"

    def __len__(self):
        """Number of basis functions."""
        return self.mu.size(1)

    def to(self, device):
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def _phi(self, t):
        return 1.0 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * t ** 2)

    def _Phi(self, t):
        return 0.5 * (1 + torch.erf(t / math.sqrt(2)))

    def _integrate_product_of_gaussians(self, mu, sigma_sq):
        sigma = torch.sqrt(self.sigma ** 2 + sigma_sq)
        return self._phi((mu - self.mu) / sigma) / sigma

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

    def integrate_t2_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t**2 * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma ** 2) / (
            self.sigma ** 2 + sigma_sq
        )
        sigma_sq_tilde = ((self.sigma ** 2) * sigma_sq) / (
            self.sigma ** 2 + sigma_sq
        )
        return S_tilde * (mu_tilde ** 2 + sigma_sq_tilde)

    def integrate_t_times_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * t * psi(t)."""
        S_tilde = self._integrate_product_of_gaussians(mu, sigma_sq)
        mu_tilde = (self.mu * sigma_sq + mu * self.sigma ** 2) / (
            self.sigma ** 2 + sigma_sq
        )
        return S_tilde * mu_tilde

    def integrate_psi_gaussian(self, mu, sigma_sq):
        """Compute integral int N(t; mu, sigma_sq) * psi(t)."""
        return self._integrate_product_of_gaussians(mu, sigma_sq)
