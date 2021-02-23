from batch_entmax_gaussian_kernels import (EntmaxGaussian1DKernel,
                                           Gaussian1DKernel,
                                           SparsemaxGaussian1DKernel,
                                           BiweightGaussian1DKernel,
                                           TriweightGaussian1DKernel)
import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)


class ContinuousEntmaxFunction(torch.autograd.Function):
    @classmethod
    def _attention(cls, ctx):
        kernel = ctx.kernel
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        r = torch.zeros((kernel._mu.shape[0], total_basis),
                        dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            r[:, start:offsets[j]] = kernel.attention(basis_functions)
            start = offsets[j]
        return r

    @classmethod
    def _jacobian(cls, ctx):
        kernel = ctx.kernel
        num_basis = [len(basis_functions) for basis_functions in ctx.psi]
        total_basis = sum(num_basis)
        J = torch.zeros((kernel._mu.shape[0], 2, total_basis),
                        dtype=ctx.dtype, device=ctx.device)
        offsets = torch.cumsum(torch.IntTensor(num_basis).to(ctx.device), dim=0)
        start = 0
        for j, basis_functions in enumerate(ctx.psi):
            g1, g2 = kernel.attention_gradient(basis_functions)
            J[:, 0, start:offsets[j]] = g1
            J[:, 1, start:offsets[j]] = g2
            start = offsets[j]
        return J

    @classmethod
    def forward(cls, ctx, theta, psi, kernel):
        # We assume a truncated parabola.
        # We have:
        # theta = [mu/sigma**2, -1/(2*sigma**2)],
        # phi(t) = [t, t**2],
        # p(t) = [theta.dot(phi(t)) - A]_+,
        # supported on [mu - a, mu + a].
        ctx.dtype = theta.dtype
        ctx.device = theta.device
        ctx.psi = psi
        ctx.kernel = kernel
        sigma = torch.sqrt(-0.5 / theta[:, 1])
        mu = theta[:, 0] * sigma ** 2
        kernel.set_parameters(mu=mu, sigma_sq=sigma**2)

        # r is dim(mu) x dim(psi)
        r = cls._attention(ctx)
        #ctx.save_for_backward(mu, sigma)
        return r

    @classmethod
    def backward(cls, ctx, grad_output):
        #mu, sigma = ctx.saved_tensors
        
        # J is dim(mu) x 2 x dim(psi)
        J = cls._jacobian(ctx)
        grad_input = torch.matmul(J, grad_output.unsqueeze(2)).squeeze(2)
        return grad_input, None, None


class ContinuousEntmax(nn.Module):
    def __init__(self, psi=None, alpha=None):
        super(ContinuousEntmax, self).__init__()
        self.psi = psi
        self.alpha = alpha
        self.kernel = EntmaxGaussian1DKernel(alpha)

    def forward(self, theta):
        return ContinuousEntmaxFunction.apply(theta, self.psi, self.kernel)


class ContinuousSoftmax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSoftmax, self).__init__()
        self.psi = psi
        self.alpha = 1
        self.kernel = Gaussian1DKernel()

    def forward(self, theta):
        return ContinuousEntmaxFunction.apply(theta, self.psi, self.kernel)


class ContinuousSparsemax(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousSparsemax, self).__init__()
        self.psi = psi
        self.alpha = 2
        self.kernel = SparsemaxGaussian1DKernel()

    def forward(self, theta):
        return ContinuousEntmaxFunction.apply(theta, self.psi, self.kernel)


class ContinuousBiweight(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousBiweight, self).__init__()
        self.psi = psi
        self.alpha = 1.5
        self.kernel = BiweightGaussian1DKernel()

    def forward(self, theta):
        return ContinuousEntmaxFunction.apply(theta, self.psi, self.kernel)


class ContinuousTriweight(nn.Module):
    def __init__(self, psi=None):
        super(ContinuousTriweight, self).__init__()
        self.psi = psi
        self.alpha = 1
        self.kernel = TriweightGaussian1DKernel()

    def forward(self, theta):
        return ContinuousEntmaxFunction.apply(theta, self.psi, self.kernel)
