import pytest

import numpy as np
import torch

from spcdist.scipy import multivariate_beta_gaussian
from spcdist.torch import MultivariateBetaGaussianDiag


@pytest.mark.parametrize('alpha', [3/2, 4/3, 2, 3])
def test_torch_scipy_agreement(alpha):
    mean = np.array([10, 11.1])
    scale_diag = np.array([0.5, 1.5])

    mbg = multivariate_beta_gaussian(mean=mean,
                                     scale=np.diag(scale_diag),
                                     alpha=alpha)

    mbg_t = MultivariateBetaGaussianDiag(torch.from_numpy(mean),
                                         torch.from_numpy(scale_diag),
                                         alpha)

    assert np.allclose(
            mbg_t.tsallis_entropy.item(),
            mbg.tsallis_entropy()
    )

    assert np.allclose(
        mbg_t.log_radius.item(),
        np.log(mbg.radius)
    )

    assert np.allclose(
        mbg_t._tau.item(),
        mbg.tau
    )
