""" test factorized scale operator """

import pytest
import torch
from spcdist.torch import FactorizedScale

@pytest.mark.parametrize('shapes', [
    ((4,3,3), (4,3,1)),
    ((4,3,3), (3,1)),
    ((3,3), (4,3,1)),
    ((3,3), (3,1)),
    # ((3,3), (3,)),  # this one raises exception
])
def test_sqrt_x(shapes):

    x_shape, v_shape = shapes

    torch.manual_seed(42)

    x = torch.randn(*x_shape)
    v = torch.randn(*v_shape)
    bmm = x.transpose(-2, -1) @ x

    fs = FactorizedScale(bmm)

    # print((fs.L @ v).shape)
    # print((fs.L_mul_X(v)).shape)

    assert torch.allclose(fs.L @ v, fs.L_mul_X(v))


@pytest.mark.parametrize('shapes', [
    ((4,3,3), (4,3,1)),
    ((4,3,3), (3,1)),
    ((3,3), (4,3,1)),
    ((3,3), (3,1)),
    # ((3,3), (3,)),  # this one raises exception
])
def test_sqrt_inv_t_x(shapes):

    torch.manual_seed(42)

    x_shape, v_shape = shapes

    x = torch.randn(*x_shape)
    v = torch.randn(*v_shape)
    bmm = x.transpose(-2, -1) @ x

    fs = FactorizedScale(bmm)

    # print((fs.L @ v).shape)
    # print((fs.L_mul_X(v)).shape)

    assert torch.allclose(fs.L_inv.transpose(-2, -1) @ v, fs.L_inv_t_mul_X(v))





