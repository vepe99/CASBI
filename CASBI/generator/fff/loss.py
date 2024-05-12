# MIT License
#
# Copyright (c) 2023 Computer Vision and Learning Lab, Heidelberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from collections import namedtuple
from math import sqrt, prod
from typing import Union, Callable

import torch
from torch.autograd import grad
from torch.autograd.forward_ad import dual_level, make_dual, unpack_dual
from CASBI.generator.fff.m_loss import project_z, project_x1, sample_v_in_tangent

SurrogateOutput = namedtuple("SurrogateOutput", ["z", "x1", "nll", "surrogate", "regularizations"])
Transform = Callable[[torch.Tensor], torch.Tensor]


def sample_v(x: torch.Tensor, hutchinson_samples: int):
    """
    Sample a random vector v of shape (x.shape, hutchinson_samples)
    with scaled orthonormal columns.

    The reference data is used for shape, device and dtype.

    Parameters
    ----------
    x: 
        Reference data.
    hutchinson_samples: 
        Number of Hutchinson samples to draw.
    
    Return
    ------
    q
    """
    batch_size, total_dim = x.shape[0], prod(x.shape[1:])
    if hutchinson_samples > total_dim:
        raise ValueError(f"Too many Hutchinson samples: got {hutchinson_samples}, expected <= {total_dim}")
    v = torch.randn(batch_size, total_dim, hutchinson_samples, device=x.device, dtype=x.dtype)
    q = torch.linalg.qr(v).Q.reshape(*x.shape, hutchinson_samples)
    return q * sqrt(total_dim)


def nll_surrogate(x: torch.Tensor, cond: torch.Tensor, encode: Transform, decode: Transform,
                  hutchinson_samples: int = 1, manifold=None) -> SurrogateOutput:
    """
    Compute the per-sample surrogate for the negative log-likelihood and the volume change estimator.
    The gradient of the surrogate is the gradient of the actual negative log-likelihood.

    Parameters
    ----------
    x: 
        Input data. Shape: (batch_size, ...)
    encode: 
        Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    decode: 
        Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    hutchinson_samples: 
        Number of Hutchinson samples to use for the volume change estimator.
    param manifold: 
    Manifold on which the latent space lies. If provided, the volume change is computed on the manifold.
    
    Returns
    -------
    Per-sample loss: 
        Shape=(batch_size,)
    """
    x = x.requires_grad_()
    z = encode(x, cond)

    metrics = {}
    # M-FFF: Project to manifold and store projection distance for regularization
    if manifold is not None:
        z, metrics["z_projection"] = project_z(z, manifold)

    surrogate = 0
    if manifold is None:
        vs = sample_v(z, hutchinson_samples)
    else:
        # M-FFF: Sample v in the tangent space of z
        vs = sample_v_in_tangent(z, hutchinson_samples, manifold)
    for k in range(hutchinson_samples):
        v = vs[..., k]

        # $ g'(z) v $ via forward-mode AD
        with dual_level():
            dual_z = make_dual(z, v)
            dual_x1 = decode(dual_z, cond)

            # M-FFF: Project to manifold and store projection distance for regularization
            if manifold is not None:
                dual_x1, metrics["x1_projection"] = project_x1(dual_x1, manifold)

            x1, v1 = unpack_dual(dual_x1)

        # $ v^T f'(x) $ via backward-mode AD
        v2, = grad(z, x, v, create_graph=True)

        # $ v^T f'(x) stop_grad(g'(z)) v $
        surrogate += sum_except_batch(v2 * v1.detach()) / hutchinson_samples

    # Per-sample negative log-likelihood
    nll = sum_except_batch((z ** 2)) / 2 - surrogate

    return SurrogateOutput(z, x1, nll, surrogate, metrics)


def fff_loss(x: torch.Tensor,
             encode: Transform, decode: Transform,
             beta: Union[float, torch.Tensor],
             hutchinson_samples: int = 1) -> torch.Tensor:
    """
    Compute the per-sample FFF/FIF loss:
    $$
    \mathcal{L} = \beta ||x - decode(encode(x))||^2 + ||encode(x)||^2 // 2 - \sum_{k=1}^K v_k^T f'(x) stop_grad(g'(z)) v_k
    $$
    where $E[v_k^T v_k] = 1$, and $ f'(x) $ and $ g'(z) $ are the Jacobians of `encode` and `decode`.

    Parameters: 
    -----------
    x:
        Input data. Shape: (batch_size, ...)
    encode: 
        Encoder function. Takes `x` as input and returns a latent representation of shape (batch_size, latent_dim).
    decode: 
        Decoder function. Takes a latent representation of shape (batch_size, latent_dim) as input and returns a reconstruction of shape (batch_size, ...).
    beta: 
        Weight of the mean squared error.
    hutchinson_samples: 
        Number of Hutchinson samples to use for the volume change estimator.
        
    Returns
    --------
    Per-sample loss. 
        Shape: (batch_size,)
    """
    surrogate = nll_surrogate(x, encode, decode, hutchinson_samples)
    mse = torch.sum((x - surrogate.x1) ** 2, dim=tuple(range(1, len(x.shape))))
    return beta * mse + surrogate.nll


def sum_except_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Sum over all dimensions except the first.
    
    Parameters
    ----------
    x: 
        Input tensor.
    
    Returns
    ---------- 
    sum:
        Sum over all dimensions except the first.
    """
    return torch.sum(x.reshape(x.shape[0], -1), dim=1)
