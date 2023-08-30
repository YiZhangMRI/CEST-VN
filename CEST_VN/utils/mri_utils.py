# author: Kerstin Hammernik (https://github.com/VLOGroup/mri-variationalnetwork)
# modified by Jianping Xu

import numpy as np
import torch

def complex_abs(data, dim=-1, keepdim=False, eps=0):
    " The input tensor: data (torch.Tensor)"
    assert data.size(dim) == 2
    return torch.sqrt((data ** 2 + eps).sum(dim=dim, keepdim=keepdim))

def real2complex(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data
    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    if axis < 0: axis = x.ndim + axis
    ctype = np.complex64 if x.dtype == np.float32 else np.complex128

    if axis < len(shape):
        newshape = tuple([i for i in range(0, axis)]) \
                   + tuple([i for i in range(axis+1, x.ndim)]) + (axis,)

        x = x.transpose(newshape)

    x = np.ascontiguousarray(x).view(dtype=ctype)
    return x.reshape(x.shape[:-1])


def complex2real(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)
    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape
    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim
    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])
        x = x.transpose(newshape)

    return x

def fft2c(img):
    """ Centered fft2 """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img))) / np.sqrt(img.shape[-2]*img.shape[-1])

def ifft2c(img):
    """ Centered ifft2 """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img))) * np.sqrt(img.shape[-2]*img.shape[-1])

def mriAdjointOp(rawdata, coilsens, mask):
    """ Adjoint MRI Cartesian Operator """
    return np.sum(ifft2c(rawdata * mask)*np.conj(coilsens), axis=0)

def mriAdjointOp_no_mask(rawdata, coilsens):
    """ Adjoint MRI Cartesian Operator """
    return np.sum(ifft2c(rawdata)*np.conj(coilsens), axis=0)

def mriForwardOp(img, coilsens, mask):
    """ Forward MRI Cartesian Operator """
    return fft2c(coilsens * img)*mask
