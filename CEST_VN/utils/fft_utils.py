# author: Kerstin Hammernik
# modified by Jianping Xu

import torch
import numpy as np

def fft2(data):
    assert data.size(-1) == 2
    data = torch.fft(data, 2, normalized=True)
    return data

def fft2c(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft2(data):
    assert data.size(-1) == 2
    data = torch.ifft(data, 2, normalized=True)
    return data

def ifft2c(data):
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def torch_abs(x):
    """
    Compute magnitude for two-channel complex torch tensor
    """
    mag = torch.sqrt(torch.sum(torch.square(x), axis=-1, keepdim=False) + 1e-9)
    return mag


""" Converting to and from complex image and two channels image """


def real_2_complex(x):
    """
    Convert real-valued, 1-channel, torch tensor to complex-valued, 2-channel
    with 0 imaginary component

    Parameters
    ----------
    x : input tensor

    Returns
    -------
    complex array with 2-channel at the end

    """
    out = x.squeeze()
    out = x.unsqueeze(-1)
    imag = torch.zeros(out.shape, dtype=out.dtype, requires_grad=out.requires_grad)
    out = torch.cat((out, imag), dim=-1)
    return out


def complex_2_numpy(x):
    """
    Convert 2-channel complex torch tensor to numpy complex number

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    out = x.numpy()
    out = np.take(out, 0, axis=-1) + np.take(out, 1, axis=-1) * 1j
    return out


def numpy_2_complex(x):
    """
    Convert numpy complex array to 2-channel complex torch tensor

    Parameters
    ----------
    x : numpy complex array
        input array

    Returns
    -------
    Equivalent 2-channel torch tensor

    """
    real = np.real(x)
    real = np.expand_dims(real, -1)
    imag = np.imag(x)
    imag = np.expand_dims(imag, -1)
    out = np.concatenate((real, imag), axis=-1)
    out = torch.from_numpy(out)
    return out

def conj(x):
    """
    Calculate the complex conjugate of x

    x is two-channels complex torch tensor
    """
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

def complex_mul(x, y):
    """ Complex multiply 2-channel complex torch tensor x,y
    """
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)

def source2CEST(x):
    img_CEST = torch.zeros([x.shape[0],26,x.shape[2],x.shape[3],x.shape[4]], dtype=x.dtype)

    for k in range(0, 26):
        img_CEST[:,k,:,:,:] = x[:, 53 - k, :, :, :] - x[:, k + 1, :, :, :]

    return img_CEST

def CEST2source(img_CEST,x):
    x_out = torch.zeros(x.shape,device=x.device)
    u_p1 = x[:, 1:27, :, :, :]  # 1 -> 26
    u_n1 = u_p1 + img_CEST  # 53 -> 28
    u_n = torch.flip(u_n1, [1])  # 28 -> 53

    x_out[:, 0, :, :, :] = x[:, 0, :, :, :]
    x_out[:, 27, :, :, :] = x[:, 27, :, :, :]
    x_out[:, 1:27, :, :, :] = u_p1
    x_out[:, 28:54, :, :, :] = u_n

    return x_out