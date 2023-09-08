# Created by Jianping Xu
# 2022/1/12

import torch
import numpy as np

def source2CEST(x):
    """
    Input:
        torch.Tensor: Real valued source data, size [N Z H W]
    Output:
        torch.Tensor: Real valued CEST data, size [N Z/2-2 H W]
    """
    num = x.shape[1]/2-1
    img_CEST = torch.zeros([x.shape[0], int(num), x.shape[2], x.shape[3]], dtype=x.dtype)

    for k in range(0, int(num)):
        img_CEST[:,k,:,:] = x[:, (x.shape[1]-1-k), :, :] - x[:, k + 1, :, :]

    return img_CEST.cuda()

def undersampling_share_data(mask,k):
    """undersample k-space data and share neighboring data
        input:
        mask: ndarray [nt,nx,ny] (e.g. [54,96,96])
        k: ndarray, k-space data [nc,nt,nx,ny] (e.g. [8,54,96,96])
        input:
        k_share: ndarray, undersampled k-space data [nc,nt,nx,ny] (e.g. [8,54,96,96])
    """
    mask_temp = np.zeros((mask.shape[0] + 2, mask.shape[1], mask.shape[2]))
    mask_temp[1:-1, :, :] = mask
    mask_temp = np.expand_dims(mask_temp, 0)  # [1,54,96,96]
    mask = mask_temp.astype('bool')

    k_temp = np.zeros((k.shape[0], k.shape[1] + 2, k.shape[2], k.shape[3]), dtype=complex)
    k_temp[:, 1:-1, :, :] = k
    k1 = k_temp

    k_share = np.zeros(k.shape, dtype=complex)  # [8,54,96,96]
    for i in range(1, k_share.shape[1] + 1):
        mask_2 = mask[:, i, :, :]
        mask2 = (~mask[:, i, :, :])
        mask1 = mask[:, i - 1, :, :]
        mask_1 = mask2 * mask1
        mask_temp = mask[:, i - 1, :, :] + mask[:, i, :, :]
        # mask_temp(mask_temp > 0) = 1
        mask_temp = ~mask_temp
        mask_3 = mask[:, i + 1, :, :] * mask_temp
        k_share[:, i - 1, :, :] = mask_1 * k1[:, i - 1, :, :] + mask_2 * k1[:, i, :, :] + mask_3 * k1[:, i + 1, :, :]

    k_share[:, 0, :, :] = k1[:, 1, :, :] * mask[:, 1, :, :]
    k_share[:, 25:28, :, :] = k1[:, 26:29, :, :] * mask[:, 26:29, :, :]  # S0 and around 0 ppm ones de not share data
    k_share = k_share.astype(np.complex64)

    return k_share
