import numpy as np

def nrmse(img, ref, axes = (0,1,2)):
    """ Compute the normalized root mean squared error (nrmse)
    :param img: input image, ZxHxW
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the nrmse is computed
    :return: (mean) nrmse
    """
    nominator = np.abs(np.sum((img - ref) * np.conj(img - ref), axis = axes))
    # denominator = np.abs(np.sum(ref * np.conj(ref), axis = axes))
    denominator = (ref.shape[0])*(ref.shape[1])*(ref.shape[2])*np.max(np.abs(ref * np.conj(ref)))
    nrmse = np.sqrt(nominator / denominator)
    return np.mean(nrmse)