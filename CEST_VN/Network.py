import torch.nn as nn
import torch.nn.functional as F
from utils.fft_utils import *
from utils.misc_utils import *

DEFAULT_OPTS = {'kernel_size': 7,
                'features_in': 1,
                'features_out': 36,
                'do_prox_map': True,
                'pad': 7,
                'vmin': -1.0, 'vmax': 1.0,
                'lamb_init': 1.0,
                'num_act_weights': 31,
                'init_type': 'linear',
                'init_scale': 0.04,
                'sampling_pattern': 'cartesian',
                'num_stages': 10,
                'momentum': 0.,
                'error_scale': 10,
                }

class RBFActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w, mu, sigma):
        """ Forward pass for RBF activation

        Parameters:
        ----------
        ctx:
        input: torch tensor (NxCxZxHxW)
            input tensor
        w: torch tensor (1 x C x 1 x 1 x 1 x # of RBF kernels)
            weight of the RBF kernels
        mu: torch tensor (# of RBF kernels)
            center of the RBF
        sigma: torch tensor (1)
            std of the RBF

        Returns:
        ----------
        torch tensor: linear weight combination of RBF of input
        """
        num_act_weights = w.shape[-1]
        output = input.new_zeros(input.shape)
        rbf_grad_input = input.new_zeros(input.shape)
        for i in range(num_act_weights):
            tmp = w[:, :, :, :, :, i] * torch.exp(-torch.square(input - mu[i]) / (2 * sigma ** 2))
            output += tmp
            rbf_grad_input += tmp * (-(input - mu[i])) / (sigma ** 2)
        del tmp
        ctx.save_for_backward(input, w, mu, sigma, rbf_grad_input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w, mu, sigma, rbf_grad_input = ctx.saved_tensors
        num_act_weights = w.shape[-1]

        # if ctx.needs_input_grad[0]:
        grad_input = grad_output * rbf_grad_input

        # if ctx.need_input_grad[1]:
        grad_w = w.new_zeros(w.shape)
        for i in range(num_act_weights):
            tmp = (grad_output * torch.exp(-torch.square(input - mu[i]) / (2 * sigma ** 2))).sum((0, 2, 3, 4))
            grad_w[:, :, :, :, :, i] = tmp.view(w.shape[0:-1])

        return grad_input, grad_w, None, None


class RBFActivation(nn.Module):
    """ RBF activation function with trainable weights """

    def __init__(self, **kwargs):
        super().__init__()
        self.options = kwargs
        x_0 = np.linspace(kwargs['vmin'], kwargs['vmax'], kwargs['num_act_weights'], dtype=np.float32)
        mu = np.linspace(kwargs['vmin'], kwargs['vmax'], kwargs['num_act_weights'], dtype=np.float32)
        self.sigma = 2 * kwargs['vmax'] / (kwargs['num_act_weights'] - 1)
        self.sigma = torch.tensor(self.sigma)
        if kwargs['init_type'] == 'linear':
            w_0 = kwargs['init_scale'] * x_0
        elif kwargs['init_type'] == 'tv':
            w_0 = kwargs['init_scale'] * np.sign(x_0)
        elif kwargs['init_type'] == 'relu':
            w_0 = kwargs['init_scale'] * np.maximum(x_0, 0)
        elif kwargs['init_type'] == 'student-t':
            alpha = 100
            w_0 = kwargs['init_scale'] * np.sqrt(alpha) * x_0 / (1 + 0.5 * alpha * x_0 ** 2)
        else:
            raise ValueError("init_type '%s' not defined!" % kwargs['init_type'])
        w_0 = np.reshape(w_0, (1, 1, 1, 1, 1, kwargs['num_act_weights']))
        w_0 = np.repeat(w_0, kwargs['features_out'], 1)
        self.w = torch.nn.Parameter(torch.from_numpy(w_0))
        self.mu = torch.from_numpy(mu)
        self.rbf_act = RBFActivationFunction.apply

    def forward(self, x):
        if not self.mu.device == x.device:
            self.mu = self.mu.to(x.device)
            self.sigma = self.sigma.to(x.device)
        output = self.rbf_act(x, self.w, self.mu, self.sigma)

        return output

class ReconCell(nn.Module):
    """ One cell of the network """

    def __init__(self, **kwargs):
        super().__init__()
        options = kwargs
        self.options = options
        conv_kernel = np.random.randn(options['features_out'], options['features_in'], options['kernel_size'],
                                      options['kernel_size'], options['kernel_size'], 2).astype(np.float32) \
                      / np.sqrt(options['kernel_size'] ** 3 * 2 * options['features_in'])
        conv_kernel -= np.mean(conv_kernel, axis=(1, 2, 3, 4, 5), keepdims=True)
        conv_kernel = torch.from_numpy(conv_kernel)
        if options['do_prox_map']:
            conv_kernel = zero_mean_norm_ball(conv_kernel, axis=(1, 2, 3, 4, 5))

        self.conv_kernel = torch.nn.Parameter(conv_kernel)
        self.activation = RBFActivation(**options)

        self.lamb = torch.nn.Parameter(torch.tensor(options['lamb_init'], dtype=torch.float32))

    def mri_forward_op(self, u, coil_sens, sampling_mask):
        """
        Parameters:
        u: complex input image [N Z H W 2]
        coil_sens: coil sensitivity map [N C Z H W 2]
        sampling_mask: [N Z H W]

        Returns:
        kspace of u with applied coil sensitivity and sampling mask
        """
        u_pad = u
        u_pad = u_pad.unsqueeze(1)  # Nx1xZxHxWx2
        coil_imgs = complex_mul(u_pad, coil_sens)  # NxCxZxHxWx2

        Fu = fft2c(coil_imgs)  # NxCxZxHxWx2

        mask = sampling_mask.unsqueeze(1)  # Nx1xZxHxW
        mask = mask.unsqueeze(5)  # Nx1xZxHxWx1
        mask = mask.repeat([1, 1, 1, 1, 1, 2])  # Nx1xZxHxWx2

        kspace = mask * Fu  # NxCxZxHxWx2
        return kspace

    def mri_adjoint_op(self, f, coil_sens, sampling_mask):
        """
        Adjoint operation that convert kspace to coil-combined under-sampled image
        by using coil_sens and sampling mask

        Parameters:
        ----------
        f: multi channel kspace [N C H W 2]
        coil_sens: coil sensitivity map [N C Z HxW 2]
        sampling_mask: [N Z H W]

        Returns:
        -----------
        Undersampled, coil-combined image
        """
        mask = sampling_mask.unsqueeze(1)  # Nx1xZxHxW
        mask = mask.unsqueeze(5)  # Nx1xZxHxWx1
        mask = mask.repeat([1, 1, 1, 1, 1, 2])  # Nx1xZxHxWx2

        Finv = ifft2c(mask * f)  # NxCxZxHxWx2
        # multiply coil images with sensitivities and sum up over channels
        img = torch.sum(complex_mul(Finv, conj(coil_sens)), 1)

        return img

    def forward(self, inputs):
        u_t_1 = inputs['u_t']  # NxZxHxWx2
        f = inputs['f']
        c = inputs['coil_sens']
        m = inputs['sampling_mask']

        u_t_2 = u_t_1.unsqueeze(1)  # Nx1xZxHxWx2
        pad = self.options['pad']
        u_t_real = u_t_2[:, :, :, :, :, 0]
        u_t_imag = u_t_2[:, :, :, :, :, 1]

        u_t_real = F.pad(u_t_real, [pad, pad, pad, pad, pad, pad], mode='constant', value=0)
        u_t_imag = F.pad(u_t_imag, [pad, pad, pad, pad, pad, pad], mode='constant', value=0)

        u_k_real = F.conv3d(u_t_real.type(torch.cuda.FloatTensor), self.conv_kernel[:, :, :, :, :, 0], stride=1, padding=5)
        u_k_imag = F.conv3d(u_t_imag.type(torch.cuda.FloatTensor), self.conv_kernel[:, :, :, :, :, 1], stride=1, padding=5)

        # add up the convolution results
        u_k = u_k_real + u_k_imag
        # apply activation function
        f_u_k = self.activation(u_k)
        # perform transpose convolution for real and imaginary part
        u_k_T_real = F.conv_transpose3d(f_u_k, self.conv_kernel[:, :, :, :, :, 0], stride=1, padding=5)
        u_k_T_imag = F.conv_transpose3d(f_u_k, self.conv_kernel[:, :, :, :, :, 1], stride=1, padding=5)

        # Rebuild complex image
        u_k_T_real = u_k_T_real.unsqueeze(-1)
        u_k_T_imag = u_k_T_imag.unsqueeze(-1)
        u_k_T = torch.cat((u_k_T_real, u_k_T_imag), dim=-1)

        # Remove padding and normalize by number of filter
        Ru = u_k_T[:, 0, pad:-pad, pad:-pad, pad:-pad, :]  # NxZxHxWx2
        Ru /= self.options['features_out']

        Au = self.mri_forward_op(u_t_1, c, m)
        At_Au_f = self.mri_adjoint_op(Au - f, c, m)
        Du = At_Au_f * self.lamb
        u_t = u_t_1 - Ru - Du
        output = {'u_t': u_t, 'f': f, 'coil_sens': c, 'sampling_mask': m}

        return output  # NxZxHxWx2

class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()
        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        cell_list = []
        for i in range(options['num_stages']):
            cell_list.append(ReconCell(**options))

        self.cell_list = nn.Sequential(*cell_list)
        self.log_img_count = 0

    def forward(self, inputs):
        output = self.cell_list(inputs)
        return output['u_t']
