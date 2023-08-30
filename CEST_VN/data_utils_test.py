# Created by Jianping Xu
# 2022/1/11

from scipy.io import loadmat
import torch.utils.data as data
import copy
import mat73
from pathlib import Path

from utils import mri_utils
from utils.fft_utils import *
from utils.CEST_utils import *

DEFAULT_OPTS = {}

class data_loader(data.Dataset):
    """
    Demo data loader: pre-processed data file saved in .mat format with keys: rawdata, sensitivities, reference and mask
    sensitivities: (n_coil, n_frame, width, height)
    rawdata: (n_coil, n_frame, width, height)
    reference: (n_frame, width, height)
    mask: (n_frame, width, height)
    """

    def __init__(self, **kwargs):
        super(data_loader, self).__init__()

        options = DEFAULT_OPTS

        for key in kwargs.keys():
            options[key] = kwargs[key]

        self.options = options
        self.data_dir = Path(self.options['data_dir'])
        self.filename = []
        self.coil_sens_list = []
        data_dir = self.data_dir

        # load undersampling mask
        self.mask_dir = data_dir / 'masks/3D/Data_sharing/'
        self.mask_name = self.mask_dir / self.options['mask']
        self.mask = loadmat(self.mask_name)
        self.mask = self.mask['mask'].astype(np.float32)

        # Load raw data and coil sensitivities name
        patient_dir = data_dir / 'test'
        data_name = patient_dir / self.options['test_data']
        self.filename.append(str(data_name))
        self.n_subj = len(self.filename)

    def __getitem__(self, idx):
        filename = self.filename[idx]
        mask = copy.deepcopy(self.mask)

        data = loadmat(filename)  # mat73.
        raw_data = data['rawdata']  # 16x63x96x96
        f = np.ascontiguousarray(raw_data).astype(np.complex64)

        coil_sens_data = data['sensitivities']  # 16x63x96x96
        c = np.ascontiguousarray(coil_sens_data).astype(np.complex64)

        ref = data['reference'].astype(np.complex64)  # 63x96x96

        # mask rawdata and share neighboring data
        f_sharing = undersampling_share_data(mask, f)

        # compute initial image input0
        input0 = mri_utils.mriAdjointOp_no_mask(f_sharing, c).astype(np.complex64)

        # normalize the data
        norm = np.max(np.abs(input0))
        f /= norm
        input0 /= norm
        ref /= norm

        input0 = numpy_2_complex(input0)
        f = numpy_2_complex(f)
        c = numpy_2_complex(c)
        mask = torch.from_numpy(mask)
        ref = numpy_2_complex(ref)

        data = {'u_t': input0, 'f': f, 'coil_sens': c, 'sampling_mask': mask, 'reference': ref}
        return data

    def __len__(self):
        return self.n_subj