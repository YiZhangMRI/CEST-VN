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

DEFAULT_OPTS = {'training_data': [p for p in range(1, 786+1)],  # for training
                'start_slice': 1, 'end_slice': 10,
                # 'val_data': [p for p in range(786, 925+1)]}   # simulated data for eval
                'val_data': ['CEST_Tumor_cao_96']}  # experimental data for eval

class data_loader(data.Dataset):
    """
    Demo data loader: pre-processed data file saved in .mat format with keys: rawdata, sensitivities, reference and mask
    sensitivities: (width, height, n_frame, n_coil)
    rawdata: (width, height, n_frame, n_coil)
    reference: (width, height, n_frame)
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
        self.mask = loadmat(self.mask_name)  # 54x96x96
        self.mask = self.mask['mask'].astype(np.float32)

        # Load raw data and coil sensitivities name
        patient_key = 'training_data'
        slice_no = [x for x in range(options['start_slice'], options['end_slice'] + 1)]

        for patient in options[patient_key]:
            patient_dir = data_dir / str(patient)
            for i in slice_no:
                slice_dir = patient_dir / 'rawdata{}.mat'.format(i)
                self.filename.append(str(slice_dir))
                coil_sens_dir = patient_dir / 'espirit{}.mat'.format(i)
                self.coil_sens_list.append(str(coil_sens_dir))

        self.n_subj = len(self.filename)

        print("Training Dataset: {} elements".format(len(self.filename)))

    def __getitem__(self, idx):
        filename = self.filename[idx]
        mask = copy.deepcopy(self.mask)
        coil_sens = self.coil_sens_list[idx]

        raw_data = mat73.loadmat(filename)  # mat73.
        raw_data = raw_data['rawdata']  # 96x96x54x16
        raw_data = np.transpose(raw_data, (3, 2, 0, 1))  # 16x54x96x96

        f = np.ascontiguousarray(raw_data).astype(np.complex64)

        coil_sens_data = mat73.loadmat(coil_sens)  # mat73.
        coil_sens_data['sensitivities'] = np.expand_dims(coil_sens_data['sensitivities'], 0).repeat(54, axis=0) # 54x96x96x16
        c = np.ascontiguousarray(np.transpose(coil_sens_data['sensitivities'], (3, 0, 1, 2))).astype(np.complex64) # 16x54x96x96

        ref = coil_sens_data['reference'].astype(np.complex64) # 96x96x54
        ref = np.transpose(ref, (2, 0, 1)) # 54x96x96

        # mask rawdata and share neighboring data
        f_sharing = undersampling_share_data(mask,f)
        f *= mask

        # compute initial image input0
        input0 = mri_utils.mriAdjointOp_no_mask(f_sharing, c).astype(np.complex64)  # input the data-shared X0,

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

class data_loader_eval(data.Dataset):
    """
    load eval data during training
    """

    def __init__(self, **kwargs):
        super(data_loader_eval, self).__init__()

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
        self.mask_name = self.mask_dir/self.options['mask']
        self.mask = loadmat(self.mask_name)  # 54x96x96
        self.mask = self.mask['mask'].astype(np.float32)

        # Load raw data and coil sensitivities name
        for patient in options['val_data']:
            patient_dir = data_dir / str(patient)
            data_dir = patient_dir / 'CEST_Tumor_cao_kz.mat' # CEST_Tumor_3_kz.mat/CEST_Tumor_cao_kz.mat
            self.filename.append(str(data_dir))
        self.n_subj = len(self.filename)

        print("Eval Dataset: {} elements".format(len(self.filename)))

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
