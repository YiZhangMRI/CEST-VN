# Created by Jianping Xu
# 2022/1/12

import argparse
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from Network import Network
from data_utils_test import data_loader
from utils.mri_utils import real2complex, complex_abs
from utils.misc_utils import print_options,  save_recon_test
from utils.eval_error import *
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Arguments')

    # Data
    parser.add_argument('--test_data', type=str, default='CEST_Tumor_cao_kz.mat', help='data to be reconstruct')
    parser.add_argument('--data_dir', type=str, default='/home/xujianping2/Dataset/CEST_train', help='directory of the data')
    parser.add_argument('--mask', type=str, default='Mask_54_96_96_acc_4_New.mat', help='undersampling mask to use')
    parser.add_argument('--save_name', type=str, default='CEST_VN_Tumor_cao_acc=4.mat', help='name of redonstruction results')

    # Testing Configuration
    parser.add_argument('--model', type=str, default='model_acc=4.pth', help='pertrained model')
    parser.add_argument('--gpus', type=str, default='0', help='gpu id to use')
    parser.add_argument('--mode', type=str, default='test', help='train or test')
    parser.add_argument('--loss_type', type=str, default='magnitude', help='compute loss on complex or magnitude image')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--save_dir', type=str, default='Exp', help='directory of the experiment')
    parser.add_argument('--loss_weight', type=float, default=10, help='trade-off between two parts of the loss')
    parser.add_argument('--loss_scale', type=float, default=100, help='scale the loss value, display purpose')
    parser.add_argument('--n_worker', type=int, default=48, help='number of workers')

    args = parser.parse_args()
    print_options(parser, args)
    args = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']
    # device = 'cuda:0'

    # Configure directory information
    project_root = '.'
    save_dir = os.path.join(project_root, 'models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    recon_dir = os.path.join(project_root, 'recon')
    if not os.path.isdir(recon_dir):
        os.makedirs(recon_dir)

    # build the model
    model = Network().cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', total_params)

    test_set = data_loader(**args)

    testing_data_loader = DataLoader(dataset=test_set, batch_size=args['batch_size'], num_workers=args['n_worker'], shuffle=True)

    path_checkpoint = os.path.join(save_dir, args['model'])
    checkpoint = torch.load(path_checkpoint)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['net'])
    print("loading checkpoint...")

    model.eval()
    base_nrmse = []
    test_nrmse = []

    t_start = time.time()
    for iteration, batch in enumerate(testing_data_loader):
        u_t = batch['u_t'].cuda()
        f = batch['f'].cuda()
        coil_sens = batch['coil_sens'].cuda()
        sampling_mask = batch['sampling_mask'].cuda()
        input = {'u_t': u_t, 'f': f, 'coil_sens': coil_sens, 'sampling_mask': sampling_mask}

        ref = batch['reference'].cuda()

        with torch.no_grad():
            recon = model(input)

        recon_real = complex_abs(recon)
        ref_real = complex_abs(ref)

        recon_complex = real2complex(recon.data.to('cpu').numpy(), axis=-1)  # NxZxHxWx2 => complex
        ref_complex = real2complex(ref.data.to('cpu').numpy(), axis=-1)
        und_complex = real2complex(u_t.data.to('cpu').numpy(), axis=-1)
        save_recon_test(args['save_name'], recon_complex, recon_dir)  # save complex recon as .mat

        for idx in range(recon_complex.shape[0]):
            test_nrmse.append(nrmse(recon_complex[idx], ref_complex[idx]))
            base_nrmse.append(nrmse(und_complex[idx], ref_complex[idx]))

        t_end = time.time()
        print(" test nRMSE:\t\t{:.6f}".format(np.mean(test_nrmse)))
        print(" base nRMSE:\t\t{:.6f}".format(np.mean(base_nrmse)))
        print(" recon time:\t\t{:.6f}".format(t_end - t_start),"s")