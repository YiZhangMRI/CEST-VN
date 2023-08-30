# Created by Jianping Xu
# 2022/1/12

import argparse
import os
import time
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from Network import Network
from data_utils import data_loader, data_loader_eval
from utils.mri_utils import real2complex, complex_abs
from utils.misc_utils import print_options, save_recon
from utils.eval_error import *
from torch.utils.tensorboard import SummaryWriter
from utils.CEST_utils import source2CEST
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Network Arguments')

    # Data
    parser.add_argument('--data_dir', type=str, default='/home/xujianping2/Dataset/CEST_train', help='directory of the data')
    parser.add_argument('--mask', type=str, default='Mask_54_96_96_acc_4_New.mat', help='undersampling mask to use')

    # Network configuration
    parser.add_argument('--num_stages', type=int, default=10, help='number of stages in the network')

    # Training and Testing Configuration
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='gpu id to use')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--loss_type', type=str, default='magnitude', help='compute loss on complex or magnitude image')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epoch')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--save_dir', type=str, default='Exp', help='directory of the experiment')
    parser.add_argument('--loss_weight', type=float, default=10, help='trade-off between two parts of the loss')
    parser.add_argument('--loss_scale', type=float, default=100, help='scale the loss value, display purpose')
    parser.add_argument('--n_worker', type=int, default=16, help='number of workers')
    parser.add_argument('--Resure', type=str, default='False', help='resume or not')

    args = parser.parse_args()
    print_options(parser, args)
    args = vars(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpus']

    # Configure directory info
    project_root = '.'
    save_dir = os.path.join(project_root, 'models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    recon_dir = os.path.join(project_root, 'recon')
    if not os.path.isdir(recon_dir):
        os.makedirs(recon_dir)

    # build the model
    model = Network()
    model = torch.nn.DataParallel(model, device_ids=[int(x) for x in args['gpus'].split(',')]).cuda()

    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters:', total_params)

    train_set = data_loader(**args)
    eval_set = data_loader_eval(**args)

    training_data_loader = DataLoader(dataset=train_set, batch_size=args['batch_size'], num_workers=args['n_worker'], shuffle=True)
    eval_data_loader = DataLoader(dataset=eval_set, batch_size=args['batch_size'], num_workers=args['n_worker'], shuffle=False)

    if args['Resure'] == 'True':
        path_checkpoint = os.path.join(save_dir, 'model_acc=4.pth')
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']+1
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
        print("loading checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        start_epoch = 0

    writer = SummaryWriter("./logs_train")
    for epoch in range(start_epoch, args['epoch'] + 1):
        model.train()
        t_start = time.time()
        train_err = 0
        train_batches = 0

        with tqdm(enumerate(training_data_loader), total = len(training_data_loader)) as tepoch:
            for iteration, batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                u_t = batch['u_t'].cuda()
                f = batch['f'].cuda()
                coil_sens = batch['coil_sens'].cuda()
                sampling_mask = batch['sampling_mask'].cuda()
                input = {'u_t': u_t, 'f': f, 'coil_sens': coil_sens, 'sampling_mask': sampling_mask}

                ref = batch['reference'].cuda()
                recon = model(input)
                recon_real = complex_abs(recon)
                ref_real = complex_abs(ref)

                recon_CEST = source2CEST(recon_real)  # [N Z/2-1 H W]
                ref_CEST = source2CEST(ref_real)

                loss_1 = criterion(recon_real + 1e-11, ref_real)  # MSE loss of source Images
                loss_2 = criterion(recon_CEST + 1e-11, ref_CEST)
                loss = loss_1 + args['loss_weight']*loss_2

                optimizer.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

                train_err += loss.item()
                train_batches += 1
                torch.cuda.empty_cache()
                tepoch.set_postfix(loss_1=args['loss_scale'] * loss_1.item(),loss_2=args['loss_scale']*args['loss_weight']*loss_2.item())

            t_end = time.time()
            train_err /= train_batches

            # eval
            model.eval()
            test_loss = []
            base_nrmse = []
            test_nrmse = []

            for iteration, batch in enumerate(eval_data_loader): # evaluation
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

                recon_CEST = source2CEST(recon_real)  # [N Z/2-1 H W]
                ref_CEST = source2CEST(ref_real)

                loss_1 = criterion(recon_real + 1e-11, ref_real)  # MSE loss of source Images
                loss_2 = criterion(recon_CEST + 1e-11, ref_CEST)
                loss = loss_1 + args['loss_weight']*loss_2
                test_loss.append(loss.item())

                recon_complex = real2complex(recon.data.to('cpu').numpy(), axis=-1)  # NxZxHxWx2 => complex
                ref_complex = real2complex(ref.data.to('cpu').numpy(), axis=-1)
                und_complex = real2complex(u_t.data.to('cpu').numpy(), axis=-1)
                save_recon(epoch, recon_complex, recon_dir)  # save complex recon as .mat

                for idx in range(recon_complex.shape[0]):
                    test_nrmse.append(nrmse(recon_complex[idx], ref_complex[idx]))
                    base_nrmse.append(nrmse(und_complex[idx], ref_complex[idx]))
            # save model
            name = "model_epoch{}_tloss{}_eloss{}.pth".format(epoch, round(args['loss_scale']*train_err,3), round(args['loss_scale']*np.mean(test_loss),3))  # save models
            torch.save(model.state_dict(), os.path.join(save_dir, name))
            checkpoint = {
                "net": model.state_dict(),
                "epoch": epoch,
                'lr_schedule': lr_schedule.state_dict()
            }
            torch.save(checkpoint, os.path.join(save_dir, name))
            lr_schedule.step()

            print(" Epoch {}/{}".format(epoch + 1, args['epoch']))
            print(" time: {}s".format(t_end - t_start))
            print(" training loss:\t\t{:.6f}".format(args['loss_scale']*train_err))
            print(" testing loss:\t\t{:.6f}".format(args['loss_scale']*np.mean(test_loss)))
            print(" test nRMSE:\t\t{:.6f}".format(np.mean(test_nrmse)))
            print(" base nRMSE:\t\t{:.6f}".format(np.mean(base_nrmse)))
            print(' learning rate:\t\t', optimizer.state_dict()['param_groups'][0]['lr'])
    writer.close()
