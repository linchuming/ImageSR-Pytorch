import os
import random

import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from dataset import DIV2K_Dataset
from model import EDSR
import tqdm

config = config['train']

scale = config['scale']
batch_size = config['batch_size']
learning_rate = float(config['learning_rate'])
weight_decay = float(config['weight_decay'])
lr_step = config['lr_step']
max_epoch = config['max_epoch']
device_mode = config['device_mode']
device_gpu_id = config['device_gpu_id']

dataset_dir = config['dataset_dir']
patch_size = config['patch_size']
crop_num_per_image = config['crop_num_per_image']

checkpoint_dir = config['checkpoint_dir']

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

if device_mode == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device("cpu")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = device_gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_fn(sr, gt):
    loss = nn.MSELoss(reduction='sum')
    output = loss(sr, gt)
    return output


if __name__ == '__main__':
    set_seed(2019)      # Set seed to produce the same training results

    model = EDSR(upscale=scale)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    writer = SummaryWriter(os.path.join(checkpoint_dir, 'tensorboard_log'),
                           flush_secs=10)
    step = 0
    dataset = DIV2K_Dataset(dataset_dir, patch_size,
                            scale, crop_num_per_image)
    dl = DataLoader(dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True,
                    pin_memory=True)

    for epoch in range(max_epoch):
        torch.cuda.empty_cache()
        dataset.crop_samples()

        print('training ...')

        epoch_step = len(dataset) // batch_size
        progress = tqdm.tqdm(total=epoch_step)

        total_loss = .0

        for i, data in enumerate(dl):
            lr = data['lr'].unsqueeze(1).to(device)
            gt = data['gt'].unsqueeze(1).to(device)
            bic = data['bic'].unsqueeze(1).to(device)
            # print(lr.size(), bic.size())
            sr = model(lr, bic)
            loss = loss_fn(sr, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_loss = loss.cpu().detach().numpy()
            total_loss += mse_loss

            step += 1
            # write to tensorboard
            if step % 40 == 0:
                writer.add_scalar('mse_loss', mse_loss, step)
                writer.add_image('bic', bic[0].clamp(0, 1), step)
                writer.add_image('sr', sr[0].clamp(0, 1), step)
                writer.add_image('gt', gt[0], step)

            progress.update(1)

        progress.close()
        scheduler.step()
        print("Epoch: {}, average MSE loss: {}".format(epoch, total_loss / (i + 1)))

        # save checkpoint
        checkpoint = {
            'state_dict': model.module.state_dict(),
            # 'opt_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint-%d.ckpt' % epoch))
