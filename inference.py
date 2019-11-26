import torch
from model import EDSR
from config import config
from scipy import misc
import utils
import os
import glob
import numpy as np

config = config['inference']

scale = config['scale']
checkpoint_path = config['checkpoint_path']
input_dir = config['input_dir']
input_suffix = config['input_suffix']
output_dir = config['output_dir']

device_mode = config['device_mode']
device_gpu_id = config['device_gpu_id']

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if device_mode == 'CPU':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    device = torch.device("cpu")
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = device_gpu_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    checkpoint = torch.load(checkpoint_path)

    model = EDSR(upscale=scale)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    fs = glob.glob(os.path.join(input_dir, input_suffix))
    psnrs = []
    for f in fs:
        img = misc.imread(f)
        lr_img = misc.imresize(img, 1.0 / scale, 'bicubic')
        bic_img = misc.imresize(lr_img, scale * 1.0, 'bicubic')
        lr_y = utils.rgb2ycbcr(lr_img)[:, :, 0]
        bic_ycbcr = utils.rgb2ycbcr(bic_img)
        bic_y = bic_ycbcr[:, :, 0]

        lr_y = torch.from_numpy(lr_y).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        bic_y = torch.from_numpy(bic_y).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        sr_y = model(lr_y, bic_y)
        sr_y = sr_y.clamp(0, 1)[0, 0] * 255.0
        sr_y = sr_y.cpu().detach().numpy()

        bic_ycbcr[:, :, 0] = sr_y
        res_img = utils.img_to_uint8(utils.ycbcr2rgb(bic_ycbcr))
        output_name = f.split(os.sep)[-1]

        misc.imsave(os.path.join(output_dir, output_name), res_img)

        gt_y = utils.rgb2ycbcr(img)[:, :, 0]
        psnr = utils.psnr(sr_y[scale:-scale, scale:-scale], gt_y[scale:-scale, scale:-scale])
        psnrs.append(psnr)

    avg_psnr = np.mean(psnrs)
    print('Average PSNR:', avg_psnr)







