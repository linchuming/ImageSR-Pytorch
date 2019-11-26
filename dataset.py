import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from scipy import misc
import numpy as np
import utils
import tqdm

class DIV2K_Dataset(Dataset):
    def __init__(self, root_dir='DIV2K_train_HR',
                 patch_size=96,
                 scale=4,
                 crop_num_per_image=128):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.scale = scale
        self.png_files = glob.glob(os.path.join(root_dir, '*.png'))
        self.crop_num_per_image = crop_num_per_image
        self.samples = []
        # self.crop_samples()

    def __len__(self):
        return len(self.png_files) * self.crop_num_per_image

    def crop_samples(self):
        self.samples = []
        print('cropping image ...')
        for png_file in tqdm.tqdm(self.png_files):
            img = misc.imread(png_file)
            height, width, _ = img.shape
            p = self.patch_size
            for i in range(self.crop_num_per_image):
                h = np.random.randint(height - p + 1)
                w = np.random.randint(width - p + 1)
                patch = img[h: h + p, w: w + p]
                gt = utils.rgb2ycbcr(patch)[:, :, 0]
                gt = np.float32(gt) / 255.0
                c1 = np.random.rand()
                c2 = np.random.rand()
                if c1 < 0.5:
                    gt = gt[::-1, :]
                if c2 < 0.5:
                    gt = gt[:, ::-1]
                lr = misc.imresize(gt, 1.0 / self.scale, 'bicubic', 'F')
                bic = misc.imresize(lr, self.scale * 1.0, 'bicubic', 'F')
                self.samples.append({'gt': gt.copy(), 'lr': lr.copy(), 'bic': bic.copy()})
                ## pytorch do not support [::-1] operation and using copy() can solve the problem


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.samples[idx]

if __name__ == '__main__':
    dataset = DIV2K_Dataset(crop_num_per_image=20)
    dataset.crop_samples()
    dl = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

    for i, batch_sample in enumerate(dl):
        print(i, batch_sample['gt'].size(), batch_sample['lr'].size())


    dataset.crop_samples()

    for i, batch_sample in enumerate(dl):
        print(i, batch_sample['gt'].size(), batch_sample['lr'].size())