import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, scale=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.scale = scale

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = x + self.scale * out
        return out

class EDSR(nn.Module):
    def __init__(self, upscale=4):
        super(EDSR, self).__init__()
        self.upscale = upscale
        self.conv1 = nn.Conv2d(1, 256, 3, 1, padding=1)

        self.resblocks = nn.Sequential()
        for i in range(16):
            self.resblocks.add_module('res2d_%d' % i, ResBlock(256))

        self.conv2 = nn.Conv2d(256, 256, 3, 1, padding=1)

        self.deconv = nn.ConvTranspose2d(256, 1, 2 * self.upscale,
                                         stride=self.upscale, padding=2)


    def forward(self, x, bic):
        out = self.conv1(x)
        out = F.relu(out)
        x1 = out
        out = self.resblocks(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = out + x1
        out = self.deconv(out)
        return out + bic

if __name__ == '__main__':
    import numpy as np
    from scipy import misc

    a = np.random.rand(100, 100)
    a_bic = misc.imresize(a, 4.0, 'bicubic', mode='F')
    #
    # t = torch.from_numpy(a)
    # t = t.view([1, 1, 100, 100])
    # t_bic = F.interpolate(t, scale_factor=4, mode='bicubic', align_corners=False)
    # t_bic = t_bic.view([400, 400]).numpy()
    #
    # print(t_bic.max(), t_bic.min(), a_bic.max(), a_bic.min())
    # print(np.sum(np.abs(a_bic - t_bic)))
    ## Can not use F.interpolate to implement bicubic method ##

    a = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float()
    a_bic = torch.from_numpy(a_bic).unsqueeze(0).unsqueeze(0).float()
    model = EDSR().cuda()
    a = a.cuda()
    a_bic = a_bic.cuda()
    out = model(a, a_bic)
    print(out.size())
    out = out.permute(0, 2, 3, 1)
    print(out.size())