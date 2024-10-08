'''
Image Similarity Measures for Medical Image Registration

Article:
    A Survey on Deep Learning in Medical Image Registration: New Technologies, Uncertainty, Evaluation Metrics, and Beyond
    J Chen*, Y Liu*, S Wei*, Z Bian, S Subramanian, A Carass, JL Prince, Y Du
    arXiv preprint arXiv:2307.15615
    *: Contributed equally

Contact:
    Junyu Chen
    jchen245@jhmi.edu
    Johns Hopkins University
'''

import torch, math
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss

    Adapted from VoxelMorph.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class MIND_loss(torch.nn.Module):
    """
    Modality independent neighbourhood descriptor for multi-modal deformable registration
    by Heinrich et al. 2012
    https://doi.org/10.1016/j.media.2012.05.008

    Adapted from https://github.com/multimodallearning
    """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

class MutualInformation(torch.nn.Module):
    """
    Mutual Information

    Adapted from VoxelMorph
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information based on non-overlapping patches

    Adapted from VoxelMorph
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)

class NCC_gauss(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss via Gaussian kernel

    More robust to intensity variations??

    Implemented by Junyu Chen, jchen245@jhmi.edu
    TODO: Under testing
    """

    def __init__(self, win=9):
        super(NCC_gauss, self).__init__()
        self.win = [win]*3
        self.filt = self.create_window_3D(win, 1).to("cuda")

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window_3D(self, window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t())
        _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                      window_size).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
        return window

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # compute filters
        pad_no = math.floor(self.win[0] / 2)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        mu1 = conv_fn(Ii, self.filt, padding=pad_no)
        mu2 = conv_fn(Ji, self.filt, padding=pad_no)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = conv_fn(Ii * Ii, self.filt, padding=pad_no) - mu1_sq
        sigma2_sq = conv_fn(Ji * Ji, self.filt, padding=pad_no) - mu2_sq
        sigma12 = conv_fn(Ii * Ji, self.filt, padding=pad_no) - mu1_mu2

        cc = (sigma12 * sigma12 + 1e-5)/(sigma1_sq * sigma2_sq + 1e-5)
        return 1-torch.mean(cc)

class PCC(torch.nn.Module):
    '''
    Pearson's correlation coefficient

    Similar to MSE but has a range [-1, 1]

    Implemented by Junyu Chen, jchen245@jhmi.edu
    '''
    def __init__(self,):
        super(PCC, self).__init__()

    def pcc(self, y_true, y_pred):
        A_bar = torch.mean(y_pred, dim=[1, 2, 3, 4], keepdim=True)
        B_bar = torch.mean(y_true, dim=[1, 2, 3, 4], keepdim=True)
        top = torch.mean((y_pred - A_bar) * (y_true - B_bar), dim=[1, 2, 3, 4], keepdim=True)
        bottom = torch.sqrt(torch.mean((y_pred - A_bar) ** 2, dim=[1, 2, 3, 4], keepdim=True) * torch.mean((y_true - B_bar) ** 2, dim=[1, 2, 3, 4], keepdim=True))
        return torch.mean(top/bottom)

    def forward(self, I, J):
        return (1-self.pcc(I,J))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    '''
    2D Structural Similarity Measure (SSIM)
    '''
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    '''
    3D Structural Similarity Measure (SSIM)
    '''
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

class CorrRatio(torch.nn.Module):
    """
    Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu

    Original Paper:
    The Correlation Ratio as a New Similarity Measure for Multimodal Image Registration
    by Roche et al. 1998
    https://link.springer.com/chapter/10.1007/BFb0056301

    Correlation Ratio as a loss function for deep learning:
    1. Chen, Junyu, et al. "Unsupervised Learning of Multi-modal Affine Registration for PET/CT." 
    2024 IEEE Nuclear Science Symposium (NSS), Medical Imaging Conference (MIC) and Room Temperature Semiconductor Detector Conference (RTSD). IEEE, 2024.
    """

    def __init__(self, bins=32, sigma_ratio=1):
        super(CorrRatio, self).__init__()
        self.num_bins = bins
        bin_centers = np.linspace(0, 1, num=bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, bins), requires_grad=False).cuda().view(1, 1, bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))#torch.exp(-0.5 * (diff ** 2) / (sigma ** 2))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape
        y_flat = Y.reshape(B, C, -1)  # Flatten spatial dimensions
        x_flat = X.reshape(B, C, -1)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B, C, 1, H*W*D]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B, C, 1, H*W*D]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True) # [B, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / (torch.sum(
            bin_counts, dim=2)+1e-5)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean()/3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/2

class LocalCorrRatio(torch.nn.Module):
    """
    Localized Correlation Ratio based on Parzen window
    Implemented by Junyu Chen, jchen245@jhmi.edu

    Correlation Ratio as a loss function for deep learning:
    1. Chen, Junyu, et al. "Unsupervised Learning of Multi-modal Affine Registration for PET/CT." 
    2024 IEEE Nuclear Science Symposium (NSS), Medical Imaging Conference (MIC) and Room Temperature Semiconductor Detector Conference (RTSD). IEEE, 2024.
    """

    def __init__(self, bins=32, sigma_ratio=1, win=9):
        super(LocalCorrRatio, self).__init__()
        self.num_bins = bins
        bin_centers = np.linspace(0, 1, num=bins)
        self.vol_bin_centers = Variable(torch.linspace(0, 1, bins), requires_grad=False).cuda().view(1, 1, bins, 1)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 2 / (2 * sigma ** 2)
        self.win = win

    def gaussian_kernel(self, diff, preterm):
        return torch.exp(- preterm * torch.square(diff))

    def correlation_ratio(self, X, Y):
        B, C, H, W, D = Y.shape

        h_r = -H % self.win
        w_r = -W % self.win
        d_r = -D % self.win
        padding = (d_r // 2, d_r - d_r // 2, w_r // 2, w_r - w_r // 2, h_r // 2, h_r - h_r // 2, 0, 0, 0, 0)
        X = F.pad(X, padding, "constant", 0)
        Y = F.pad(Y, padding, "constant", 0)

        B, C, H, W, D = Y.shape
        num_patch = (H // self.win) * (W // self.win) * (D // self.win)
        x_patch = torch.reshape(X, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        x_flat = x_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B*num_patch, C, self.win ** 3)

        y_patch = torch.reshape(Y, (B, C, H // self.win, self.win, W // self.win, self.win, D // self.win, self.win))
        y_flat = y_patch.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B * num_patch, C, self.win ** 3)

        bins = self.vol_bin_centers

        # Calculate distances from each pixel to each bin
        y_expanded = y_flat.unsqueeze(2)  # [B*num_patch, C, 1, win**3]
        diff = y_expanded - bins  # Broadcasted subtraction

        # Apply Parzen window approximation
        weights = self.gaussian_kernel(diff, preterm=self.preterm)
        weights_norm = weights / (torch.sum(weights, dim=-1, keepdim=True)+1e-5)
        # Compute weighted mean intensity in y_pred for each bin
        x_flat_expanded = x_flat.unsqueeze(2)  # Shape: [B*num_patch, C, 1, win**3]
        mean_intensities = torch.sum(weights_norm * x_flat_expanded, dim=3)  # conditional mean, [B*num_patch, C, bin]
        bin_counts = torch.sum(weights, dim=3)
        # mean_intensities = weighted_sums / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero

        # Compute total mean of y_pred
        total_mean = torch.mean(x_flat, dim=2, keepdim=True)  # [B*num_patch, C, 1]

        # Between-group variance
        between_group_variance = torch.sum(bin_counts * (mean_intensities - total_mean) ** 2, dim=2) / torch.sum(
            bin_counts, dim=2)

        # Total variance
        total_variance = torch.var(x_flat, dim=2)

        # Correlation ratio
        eta_square = between_group_variance / (total_variance + 1e-5)

        return eta_square.mean() / 3

    def forward(self, y_true, y_pred):
        CR = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true) #make it symmetric

        shift_size = self.win//2
        y_true = torch.roll(y_true, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))
        y_pred = torch.roll(y_pred, shifts=(-shift_size, -shift_size, -shift_size), dims=(2, 3, 4))

        CR_shifted = self.correlation_ratio(y_true, y_pred) + self.correlation_ratio(y_pred, y_true)
        return -CR/4 - CR_shifted/4
