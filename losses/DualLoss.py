import torch
import torch.nn.functional as F
# from torchvision.transforms.functional import pad
import numpy as np
import cv2


def compute_grad_mag(E, cuda=True):
    E_ = convTri(E, 4, cuda)        # 
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    mag = torch.sqrt(torch.mul(Ox, Ox) + torch.mul(Oy, Oy) + 1e-6)
    mag = mag / mag.max()

    return mag


def convTri(input, r, cuda=True):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is   用二维三角形滤波器对图像进行卷积
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius  整数过滤半径
    :param cuda: move the kernel to gpu
    """
    n, c, h, w = input.shape
    # return input
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    kernel = torch.Tensor([f]) / (r + 1) ** 2
    if type(cuda) is int:
        if cuda != -1:
            kernel = kernel.cuda(device=cuda)
    else:
        if cuda is True:
            kernel = kernel.cuda()

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode='replicate')
    input_ = F.pad(input_, (r, r, 0, 0), mode='reflect')
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode='replicate')
    input_ = F.pad(input_, (0, 0, r, r), mode='reflect')
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(input_, kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), padding=0, groups=c)
    output = F.conv2d(output, kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), padding=0, groups=c)

    return output


def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    x, y = gradient_central_diff(input, cuda)
    return x, y


def gradient_central_diff(input, cuda):
    return input, input

# def gradient_central_diff(input, cuda):
#     return input, input
#     kernel = [[1, 0, -1]]
#     kernel_t = 0.5 * torch.Tensor(kernel) * -1.  # pytorch implements correlation instead of conv
#     if type(cuda) is int:
#         if cuda != -1:
#             kernel_t = kernel_t.cuda(device=cuda)
#     else:
#         if cuda is True:
#             kernel_t = kernel_t.cuda()
#     n, c, h, w = input.shape

#     x = conv2d_same(input, kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
#     y = conv2d_same(input, kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]), c)
#     return x, y



class DualLoss(torch.nn.Module):
    def __init__(self, cuda=False):
        super(DualLoss, self).__init__()
        self._cuda = cuda
        return

    def forward(self, input_logits, gts):
        """
        :param input_logits: NxCxHxW
        :param gt_semantic_masks: NxCxHxW
        :return: final loss
        """
        N, C, H, W = input_logits.shape
        th = 1e-8  # 1e-10
        eps = 1e-10

        g = input_logits
        g = compute_grad_mag(g)
        g_hat = compute_grad_mag(gts)


        # in_cpu = input_logits[0,:,:,:].permute((1,2,0)).cpu().detach().numpy()
        # cv2.imwrite('/gemini/code/segmentation0329/figure/in_cpu2.tif', in_cpu)
        # gt_cpu = gts[0,:,:,:].permute((1,2,0)).cpu().detach().numpy()
        # cv2.imwrite('/gemini/code/segmentation0329/figure/gt_cpu2.tif', gt_cpu)

        # g_cpu = g[0,:,:,:].permute((1,2,0)).cpu().detach().numpy()
        # cv2.imwrite('/gemini/code/segmentation0329/figure/g_cpu2.tif', g_cpu)
        # g_hat_cpu = g_hat[0,:,:,:].permute((1,2,0)).cpu().detach().numpy()
        # cv2.imwrite('/gemini/code/segmentation0329/figure/g_hat_cpu2.tif', g_hat_cpu)


        g = g.view(N, -1)
        g_hat = g_hat.view(N, -1)
        loss_ewise = F.l1_loss(g, g_hat, reduction='none', reduce=False)

        p_plus_g_mask = (g >= th).detach().float()
        loss_p_plus_g = torch.sum(loss_ewise * p_plus_g_mask) / (torch.sum(p_plus_g_mask) + eps)

        p_plus_g_hat_mask = (g_hat >= th).detach().float()
        loss_p_plus_g_hat = torch.sum(loss_ewise * p_plus_g_hat_mask) / (torch.sum(p_plus_g_hat_mask) + eps)

        total_loss = 0.5 * loss_p_plus_g + 0.5 * loss_p_plus_g_hat

        return total_loss