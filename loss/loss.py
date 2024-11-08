import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from models.discriminator import Discriminator_STE
from PIL import Image
import numpy as np


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def visual(image):
    im = image.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()


def dice_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss


class LossWithGAN_STE(nn.Module):
    def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()
        self.l1_noreduce = nn.L1Loss(reduction='none')
        self.extractor = extractor
        self.discriminator = Discriminator_STE(3)  ## local_global sn patch gan
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        self.lamda = Lamda
        self.writer = SummaryWriter(logPath)

    def forward(self, input_, gt, mask_gt, stroke_gt, x_o1, x_o2, x_o3, output, mask_model, stroke_model, count, epoch):
        self.discriminator.zero_grad()
        D_gt = self.discriminator(gt, mask_gt)
        D_gt = D_gt.mean().sum() * -1
        D_model = self.discriminator(output, mask_gt)
        D_model = D_model.mean().sum() * 1
        D_loss = torch.mean(F.relu(1. + D_gt)) + torch.mean(F.relu(1. + D_model))  # SN-patch-GAN loss
        D_model = -torch.mean(D_model)  # SN-Patch-GAN loss
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)

        self.writer.add_scalar('LossD/Discriminator loss', D_loss.item(), count)

        output_comp = mask_gt * input_ + (1 - mask_gt) * output

        mask_block_loss = dice_loss(mask_model, 1 - mask_gt)
        mask_block_loss *= 0.4
        # --- MSR Reconstruction loss ---
        masks_a = F.interpolate(mask_gt, scale_factor=0.25)
        masks_b = F.interpolate(mask_gt, scale_factor=0.5)
        imgs1 = F.interpolate(gt, scale_factor=0.25)
        imgs2 = F.interpolate(gt, scale_factor=0.5)
        # Eq. 9 in paper is wrong?
        recon_loss = (
            8 * self.l1((1 - mask_gt) * x_o3, (1 - mask_gt) * gt)
              + 0.8 * self.l1(mask_gt * x_o3, mask_gt * gt) +
            6 * self.l1((1 - masks_b) * x_o2, (1 - masks_b) * imgs2)
              + 0.8 * self.l1(masks_b * x_o2, masks_b * imgs2) +
            5 * self.l1((1 - masks_a) * x_o1, (1 - masks_a) * imgs1)
              + 0.8 * self.l1(masks_a * x_o1, masks_a * imgs1) +
            10 * self.l1((1 - mask_gt) * output, (1 - mask_gt) * gt)
              + 2 * self.l1(mask_gt * output, mask_gt * gt)
        )

        # --- Stroke SN Loss ---
        stroke_loss = self.l1_noreduce(stroke_gt, stroke_model)
        stroke_gt: torch.Tensor # B, C, W, H
        with torch.no_grad():
            p_positive = stroke_gt.sum([1, 2, 3]) / (stroke_gt.shape[2] * stroke_gt.shape[3])
            p_positive = torch.clamp(p_positive, 3e-3).reshape(-1, 1, 1, 1) # B, C, W, H
            p_pixels = (1 - p_positive) * (1 - stroke_gt) + p_positive * stroke_gt
        stroke_loss = (stroke_loss / p_pixels).mean()


        # --- Perceptual Loss ---
        perceptual_loss = 0.0
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)
        for i in range(3):
            perceptual_loss += self.l1(feat_output[i], feat_gt[i])
            perceptual_loss += self.l1(feat_output_comp[i], feat_gt[i])
        perceptual_loss *= 0.01 # TODO: Check paper

        # --- Style Loss ---
        style_loss = 0.0
        for i in range(3):
            style_loss += self.l1(gram_matrix(feat_output[i]),
                                       gram_matrix(feat_gt[i]))
            style_loss += self.l1(gram_matrix(feat_output_comp[i]),
                                       gram_matrix(feat_gt[i]))
        style_loss *= 120

        self.writer.add_scalar('LossG/msr recon loss', recon_loss.item(), count)
        self.writer.add_scalar('LossPrc/Perceptual loss', perceptual_loss.item(), count)
        self.writer.add_scalar('LossStyle/style loss', style_loss.item(), count)
        self.writer.add_scalar('LossStyle/mask loss', mask_block_loss.item(), count)
        self.writer.add_scalar('LossStyle/stroke loss', stroke_loss.item(), count)

        GLoss = recon_loss + perceptual_loss + style_loss + 0.1 * D_model + mask_block_loss + stroke_loss
        self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)
        return GLoss.sum()
