import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.enc_4 = nn.Sequential(*vgg16.features[17:23])

        # print(self.enc_1)
        # print(self.enc_2)
        # print(self.enc_3)
        # print(self.enc_4)

        # fix the encoder
        for i in range(4):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
    

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        lossvalue = torch.norm(output-target, p=2, dim=1).mean()
        return lossvalue


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = VGG16FeatureExtractor()
        self.l2 = L2Loss()

    def forward(self, csa, csa_d, target, mask):
        vgg_gt = self.vgg(target)
        vgg_gt = vgg_gt[-1]

        mask_r = F.interpolate(mask, size=list(csa.shape)[2:4])

        lossvalue = (self.l2(csa*mask_r, vgg_gt*mask_r) + self.l2(csa_d*mask_r, vgg_gt*mask_r))/2
        return lossvalue


def calc_gan_loss(discriminator, output, target):
    y_pred_fake = discriminator(output)
    y_pred = discriminator(target)

    g_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) + 1.) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) - 1.) ** 2))/2
    d_loss = (torch.mean((y_pred - torch.mean(y_pred_fake) - 1.) ** 2) + torch.mean((y_pred_fake - torch.mean(y_pred) + 1.) ** 2))/2

    return g_loss, d_loss
