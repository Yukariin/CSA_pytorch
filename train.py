import argparse
import os

import numpy as np
import PIL
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from data import DS, InfiniteSampler
from loss import ConsistencyLoss, calc_gan_loss
from model import InpaintNet, FeaturePatchDiscriminator, PatchDiscriminator


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='./root')
parser.add_argument('--save_dir', type=str, default='./snapshots')
parser.add_argument('--lr', type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--resume', type=int)
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.backends.cudnn.benchmark = True

if not os.path.exists(args.save_dir):
    os.makedirs('{:s}/ckpt'.format(args.save_dir))

writer = SummaryWriter()

size = (args.image_size, args.image_size)
train_tf = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
    ]),
    transforms.RandomAffine(10, (0.1,0.1), (0.8,1.2), 5, resample=PIL.Image.BILINEAR),
    transforms.ToTensor(),
])

train_set = DS(args.root, train_tf)
iterator_train = iter(data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(train_set)),
    num_workers=args.n_threads))
print(len(train_set))

g_model = InpaintNet().to(device)
fd_model = FeaturePatchDiscriminator().to(device)
pd_model = PatchDiscriminator().to(device)
l1 = nn.L1Loss().to(device)
cons = ConsistencyLoss().to(device)

start_iter = 0
g_optimizer = torch.optim.Adam(
    g_model.parameters(),
    args.lr, (args.b1, args.b2))
fd_optimizer = torch.optim.Adam(
    fd_model.parameters(),
    args.lr, (args.b1, args.b2))
pd_optimizer = torch.optim.Adam(
    pd_model.parameters(),
    args.lr, (args.b1, args.b2))

if args.resume:
    g_checkpoint = torch.load(f'{args.save_dir}/ckpt/G_{args.resume}.pth', map_location=device)
    g_model.load_state_dict(g_checkpoint)
    fd_checkpoint = torch.load(f'{args.save_dir}/ckpt/FD_{args.resume}.pth', map_location=device)
    fd_model.load_state_dict(fd_checkpoint)
    pd_checkpoint = torch.load(f'{args.save_dir}/ckpt/PD_{args.resume}.pth', map_location=device)
    pd_model.load_state_dict(pd_checkpoint)
    print('Models restored')

for i in tqdm(range(start_iter, args.max_iter)):
    img, mask = [x.to(device) for x in next(iterator_train)]
    img = 2. * img - 1. # [0,1] -> [-1,1]
    masked = img * (1. - mask)

    coarse_result, refine_result, csa, csa_d = g_model(masked, mask)

    fg_loss, fd_loss = calc_gan_loss(fd_model, refine_result, img)
    pg_loss, pd_loss = calc_gan_loss(pd_model, refine_result, img)

    recon_loss = l1(coarse_result, img) + l1(refine_result, img)
    gan_loss = fg_loss + pg_loss
    cons_loss = cons(csa, csa_d, img, mask)
    total_loss = 1*recon_loss + 0.01*cons_loss + 0.002*gan_loss
    g_optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    g_optimizer.step()

    fd_optimizer.zero_grad()
    fd_loss.backward(retain_graph=True)
    fd_optimizer.step()

    pd_optimizer.zero_grad()
    pd_loss.backward()
    pd_optimizer.step()

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        #torch.save(g_model.state_dict(), f'{args.save_dir}/ckpt/G_{i + 1}.pth')
        #torch.save(fd_model.state_dict(), f'{args.save_dir}/ckpt/FD_{i + 1}.pth')
        #torch.save(pd_model.state_dict(), f'{args.save_dir}/ckpt/PD_{i + 1}.pth')
        torch.save(g_model.state_dict(), f'{args.save_dir}/ckpt/G_10000.pth')
        torch.save(fd_model.state_dict(), f'{args.save_dir}/ckpt/FD_10000.pth')
        torch.save(pd_model.state_dict(), f'{args.save_dir}/ckpt/PD_10000.pth')

    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('g_loss/recon_loss', recon_loss.item(), i + 1)
        writer.add_scalar('g_loss/cons_loss', cons_loss.item(), i + 1)
        writer.add_scalar('g_loss/gan_loss', gan_loss.item(), i + 1)
        writer.add_scalar('g_loss/total_loss', total_loss.item(), i + 1)
        writer.add_scalar('d_loss/fd_loss', fd_loss.item(), i + 1)
        writer.add_scalar('d_loss/pd_loss', pd_loss.item(), i + 1)

    def denorm(x):
        out = (x + 1) / 2 # [-1,1] -> [0,1]
        return out.clamp_(0, 1)
    if (i + 1) % args.vis_interval == 0:
        ims = torch.cat([img, masked, coarse_result, refine_result], dim=3)
        writer.add_images('raw_masked_coarse_refine', denorm(ims), i + 1)

writer.close()
