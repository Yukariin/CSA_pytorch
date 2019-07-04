import argparse

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from model import InpaintNet


def norm(x):
    return 2. * x - 1.  # [0,1] -> [-1,1]


def denorm(x):
    out = (x + 1) / 2  # [-1,1] -> [0,1]
    return out.clamp_(0, 1)


parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint', type=str,
                    help='The filename of pickle checkpoint.')


if __name__ == "__main__":
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    g_model = InpaintNet().to(device)
    g_checkpoint = torch.load(args.checkpoint, map_location=device)
    g_model.load_state_dict(g_checkpoint)
    g_model.eval()

    to_tensor = transforms.ToTensor()

    img = Image.open(args.image).convert('RGB')
    mask = Image.open(args.mask).convert('RGB')
    img = to_tensor(img)
    mask = to_tensor(mask)
    _, h, w = img.shape
    grid = 256
    img = img[:, :h//grid*grid, :w//grid*grid]
    mask = mask[:, :h//grid*grid, :w//grid*grid]
    img = img.unsqueeze_(0)  # CHW -> BCHW
    mask = mask.unsqueeze_(0)  # CHW -> BCHW
    img = norm(img)  # [0,1] -> [-1,1]
    mask = mask[:, 0:1, :, :]  #Bx3xHxW -> Bx1xHxW
    img = img * (1. - mask)
    img = img.to(device)
    mask = mask.to(device)
    print(img.shape)

    import time
    start_time = time.time()
    _, result, _, _ = g_model(img, mask)
    print("Done in %.3f seconds!" % (time.time() - start_time))

    save_image(denorm(result), args.output)
