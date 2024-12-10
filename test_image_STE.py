import os
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.dataloader import ErasingData
from models.ensexam import EnsExamNet


parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--savePath', type=str, default='./results/sn_tv/')
parser.add_argument('--compress', type=bool, required=True, action=argparse.BooleanOptionalAction)
parser.add_argument('--dilation', type=bool, required=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
save_path = args.savePath
result_with_mask = save_path + 'WithMaskOutput/'
result_with_stroke_mask = save_path + 'WithStrokeMaskOutput/'
result_straight = save_path + 'StrOutput/'
result_stroke = save_path + 'StrokeOutput/'
result_mm = save_path + 'MaskOutput/'

if os.path.exists(save_path):
    shutil.rmtree(save_path)

for x in [result_with_mask, result_with_stroke_mask, result_straight, result_stroke, result_mm]:
    os.makedirs(x)


erase_data = ErasingData(dataRoot, loadSize, training=False)
erase_data = DataLoader(erase_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False)


netG = EnsExamNet()

netG.load_state_dict(torch.load(args.pretrained))

#
if cuda:
    netG = netG.cuda()

for param in netG.parameters():
    param.requires_grad = False

print('OK!')

import time
start = time.time()
netG.eval()
for original_img, _, _, _, path in erase_data:
    if cuda: original_img = original_img.cuda()
    x_o1, x_o2, x_o3, output, mm, stroke_mm = netG(original_img)
    output = output.data.cpu()
    original_img = original_img.data.cpu()
    print(f"dilation = {args.dilation}", path)
    if args.dilation:
        dilation_size = 13
        assert dilation_size % 2 == 1
        padding = dilation_size // 2
        stroke_mm: torch.Tensor = torch.clip(stroke_mm.data.cpu(), 0.0, 1.0)
        kernel = torch.ones((3, 3, dilation_size, dilation_size))
        stroke_mm = F.conv2d(stroke_mm, kernel, padding=padding)
        stroke_mm = torch.threshold(stroke_mm, 0.5, 0)

    mm = torch.clip(mm.data.cpu(), 0.0, 1.0)
    stroke_mm = torch.clip(stroke_mm.data.cpu(), 0.0, 1.0)

    output_with_stroke_mask = output * stroke_mm + original_img * (1 - stroke_mm)
    output_with_mask = output * mm + original_img * (1 - mm)

    p = path[0]
    if not args.compress: p = p.replace('jpg', 'png')
    save_image(output_with_mask, result_with_mask+p)
    save_image(output_with_stroke_mask, result_with_stroke_mask+p)
    save_image(output, result_straight+p)
    save_image(stroke_mm, result_stroke+p)
    save_image(mm, result_mm+p)
