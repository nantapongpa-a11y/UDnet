"""Enhance images with a TrainU.py checkpoint.

Applies the same phase lock the model was trained with -- the lock is part of
the model's output, so it must also run at inference.
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from model_utils.dataset import is_image_file
from model_utils.udnet_u import UDnetG, apply_phase_lock


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "t", "y")


parser = argparse.ArgumentParser(description='PyTorch UDnet-U inference')
parser.add_argument('--model', required=True, help='final.pth or last.pth from TrainU.py')
parser.add_argument('--input_dir', required=True)
parser.add_argument('--output_dir', required=True)
parser.add_argument('--gpu_mode', type=str2bool, default=True)
parser.add_argument('--device', type=str, default='cuda:0')

if __name__ == '__main__':
    opt = parser.parse_args()
    device = torch.device(opt.device if opt.gpu_mode else 'cpu')
    ck = torch.load(opt.model, map_location='cpu')
    a = ck['args']
    G = UDnetG()
    G.load_state_dict(ck['G'])
    G.to(device).eval()
    print('model {} (iter {}) | lock={} radius={}'.format(opt.model, ck['iter'], a['lock'], a['lock_radius']))

    os.makedirs(opt.output_dir, exist_ok=True)
    to_tensor = ToTensor()
    files = sorted(f for f in os.listdir(opt.input_dir) if is_image_file(f))
    with torch.no_grad():
        for f in files:
            img = Image.open(os.path.join(opt.input_dir, f)).convert('RGB')
            w, h = img.size
            img = img.resize((w - w % 8, h - h % 8))          # same as Inference.py
            x = to_tensor(img).unsqueeze(0).to(device)
            out = apply_phase_lock(G(x), x, a['lock'], a['lock_radius'])
            out = (out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            Image.fromarray(out).save(os.path.join(opt.output_dir, os.path.splitext(f)[0] + '.png'))
    print('===> {} images saved to {}'.format(len(files), opt.output_dir))
