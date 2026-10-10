"""Check that a checkpoint really matches the current UDnet code.

Inference.py loads weights with strict=False, so any layer whose name or shape
changed is silently left at random init. Run this before trusting results:

    python CheckWeights.py --model weights/UDnet.pth
"""
import sys
import argparse
import torch
import torch.nn as nn
from model_utils.UDnet import Decoder

parser = argparse.ArgumentParser(description='Check UDnet checkpoint keys')
parser.add_argument('--model', default='weights/UDnet.pth', help='checkpoint to check')

# Keys used only by the training (posterior) branch; not needed by Inference.py
TRAIN_ONLY = ('decoder.po_', 'decoder.compute_z_po.', 'decoder.colornet.')
# Frozen VGG16 of the perceptual loss; stored in checkpoints but never trained
IGNORED = ('VGG16.', 'criterion.')


class _Net(nn.Module):
    # same key layout as mynet, without building VGG16 (avoids the download)
    def __init__(self):
        super(_Net, self).__init__()
        self.decoder = Decoder(device=torch.device('cpu'))


if __name__ == '__main__':
    opt = parser.parse_args()
    ckpt = torch.load(opt.model, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    ckpt = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in ckpt.items()}
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith(IGNORED)}
    model = _Net().state_dict()

    missing = [k for k in model if k not in ckpt]
    unexpected = [k for k in ckpt if k not in model]
    mismatch = [k for k in model if k in ckpt and ckpt[k].shape != model[k].shape]

    missing_inf = [k for k in missing if not k.startswith(TRAIN_ONLY)]
    missing_train = [k for k in missing if k.startswith(TRAIN_ONLY)]

    print('===> {}: {} model tensors, {} checkpoint tensors'.format(opt.model, len(model), len(ckpt)))
    for title, keys in [('MISSING (used at inference -> random init!)', missing_inf),
                        ('SHAPE MISMATCH', mismatch),
                        ('missing, training branch only (matters for --resume_train)', missing_train),
                        ('unexpected (in checkpoint, not in code)', unexpected)]:
        if keys:
            print('\n{} [{}]'.format(title, len(keys)))
            for k in keys:
                extra = '  ckpt {} vs model {}'.format(tuple(ckpt[k].shape), tuple(model[k].shape)) if k in mismatch else ''
                print('   ' + k + extra)

    if missing_inf or mismatch:
        print('\n===> FAIL: checkpoint does not fully match the inference path.')
        sys.exit(1)
    print('\n===> OK: every tensor used by Inference.py is loaded from the checkpoint.')
