import os
import csv
import argparse
import cv2
import numpy as np
from model_utils.dataset import is_image_file
from model_utils.metrics import psnr, ssim, gmsd, uiqm, uciqe


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "t", "y")


parser = argparse.ArgumentParser(description='Evaluate enhanced underwater images')
parser.add_argument('--pred_dir', type=str, required=True, help='folder of enhanced images (e.g. Inference.py output)')
parser.add_argument('--gt_dir', type=str, default='', help='folder of reference images; enables PSNR/SSIM/GMSD')
parser.add_argument('--nr_size', type=int, default=0, help='if >0, resize to nr_size x nr_size before UIQM/UCIQE')
parser.add_argument('--pyiqa', type=str2bool, default=False, help='also compute NIQE and MUSIQ (needs pip install pyiqa)')
parser.add_argument('--csv', type=str, default='', help='optional path to save per-image scores')


def load_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError('cannot read ' + path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def list_images(folder):
    # stem -> full path, so "x.png" and "x.jpg" still pair up
    return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in sorted(os.listdir(folder)) if is_image_file(f)}


def build_pyiqa():
    import torch
    import pyiqa
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    niqe = pyiqa.create_metric('niqe', device=device)
    musiq = pyiqa.create_metric('musiq', device=device)

    def score(img):
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        with torch.no_grad():
            return {'NIQE': niqe(t).item(), 'MUSIQ': musiq(t).item()}
    return score


if __name__ == '__main__':
    opt = parser.parse_args()
    preds = list_images(opt.pred_dir)
    gts = list_images(opt.gt_dir) if opt.gt_dir else {}
    iqa = build_pyiqa() if opt.pyiqa else None

    if gts:
        names = [n for n in preds if n in gts]
        print('===> {} pairs ({} preds without GT, {} GT without pred)'.format(
            len(names), len(set(preds) - set(gts)), len(set(gts) - set(preds))))
    else:
        names = list(preds)
        print('===> {} images (no GT: no-reference metrics only)'.format(len(names)))
    if not names:
        raise SystemExit('No images to evaluate.')

    rows = []
    resized = 0
    for n in names:
        pred = load_rgb(preds[n])
        row = {'name': n}

        if gts:
            gt = load_rgb(gts[n])
            if pred.shape != gt.shape:
                # Inference.py crops sizes to a multiple of 8; compare at GT resolution
                pred_fr = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
                resized += 1
            else:
                pred_fr = pred
            row['PSNR'] = psnr(pred_fr, gt)
            row['SSIM'] = ssim(pred_fr, gt)
            row['GMSD'] = gmsd(pred_fr, gt)

        nr = cv2.resize(pred, (opt.nr_size, opt.nr_size), interpolation=cv2.INTER_CUBIC) if opt.nr_size > 0 else pred
        row['UIQM'] = uiqm(nr)
        # UCIQE: common Python port (comparable to papers) and the a/b-centred fix,
        # which stays in range on dark images where the common port blows up
        row['UCIQE'] = uciqe(nr)
        row['UCIQEc'] = uciqe(nr, centered_ab=True)
        if iqa:
            row.update(iqa(pred))

        rows.append(row)
        print('{:<30s} '.format(n) + '  '.join('{}={:.4f}'.format(k, v) for k, v in row.items() if k != 'name'))

    if resized:
        print('===> {} predictions resized to GT size for full-reference metrics'.format(resized))

    keys = [k for k in rows[0] if k != 'name']
    better = {'PSNR': '+', 'SSIM': '+', 'GMSD': '-', 'UIQM': '+', 'UCIQE': '+', 'UCIQEc': '+', 'NIQE': '-', 'MUSIQ': '+'}
    print('\n===> Mean over {} images  (+ higher is better, - lower is better)'.format(len(rows)))
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        print('  {:<6s}({}) {:.4f}  (std {:.4f})'.format(k, better.get(k, '?'), vals.mean(), vals.std()))

    if opt.csv:
        with open(opt.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name'] + keys)
            w.writeheader()
            w.writerows(rows)
        print('===> Per-image scores saved to {}'.format(opt.csv))
