"""Train UDnet-U (UDnet + clean-domain GAN + phase lock + physical loss).

Stage 1: UDnet as in the repo (posterior sees the pseudo-label, L_e = MSE + VGG)
         plus lam_adv * adversarial loss on a clean domain.
Stage 2: unpaired -- after --warmup_iters the posterior sees a clean image instead
         of the pseudo-label, and L_e fades out over --le_anneal iterations.

Every term can be switched off (lam_* = 0, --lock none) for ablations.
Training length is a fixed number of iterations so runs with 200/400/800 images
take the same number of steps. Resumes automatically from save_folder/last.pth.
"""
import os
import csv
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
from model_utils.data import get_training_set
from model_utils.dataset import is_image_file
from model_utils.UDnet import PerceptionLoss
from model_utils.udnet_u import (UDnetG, PatchDiscriminator, PhysicalLoss, underwater_dcp, apply_phase_lock,
                                 lsgan_d_loss, lsgan_g_loss, load_repo_weights)


def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "t", "y")


parser = argparse.ArgumentParser(description='PyTorch UDnet-U')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--gpu_mode', type=str2bool, default=True)
parser.add_argument('--data_dir', type=str, default='./dataset/', help='parent folder of the underwater images')
parser.add_argument('--data_train_dataset', type=str, default='image')
parser.add_argument('--clean_dir', type=str, default='', help='clean-domain images (needed if lam_adv > 0 or stage 2)')
parser.add_argument('--stage', type=int, default=1, choices=[1, 2])
parser.add_argument('--iters', type=int, default=40000, help='total generator updates')
parser.add_argument('--batchSize', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--data_augmentation', type=str2bool, default=True)
parser.add_argument('--lam_e', type=float, default=1.0, help='weight of L_e = MSE + VGG vs pseudo-label')
parser.add_argument('--beta_kl', type=float, default=1.0, help='weight of KL (L_m + L_s)')
parser.add_argument('--lam_adv', type=float, default=0.01, help='weight of adversarial loss; 0 = no GAN')
parser.add_argument('--adv_warmup', type=int, default=2000, help='iterations to ramp lam_adv from 0')
parser.add_argument('--lam_phys', type=float, default=0.0, help='weight of physical reconstruction loss')
parser.add_argument('--lock', type=str, default='none', choices=['none', 'full', 'low', 'high'])
parser.add_argument('--lock_radius', type=float, default=0.5)
parser.add_argument('--warmup_iters', type=int, default=0, help='stage 2: iterations trained like stage 1 first')
parser.add_argument('--le_anneal', type=int, default=0, help='stage 2: iterations for L_e to fade to 0 after warmup')
parser.add_argument('--init', type=str, default='', help='start the generator from a Train.py/author checkpoint')
parser.add_argument('--save_folder', default='weights_u/')
parser.add_argument('--save_every', type=int, default=500)
parser.add_argument('--keep_every', type=int, default=10000, help='also keep a generator snapshot every N iterations (0 = off)')
parser.add_argument('--log_every', type=int, default=100)


class CleanFolder(Dataset):
    """Unpaired clean-domain images: random crop + flips, no labels."""
    def __init__(self, folder, patch_size):
        self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if is_image_file(f))
        if not self.files:
            raise RuntimeError('no images in ' + folder)
        self.patch_size = patch_size
        self.to_tensor = ToTensor()

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        w, h = img.size
        if min(w, h) < self.patch_size:
            s = self.patch_size / min(w, h)
            img = img.resize((max(self.patch_size, round(w * s)), max(self.patch_size, round(h * s))), Image.BICUBIC)
            w, h = img.size
        x, y = random.randint(0, w - self.patch_size), random.randint(0, h - self.patch_size)
        img = img.crop((x, y, x + self.patch_size, y + self.patch_size))
        if random.random() < 0.5:
            img = ImageOps.flip(img)
        if random.random() < 0.5:
            img = ImageOps.mirror(img)
        return self.to_tensor(img)

    def __len__(self):
        return len(self.files)


def forever(loader):
    while True:
        for batch in loader:
            yield batch


def save(path, **state):
    torch.save(state, path + '.tmp')
    os.replace(path + '.tmp', path)          # no half-written checkpoint if Colab disconnects


if __name__ == '__main__':
    opt = parser.parse_args()
    if not opt.gpu_mode:
        opt.device = 'cpu'
    elif not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --gpu_mode False")
    device = torch.device(opt.device)
    print(opt)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_folder, exist_ok=True)

    use_clean = opt.lam_adv > 0 or opt.stage == 2
    if use_clean and not opt.clean_dir:
        raise SystemExit('--clean_dir is required when lam_adv > 0 or stage 2')

    print('===> Loading datasets')
    train_set = get_training_set(opt.data_dir, opt.data_train_dataset, opt.data_train_dataset,
                                 opt.patch_size, opt.data_augmentation)
    xs = forever(DataLoader(train_set, batch_size=opt.batchSize, shuffle=True,
                            num_workers=opt.threads, drop_last=True))
    ys = forever(DataLoader(CleanFolder(opt.clean_dir, opt.patch_size), batch_size=opt.batchSize,
                            shuffle=True, num_workers=opt.threads, drop_last=True)) if use_clean else None
    print('underwater images: {} | clean images: {}'.format(
        len(train_set), len(os.listdir(opt.clean_dir)) if use_clean else 0))

    G = UDnetG().to(device)
    D = PatchDiscriminator().to(device)
    P = PhysicalLoss().to(device)
    vgg = PerceptionLoss().to(device)
    mse = nn.MSELoss()
    optG = optim.Adam(list(G.parameters()) + list(P.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    optD = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    last = os.path.join(opt.save_folder, 'last.pth')
    start = 1
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D']); P.load_state_dict(ck['P'])
        optG.load_state_dict(ck['optG']); optD.load_state_dict(ck['optD'])
        start = ck['iter'] + 1
        print('===> Resumed from iteration {}'.format(ck['iter']))
    elif opt.init:
        load_repo_weights(G, opt.init)

    n_inf = sum(p.numel() for n, p in G.named_parameters()
                if not n.startswith(('decoder.po_', 'decoder.compute_z_po', 'decoder.colornet')))
    print('generator params at inference: {:.3f}M | discriminator {:.2f}M (training only)'.format(
        n_inf / 1e6, sum(p.numel() for p in D.parameters()) / 1e6))

    log_path = os.path.join(opt.save_folder, 'loss_log.csv')
    fields = ['iter', 'total', 'L_e', 'w_e', 'L_m', 'L_s', 'adv', 'D', 'phys', 'sec_per_it']
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, 'a', newline='')
    log_w = csv.DictWriter(log_f, fieldnames=fields, restval='')
    if new_log:
        log_w.writeheader()

    G.train(); D.train()
    acc, n_acc, t0 = {}, 0, time.time()
    for it in range(start, opt.iters + 1):
        inp, pseudo, _ = next(xs)
        inp, pseudo = inp.to(device), pseudo.to(device)
        clean = next(ys).to(device) if use_clean else None

        unpaired = opt.stage == 2 and it > opt.warmup_iters
        if not unpaired:
            w_e = opt.lam_e
        elif opt.le_anneal > 0:
            w_e = opt.lam_e * max(0.0, 1.0 - (it - opt.warmup_iters) / opt.le_anneal)
        else:
            w_e = 0.0
        ramp = min(1.0, it / max(1, opt.adv_warmup))

        out, L_m, L_s = G(inp, clean if unpaired else pseudo)
        out = apply_phase_lock(out, inp, opt.lock, opt.lock_radius)

        terms = {'L_m': L_m.item(), 'L_s': L_s.item(), 'w_e': w_e}
        if opt.lam_adv > 0:
            optD.zero_grad(set_to_none=True)
            loss_D = lsgan_d_loss(D(clean), D(out.detach()))
            loss_D.backward()
            optD.step()
            terms['D'] = loss_D.item()

        loss = opt.beta_kl * (L_m + L_s)
        if w_e > 0:
            L_e = mse(out, pseudo) + vgg(out, pseudo)
            loss = loss + w_e * L_e
            terms['L_e'] = L_e.item()
        if opt.lam_adv > 0:
            L_adv = lsgan_g_loss(D(out))
            loss = loss + ramp * opt.lam_adv * L_adv
            terms['adv'] = L_adv.item()
        if opt.lam_phys > 0:
            with torch.no_grad():
                t, B_inf = underwater_dcp(inp)
            L_phys, _ = P(out, inp, t, B_inf)
            loss = loss + opt.lam_phys * L_phys
            terms['phys'] = L_phys.item()
        terms['total'] = loss.item()

        optG.zero_grad(set_to_none=True)
        loss.backward()
        optG.step()

        for k, v in terms.items():
            acc[k] = acc.get(k, 0.0) + v
        n_acc += 1
        if it % opt.log_every == 0 or it == opt.iters:
            row = {k: v / n_acc for k, v in acc.items()}
            row['iter'], row['sec_per_it'] = it, (time.time() - t0) / n_acc
            log_w.writerow(row)
            log_f.flush()
            print('===> Iter {}/{} '.format(it, opt.iters) + ' '.join(
                '{}={:.4f}'.format(k, row[k]) for k in fields[1:] if k in row), flush=True)
            acc, n_acc, t0 = {}, 0, time.time()

        if it % opt.save_every == 0 or it == opt.iters:
            save(last, G=G.state_dict(), D=D.state_dict(), P=P.state_dict(), optG=optG.state_dict(),
                 optD=optD.state_dict(), iter=it, args=vars(opt))
        if opt.keep_every and it % opt.keep_every == 0 and it < opt.iters:
            # small generator-only snapshots, to check afterwards whether fewer iterations would do
            save(os.path.join(opt.save_folder, 'iter_{}.pth'.format(it)), G=G.state_dict(), iter=it, args=vars(opt))
    log_f.close()

    save(os.path.join(opt.save_folder, 'final.pth'), G=G.state_dict(), iter=opt.iters, args=vars(opt))
    print('===> Checkpoint saved to {}'.format(os.path.join(opt.save_folder, 'final.pth')))
