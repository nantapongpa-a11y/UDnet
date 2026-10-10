"""UDnet-U building blocks.

underwater_dcp, phase_lock, PhysicalLoss, PatchDiscriminator and the LSGAN
losses are copied unchanged from UDnet_U_demo.ipynb. The rest connects them to
the repo's UDnet (UDnetG) and adds tools to compare phase-lock variants.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.distributions import kl
from model_utils.UDnet import Decoder


# ----------------------------------------------------------------------------
# From UDnet_U_demo.ipynb (unchanged)
# ----------------------------------------------------------------------------
def _min_pool(x, k):
    """ค่าต่ำสุดในหน้าต่าง k x k — ไม่มีพารามิเตอร์"""
    return -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k // 2)


def underwater_dcp(I, window=15, omega=0.95, t_min=0.1):
    """I : (B,3,H,W) RGB ใน [0,1]
       คืน t : (B,1,H,W) และ B_inf : (B,3)"""
    B, _, H, W = I.shape
    gb = I[:, 1:3]                                        # G,B เท่านั้น
    dark = _min_pool(gb.min(dim=1, keepdim=True).values, window)

    k = max(1, int(0.001 * H * W))                        # 0.1% ที่สว่างที่สุดใน dark channel
    idx = dark.flatten(1).topk(k, dim=1).indices
    B_inf = I.flatten(2).gather(2, idx.unsqueeze(1).expand(-1, 3, -1)).mean(-1)

    norm = I / (B_inf.view(B, 3, 1, 1) + 1e-6)
    t = 1.0 - omega * _min_pool(norm[:, 1:3].min(dim=1, keepdim=True).values, window)
    return t.clamp(t_min, 1.0), B_inf


def phase_lock(J, I, lock_radius=None):
    """เก็บ amplitude ของ J, ใช้ phase ของ I"""
    H, W = J.shape[-2:]
    Jf = torch.fft.rfft2(J.float(), norm="ortho")
    If = torch.fft.rfft2(I.float(), norm="ortho")

    pha = torch.angle(If)
    if lock_radius is not None:
        fy = torch.fft.fftfreq(H, device=J.device).abs().view(-1, 1)
        fx = torch.fft.rfftfreq(W, device=J.device).abs().view(1, -1)
        keep = ((fy ** 2 + fx ** 2).sqrt() <= lock_radius * 0.5)
        pha = torch.where(keep, pha, torch.angle(Jf))

    out = torch.fft.irfft2(torch.polar(Jf.abs(), pha), s=(H, W), norm="ortho")
    return out.clamp(0.0, 1.0)


class PhysicalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma_D = nn.Parameter(torch.ones(3))
        self.gamma_B = nn.Parameter(torch.ones(3))

    def forward(self, J, I, t, B_inf):
        gD = self.gamma_D.clamp(0.5, 2.0).view(1, 3, 1, 1)
        gB = self.gamma_B.clamp(0.5, 2.0).view(1, 3, 1, 1)
        I_recon = J * t.pow(gD) + B_inf.view(-1, 3, 1, 1) * (1.0 - t.pow(gB))
        return (I_recon - I).abs().mean(), I_recon


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [spectral_norm(nn.Conv2d(in_ch, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [spectral_norm(nn.Conv2d(ndf * prev, ndf * mult, 4, 2, 1)),
                       nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [spectral_norm(nn.Conv2d(ndf * prev, ndf * mult, 4, 1, 1)),
                   nn.InstanceNorm2d(ndf * mult), nn.LeakyReLU(0.2, True),
                   spectral_norm(nn.Conv2d(ndf * mult, 1, 4, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def lsgan_d_loss(d_real, d_fake):
    return 0.5 * ((d_real - 1.0) ** 2).mean() + 0.5 * (d_fake ** 2).mean()


def lsgan_g_loss(d_fake):
    return ((d_fake - 1.0) ** 2).mean()


# ----------------------------------------------------------------------------
# Glue to the repo's UDnet
# ----------------------------------------------------------------------------
class UDnetG(nn.Module):
    """The repo's UDnet Decoder as a generator.

    train mode: forward(x, cond) -> out, KL_mean, KL_std   (cond = what the posterior sees)
    eval mode : forward(x)       -> out                    (prior mean, same as Inference.py)
    State-dict keys are "decoder.*", the same as checkpoints saved by Train.py.
    """
    def __init__(self):
        super().__init__()
        self.decoder = Decoder(device=torch.device("cpu"))

    def forward(self, x, cond=None):
        if self.training:
            out, pr_u, pr_s, po_u, po_s = self.decoder(x, cond, training=True)
            return out, kl.kl_divergence(po_u, pr_u).mean(), kl.kl_divergence(po_s, pr_s).mean()
        return self.decoder(x, x, training=False)


def load_repo_weights(G, path):
    """Initialise UDnetG from a Train.py / author checkpoint; report what did not load."""
    sd = torch.load(path, map_location="cpu")
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k.startswith("decoder.")}
    res = G.load_state_dict(sd, strict=False)
    print("init from {}: {} tensors loaded, missing {}, unexpected {}".format(
        path, len(sd), list(res.missing_keys), list(res.unexpected_keys)))


def _low_mask(H, W, radius, device):
    # True for frequencies inside radius -- same convention as phase_lock(lock_radius=...)
    fy = torch.fft.fftfreq(H, device=device).abs().view(-1, 1)
    fx = torch.fft.rfftfreq(W, device=device).abs().view(1, -1)
    return (fy ** 2 + fx ** 2).sqrt() <= radius * 0.5


def _locked_mask(mode, H, W, radius, device):
    # True where the output phase is forced to the input phase
    if mode == "full":
        return torch.ones(H, W // 2 + 1, dtype=torch.bool, device=device)
    if mode == "low":
        return _low_mask(H, W, radius, device)
    if mode == "high":
        return ~_low_mask(H, W, radius, device)
    raise ValueError("unknown lock mode " + mode)


def apply_phase_lock(J, I, mode="full", radius=0.5):
    """none | full | low (demo's lock_radius: lock low-mid, free high) | high (lock high, free low)."""
    if mode == "none":
        return J
    if mode == "full":
        return phase_lock(J, I, None)
    if mode == "low":
        return phase_lock(J, I, radius)
    H, W = J.shape[-2:]
    Jf = torch.fft.rfft2(J.float(), norm="ortho")
    If = torch.fft.rfft2(I.float(), norm="ortho")
    pha = torch.where(_locked_mask(mode, H, W, radius, J.device), torch.angle(If), torch.angle(Jf))
    return torch.fft.irfft2(torch.polar(Jf.abs(), pha), s=(H, W), norm="ortho").clamp(0.0, 1.0)


def best_under_phase_lock(R, I, mode="full", radius=0.5):
    """Closest image to R (L2) that a phase-locked model could possibly output.

    Locked frequencies: project R onto the input's phase direction (amplitude >= 0).
    Free frequencies: copy R. This is an upper bound on PSNR for that lock mode.
    """
    H, W = I.shape[-2:]
    If = torch.fft.rfft2(I.float(), norm="ortho")
    Rf = torch.fft.rfft2(R.float(), norm="ortho")
    u = If / (If.abs() + 1e-12)
    proj = (Rf * u.conj()).real.clamp(min=0) * u
    out = torch.where(_locked_mask(mode, H, W, radius, I.device), proj, Rf)
    return torch.fft.irfft2(out, s=(H, W), norm="ortho").clamp(0.0, 1.0)
