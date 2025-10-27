import argparse
import csv
import math
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset (CSV-based)
# =========================

class CSVDigitsDataset(Dataset):
    """
    A dataset that reads (filename, label) from a CSV file under a given image folder.
    - image_folder: folder containing images
    - csv_path: CSV with header, columns include 'filename' and either 'label' or 'digit'
    - domain_idx: 0 for mnistm, 1 for svhn
    - image_size: final size (28)
    """
    def __init__(self, image_folder: str, csv_path: str, domain_idx: int, image_size: int = 28):
        self.image_folder = Path(image_folder)
        self.csv_path = Path(csv_path)
        self.domain_idx = int(domain_idx)

        if not self.image_folder.exists():
            raise FileNotFoundError(f"[Dataset] image folder not found: {self.image_folder}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"[Dataset] csv not found: {self.csv_path}")

        # read CSV
        self.items: List[Tuple[str, int]] = []
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # detect label column
            cols = [c.strip().lower() for c in reader.fieldnames or []]
            if 'image_name' not in cols:
                raise ValueError(f"[Dataset] CSV must contain 'filename' column. Found: {cols}")
            if 'label' in cols:
                label_key = 'label'
            elif 'digit' in cols:
                label_key = 'digit'
            else:
                raise ValueError(f"[Dataset] CSV must contain a 'label' or 'digit' column. Found: {cols}")

            fname_key = 'image_name'
            for row in reader:
                fn = row[fname_key].strip()
                lab = int(row[label_key])
                if lab < 0 or lab > 9:
                    raise ValueError(f"[Dataset] label out of range [0..9]: {lab} in {self.csv_path}")
                self.items.append((fn, lab))

        if len(self.items) == 0:
            raise ValueError(f"[Dataset] Empty CSV: {self.csv_path}")

        # transforms (ensure RGB, 28x28, [-1,1])
        self.tx = transforms.Compose([
            transforms.Lambda(lambda p: p.convert('RGB') if p.mode != 'RGB' else p),
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, digit = self.items[idx]
        img_path = self.image_folder / fname
        if not img_path.exists():
            raise FileNotFoundError(f"[Dataset] image not found: {img_path}")
        img = Image.open(img_path)
        img = self.tx(img)
        domain = self.domain_idx
        return img, digit, domain


# =========================
# Diffusion Schedules
# =========================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


# =========================
# Model: Conditional U-Net (28x28)
# (identical to train.py so that checkpoints are compatible)
# =========================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device) * (-1.0))
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, cond_vec):
        scale, shift = self.cond(cond_vec).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = self.conv2(self.dropout(self.act2(h)))
        return h + self.skip(x)

class SelfAttention2d(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads
    def forward(self, x):
        b, c, h, w = x.shape
        hds = self.num_heads
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(b, hds, c // hds, h * w)
        k = self.k(x_norm).reshape(b, hds, c // hds, h * w)
        v = self.v(x_norm).reshape(b, hds, c // hds, h * w)
        attn = torch.softmax((q.transpose(2, 3) @ k) / math.sqrt(c // hds), dim=-1)
        out = (attn @ v.transpose(2, 3)).transpose(2, 3)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return out + x

class CondUNetSmall(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, dim_t=128, dim_digit=64, dim_domain=16, use_attention=True):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalPosEmb(dim_t),
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t)
        )
        self.digit_emb = nn.Embedding(11, dim_digit)    # 0..9 + null(10)
        self.domain_emb = nn.Embedding(3, dim_domain)   # 0,1 + null(2)
        cond_dim = dim_t + dim_digit + dim_domain

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.rb1 = ResBlock(base_ch, base_ch, cond_dim)
        self.rb2 = ResBlock(base_ch, base_ch, cond_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1)  # 28->14

        self.rb3 = ResBlock(base_ch * 2, base_ch * 2, cond_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1)  # 14->7

        self.rb4 = ResBlock(base_ch * 4, base_ch * 4, cond_dim)
        self.mid_attn = SelfAttention2d(base_ch * 4) if use_attention else nn.Identity()
        self.rb5 = ResBlock(base_ch * 4, base_ch * 4, cond_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1)  # 7->14
        self.rb6 = ResBlock(base_ch * 2 + base_ch * 2, base_ch * 2, cond_dim)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, 4, stride=2, padding=1)      # 14->28
        self.rb7 = ResBlock(base_ch + base_ch, base_ch, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, digit, domain):
        t_emb = self.t_embed(t)
        d_emb = self.digit_emb(digit)
        dm_emb = self.domain_emb(domain)
        cond = torch.cat([t_emb, d_emb, dm_emb], dim=1)

        x0 = self.in_conv(x)
        x1 = self.rb1(x0, cond)
        x1 = self.rb2(x1, cond)
        x2 = self.down1(x1)

        x2 = self.rb3(x2, cond)
        x3 = self.down2(x2)

        x3 = self.rb4(x3, cond)
        x3 = self.mid_attn(x3)
        x3 = self.rb5(x3, cond)

        u1 = self.up1(x3)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.rb6(u1, cond)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.rb7(u2, cond)

        out = self.out_conv(self.out_act(self.out_norm(u2)))
        return out


# =========================
# EMA
# =========================

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register(model)
    def register(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =========================
# Diffusion helpers
# =========================

def prepare_noise_schedule(T: int, device: torch.device):
    betas = cosine_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
    }

def sample_timesteps(bs: int, T: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, T, (bs,), device=device, dtype=torch.long)

def loss_eps_pred(model: nn.Module, x0: torch.Tensor, digit: torch.Tensor, domain: torch.Tensor,
                  sched: dict, T: int, p_uncond: float):
    device = x0.device
    b = x0.size(0)
    t = sample_timesteps(b, T, device)
    eps = torch.randn_like(x0)
    sqrt_ab = sched["sqrt_alphas_cumprod"][t].view(b, 1, 1, 1)
    sqrt_om = sched["sqrt_one_minus_alphas_cumprod"][t].view(b, 1, 1, 1)
    xt = sqrt_ab * x0 + sqrt_om * eps

    if p_uncond > 0.0:
        drop = (torch.rand(b, device=device) < p_uncond)
        digit = digit.clone()
        domain = domain.clone()
        digit[drop] = 10
        domain[drop] = 2

    eps_hat = model(xt, t, digit, domain)
    return F.mse_loss(eps_hat, eps)


# =========================
# Train
# =========================

def build_loader(args):
    datasets = []
    if args.mnistm_imgdir and args.mnistm_csv:
        datasets.append(CSVDigitsDataset(
            image_folder=str(Path(args.data_root) / args.mnistm_imgdir),
            csv_path=str(Path(args.data_root) / args.mnistm_csv),
            domain_idx=0, image_size=args.image_size
        ))
    if args.svhn_imgdir and args.svhn_csv:
        datasets.append(CSVDigitsDataset(
            image_folder=str(Path(args.data_root) / args.svhn_imgdir),
            csv_path=str(Path(args.data_root) / args.svhn_csv),
            domain_idx=1, image_size=args.image_size
        ))
    if not datasets:
        raise ValueError("Please provide at least one dataset via --mnistm_imgdir/csv and/or --svhn_imgdir/csv")
    ds = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    return loader

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    loader = build_loader(args)

    model = CondUNetSmall(
        in_ch=3, base_ch=args.base_channels, dim_t=args.time_dim,
        dim_digit=args.digit_dim, dim_domain=args.domain_dim,
        use_attention=not args.no_attention
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.999), weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    ema = EMA(model, decay=args.ema_decay)

    sched = prepare_noise_schedule(args.num_steps, device)

    total_steps = 0
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "p1_ddpm_cond_unet.pt"

    model.train()
    for epoch in range(1, args.epochs + 1):
        for imgs, digits, domains in loader:
            imgs = imgs.to(device, non_blocking=True)
            digits = digits.to(device, non_blocking=True).long()
            domains = domains.to(device, non_blocking=True).long()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                loss = loss_eps_pred(model, imgs, digits, domains, sched, args.num_steps, args.p_uncond)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ema.update(model)
            total_steps += 1
            if total_steps % args.log_every == 0:
                print(f"[epoch {epoch}] step {total_steps} | loss={loss.item():.4f}", flush=True)

            if total_steps % args.save_every == 0:
                ema.apply_shadow(model)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "steps": total_steps, "args": vars(args)}, ckpt_path)
                ema.restore(model)
                print(f"[ckpt] saved to {ckpt_path} at step {total_steps}", flush=True)

        # end epoch save
        ema.apply_shadow(model)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "steps": total_steps, "args": vars(args)}, ckpt_path)
        ema.restore(model)
        print(f"[ckpt] saved (end epoch {epoch}) to {ckpt_path}", flush=True)

    print("[done] training completed.")

def parse_args():
    p = argparse.ArgumentParser(description="P1 Training (CSV-based) - Conditional DDPM on MNIST-M & SVHN")
    # data root + per-dataset paths (relative to data_root or absolute)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--mnistm_imgdir", type=str, default="", help="e.g., mnistm/images")
    p.add_argument("--mnistm_csv", type=str, default="", help="e.g., mnistm/labels.csv")
    p.add_argument("--svhn_imgdir", type=str, default="", help="e.g., svhn/images")
    p.add_argument("--svhn_csv", type=str, default="", help="e.g., svhn/labels.csv")
    p.add_argument("--image_size", type=int, default=28)
    p.add_argument("--workers", type=int, default=4)

    # model
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--time_dim", type=int, default=128)
    p.add_argument("--digit_dim", type=int, default=64)
    p.add_argument("--domain_dim", type=int, default=16)
    p.add_argument("--no_attention", action="store_true")

    # diffusion
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--p_uncond", type=float, default=0.15)

    # train
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2025)

    # ckpt
    p.add_argument("--save_dir", type=str, default="checkpoints_csv")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
