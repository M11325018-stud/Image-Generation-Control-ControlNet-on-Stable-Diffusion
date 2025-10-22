import argparse
import math
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import time  # <<< NEW

# ------------------ Utils ------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_image(x: torch.Tensor) -> torch.Tensor:
    # expects x in [-1,1]; returns [0,1]
    return (x.clamp(-1, 1) + 1.0) * 0.5

# ------------------ Schedule ------------------
def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-5, 0.999)

# ------------------ Tiny Conditional U-Net (must match training) ------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
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

# ------------------ Sampler ------------------
@torch.no_grad()
def generate_p1_images(model, out_root: Path, num_steps: int, sample_steps: int,
                       cfg_w: float, seed: int, device: torch.device,
                       trace_digits=None, trace_count: int = 6, method: str = "auto"):
    out_root = Path(out_root)
    (out_root / "mnistm").mkdir(parents=True, exist_ok=True)
    (out_root / "svhn").mkdir(parents=True, exist_ok=True)
    trace_dir = out_root / "trace"
    if trace_digits:
        trace_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    model.eval().to(device)

    betas = cosine_beta_schedule(num_steps).to(device)              # [T]
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)                   # a_bar[t]

    use_ddim = (sample_steps < num_steps) if method == "auto" else (method.lower() == "ddim")

    if use_ddim:
        idxs = torch.linspace(0, num_steps - 1, sample_steps, dtype=torch.long, device=device)
        time_seq = list(reversed(idxs.tolist()))
    else:
        time_seq = list(range(num_steps - 1, -1, -1))

    def pred_eps(x, t, digit_idx, domain_idx):
        t_tensor = torch.tensor([t], device=device, dtype=torch.long)
        eps_c  = model(x, t_tensor, digit=torch.tensor([digit_idx], device=device),
                       domain=torch.tensor([domain_idx], device=device))
        eps_uc = model(x, t_tensor, digit=torch.tensor([10], device=device),
                       domain=torch.tensor([2], device=device))
        return (1.0 + cfg_w) * eps_c - cfg_w * eps_uc

    # Precompute indices for traces
    trace_sel = None
    if trace_digits:
        trace_sel = set(np.linspace(0, len(time_seq)-1, trace_count, dtype=int).tolist())

    # --------- TIMING START ---------
    t_global = time.time()
    total_imgs = 0
    # --------------------------------

    for d in range(10):
        domain_name = "mnistm" if d % 2 == 0 else "svhn"
        domain_idx  = 0 if d % 2 == 0 else 1
        save_dir    = out_root / domain_name

        t_digit = time.time()  # timing per digit

        for k in range(1, 51):
            x = torch.randn(1, 3, 28, 28, device=device)

            for i, t in enumerate(time_seq):
                eps = pred_eps(x, t, d, domain_idx)

                a_bar_t = alphas_cumprod[t]
                if i == len(time_seq) - 1:
                    a_bar_prev = torch.tensor(1.0, device=device)
                else:
                    a_bar_prev = alphas_cumprod[ time_seq[i+1] ]

                # predict x0
                x0_hat = (x - torch.sqrt(1.0 - a_bar_t) * eps) / torch.sqrt(a_bar_t)
                x0_hat = x0_hat.clamp(-1.0, 1.0)

                if use_ddim:
                    # DDIM (eta=0)
                    x = torch.sqrt(a_bar_prev) * x0_hat + torch.sqrt(1.0 - a_bar_prev) * eps
                else:
                    # DDPM ancestral step
                    beta_t = betas[t]
                    alpha_t = 1.0 - beta_t
                    coef1 = torch.sqrt(a_bar_prev) * beta_t / (1.0 - a_bar_t)
                    coef2 = torch.sqrt(alpha_t) * (1.0 - a_bar_prev) / (1.0 - a_bar_t)
                    mean = coef1 * x0_hat + coef2 * x
                    if t > 0:
                        var = (1.0 - a_bar_prev) / (1.0 - a_bar_t) * beta_t
                        noise = torch.randn_like(x)
                        x = mean + torch.sqrt(var) * noise
                    else:
                        x = mean

                # Save trace frames
                if trace_digits and (d in trace_digits) and k == 1:
                    if i in trace_sel:
                        save_image(to_image(x.detach().cpu()), str(trace_dir / f"{domain_name}_{d}_trace_{i:04d}.png"))

            # save final image
            save_image(to_image(x.detach().cpu()), str(save_dir / f"{d}_{k:03d}.png"))

            total_imgs += 1
            if k % 10 == 0:
                print(f"[digit {d}] saved {k}/50 to {domain_name}", flush=True)

        dt_digit = time.time() - t_digit
        ms_per_img = (dt_digit / 50.0) * 1000.0
        print(f"[time] digit {d} ({domain_name}): {dt_digit:.2f}s, {50/dt_digit:.2f} img/s, {ms_per_img:.1f} ms/img", flush=True)

    dt_total = time.time() - t_global
    print(f"[time] total: {dt_total:.2f}s for {total_imgs} images "
          f"({total_imgs/dt_total:.2f} img/s, {(dt_total/total_imgs)*1000.0:.1f} ms/img)", flush=True)

# ------------------ Load & CLI ------------------
def load_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})
    model = CondUNetSmall(
        in_ch=3,
        base_ch=args.get("base_channels", 64),
        dim_t=args.get("time_dim", 128),
        dim_digit=args.get("digit_dim", 64),
        dim_domain=args.get("domain_dim", 16),
        use_attention=not args.get("no_attention", False)
    )
    model.load_state_dict(ckpt["model"], strict=True)
    return model

def main():
    ap = argparse.ArgumentParser(description="P1 Inference — generate 500 images under the HW spec")
    ap.add_argument("--output_dir", type=str, required=True, help="Root output folder (will contain mnistm/ and svhn/)")
    ap.add_argument("--ckpt", type=str, default="checkpoints/p1_ddpm_cond_unet.pt", help="Checkpoint path")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--num_steps", type=int, default=1000, help="Training diffusion steps (T) used in schedule")
    ap.add_argument("--sample_steps", type=int, default=250, help="Sampling steps; < num_steps uses DDIM(eta=0)")
    ap.add_argument("--cfg_w", type=float, default=2.0, help="Classifier-Free Guidance weight (1.5~2.5 typical)")
    ap.add_argument("--method", type=str, default="auto", choices=["auto","ddpm","ddim"], help="Force sampler or auto by steps")
    ap.add_argument("--trace_digits", type=str, default="", help="e.g. '0,1' to save reverse-process frames for digits")
    ap.add_argument("--trace_count", type=int, default=6, help="How many frames to save along the trajectory")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_root = Path(args.output_dir); out_root.mkdir(parents=True, exist_ok=True)

    model = load_model(Path(args.ckpt))

    trace_digits = None
    if args.trace_digits.strip():
        trace_digits = [int(x) for x in args.trace_digits.split(",")]

    generate_p1_images(
        model=model,
        out_root=out_root,
        num_steps=args.num_steps,
        sample_steps=args.sample_steps,
        cfg_w=args.cfg_w,
        seed=args.seed,
        device=device,
        trace_digits=trace_digits,
        trace_count=args.trace_count,
        method=args.method,
    )

if __name__ == "__main__":
    main()
