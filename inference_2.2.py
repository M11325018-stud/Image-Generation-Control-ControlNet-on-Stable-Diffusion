# inference_hw2.2.py
import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image

# ---------- helpers ----------
def to_image(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5

def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_noises(noise_dir: Path, device):
    files = [noise_dir / f"{i:02d}.pt" for i in range(10)]
    if not all(f.exists() for f in files):
        missing = [f.name for f in files if not f.exists()]
        raise FileNotFoundError(f"Missing: {missing}")
    noises = []
    for f in files:
        n = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(n, dict) and "noise" in n:
            n = n["noise"]
        n = torch.as_tensor(n, dtype=torch.float32)
        noises.append(n.to(device))
    return noises

def make_1based_equal_schedule(T: int, steps: int):
    stride = (T - 1) // (steps - 1)
    t_seq  = [1 + i * stride for i in range(steps)]
    return t_seq[::-1]

# ---------- 唯一的後處理：scale ----------
def postprocess_scale(x, target_range=(-1.0, 1.0)):
    """線性縮放到目標範圍"""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-8:
        return x
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized * (target_range[1] - target_range[0]) + target_range[0]

# ---------- DDIM (eta=0) ----------
@torch.no_grad()
def ddim_eta0(model, noises, betas, device, steps: int):
    """
    DDIM 採樣，最後應用 scale 後處理
    """
    betas = betas.to(device=device, dtype=torch.float32)
    T = betas.numel()
    
    alphas = (1.0 - betas).to(dtype=torch.float32)
    abar   = torch.cumprod(alphas, dim=0).to(dtype=torch.float32)

    t_seq = make_1based_equal_schedule(T, steps)
    model.eval().to(device)

    results = []
    
    for i, z in enumerate(noises):
        x = z.unsqueeze(0) if z.dim() == 3 else z
        x = x.to(device, dtype=torch.float32)

        # DDIM 採樣循環
        for si, t_cur in enumerate(t_seq):
            a_bar_t = abar[t_cur - 1]
            if si == len(t_seq) - 1:
                a_bar_prev = torch.tensor(1.0, device=device, dtype=torch.float32)
            else:
                a_bar_prev = abar[t_seq[si + 1] - 1]

            t_tensor = torch.full((x.size(0),), t_cur, device=device, dtype=torch.long)
            eps = model(x, t_tensor)

            sqrt_a_bar_t = torch.sqrt(a_bar_t)
            sqrt_one_minus_a_bar_t = torch.sqrt(1.0 - a_bar_t)
            x0_hat = (x - sqrt_one_minus_a_bar_t * eps) / sqrt_a_bar_t

            sqrt_a_bar_prev = torch.sqrt(a_bar_prev)
            sqrt_one_minus_a_bar_prev = torch.sqrt(1.0 - a_bar_prev)
            x = sqrt_a_bar_prev * x0_hat + sqrt_one_minus_a_bar_prev * eps

        # ⭐ 應用 scale 後處理
        x = postprocess_scale(x, (-1.0, 1.0))
        
        results.append(x[0])  # [C, H, W]
    
    return results

def main():
    ap = argparse.ArgumentParser(description="DDIM inference with scale postprocessing")
    ap.add_argument("--noise_dir", type=str, required=True)
    ap.add_argument("--out_dir",   type=str, required=True)
    ap.add_argument("--weight",    type=str, required=True)
    ap.add_argument("--steps",     type=int, default=50)
    ap.add_argument("--device",    type=str, default="cuda")
    ap.add_argument("--linear_start", type=float, default=1e-4)
    ap.add_argument("--linear_end",   type=float, default=2e-2)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_deterministic()

    print("=" * 70)
    print("DDIM Inference with Scale Postprocessing")
    print("=" * 70)

    # 載入模型
    from UNet import UNet
    model = UNet()
    ckpt = torch.load(args.weight, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print("✓ Model loaded")

    # 載入 betas
    import utils
    betas = utils.beta_scheduler(n_timestep=1000, linear_start=args.linear_start, linear_end=args.linear_end)
    betas = betas.to(device=device, dtype=torch.float32)
    print("✓ Betas loaded")

    # 載入噪聲
    noises = load_noises(Path(args.noise_dir), device)
    print(f"✓ Loaded {len(noises)} noises")

    # 生成圖片
    print("\nGenerating images with scale postprocessing...")
    outputs = ddim_eta0(model, noises, betas, device, args.steps)
    
    # 保存結果
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i, output in enumerate(outputs):
        save_image(to_image(output.cpu()), str(out_dir / f"{i:02d}.png"))
        print(f"Generated {i:02d}.png, range: [{output.min():.4f}, {output.max():.4f}]")
    
    print(f"\n✓ Results saved to {out_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main() 
