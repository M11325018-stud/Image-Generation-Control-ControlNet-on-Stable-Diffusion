import os
import sys
import json
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== IMPORTANT: use the provided Stable-Diffusion repo (TA installs with -e) ====
PROJ_DIR = os.path.dirname(__file__)
SD_DIR = os.path.join(PROJ_DIR, "../stable-diffusion")
if SD_DIR not in sys.path:
    sys.path.insert(0, SD_DIR)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import timestep_embedding

# ==== ControlNet & Dataset ====
from ControlNet import ControlNet
from dataset import Fill50kDataset


# -------------------------- Utils --------------------------
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_sd_model(config_path: str, ckpt_path: str, device: str):
    """Load SD v1.x backbone (frozen)."""
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0:
        print("[WARN] Missing keys:", len(missing))
    if len(unexpected) > 0:
        print("[WARN] Unexpected keys:", len(unexpected))
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# --------- 關閉 SD 的 gradient checkpoint ---------
def disable_sd_checkpointing(unet):
    """
    強制關閉 SD UNet 的 gradient checkpointing。
    使用多種方法確保完全關閉。
    """
    # 方法 1: Monkeypatch checkpoint 函數
    try:
        import ldm.modules.diffusionmodules.util as dm_util
        def _identity_ckpt(func, *args, **kwargs):
            return func(*args, **kwargs)
        dm_util.checkpoint = _identity_ckpt
        print("[INFO] Monkeypatched ldm checkpoint function")
    except Exception as e:
        print(f"[WARN] Failed to monkeypatch checkpoint: {e}")
    
    # 方法 2: 遞歸關閉所有模組的 checkpoint 標誌
    def _recursive_disable(module, depth=0):
        for attr in ['use_checkpoint', 'checkpoint', 'gradient_checkpointing']:
            if hasattr(module, attr):
                try:
                    setattr(module, attr, False)
                except:
                    pass
        for child in module.children():
            _recursive_disable(child, depth + 1)
    
    _recursive_disable(unet)
    print("[INFO] Disabled all checkpoint flags recursively")
    
    # 方法 3: 替換 torch.utils.checkpoint
    try:
        import torch.utils.checkpoint as ckpt_module
        def dummy_checkpoint(function, *args, **kwargs):
            return function(*args)
        ckpt_module.checkpoint = dummy_checkpoint
        ckpt_module.checkpoint_sequential = lambda functions, segments, *args: functions(*args)
        print("[INFO] Patched torch.utils.checkpoint")
    except Exception as e:
        print(f"[WARN] Failed to patch torch checkpoint: {e}")
    
    print("[INFO] ✅ Gradient checkpointing fully disabled")


# ---------------------- 修正版：使用 Hook 機制進行控制注入 ----------------------
def _resize_like(tensor, ref):
    """Resize tensor to match reference shape."""
    if tensor.shape[-2:] != ref.shape[-2:]:
        tensor = F.interpolate(tensor, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    return tensor


def _maybe_match_channels(ctrl, ref):
    """Match channel dimensions between control and reference."""
    c_ctrl, c_ref = ctrl.shape[1], ref.shape[1]
    if c_ctrl == c_ref:
        return ctrl
    if c_ctrl > c_ref:
        return ctrl[:, :c_ref, :, :]
    pad = torch.zeros(ctrl.shape[0], c_ref - c_ctrl, ctrl.shape[2], ctrl.shape[3],
                      device=ctrl.device, dtype=ctrl.dtype)
    return torch.cat([ctrl, pad], dim=1)


def register_control_hooks(unet, controls: List[torch.Tensor], strength: float = 1.0):
    """
    註冊 forward hooks 來注入控制信號。
    這樣可以確保梯度正確回傳。
    
    Args:
        unet: UNet 模型
        controls: 來自 ControlNet 的控制信號列表
        strength: 控制強度（訓練時通常設為 1.0）
    
    Returns:
        handles: hook handles 列表（用於移除）
    """
    handles = []
    
    # Input blocks (12 個)
    for i, block in enumerate(unet.input_blocks):
        def _make_hook(idx):
            def hook(module, inputs, output):
                ctrl = controls[idx]
                ctrl = _resize_like(ctrl, output)
                ctrl = _maybe_match_channels(ctrl, output)
                return output + strength * ctrl
            return hook
        handles.append(block.register_forward_hook(_make_hook(i)))
    
    # Middle block
    def mid_hook(module, inputs, output):
        ctrl = controls[12]
        ctrl = _resize_like(ctrl, output)
        ctrl = _maybe_match_channels(ctrl, output)
        return output + strength * ctrl
    handles.append(unet.middle_block.register_forward_hook(mid_hook))
    
    return handles


# -------------------------- Training --------------------------
def train():
    ap = argparse.ArgumentParser(description="HW2 P3 – Train ControlNet (SD frozen, no checkpoint)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--sd_config", default="../stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    ap.add_argument("--sd_ckpt",   default="../models/ldm/stable-diffusion-v1/model.ckpt")
    ap.add_argument("--out_dir",   default="./p3_outputs")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--control_strength", type=float, default=1.0, help="Control signal strength")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Control strength: {args.control_strength}")

    # ---- 1. Load frozen SD backbone ----
    sd_model = load_sd_model(args.sd_config, args.sd_ckpt, device)
    unet = sd_model.model.diffusion_model
    disable_sd_checkpointing(unet)

    # ---- 2. Create trainable ControlNet ----
    controlnet = ControlNet().to(device)
    for p in controlnet.parameters():
        p.requires_grad_(True)
    
    if args.resume and os.path.isfile(args.resume):
        cn_sd = torch.load(args.resume, map_location="cpu")
        controlnet.load_state_dict(cn_sd, strict=True)
        print(f"[INFO] Resumed from {args.resume}")

    # ---- 3. Dataset / Dataloader ----
    dataset = Fill50kDataset(args.data_root, size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # ---- 4. Optimizer ----
    opt = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    best_loss = float("inf")
    global_step = 0

    num_timesteps = sd_model.num_timesteps
    alphas = sd_model.alphas_cumprod.to(device)

    # 🔥 統計變量
    loss_history = []
    grad_history = []
    t_history = []

    # ---- 5. Training Loop ----
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        controlnet.train()
        epoch_loss = 0.0
        
        batch_pbar = tqdm(
            loader, 
            desc=f"Epoch {epoch+1}/{args.epochs}", 
            position=1, 
            leave=False,
            ncols=100
        )

        for batch_idx, batch in enumerate(batch_pbar):
            # 兼容 tuple / dict
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    hint, img = batch[0], batch[1]
                else:
                    raise ValueError(f"Unexpected tuple batch length: {len(batch)}")
                texts = [""] * hint.shape[0]
            elif isinstance(batch, dict):
                hint = None
                for k in ["hint","source","hints","sketch"]:
                    if k in batch and batch[k] is not None:
                        hint = batch[k]; break
                if hint is None:
                    raise KeyError("Batch missing hint/source")
                img = None
                for k in ["image","target","gt","images"]:
                    if k in batch and batch[k] is not None:
                        img = batch[k]; break
                if img is None:
                    img = hint
                texts = batch.get("text") or batch.get("prompt") or ["" for _ in range(hint.shape[0])]
            else:
                raise TypeError(f"Unexpected batch type: {type(batch)}")

            hint, img = hint.to(device), img.to(device)
            if isinstance(texts, torch.Tensor):
                texts = ["" for _ in range(hint.shape[0])]
            B = img.shape[0]

            # 🔥 DEBUG: 第一個 batch
            if global_step == 0:
                tqdm.write("=" * 80)
                tqdm.write("[DEBUG] First Batch Information:")
                tqdm.write(f"  Hint shape: {hint.shape}")
                tqdm.write(f"  Image shape: {img.shape}")
                tqdm.write(f"  Hint range: [{hint.min():.4f}, {hint.max():.4f}]")
                tqdm.write(f"  Image range: [{img.min():.4f}, {img.max():.4f}]")
                tqdm.write(f"  Batch size: {B}")
                tqdm.write(f"  First prompt: {texts[0][:50]}...")
                tqdm.write("=" * 80)

            # ---- text conditioning ----
            with torch.no_grad():
                c = sd_model.get_learned_conditioning(texts)

            # ---- encode GT image to latent z0 ----
            with torch.no_grad():
                posterior = sd_model.encode_first_stage(img)
                z0 = sd_model.get_first_stage_encoding(posterior)

            # ---- sample t + add noise ----
            t = torch.randint(0, num_timesteps, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(z0)
            sqrt_a = torch.sqrt(alphas[t]).view(B,1,1,1)
            sqrt_m1_a = torch.sqrt(1 - alphas[t]).view(B,1,1,1)
            zt = sqrt_a * z0 + sqrt_m1_a * noise

            # 🔥 DEBUG: 時間步分佈（前 10 步）
            if global_step <= 10:
                t_list = t.tolist()
                t_min, t_max = t.min().item(), t.max().item()
                t_mean = t.float().mean().item()
                t_history.append(t_mean)
                
                if global_step <= 2:
                    tqdm.write(f"\n[DEBUG] Step {global_step}:")
                    tqdm.write(f"  🔥 Timestep t: {t_list}")
                    tqdm.write(f"  🔥 t range: [{t_min}, {t_max}]")
                    tqdm.write(f"  🔥 t mean: {t_mean:.1f} (should be ~500)")
                    
                    if t_mean < 200:
                        tqdm.write(f"  ⚠️ WARNING: t mean is TOO SMALL! This will cause low loss and small gradients!")
                    
                    tqdm.write(f"  z0 range: [{z0.min():.4f}, {z0.max():.4f}]")
                    tqdm.write(f"  noise range: [{noise.min():.4f}, {noise.max():.4f}]")
                    tqdm.write(f"  zt range: [{zt.min():.4f}, {zt.max():.4f}]")

            # ---- 🔥 修正版：ControlNet forward 並使用 Hook 機制 ----
            # 1. 生成時間嵌入（不需要 no_grad，因為 UNet 已經被凍結）
            with torch.no_grad():
                t_emb = unet.time_embed(timestep_embedding(t, unet.model_channels))
            
            # 2. ControlNet 生成控制信號（需要梯度）
            controls = controlnet(zt, hint, t_emb, c)
            
            # 🔥 DEBUG: 控制信號
            if global_step <= 2:
                tqdm.write(f"  Controls count: {len(controls)}")
                for i in range(min(3, len(controls))):
                    ctrl = controls[i]
                    tqdm.write(f"    Control[{i}]: shape={ctrl.shape}, "
                              f"mean={ctrl.mean():.6f}, std={ctrl.std():.6f}, "
                              f"requires_grad={ctrl.requires_grad}")
                tqdm.write(f"    ... (total {len(controls)} controls)")
            
            # 3. 註冊 hooks 並進行 UNet forward（梯度會通過 hooks 回傳）
            handles = register_control_hooks(unet, controls, strength=args.control_strength)
            
            # 4. UNet forward（控制信號會透過 hooks 自動加入）
            eps_pred = unet(zt, t, context=c)
            
            # 5. 移除 hooks
            for hdl in handles:
                hdl.remove()

            if global_step <= 2:
                tqdm.write(f"  eps_pred range: [{eps_pred.min():.4f}, {eps_pred.max():.4f}]")
                tqdm.write(f"  eps_pred requires_grad: {eps_pred.requires_grad}")

            # ---- loss & update ----
            loss = F.mse_loss(eps_pred, noise)
            epoch_loss += loss.item()
            loss_history.append(loss.item())

            # 🔥 DEBUG: Loss 監控（Fill50k 任務 loss 會較小）
            if global_step <= 10:
                if global_step <= 2:
                    tqdm.write(f"  🔥 Loss: {loss.item():.6f}")
                    
                    # Fill50k 是簡單任務，loss 在 0.01-0.08 是正常的
                    if loss.item() < 0.005:
                        tqdm.write(f"  ⚠️ WARNING: Loss is EXTREMELY LOW! Possible training issue.")
                    elif loss.item() > 0.15:
                        tqdm.write(f"  ⚠️ WARNING: Loss is TOO HIGH for Fill50k task!")
                    else:
                        tqdm.write(f"  ✅ Loss looks normal for Fill50k task (simple geometry + color)")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            
            # 🔥 DEBUG: 梯度檢查
            if global_step <= 10:
                # 總梯度 norm
                grad_norm = 0.0
                for name, param in controlnet.named_parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                grad_history.append(grad_norm)
                
                if global_step <= 2:
                    tqdm.write(f"  🔥 Total gradient norm: {grad_norm:.6f}")
                    
                    if grad_norm < 0.01:
                        tqdm.write(f"  ⚠️ WARNING: Gradient is TOO SMALL! Expected > 0.1")
                    elif grad_norm > 100:
                        tqdm.write(f"  ⚠️ WARNING: Gradient is TOO LARGE! Possible gradient explosion!")
                    else:
                        tqdm.write(f"  ✅ Gradient norm looks healthy!")
            
            # 🔥 統計摘要（第 10 步）
            if global_step == 10:
                tqdm.write("\n" + "=" * 80)
                tqdm.write("[SUMMARY] First 10 Steps Statistics:")
                tqdm.write(f"  Average t: {sum(t_history)/len(t_history):.1f} (should be ~500)")
                tqdm.write(f"  Average loss: {sum(loss_history)/len(loss_history):.4f} (should be 0.05-0.12)")
                tqdm.write(f"  Average gradient norm: {sum(grad_history)/len(grad_history):.4f} (should be > 0.1)")
                tqdm.write(f"  Loss trend: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
                tqdm.write(f"  Gradient trend: {grad_history[0]:.4f} → {grad_history[-1]:.4f}")
                
                # 診斷
                avg_t = sum(t_history) / len(t_history)
                avg_loss = sum(loss_history) / len(loss_history)
                avg_grad = sum(grad_history) / len(grad_history)
                
                tqdm.write("\n[DIAGNOSIS]:")
                issues = []
                if avg_t < 200:
                    issues.append("⚠️ Timesteps too small → causing low loss")
                if avg_loss < 0.04:
                    issues.append("⚠️ Loss too low → task too easy or model cheating")
                if avg_grad < 0.05:
                    issues.append("⚠️ Gradients too small → learning too slow")
                if len(grad_history) > 1 and grad_history[-1] < grad_history[0] * 0.3:
                    issues.append("⚠️ Gradient collapsing → serious training issue")
                
                if issues:
                    for issue in issues:
                        tqdm.write(f"  {issue}")
                    tqdm.write("\n  Suggested fixes:")
                    if avg_grad < 0.05:
                        tqdm.write("    - Increase learning rate to 5e-4 or 1e-3")
                        tqdm.write("    - Increase control strength to 2.0 or 3.0")
                    if avg_t < 200:
                        tqdm.write("    - Check random seed is working correctly")
                else:
                    tqdm.write("  ✅ Training looks normal!")
                
                tqdm.write("=" * 80 + "\n")
            
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
            opt.step()

            global_step += 1
            
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'step': global_step
            })
            
            if global_step % args.save_every == 0:
                ckpt = os.path.join(
                    args.out_dir, 
                    f"controlnet_step{global_step:06d}_loss{loss.item():.4f}.pth"
                )
                torch.save(controlnet.state_dict(), ckpt)
                tqdm.write(f"[SAVE] {ckpt}")

        # ---- Epoch 結束 ----
        avg = epoch_loss / max(1, len(loader))
        
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg:.4f}',
            'best': f'{best_loss:.4f}' if best_loss != float("inf") else 'N/A'
        })
        
        if avg < best_loss:
            best_loss = avg
            best_path = os.path.join(
                args.out_dir, 
                f"controlnet_best_epoch{epoch+1:02d}_loss{avg:.4f}.pth"
            )
            torch.save(controlnet.state_dict(), best_path)
            tqdm.write(f"[BEST] {best_path}")

    final = os.path.join(
        args.out_dir, 
        f"controlnet_final_epoch{args.epochs}_loss{avg:.4f}.pth"
    )
    torch.save(controlnet.state_dict(), final)
    print(f"\n[DONE] Saved final model | best loss: {best_loss:.4f}")


if __name__ == "__main__":
    train()
