import os
import sys
import json
import argparse
from typing import List, Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ==== IMPORTANT: use the provided Stable-Diffusion repo (TA installs with -e) ====
PROJ_DIR = os.path.dirname(__file__)
SD_DIR = os.path.join(PROJ_DIR, "../stable-diffusion")
if SD_DIR not in sys.path:
    sys.path.insert(0, SD_DIR)

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.util import timestep_embedding

from ControlNet import ControlNet

# -------------------------- IO helpers --------------------------
def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_image(path: str, size: int = 256) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    # map to [-1, 1]
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr * 2.0) - 1.0
    ten = torch.from_numpy(arr).permute(2, 0, 1)  # [3,H,W]
    return ten.unsqueeze(0)  # [1,3,H,W]

def save_image(t: torch.Tensor, path: str):
    # t: [1,3,H,W] in [0,1]
    t = t.detach().cpu().clamp(0.0, 1.0)[0]
    arr = (t.permute(1,2,0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)

def load_sd_model(config_path: str, ckpt_path: str, device: str):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model

# ---------------------- multi-scale hooks (same as train) ----------------------
def _resize_like(tensor, ref):
    if tensor.shape[-2:] != ref.shape[-2:]:
        tensor = F.interpolate(tensor, size=ref.shape[-2:], mode='bilinear', align_corners=False)
    return tensor

def _maybe_match_channels(ctrl, ref):
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
    Register forward hooks to inject control signals.
    Same implementation as training for consistency.
    """
    handles = []
    for i, block in enumerate(unet.input_blocks):
        def _make_hook(idx):
            def hook(module, inputs, output):
                ctrl = controls[idx]
                ctrl = _resize_like(ctrl, output)
                ctrl = _maybe_match_channels(ctrl, output)
                return output + strength * ctrl
            return hook
        handles.append(block.register_forward_hook(_make_hook(i)))

    def mid_hook(module, inputs, output):
        ctrl = controls[12]
        ctrl = _resize_like(ctrl, output)
        ctrl = _maybe_match_channels(ctrl, output)
        return output + strength * ctrl
    handles.append(unet.middle_block.register_forward_hook(mid_hook))
    return handles

# -------------------------- Sampling --------------------------
@torch.no_grad()
def ddim_sample(sd_model, controlnet, hint, prompt, num_samples=1, ddim_steps=50,
                scale=7.5, eta=0.0, seed=42, device='cuda', control_strength=1.0):
    """
    DDIM sampling with ControlNet control signals.
    
    Args:
        sd_model: Stable Diffusion model
        controlnet: ControlNet model
        hint: Control image [1,3,H,W] in [-1,1]
        prompt: Text prompt (string)
        num_samples: Number of samples to generate
        ddim_steps: Number of DDIM steps (default 50)
        scale: Classifier-free guidance scale (default 7.5)
        eta: DDIM eta parameter (0=deterministic)
        seed: Random seed for reproducibility
        device: Device to run on
        control_strength: Control signal strength (1.0=full control)
    
    Returns:
        x_samples: Generated images [num_samples,3,H,W] in [0,1]
    """
    seed_everything(seed)

    # Text conditioning
    c  = sd_model.get_learned_conditioning([prompt] * num_samples)
    uc = sd_model.get_learned_conditioning([""] * num_samples)

    hint = hint.to(device).repeat(num_samples, 1, 1, 1)

    H, W = 512, 512
    h, w = H // 8, W // 8
    shape = (num_samples, 4, h, w)
    x = torch.randn(shape, device=device)

    unet = sd_model.model.diffusion_model
    num_timesteps = sd_model.num_timesteps
    
    # DDIM timestep schedule (uniform spacing)
    timesteps = np.linspace(num_timesteps - 1, 0, ddim_steps).astype(int)

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

        # Generate time embedding (no_grad is fine since UNet is frozen)
        t_emb = unet.time_embed(timestep_embedding(t_tensor, unet.model_channels))
        
        # Generate control signals from ControlNet
        controls = controlnet(x, hint, t_emb, c)
        
        # Register hooks for this step
        handles = register_control_hooks(unet, controls, strength=control_strength)

        # Predict noise with classifier-free guidance
        noise_pred_uncond = unet(x, t_tensor, context=uc)
        noise_pred_cond   = unet(x, t_tensor, context=c)
        
        # Remove hooks after forward pass
        for hdl in handles:
            hdl.remove()

        # Classifier-free guidance
        noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)

        # DDIM update step
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            alpha_t = sd_model.alphas_cumprod[t]
            alpha_t_next = sd_model.alphas_cumprod[t_next]
            
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            # Direction pointing to xt
            dir_xt = torch.sqrt(torch.clamp(1 - alpha_t_next, min=1e-8)) * noise_pred
            
            # Add noise (eta=0 for deterministic)
            noise = torch.randn_like(x) * eta * torch.sqrt(
                torch.clamp((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next), min=0.0)
            )
            
            # Update x
            x = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt + noise
        else:
            # Final step: predict x0 directly
            alpha_t = sd_model.alphas_cumprod[t]
            x = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

    # Decode latents to images
    x_samples = sd_model.decode_first_stage(x)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, 0.0, 1.0)
    return x_samples

# -------------------------- Main --------------------------
def main():
    parser = argparse.ArgumentParser(description="HW2 P3 - ControlNet Inference")
    parser.add_argument("--json_path", type=str, required=True,
                        help="Path to JSON file with test conditions")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Path to directory containing source images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("--controlnet_path", type=str, required=True,
                        help="Path to trained ControlNet checkpoint")
    parser.add_argument("--sd_config", type=str, 
                        default="stable-diffusion/configs/stable-diffusion/v1-inference.yaml",
                        help="Path to Stable Diffusion config")
    parser.add_argument("--sd_ckpt", type=str, 
                        default="models/ldm/stable-diffusion-v1/model.ckpt",
                        help="Path to Stable Diffusion checkpoint")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--scale", type=float, default=7.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta parameter (0=deterministic)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size (width=height)")
    parser.add_argument("--control_strength", type=float, default=1.0,
                        help="Control signal strength (0.0-2.0)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Loading Stable Diffusion from: {args.sd_ckpt}")
    print(f"[INFO] Loading ControlNet from: {args.controlnet_path}")
    print(f"[INFO] DDIM steps: {args.ddim_steps}")
    print(f"[INFO] Guidance scale: {args.scale}")
    print(f"[INFO] Control strength: {args.control_strength}")
    print(f"[INFO] Seed: {args.seed}")

    # ---- Load Stable Diffusion ----
    sd_model = load_sd_model(args.sd_config, args.sd_ckpt, device)
    
    # ---- Load ControlNet ----
    controlnet = ControlNet().to(device)
    cn_sd = torch.load(args.controlnet_path, map_location="cpu")
    controlnet.load_state_dict(cn_sd, strict=True)
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad = False
    print("[INFO] Models loaded successfully")

    # ---- Read JSON file ----
    # Support multiple JSON formats:
    # 1. Standard JSON array: [{"source": "0.png", ...}, ...]
    # 2. JSONL format: one JSON object per line
    # 3. Multiple JSON objects without array wrapper
    
    try:
        with open(args.json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        # Try standard JSON array first
        try:
            data = json.loads(content)
            if not isinstance(data, list):
                data = [data]  # Wrap single object in list
        except json.JSONDecodeError as e:
            # Try JSONL format (one JSON per line)
            print(f"[INFO] Standard JSON failed, trying JSONL format...")
            data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except json.JSONDecodeError as line_error:
                    print(f"[WARN] Line {line_num} is not valid JSON: {line[:50]}...")
                    continue
            
            if not data:
                print(f"[ERROR] Could not parse JSON file: {args.json_path}")
                print(f"Original error: {e}")
                return
    
    except FileNotFoundError:
        print(f"[ERROR] JSON file not found: {args.json_path}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to read JSON file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"[INFO] Processing {len(data)} samples...")

    # Robust key extraction helper
    def _get(item: Dict, *keys, default=None):
        """Get value from dict with multiple possible keys"""
        for k in keys:
            if k in item and item[k] is not None:
                return item[k]
        return default

    # ---- Process each sample ----
    for idx, item in enumerate(data):
        # Extract source image filename
        src_name = _get(item, "source", "source_img", "source_image", "source_file", "input", "image")
        if src_name is None:
            print(f"[WARN] Sample {idx}: Missing source image name, skipping")
            continue
        
        # Extract target filename (use source name as fallback)
        tgt_name = _get(item, "target", "file_name", "output", default=src_name)
        
        # Extract prompt (use empty string as fallback)
        prompt = _get(item, "text", "prompt", default="")
        
        # Construct paths
        src_path = os.path.join(args.source_dir, src_name)
        out_path = os.path.join(args.output_dir, tgt_name)
        
        # Check if source file exists
        if not os.path.exists(src_path):
            print(f"[ERROR] Sample {idx}: Source file not found: {src_path}")
            continue
        
        # Ensure output directory exists
        out_dir = os.path.dirname(out_path)
        if out_dir:  # Only create if there's a directory component
            os.makedirs(out_dir, exist_ok=True)
        
        print(f"[{idx+1}/{len(data)}] Processing: {src_name} -> {tgt_name}")
        print(f"           Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        
        try:
            # Load control image
            hint = load_image(src_path, size=args.image_size).to(device)
            
            # Generate image
            samples = ddim_sample(
                sd_model, 
                controlnet, 
                hint, 
                prompt,
                num_samples=1, 
                ddim_steps=args.ddim_steps,
                scale=args.scale, 
                eta=args.eta, 
                seed=args.seed, 
                device=device,
                control_strength=args.control_strength
            )
            
            # Save output
            save_image(samples, out_path)
            print(f"           ✓ Saved to: {out_path}")
            
        except Exception as e:
            print(f"[ERROR] Sample {idx}: Failed to process - {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("[DONE] Inference finished successfully!")
    print(f"[INFO] Generated images saved to: {args.output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()