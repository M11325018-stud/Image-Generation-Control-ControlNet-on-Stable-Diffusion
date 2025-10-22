🧩 Conditional Diffusion Model for Handwritten Digits

Architecture: U-Net with digit / domain embeddings and sinusoidal time encoding.

Training: Classifier-free guidance, cosine β schedule (T = 1000), AdamW optimizer + EMA stabilization.

Dataset: Even digits → MNIST-M domain, Odd digits → SVHN domain.

Visualization: Reverse diffusion (t = 1000 → 0) shows progressive denoising and digit emergence.

Results:

MNIST-M Accuracy = 0.920

SVHN Accuracy = 0.988

Overall = 0.954
→ Demonstrates stable cross-domain conditional generation and clear digit reconstruction.

🌀 ControlNet on Stable Diffusion

Base Architecture: Stable Diffusion v1-4 + custom ControlNet module.

Integration: Forward hooks inject ControlNet features into UNet layers, dynamically aligning spatial and channel dimensions.

Training: Only ControlNet parameters optimized; Stable Diffusion weights frozen to preserve semantic priors.

Experiment: “Two-circle” control images used to test structural and color guidance.

Results:

Mean Score (IoU×CLIP) = 0.6581

Final Mean IoU = 0.6581 (≈ 20% above class average)
→ Model accurately controls geometry / position but shows limited color fidelity due to data and prompt imbalance.

🧠 Key Techniques

Diffusion Model (DDPM / DDIM)

Classifier-Free Guidance

U-Net Conditional Architecture

ControlNet Integration on Stable Diffusion

Cosine Noise Scheduling

Gradient Clipping & EMA Optimization



🔗 Reference
Diffusion Models

Jonathan Ho, Ajay Jain, Pieter Abbeel.
Denoising Diffusion Probabilistic Models. NeurIPS 2020.
[arXiv 2006.11239]

Diffusion Sampling

Jiaming Song, Chenlin Meng, Stefano Ermon.
Denoising Diffusion Implicit Models. ICLR 2021.
[arXiv 2010.02502]

ControlNet

Lvmin Zhang, Maneesh Agrawala.
Adding Conditional Control to Text-to-Image Diffusion Models. arXiv 2023.
[arXiv 2302.05543]

Official Implementation: https://github.com/lllyasviel/ControlNet

Evaluation Framework

SAM-2: https://github.com/facebookresearch/sam2

CLIP: https://github.com/openai/CLIP
