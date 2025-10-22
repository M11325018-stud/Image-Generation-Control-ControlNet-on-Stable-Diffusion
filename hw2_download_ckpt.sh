#!/bin/bash

# 下載 Stable Diffusion 1.4 預訓練權重
echo "Downloading Stable Diffusion v1.4 weights..."

echo "Downloading hw2.1 chickpoints ..."

# 創建存放模型的資料夾
mkdir -p models/ldm/stable-diffusion-v1

# 下載 SD 1.4 權重（約 4GB）
#wget -O models/ldm/stable-diffusion-v1/model.ckpt \
    #"https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"

echo "Downloading hw2.1 chickpoints ..."    

wget -O p1_ddpm_cond_unet.pt  "https://www.dropbox.com/scl/fi/hi7mfzrh0wff1fufl3mnb/p1_ddpm_cond_unet.pt?rlkey=97xu32fyxb4rp1zmj2jwgsnib&st=59yk40qo&dl=0"


echo "Downloading hw2.3_checkpoints.pth ..."
wget -O hw2.3_checkpoints.pth "https://www.dropbox.com/scl/fi/95m0vl2hmy2vglzln0xb9/hw2.3_checkpoints.pth?rlkey=yv6yt5ugpfx2tshawqnahge7y&st=evalgplg&dl=0"
echo "Download complete!"