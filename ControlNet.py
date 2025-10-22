import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class ControlNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        hint_channels: int = 3,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Input Hint Block
        self.input_hint_block = nn.Sequential(
            nn.Conv2d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(256, model_channels, 3, padding=1),
        )
        
        # 🔥 直接創建 13 個零卷積，輸出統一的通道數
        # 讓 _maybe_match_channels 在 hook 中自動調整
        self.zero_convs = nn.ModuleList([
            self._make_zero_conv(model_channels) for _ in range(13)
        ])
        
        print(f"✅ ControlNet initialized")
        print(f"   - 13 zero convs with {model_channels} channels each")
    
    def _make_zero_conv(self, channels):
        conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return conv
    
    def forward(self, x, hint, emb=None, context=None):
        # 編碼 hint
        guided_hint = self.input_hint_block(hint)  # [B, 320, H/8, W/8]
        
        # 🔥 所有控制信號都用相同的 320 通道特徵
        controls = []
        for zero_conv in self.zero_convs:
            ctrl = zero_conv(guided_hint)
            controls.append(ctrl)
        
        return controls