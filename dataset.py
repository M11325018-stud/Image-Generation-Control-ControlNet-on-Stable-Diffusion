import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class Fill50kDataset(Dataset):
    """Fill50k 資料集載入器 - 支援 JSONL 格式"""
    
    def __init__(self, data_root, size=512):
        self.data_root = data_root
        self.size = size
        
        # 載入 JSONL 格式的 prompt
        prompt_path = os.path.join(data_root, 'prompt.json')
        
        print(f"Loading prompts from {prompt_path}...")
        self.data = []
        
        with open(prompt_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # 跳過空行
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Line {line_num} is invalid JSON, skipping...")
                        continue
        
        print(f"✅ Loaded {len(self.data)} samples")
        
        # 🔥 修正：圖像轉換（正確的 RGB normalize）
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ✅ RGB 三通道
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 注意：source 和 target 已經包含了路徑
        # 例如："source": "source/9.png"
        source_path = os.path.join(self.data_root, item['source'])
        target_path = os.path.join(self.data_root, item['target'])
        
        # 檢查文件是否存在
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source image not found: {source_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target image not found: {target_path}")
        
        # 讀取圖像
        source = Image.open(source_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')
        
        # 轉換
        source = self.transform(source)
        target = self.transform(target)
        
        return {
            'hint': source,           # 控制圖像（黑底白圈）
            'target': target,         # 目標圖像（彩色圓圈）
            'prompt': item['prompt']  # 文字提示
        }  # ✅ 加上右括號