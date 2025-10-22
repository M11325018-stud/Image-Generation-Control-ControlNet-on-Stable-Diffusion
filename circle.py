#!/usr/bin/env python3
"""
創建包含兩個圓圈的測試數據
用於 Problem 3 Report 第二題
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import json

def create_two_circles_control_image(output_path, config):
    """
    創建包含兩個圓圈的控制圖像
    
    Args:
        output_path: 輸出圖像路徑
        config: 配置字典，包含兩個圓圈的參數
            {
                'circle1': {'center': (x, y), 'radius': r},
                'circle2': {'center': (x, y), 'radius': r},
                'size': (width, height)
            }
    """
    size = config.get('size', (512, 512))
    
    # 創建黑色背景
    img = Image.new('RGB', size, color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # 畫第一個圓圈（白色輪廓）
    c1 = config['circle1']
    x1, y1 = c1['center']
    r1 = c1['radius']
    draw.ellipse([x1-r1, y1-r1, x1+r1, y1+r1], outline=(255, 255, 255), width=3)
    
    # 畫第二個圓圈（白色輪廓）
    c2 = config['circle2']
    x2, y2 = c2['center']
    r2 = c2['radius']
    draw.ellipse([x2-r2, y2-r2, x2+r2, y2+r2], outline=(255, 255, 255), width=3)
    
    # 保存
    img.save(output_path)
    print(f"✓ Created control image: {output_path}")
    return img

def create_test_cases():
    """創建兩組測試案例"""
    
    # 創建輸出目錄
    os.makedirs("two_circles_test", exist_ok=True)
    os.makedirs("two_circles_test/source", exist_ok=True)
    
    # 測試案例 1: 兩個大小相似的圓（上下排列）
    test1_config = {
        'size': (512, 512),
        'circle1': {
            'center': (256, 180),  # 上方
            'radius': 80
        },
        'circle2': {
            'center': (256, 340),  # 下方
            'radius': 80
        }
    }
    
    test1_prompt = "red circle and blue circle on white background"
    
    # 測試案例 2: 兩個大小不同的圓（左右排列）
    test2_config = {
        'size': (512, 512),
        'circle1': {
            'center': (180, 256),  # 左邊，較大
            'radius': 100
        },
        'circle2': {
            'center': (380, 256),  # 右邊，較小
            'radius': 60
        }
    }
    
    test2_prompt = "large yellow circle and small green circle on purple background"
    
    # 生成控制圖像
    create_two_circles_control_image("two_circles_test/source/test1.png", test1_config)
    create_two_circles_control_image("two_circles_test/source/test2.png", test2_config)
    
    # 創建 JSON 文件
    json_data = [
        {
            "source": "test1.png",
            "target": "test1.png",
            "prompt": test1_prompt
        },
        {
            "source": "test2.png",
            "target": "test2.png",
            "prompt": test2_prompt
        }
    ]
    
    with open("two_circles_test/prompt.json", "w") as f:
        json.dump(json_data, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ Test cases created successfully!")
    print("="*80)
    print("\nTest Case 1:")
    print(f"  Config: Two similar-sized circles (top-bottom)")
    print(f"  Circle 1: center=(256, 180), radius=80")
    print(f"  Circle 2: center=(256, 340), radius=80")
    print(f"  Prompt: {test1_prompt}")
    
    print("\nTest Case 2:")
    print(f"  Config: Two different-sized circles (left-right)")
    print(f"  Circle 1: center=(180, 256), radius=100")
    print(f"  Circle 2: center=(380, 256), radius=60")
    print(f"  Prompt: {test2_prompt}")
    
    print("\n" + "="*80)
    print("Files created:")
    print("  two_circles_test/source/test1.png")
    print("  two_circles_test/source/test2.png")
    print("  two_circles_test/prompt.json")
    print("="*80)
    
    return json_data

def create_additional_test_cases():
    """創建額外的測試案例（用於深入分析）"""
    
    # 測試案例 3: 兩個重疊的圓
    test3_config = {
        'size': (512, 512),
        'circle1': {
            'center': (230, 256),
            'radius': 90
        },
        'circle2': {
            'center': (282, 256),
            'radius': 90
        }
    }
    test3_prompt = "overlapping red circle and blue circle on gray background"
    
    # 測試案例 4: 兩個距離很遠的圓
    test4_config = {
        'size': (512, 512),
        'circle1': {
            'center': (150, 150),
            'radius': 60
        },
        'circle2': {
            'center': (370, 370),
            'radius': 60
        }
    }
    test4_prompt = "pink circle and orange circle on light blue background"
    
    create_two_circles_control_image("two_circles_test/source/test3.png", test3_config)
    create_two_circles_control_image("two_circles_test/source/test4.png", test4_config)
    
    print("\n✓ Additional test cases created:")
    print("  Test 3: Overlapping circles")
    print("  Test 4: Distant circles")

if __name__ == "__main__":
    print("="*80)
    print("Creating Two-Circle Test Cases for Problem 3 Report")
    print("="*80)
    
    # 創建基本測試案例
    create_test_cases()
    
    # 創建額外測試案例（可選）
    print("\n" + "="*80)
    create_additional = input("\nCreate additional test cases? (y/n): ").lower()
    if create_additional == 'y':
        create_additional_test_cases()
    
    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print("1. Run inference:")
    print("   bash hw2_3.sh \\")
    print("       two_circles_test/prompt.json \\")
    print("       two_circles_test/source \\")
    print("       two_circles_test/output \\")
    print("       <your_model_path>")
    print("\n2. Check results in: two_circles_test/output/")
    print("\n3. Analyze the generated images for your report")
    print("="*80)