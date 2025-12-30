from PIL import Image
import glob
import os
import numpy as np

# 输入目录，放图片
input_dir = "./input_images"
# 输出目录
output_dir = "./input_raw"
os.makedirs(output_dir, exist_ok=True)

# 获取所有图片
image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")))  # 可改成 *.jpg

# 遍历图片
for idx, img_path in enumerate(image_files):
    print(f"Processing {img_path}")
    with Image.open(img_path) as img:
        # 裁剪/调整大小到 640x480 (宽x高)
        img = img.resize((640, 480))  # 640w x 480h
        # 转成 RGB
        img = img.convert("RGB")
        # 转成 numpy array
        data = np.array(img, dtype=np.float32)  # uint8 -> float32
        # 可选：归一化到 [0,1]，如果模型要求
        data /= 255.0
        # 转换为 NCHW 顺序 (C,H,W)
        data = data.transpose(2, 0, 1)
        # 写入 float32 raw
        raw_path = os.path.join(output_dir, f"image_{idx+1}.raw")
        data.tofile(raw_path)

    print(f"Saved float32 raw: {raw_path}")