"""
对医学图像切片进行以标注区域为中心的裁剪：
1. 读取配对的 JPG 图像和 PNG 标签文件
2. 计算标签前景区域的质心作为裁剪中心点
3. 以质心为中心裁剪出固定大小（224x224）的区域，靠近边界时自动调整
4. 将裁剪后的图像和标签分别保存至输出文件夹
"""

import os
import cv2
import numpy as np

image_dir = "/image"
label_dir = "/label"
output_img_dir = "/image1"
output_lbl_dir = "/label1"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

crop_size = 224
half_crop = crop_size // 2

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

for img_file in image_files:
    name = os.path.splitext(img_file)[0]
    img_path = os.path.join(image_dir, img_file)
    lbl_path = os.path.join(label_dir, name + ".png")

    if not os.path.exists(lbl_path):
        print(f"标签缺失: {lbl_path}")
        continue

    image = cv2.imread(img_path)
    label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)

    # 提取所有前景像素（标签值>0）的坐标
    coords = np.column_stack(np.where(label > 0))
    if coords.size == 0:
        print(f"跳过空标签: {lbl_path}")
        continue

    # 计算前景区域的质心作为裁剪中心点
    center_y, center_x = np.mean(coords, axis=0).astype(int)

    # 以质心为中心计算初始裁剪边界
    h, w = label.shape
    x1 = max(0, center_x - half_crop)
    y1 = max(0, center_y - half_crop)
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)

    # 靠近图像边界时，反向调整起点，确保裁剪尺寸始终为 224x224
    if x2 - x1 < crop_size:
        x1 = max(0, x2 - crop_size)
    if y2 - y1 < crop_size:
        y1 = max(0, y2 - crop_size)

    cropped_img = image[y1:y2, x1:x2]
    cropped_lbl = label[y1:y2, x1:x2]

    out_img_path = os.path.join(output_img_dir, name + ".jpg")
    out_lbl_path = os.path.join(output_lbl_dir, name + ".png")
    cv2.imwrite(out_img_path, cropped_img)
    cv2.imwrite(out_lbl_path, cropped_lbl)

    print(f"已裁剪并保存: {out_img_path}, {out_lbl_path}")