"""
3D NIfTI医学数据转换为2D切片：
1. 遍历输入根目录下的每个病例子文件夹（包含 t1c.nii.gz 和 seg.nii.gz）
2. 筛选出包含分割标注的切片，去除首尾各4张边缘切片
3. 将图像切片归一化后保存为 JPG，标注切片二值化后保存为 PNG
4. 所有切片逆时针旋转90度后输出至 image/ 和 label/ 文件夹
"""

import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
import imageio.v2 as imageio


def load_nii(path):
    """加载 NIfTI 文件，转为标准方向后返回3D numpy数组"""
    ni = nib.load(str(path))
    ni = nib.as_closest_canonical(ni)  # 统一转为标准解剖方向（RAS+）
    data = np.asanyarray(ni.dataobj)
    data = np.squeeze(data)  # 去除多余维度
    if data.ndim != 3:
        raise ValueError(f"{path} is not 3D after squeeze, got shape {data.shape}")
    return data


def vol_to_uint8(volume, pmin=0.5, pmax=99.5):
    """将体素值按百分位数归一化并映射到0-255的uint8范围"""
    v = volume.astype(np.float32)
    lo, hi = np.percentile(v[np.isfinite(v)], [pmin, pmax])
    if hi <= lo:  # 异常情况：退回到全局最值
        lo, hi = np.min(v), np.max(v)
        if hi == lo:
            return np.zeros_like(v, dtype=np.uint8)
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return v.astype(np.uint8)


def choose_axis(axis_str):
    """将字符串轴名（x/y/z）转换为对应的数组维度索引"""
    axis_str = axis_str.lower()
    if axis_str == "z":
        return 2
    if axis_str == "y":
        return 1
    if axis_str == "x":
        return 0
    raise ValueError("axis must be one of x/y/z")


def save_jpg(arr_uint8, path, quality=95):
    """将uint8灰度数组保存为高质量JPEG图像"""
    im = Image.fromarray(arr_uint8)
    im.save(path, format="JPEG", quality=quality, subsampling=0)


def save_label_png(label_slice, path):
    """将标注切片二值化（非零区域为255）并保存为PNG"""
    mask = (label_slice != 0).astype(np.uint8) * 255
    imageio.imwrite(path, mask)


def export_case(case_dir, out_img_dir, out_lab_dir, axis, digits):
    """
    处理单个病例：提取有效切片并保存图像和标注
    返回 (已保存切片数, 总切片数)
    """
    case_name = Path(case_dir).name
    img_path = Path(case_dir) / "t1c.nii.gz"
    seg_path = Path(case_dir) / "seg.nii.gz"

    if not img_path.exists() or not seg_path.exists():
        print(f"[skip] {case_name}: missing t1c.nii.gz or seg.nii.gz")
        return 0, 0

    vol_img = load_nii(img_path)
    vol_seg = load_nii(seg_path)

    if vol_img.shape != vol_seg.shape:
        print(f"[warn] {case_name}: shape mismatch {vol_img.shape} vs {vol_seg.shape}, skip.")
        return 0, 0

    vol_img_u8 = vol_to_uint8(vol_img)
    ax = choose_axis(axis)
    num_slices = vol_img.shape[ax]

    # 遍历所有切片，收集包含分割标注（非全零）的切片索引
    non_empty_indices = []
    for i in range(num_slices):
        if ax == 0:
            seg_sl = vol_seg[i, :, :]
        elif ax == 1:
            seg_sl = vol_seg[:, i, :]
        else:
            seg_sl = vol_seg[:, :, i]
        if np.any(seg_sl != 0):
            non_empty_indices.append(i)

    # 有效标注切片不足10张时跳过该病例，避免数据质量过低
    if len(non_empty_indices) < 10:
        print(f"[skip] {case_name}: only {len(non_empty_indices)} labeled slices (<10).")
        return 0, num_slices

    # 去掉首尾各4张，避免边缘切片标注不完整
    valid_indices = non_empty_indices[4:-4]
    if len(valid_indices) == 0:
        print(f"[skip] {case_name}: after trimming, no slices left.")
        return 0, num_slices

    saved = 0
    for i in valid_indices:
        # 按指定轴提取当前切片
        if ax == 0:
            img_sl = vol_img_u8[i, :, :]
            seg_sl = vol_seg[i, :, :]
        elif ax == 1:
            img_sl = vol_img_u8[:, i, :]
            seg_sl = vol_seg[:, i, :]
        else:
            img_sl = vol_img_u8[:, :, i]
            seg_sl = vol_seg[:, :, i]

        # 逆时针旋转90度，使图像方向符合常规显示习惯
        img_sl = np.rot90(img_sl, k=1)
        seg_sl = np.rot90(seg_sl, k=1)

        # 文件命名：病例名_切片索引（补零对齐）
        idx = str(i).zfill(digits)
        img_out = out_img_dir / f"{case_name}_{idx}.jpg"
        lab_out = out_lab_dir / f"{case_name}_{idx}.png"

        save_jpg(img_sl, img_out)
        save_label_png(seg_sl, lab_out)
        saved += 1

    print(f"[done] {case_name}: saved {saved} labeled slices (after trimming & rotation).")
    return saved, num_slices


def main():
    parser = argparse.ArgumentParser(description="Convert 3D NIfTI to 2D slices (image JPG / label PNG), skipping empty-label slices and trimming head/tail, rotated 90° CCW.")
    parser.add_argument("--input_root", default=r'F:\process', type=str, help="Root folder containing case subfolders (each with t1c.nii.gz and seg.nii.gz).")
    parser.add_argument("--output_root", type=str, default=r'D:\test\2', help="Output root folder that will contain 'image' and 'label'.")
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"], help="Slice axis (default: z/axial).")
    parser.add_argument("--prefix_digits", type=int, default=4, help="Zero-padding digits for slice index.")
    args = parser.parse_args()

    in_root = Path(args.input_root)
    out_root = Path(args.output_root)
    out_img = out_root / "image"
    out_lab = out_root / "label"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    case_dirs = [p for p in in_root.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"No case folders found under {in_root}")
        return

    total_saved = 0
    total_cases = 0

    for case in sorted(case_dirs):
        s, _ = export_case(case, out_img, out_lab, args.axis, args.prefix_digits)
        total_saved += s
        total_cases += 1

    print(f"\nAll done. Cases processed: {total_cases}, labeled slices saved: {total_saved}")
    print(f"Images: {out_img}")
    print(f"Labels: {out_lab}")


if __name__ == "__main__":
    main()