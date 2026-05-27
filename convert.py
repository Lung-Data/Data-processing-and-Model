"""
3D NIfTI数据转换为2D切片：
1. 遍历输入根目录下的每个病例子文件夹（包含 t1c.nii.gz 和 seg.nii.gz）
2. 将图像切片归一化后保存为 JPG，标注切片二值化后保存为 PNG
3. 所有切片逆时针旋转90度后输出至 image/ 和 label/ 文件夹
"""
import os
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image
import imageio.v2 as imageio


def load_nii(path):
    ni = nib.load(str(path))
    ni = nib.as_closest_canonical(ni) 
    data = np.asanyarray(ni.dataobj)
    data = np.squeeze(data) 
    if data.ndim != 3:
        raise ValueError(f"{path} is not 3D after squeeze, got shape {data.shape}")
    return data


def vol_to_uint8(volume, pmin=0.5, pmax=99.5):
    v = volume.astype(np.float32)
    lo, hi = np.percentile(v[np.isfinite(v)], [pmin, pmax])
    if hi <= lo:  
        lo, hi = np.min(v), np.max(v)
        if hi == lo:
            return np.zeros_like(v, dtype=np.uint8)
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo) * 255.0
    return v.astype(np.uint8)


def choose_axis(axis_str):
    axis_str = axis_str.lower()
    if axis_str == "z":
        return 2
    if axis_str == "y":
        return 1
    if axis_str == "x":
        return 0
    raise ValueError("axis must be one of x/y/z")


def save_jpg(arr_uint8, path, quality=95):
    im = Image.fromarray(arr_uint8)
    im.save(path, format="JPEG", quality=quality, subsampling=0)


def save_label_png(label_slice, path):
    mask = (label_slice != 0).astype(np.uint8) * 255
    imageio.imwrite(path, mask)


def export_case(case_dir, out_img_dir, out_lab_dir, axis, digits):
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

    # 如果该病例没有任何包含标注的切片，则跳过
    if len(non_empty_indices) == 0:
        print(f"[skip] {case_name}: no labeled slices found.")
        return 0, num_slices

    saved = 0
    for i in non_empty_indices:
        if ax == 0:
            img_sl = vol_img_u8[i, :, :]
            seg_sl = vol_seg[i, :, :]
        elif ax == 1:
            img_sl = vol_img_u8[:, i, :]
            seg_sl = vol_seg[:, i, :]
        else:
            img_sl = vol_img_u8[:, :, i]
            seg_sl = vol_seg[:, :, i]

        img_sl = np.rot90(img_sl, k=1)
        seg_sl = np.rot90(seg_sl, k=1)

        idx = str(i).zfill(digits)
        img_out = out_img_dir / f"{case_name}_{idx}.jpg"
        lab_out = out_lab_dir / f"{case_name}_{idx}.png"

        save_jpg(img_sl, img_out)
        save_label_png(seg_sl, lab_out)
        saved += 1

    print(f"[done] {case_name}: saved {saved} labeled slices (rotation applied).")
    return saved, num_slices


def main():
    parser = argparse.ArgumentParser(description="Convert 3D NIfTI to 2D slices (image JPG / label PNG), skipping empty-label slices, rotated 90° CCW.")
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
