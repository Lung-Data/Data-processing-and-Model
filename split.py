import os
import shutil
import random
from math import floor


def create_dirs(base_path, splits, sub_dirs):
    for split in splits:
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(base_path, split, sub_dir), exist_ok=True)


def split_dataset_by_patient(img_dir, mask_dir, out_dir, img_ext='.jpg', mask_ext='.png', seed=42):
    random.seed(seed)

    splits = ['train', 'val', 'test']
    sub_dirs = ['images', 'masks']
    create_dirs(out_dir, splits, sub_dirs)

    all_files = [f for f in os.listdir(img_dir) if f.endswith(img_ext)]

    patient_ids = list(set([f.split('_')[0] for f in all_files]))
    patient_ids.sort()
    random.shuffle(patient_ids)

    total_patients = len(patient_ids)
    train_count = floor(total_patients * 0.8)
    val_count = floor(total_patients * 0.1)

    train_pids = set(patient_ids[:train_count])
    val_pids = set(patient_ids[train_count:train_count + val_count])
    test_pids = set(patient_ids[train_count + val_count:])

    print(f"Total patients: {total_patients}")
    print(f"Train: {len(train_pids)}, Val: {len(val_pids)}, Test: {len(test_pids)}")

    for file_name in all_files:
        pid = file_name.split('_')[0]

        if pid in train_pids:
            current_split = 'train'
        elif pid in val_pids:
            current_split = 'val'
        else:
            current_split = 'test'

        base_name = os.path.splitext(file_name)[0]
        mask_name = base_name + mask_ext

        src_img_path = os.path.join(img_dir, file_name)
        src_mask_path = os.path.join(mask_dir, mask_name)

        dst_img_path = os.path.join(out_dir, current_split, 'images', file_name)
        dst_mask_path = os.path.join(out_dir, current_split, 'masks', mask_name)

        shutil.copy2(src_img_path, dst_img_path)

        if os.path.exists(src_mask_path):
            shutil.copy2(src_mask_path, dst_mask_path)
        else:
            print(f"Warning: Mask file not found -> {src_mask_path}")

    print("Dataset splitting completed successfully.")


if __name__ == "__main__":
    IMAGE_DIRECTORY = "./original_images"
    MASK_DIRECTORY = "./original_masks"
    OUTPUT_DIRECTORY = "./split_dataset"

    IMAGE_EXTENSION = ".jpg"
    MASK_EXTENSION = ".png"

    split_dataset_by_patient(
        img_dir=IMAGE_DIRECTORY,
        mask_dir=MASK_DIRECTORY,
        out_dir=OUTPUT_DIRECTORY,
        img_ext=IMAGE_EXTENSION,
        mask_ext=MASK_EXTENSION
    )