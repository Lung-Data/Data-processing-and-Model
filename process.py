"""
处理医学CT图像：
1. 遍历输入文件夹中的每个子文件夹
2. 将子文件夹中与文件夹同名的 .nii.gz 文件复制并重命名为 seg.nii.gz
3. 将子文件夹中所有 DICOM (.dcm) 文件按 InstanceNumber 排序后合并，
   修正方向矩阵后转换为 t1c.nii.gz
4. 所有结果保存至输出文件夹，已处理的子文件夹自动跳过
"""

import SimpleITK as sitk
import os
import shutil


def process_local_files():
    input_folder = './dataset'
    output_folder = './process'

    os.makedirs(output_folder, exist_ok=True)

    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        print("=============================================================================>")

        if not os.path.isdir(subfolder_path):
            continue

        output_subfolder = os.path.join(output_folder, subfolder)

        if os.path.exists(output_subfolder):
            print(f"Skipping {subfolder} as it already exists in output.")
            continue

        os.makedirs(output_subfolder, exist_ok=True)

        try:
            nii_file = os.path.join(subfolder_path, f'{subfolder}.nii.gz')
            local_seg_path = os.path.join(output_subfolder, 'seg.nii.gz')

            if os.path.exists(nii_file):
                shutil.copy2(nii_file, local_seg_path)
                print(f'Copied {nii_file} -> {local_seg_path}')
            else:
                print(f"{nii_file} does not exist, skipping seg file.")

            dicom_file_paths = sorted(
                [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith('.dcm')],
                key=get_instance_number
            )

            if dicom_file_paths:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(dicom_file_paths)
                image = reader.Execute()

                direction = list(image.GetDirection())
                direction_matrix = [direction[i:i+3] for i in range(0, 9, 3)]
                direction_matrix[2] = [-v for v in direction_matrix[2]]
                image.SetDirection([item for row in direction_matrix for item in row])

                t1c_path = os.path.join(output_subfolder, 't1c.nii.gz')
                sitk.WriteImage(image, t1c_path)
                print(f'DICOM converted -> {t1c_path}')
            else:
                print(f"No DICOM files found in {subfolder_path}, skipping t1c conversion.")

        except Exception as e:
            print(f"Error processing {subfolder_path}: {e}")

    print("\nAll done!")


def get_instance_number(dicom_file):
    instance_number = sitk.ReadImage(dicom_file).GetMetaData('0020|0013')
    return int(instance_number) if instance_number else float('inf')


if __name__ == '__main__':
    process_local_files()