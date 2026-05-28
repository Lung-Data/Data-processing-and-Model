# Data-processing-and-Model
This repository contains the dataset processing code for the Lung tumor CT segmentation dataset, as well as the model code for segmenting the target region.
## Dataset Acquisition

The Lung tumor CT segmentation dataset is hosted on the **ScienceDB**. To download the data, go to: [Lung tumor CT segmentation dataset](https://www.scidb.cn/en/s/NVB3y2)
<img width="372" height="203" alt="{47BC6FC8-68B6-4300-9F63-EA500AF4062E}" src="https://github.com/user-attachments/assets/9e5a0b06-61c2-4d6a-898e-bf96bed0cb00" />

   The download includes:

   - *Meta-information* 

   - *CT Image Files* — de-identified DICOMs  

   - *Annotation Files* — CT in NIfTI format (from DICOM conversion)
  
## Model
## 🧠Network Architecture
   <img width="556" height="322" alt="image" src="https://github.com/user-attachments/assets/a181d475-b66a-4baf-a0ff-5e84719daab4" />


## 📦 Document Description
🔹 **process.py :** This script converts the DICOM series into unified 3D NIfTI volumes. It automatically sorts the DICOM slices by InstanceNumber and rigorously corrects the Z-axis direction matrix, ensuring strict spatial consistency between the CT imagery and the matched segmentation masks.

🔹 **convert.py :** The script extracts two-dimensional slices from regions containing tumor annotations. To address spatial confusion between different scanner coordinate systems, spatial normalization aligns all images to standard anatomical viewing orientations.

🔹 **crop.py :** The script automatically crops a 224 × 224 ROI centered on the tumor. Automatic adjustment of the starting coordinates near image edges ensures uniformity of input size, allowing the model to focus more on the anatomical region and achieve better segmentation results.

🔹 **split.py :** The script divides the dataset at the patient level into a training set (80%), a validation set (10%), and a test set (10%). This ensures that all slices belonging to the same patient are restricted to specific subsets, preventing slice-level data leakage.

🔹 **network/network.py :** This script defines the core deep learning architecture for lung tumor segmentation. It adapts the pre-trained SAM2 as a robust feature encoder to effectively capture and refine tumor-specific spatial features and boundary details.

🔹 **network/train.py :** This script contains the training code for the model. It uses Dice loss and cross-entropy loss to balance region overlap and pixel-level classification, ensuring stable model convergence and automatically saving the weights of the best-performing model.

🔹 **network/test.py :** This script contains the test code for the model. It tests two metrics of the model: Dice and IoU to evaluate the model's performance.

## ⚠️ Notice
Remember to modify the path to the dataset before running the test and training code. Different image resolutions require modifying the relevant parameter parameters.

## 📫 Contact
If you have any questions, please feel free to contact us:
chenchang@ahut.edu.cn
