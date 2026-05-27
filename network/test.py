import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.dataset import FullDataset
from networks.network import NetworkCC
import logging
from tqdm import tqdm
from medpy import metric

parser = argparse.ArgumentParser()
parser.add_argument("--sam2_checkpoint", type=str,default="/checkpoint/sam2_hiera_large.pt",help="Path to SAM2 checkpoint")
parser.add_argument("--test_image_path", type=str,default="/data/lung/test/images/",help="Path to test images")
parser.add_argument("--test_mask_path", type=str,default="/data/lung/test/labels/",help="Path to test masks")
parser.add_argument("--model_path", type=str,default="./checkpoints_lung/best_model.pth",help="Path to trained model checkpoint")
parser.add_argument("--save_path", type=str,default="./test_results/Lung",help="Path to save test results")
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_chans", type=int, default=3)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--freeze_sam2", action="store_true", default=True)
parser.add_argument("--eval_threshold", type=float, default=0.5)
args = parser.parse_args()

def calculate_metrics(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2.0 * intersection) / (union + 1e-8)
    iou = intersection / (pred_binary.sum() + target.sum() - intersection + 1e-8)
    return dice.item(), iou.item()


def calculate_hd95(pred_bin_t, target_t):
    if pred_bin_t.dim() == 4:
        pred_bin_t = pred_bin_t.squeeze(1)
    if target_t.dim() == 4:
        target_t = target_t.squeeze(1)

    hd_list = []
    for b in range(pred_bin_t.shape[0]):
        pred_np = (pred_bin_t[b].detach().cpu().numpy() > 0).astype(np.bool_)
        gt_np = (target_t[b].detach().cpu().numpy() > 0).astype(np.bool_)

        if pred_np.sum() == 0 or gt_np.sum() == 0:
            hd = 0.0
        else:
            try:
                hd = metric.binary.hd95(pred_np, gt_np)
            except Exception:
                hd = 0.0
        hd_list.append(hd)

    return float(np.mean(hd_list)) if hd_list else 0.0


def setup_logging(save_path):
    log_file = os.path.join(save_path, "testing.log")
    os.makedirs(save_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def test(args):
    os.makedirs(args.save_path, exist_ok=True)

    logger = setup_logging(args.save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading test dataset...")
    test_dataset = FullDataset(
        args.test_image_path,
        args.test_mask_path,
        args.img_size,
        mode="test"
    )
    logger.info(f"Test set size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info("Initializing model...")
    model = NetworkCC(
        in_chns=args.in_chans,
        class_num=args.num_classes,
        sam2_checkpoint_path=args.sam2_checkpoint,
        freeze_sam2=args.freeze_sam2
    )
    model.to(device)

    logger.info(f"Loading model from {args.model_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        logger.info("Checkpoint loaded as dict with 'model_state_dict'.")
    else:
        state_dict = ckpt
        logger.info("Checkpoint loaded as raw state_dict.")

    model.load_state_dict(state_dict)
    model.eval()

    dice_scores = []
    iou_scores = []
    hd95_scores = []

    logger.info("Starting testing...")

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            images = batch["image"].to(device)
            targets = batch["label"].to(device)

            outputs = model(images)
            final_output = outputs[-1]

            if final_output.shape[1] > 1:
                pred = final_output[:, 1:2, :, :]
            else:
                pred = final_output

            dice, iou = calculate_metrics(pred, targets, threshold=args.eval_threshold)
            dice_scores.append(dice)
            iou_scores.append(iou)

            pred_bin = (torch.sigmoid(pred) > args.eval_threshold).float()
            hd95 = calculate_hd95(pred_bin, targets)
            hd95_scores.append(hd95)

            logger.info(f"Batch {idx+1} - Dice: {dice:.4f}, IoU: {iou:.4f}, HD95: {hd95:.2f}")

    avg_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    avg_hd95 = float(np.mean(hd95_scores)) if hd95_scores else 0.0

    std_dice = float(np.std(dice_scores)) if dice_scores else 0.0
    std_iou = float(np.std(iou_scores)) if iou_scores else 0.0
    std_hd95 = float(np.std(hd95_scores)) if hd95_scores else 0.0

    logger.info("=" * 50)
    logger.info("Testing Results Summary:")
    logger.info(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}")
    logger.info(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}")
    logger.info(f"Average HD95: {avg_hd95:.2f} ± {std_hd95:.2f}")
    logger.info("=" * 50)

    results_file = os.path.join(args.save_path, "test_results.txt")
    with open(results_file, "w") as f:
        f.write("Testing Results Summary:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test Dataset: {args.test_image_path}\n")
        f.write(f"Number of test images: {len(test_dataset)}\n")
        f.write(f"Threshold: {args.eval_threshold}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Dice: {avg_dice:.4f} ± {std_dice:.4f}\n")
        f.write(f"Average IoU: {avg_iou:.4f} ± {std_iou:.4f}\n")
        f.write(f"Average HD95: {avg_hd95:.2f} ± {std_hd95:.2f}\n")
        f.write("=" * 50 + "\n")

    logger.info(f"Results saved to {results_file}")
    logger.info("Testing completed!")


if __name__ == "__main__":
    test(args)