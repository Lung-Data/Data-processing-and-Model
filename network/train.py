import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
from utils.dataset import FullDataset
from networks.network import NetworkCC

parser = argparse.ArgumentParser("LCASAM2 Training")
parser.add_argument("--sam2_checkpoint", type=str,
                    default="./checkpoint/sam2_hiera_large.pt",
                    help="Path to SAM2 checkpoint")
parser.add_argument("--train_image_path", type=str,
                    default="./data/lung/train/",
                    help="Path to training images")
parser.add_argument("--train_mask_path", type=str,
                    default="./data/lung/train_masks/",
                    help="Path to training masks")
parser.add_argument("--test_image_path", type=str,
                    default="./data/lung/test/",
                    help="Path to test images")
parser.add_argument("--test_mask_path", type=str,
                    default="./data/lung/test_masks/",
                    help="Path to test masks")
parser.add_argument('--save_path', type=str,
                    default='./checkpoints_lung',
                    help="Path to save model checkpoints")
parser.add_argument("--epoch", type=int, default=300)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--in_chans", type=int, default=3)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--lambda_dice", type=float, default=0.6, help="Lambda for Dice loss")
parser.add_argument("--freeze_sam2", action="store_true", default=True)
parser.add_argument("--val_interval", type=int, default=1)
parser.add_argument("--save_interval", type=int, default=50, help="Interval to save regular checkpoints")

args = parser.parse_args()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class HybridLoss(nn.Module):
    def __init__(self, lambda_dice=0.5):
        super().__init__()
        self.lambda_dice = lambda_dice
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.lambda_dice * dice + (1 - self.lambda_dice) * bce


def setup_optimizer(model, base_lr):
    sam2_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'hiera_encoder' in name and 'adapter' in name:
                sam2_params.append(param)
            else:
                other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': sam2_params, 'lr': base_lr * 0.1, 'weight_decay': 1e-5},
        {'params': other_params, 'lr': base_lr, 'weight_decay': 1e-5}
    ], betas=(0.9, 0.999))

    return optimizer


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        images = batch['image'].to(device)
        targets = batch['label'].to(device).squeeze(1).long()

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, (list, tuple)):
            loss = sum([criterion(out, targets) for out in outputs])
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            images = batch['image'].to(device)
            targets = batch['label'].to(device).squeeze(1).long()

            outputs = model(images)

            if isinstance(outputs, (list, tuple)):
                loss = sum([criterion(out, targets) for out in outputs])
                main_output = outputs[0]
            else:
                loss = criterion(outputs, targets)
                main_output = outputs

            total_loss += loss.item()

            preds = torch.argmax(main_output, dim=1)
            target_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
            pred_one_hot = F.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2).float()

            intersection = (pred_one_hot * target_one_hot).sum(dim=(2, 3))
            union = pred_one_hot.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

            smooth = 1e-5
            dice = (2. * intersection + smooth) / (union + smooth)

            if num_classes > 1:
                batch_dice = dice[:, 1:].mean().item()
            else:
                batch_dice = dice.mean().item()

            total_dice += batch_dice

    return total_loss / len(val_loader), total_dice / len(val_loader)


def setup_logging(save_path):
    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main(args):
    logger = setup_logging(args.save_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, args.img_size, mode='train')
    val_dataset = FullDataset(args.test_image_path, args.test_mask_path, args.img_size, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = NetworkCC(
        in_chns=args.in_chans,
        class_num=args.num_classes,
        sam2_checkpoint_path=args.sam2_checkpoint,
        freeze_sam2=args.freeze_sam2
    ).to(device)

    criterion = HybridLoss(lambda_dice=args.lambda_dice)
    optimizer = setup_optimizer(model, args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    logger.info("=" * 60)
    logger.info(f"Training Configuration:")
    logger.info(f"  Optimizer: AdamW (β1=0.9, weight_decay=1e-5)")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Image size: {args.img_size}×{args.img_size}")
    logger.info(f"  Loss: λ={args.lambda_dice}*Dice + {1 - args.lambda_dice}*BCE")
    logger.info("=" * 60)

    best_dice = 0.0

    for epoch in range(args.epoch):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch + 1}/{args.epoch} - Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}")

        if (epoch + 1) % args.val_interval == 0:
            val_loss, val_dice = validate(model, val_loader, criterion, device, args.num_classes)
            logger.info(f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model.pth'))
                logger.info(f"Best model saved! Val Dice: {best_dice:.4f}")

        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.save_path, f'epoch_{epoch + 1}.pth'))

    torch.save(model.state_dict(), os.path.join(args.save_path, 'final_model.pth'))
    logger.info(f"Training completed! Best Val Dice: {best_dice:.4f}")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(42)
    main(args)