import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================
# CONFIG
# =============================
DATA_ROOT = "/kaggle/input/datasets/dasmehdixtr/uavid-v1"
SAM_CKPT  = "/kaggle/input/datasets/nivedsarat/checkpoints/sam2.1_hiera_large.pt"
SAVE_DIR  = "/kaggle/working/checkpoints_final_model"

IMAGE_SIZE = 1024
NUM_CLASSES = 8
BATCH_SIZE = 2
EPOCHS = 80
LR = 6e-4   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# COLOR MAP
# =============================
COLOR_MAP = {
    (128, 0, 0): 0,
    (128, 64, 128): 1,
    (0, 128, 0): 2,
    (128, 128, 0): 3,
    (64, 0, 128): 4,
    (192, 0, 192): 5,
    (64, 64, 0): 6,
    (0, 0, 0): 7,
}

# =============================
# DATASET
# =============================
class UAVidDataset(Dataset):
    def __init__(self, root_dir, split="uavid_train"):
        self.samples = []

        split_dir = os.path.join(root_dir, split)
        for seq in sorted(os.listdir(split_dir)):
            img_dir = os.path.join(split_dir, seq, "Images")
            mask_dir = os.path.join(split_dir, seq, "Labels")

            if not os.path.isdir(img_dir):
                continue

            for img_name in sorted(os.listdir(img_dir)):
                self.samples.append((
                    os.path.join(img_dir, img_name),
                    os.path.join(mask_dir, img_name)
                ))

        print(f"{split}: {len(self.samples)} samples")

        self.img_tf = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        self.mask_tf = T.Resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=T.InterpolationMode.NEAREST
        )

        #  AUGMENTATION
        if "train" in split:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=0,
                    p=0.5
                ),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        img = np.array(img)
        mask = np.array(mask)

        #AUGMENT FIRST (same size)
        if self.aug:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # THEN resize
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        img = self.img_tf(img)
        mask = self.mask_tf(mask)
        mask = np.array(mask)

        # Convert to class mask
        class_mask = np.full((mask.shape[0], mask.shape[1]), 255, dtype=np.int64)
        for color, cid in COLOR_MAP.items():
            class_mask[np.all(mask == color, axis=-1)] = cid

        return img, torch.from_numpy(class_mask)

# =============================
# CLASS WEIGHTS
# =============================

def compute_class_weights(dataset):
    print("Computing class weights...")
    counts = torch.zeros(NUM_CLASSES)

    for _, mask in dataset:
        unique, c = torch.unique(mask, return_counts=True)
        for u, cnt in zip(unique, c):
            if u < NUM_CLASSES:
                counts[u] += cnt

    weights = 1.0 / (counts + 1e-6)
    weights = torch.sqrt(weights)   # stabilize
    weights = weights / weights.sum() * NUM_CLASSES

    print("Weights:", weights)
    return weights

# =============================
# DICE LOSS
# =============================
class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        onehot = F.one_hot(
            targets.clamp(0, self.num_classes-1),
            self.num_classes
        ).permute(0,3,1,2).float()

        dims = (0,2,3)
        inter = torch.sum(probs * onehot, dims)
        union = torch.sum(probs + onehot, dims)

        dice = (2*inter + 1e-6)/(union + 1e-6)
        return 1 - dice.mean()

# =============================
# METRICS
# =============================
def compute_confusion_matrix(pred, target,NUM_CLASSES):
    pred = pred.view(-1)
    target = target.view(-1)
    mask = (target >= 0) & (target < NUM_CLASSES)

    cm = torch.bincount(
        NUM_CLASSES * target[mask] + pred[mask],
        minlength=NUM_CLASSES**2
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    return cm

def compute_metrics_from_cm(cm, eps=1e-6):
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    accuracy = tp.sum() / cm.sum()

    return {
        "accuracy": accuracy.item(),
        "precision_per_class": precision.cpu().numpy(),
        "recall_per_class": recall.cpu().numpy(),
        "iou_per_class": iou.cpu().numpy(),
        "dice_per_class": dice.cpu().numpy(),
        "miou": iou.mean().item(),
        "mdice": dice.mean().item(),
    }

# =============================
# MAIN
# =============================
def main():

    train_ds = UAVidDataset(DATA_ROOT, "uavid_train")
    val_ds   = UAVidDataset(DATA_ROOT, "uavid_val")

    train_loader = DataLoader(train_ds, BATCH_SIZE, True, num_workers=4)
    val_loader   = DataLoader(val_ds, 1, False, num_workers=4)

    model = SAM2UNet(SAM_CKPT, NUM_CLASSES).to(DEVICE)

    weights = compute_class_weights(train_ds).to(DEVICE)

    ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
    dice_loss = DiceLoss(NUM_CLASSES)

    def combined_loss(logits, targets):
        return ce_loss(logits, targets) + 0.7 * dice_loss(logits, targets)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    scaler = GradScaler("cuda")

    best_miou = 0.0
    log_path = os.path.join(SAVE_DIR, "train_log.csv")

    for epoch in range(1, EPOCHS + 1):

        model.train()
        train_loss = 0.0
        train_main = 0.0
        train_aux  = 0.0

        for img, mask in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):

            img = img.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            with autocast("cuda"):
                out, out1, out2 = model(img)

                loss_main = combined_loss(out, mask)
                loss_aux1 = combined_loss(out1, mask)
                loss_aux2 = combined_loss(out2, mask)

                loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_main += loss_main.item()
            train_aux  += (loss_aux1.item() + loss_aux2.item()) / 2

        train_loss /= len(train_loader)
        train_main /= len(train_loader)
        train_aux  /= len(train_loader)

        # =============================
        # VALIDATION
        # =============================
        model.eval()
        total_cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, device=DEVICE)
        val_loss = 0.0

        with torch.no_grad():
            for img, mask in val_loader:

                img = img.to(DEVICE)
                mask = mask.to(DEVICE)

                with autocast("cuda"):
                    out, out1, out2 = model(img)

                    loss_main = combined_loss(out, mask)
                    loss_aux1 = combined_loss(out1, mask)
                    loss_aux2 = combined_loss(out2, mask)

                    loss = loss_main + 0.4 * loss_aux1 + 0.4 * loss_aux2

                pred = torch.argmax(out, dim=1)
                total_cm += compute_confusion_matrix(pred, mask, NUM_CLASSES)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        metrics = compute_metrics_from_cm(total_cm)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n========== Epoch {epoch:03d}/{EPOCHS} ==========")
        print(f"LR: {current_lr:.6f}")

        print("\nTrain:")
        print(f"  Total Loss : {train_loss:.4f}")
        print(f"  Main Loss  : {train_main:.4f}")
        print(f"  Aux Loss   : {train_aux:.4f}")

        print("\nValidation:")
        print(f"  Val Loss   : {val_loss:.4f}")
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  Mean IoU   : {metrics['miou']:.4f}")
        print(f"  Mean Dice  : {metrics['mdice']:.4f}")

        print("\nPer-class metrics:")
        for cls in range(NUM_CLASSES):
            print(
                f"  Class {cls}: "
                f"IoU={metrics['iou_per_class'][cls]:.4f}, "
                f"Dice={metrics['dice_per_class'][cls]:.4f}, "
                f"Prec={metrics['precision_per_class'][cls]:.4f}, "
                f"Rec={metrics['recall_per_class'][cls]:.4f}"
            )
        # ================================
        # CSV LOG
        # ================================
        with open(log_path, "a") as f:

            # Write header only once
            if epoch == 1:
                header = [
                    "epoch",
                    "lr",
                    "train_loss", "train_main", "train_aux",
                    "val_loss",
                    "accuracy", "miou", "mdice"
                ]

                # Per-class metrics
                for cls in range(NUM_CLASSES):
                    header += [
                        f"iou_c{cls}",
                        f"dice_c{cls}",
                        f"prec_c{cls}",
                        f"rec_c{cls}"
                    ]

                f.write(",".join(header) + "\n")

            # Current LR
            current_lr = optimizer.param_groups[0]["lr"]

            # Row data
            row = [
                epoch,
                current_lr,
                train_loss, train_main, train_aux,
                val_loss,
                metrics['accuracy'],
                metrics['miou'],
                metrics['mdice']
            ]

            # Add per-class values
            for cls in range(NUM_CLASSES):
                row += [
                    metrics['iou_per_class'][cls],
                    metrics['dice_per_class'][cls],
                    metrics['precision_per_class'][cls],
                    metrics['recall_per_class'][cls]
                ]

            # Write row
            f.write(",".join(
                [f"{x:.6f}" if isinstance(x, float) else str(x) for x in row]
            ) + "\n")
            
        

        if metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR,"best_model.pth"))
            print(f"\n✓ New Best mIoU: {best_miou:.4f} — Model Saved")

    print("\nTraining Complete")

if __name__ == "__main__":
    main()
