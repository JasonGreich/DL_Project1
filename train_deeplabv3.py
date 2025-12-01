import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from segmentation_models.deeplabv3 import DeepLabV3Predictor
from dataset_preprocess.preprocess import get_semantic_dataloaders


class DeepLabV3Trainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.best_miou = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
        self.num_classes = None  # will be set in load_model

    # -------------------------------------------------------
    # Model / optimizer / loss
    # -------------------------------------------------------
    def load_model(self, num_classes=183):
        """
        Load pretrained DeepLabV3-ResNet50 backbone + replace classifier head
        to output `num_classes` semantic labels.
        """
        print("[INFO] Loading pre-trained DeepLabV3-ResNet50...")
        predictor = DeepLabV3Predictor(device=self.device)
        self.model = predictor.model

        # Replace last classifier layer (Conv2d) to have 183 outputs
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

        self.model.to(self.device)
        self.num_classes = num_classes

        print(f"[INFO] DeepLab classifier head set to {num_classes} classes.")
        return self.model

    def setup_optimizer(self, lr=0.001, weight_decay=0.0005):
        """
        Use different lr for backbone vs classifier (common fine-tuning trick).
        """
        params_to_optimize = [
            {'params': [p for p in self.model.backbone.parameters() if p.requires_grad], 'lr': lr * 0.1},
            {'params': [p for p in self.model.classifier.parameters() if p.requires_grad], 'lr': lr},
        ]

        self.optimizer = optim.Adam(params_to_optimize, lr=lr, weight_decay=weight_decay)
        return self.optimizer

    def setup_loss(self):
        """
        CrossEntropy with ignore_index=255 for 'void' pixels.
        """
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=255,
            reduction='mean'
        )
        return self.criterion

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------
    def is_valid_batch(self, masks):
        """
        Ensure mask values are in allowed range:
          - >= 0
          - <= 255
          - and all non-255 labels < num_classes
        """
        if self.num_classes is None:
            raise ValueError("num_classes is not set. Call load_model() first.")

        min_val = masks.min().item()
        max_val = masks.max().item()

        # Basic sanity
        if min_val < 0:
            return False
        if max_val > 255:
            return False

        # Allow ignore_index=255, but all other labels must be < num_classes
        if max_val >= self.num_classes and max_val != 255:
            return False

        return True

    # -------------------------------------------------------
    # Training / validation
    # -------------------------------------------------------
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        num_valid_batches = 0
        num_skipped = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)          # [N, 3, H, W]
            masks = batch['mask'].long().to(self.device)     # [N, H, W]

            if not self.is_valid_batch(masks):
                print(f"  Skipping batch {batch_idx+1} (invalid mask values: "
                      f"min={masks.min().item()}, max={masks.max().item()})")
                num_skipped += 1
                continue

            outputs = self.model(images)['out']  # [N, C, H, W]

            # (Optional) debug first batch once
            # if batch_idx == 0 and epoch == 1:
            #     print("DEBUG outputs.shape:", outputs.shape)
            #     print("DEBUG masks min/max:", masks.min().item(), masks.max().item())

            loss = self.criterion(outputs, masks)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_valid_batches += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {avg_loss:.4f} | Skipped: {num_skipped}")

        avg_epoch_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0
        print(f"[INFO] Skipped {num_skipped}/{len(train_loader)} batches due to invalid masks")
        return avg_epoch_loss

    def compute_iou_per_class(self, pred_logits, target):
        """
        pred_logits: [N, C, H, W]
        target:      [N, H, W], with ignore_index=255 possibly
        """
        num_classes = self.num_classes
        pred = pred_logits.argmax(dim=1)        # [N, H, W]
        target = target.long()                  # [N, H, W]

        # Ignore index handling: do not count 255 pixels in IoU
        ignore_mask = (target == 255)
        pred = pred.clone()
        target = target.clone()
        pred[ignore_mask] = -1
        target[ignore_mask] = -1

        iou_per_class = []

        for cls in range(num_classes):
            pred_mask = (pred == cls).float()
            target_mask = (target == cls).float()

            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection

            if union == 0:
                # If no pixels for this class in both pred and target, define IoU as 1
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union

            iou_per_class.append(iou.item())

        return np.mean(iou_per_class)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_miou = 0.0
        num_valid_batches = 0
        num_skipped = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                images = batch['image'].to(self.device)
                masks = batch['mask'].long().to(self.device)

                if not self.is_valid_batch(masks):
                    num_skipped += 1
                    continue

                outputs = self.model(images)['out']
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                miou = self.compute_iou_per_class(outputs, masks)
                total_miou += miou
                num_valid_batches += 1

        avg_loss = total_loss / num_valid_batches if num_valid_batches > 0 else 0
        avg_miou = total_miou / num_valid_batches if num_valid_batches > 0 else 0

        print(f"[INFO] Validation: Skipped {num_skipped} batches due to invalid masks")
        return avg_loss, avg_miou

    # -------------------------------------------------------
    # Checkpointing / history
    # -------------------------------------------------------
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_dir = Path('checkpoints/deeplabv3')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'num_classes': self.num_classes,
        }

        if is_best:
            path = checkpoint_dir / 'best_model.pth'
            print(f"[INFO] Saving best model to {path}")
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def save_training_history(self):
        history_dir = Path('training_logs')
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f'deeplabv3_history_{timestamp}.json'

        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

        print(f"[INFO] Training history saved to {history_path}")

    # -------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------
    def train(self, data_dir, num_epochs=30, batch_size=4, lr=0.001, img_size=512, num_classes=183):
        print(f"\n{'='*70}")
        print("FINE-TUNING DEEPLABV3 ON COCO SEMANTIC SEGMENTATION (183 CLASSES)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs} | Batch size: {batch_size} | LR: {lr} | Image size: {img_size}")
        print(f"Number of classes: {num_classes}")
        print(f"{'='*70}\n")

        # Model / optimizer / loss
        self.load_model(num_classes=num_classes)
        self.setup_optimizer(lr=lr)
        self.setup_loss()

        print("[INFO] Loading COCO semantic segmentation dataloaders...")
        train_loader, val_loader = get_semantic_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=0
        )

        print(f"[INFO] Training samples: {len(train_loader.dataset)}")
        print(f"[INFO] Validation samples: {len(val_loader.dataset)}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)

            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)

            val_loss, val_miou = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_miou'].append(val_miou)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

            is_best = val_miou > self.best_miou
            if is_best:
                self.best_miou = val_miou
                print(f"[INFO] Best validation mIoU: {self.best_miou:.4f}")

            self.save_checkpoint(epoch, is_best=is_best)

        self.save_training_history()
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"Best validation mIoU: {self.best_miou:.4f}")
        print(f"{'='*70}\n")


if __name__ == '__main__':
    trainer = DeepLabV3Trainer(device='cuda' if torch.cuda.is_available() else 'cpu')

    trainer.train(
        data_dir='dataset',
        num_epochs=30,
        batch_size=4,
        lr=0.001,
        img_size=512,
        num_classes=183
    )
