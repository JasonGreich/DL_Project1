"""
Training script for semantic segmentation models.
Supports UNet, SegFormer, Mask R-CNN, and SegNet.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np

from dataset_preprocess.preprocess import get_segmentation_dataloaders
from segmentation_models.unet import ResNetUNet, SegmentationLoss
from utils import (
    get_device, save_checkpoint, load_checkpoint,
    plot_training_curves, visualize_segmentation,
    compute_iou, compute_dice
)


def train_epoch(model, dataloader, criterion, optimizer, device, model_name='unet'):
    """
    Train for one epoch.

    Args:
        model: Segmentation model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        model_name: Name of the model

    Returns:
        Average training loss, average mIoU
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            iou = compute_iou(preds, masks, num_classes=183)

        running_loss += loss.item()
        running_iou += iou
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{iou:.4f}'
        })

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches

    return avg_loss, avg_iou


@torch.no_grad()
def validate(model, dataloader, criterion, device, model_name='unet'):
    """
    Validate the model.

    Args:
        model: Segmentation model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to use
        model_name: Name of the model

    Returns:
        Average validation loss, average mIoU, average Dice
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc='Validation')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        # Compute metrics
        preds = torch.argmax(outputs, dim=1)
        iou = compute_iou(preds, masks, num_classes=183)
        dice = compute_dice(preds, masks, num_classes=183)

        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{iou:.4f}',
            'Dice': f'{dice:.4f}'
        })

    avg_loss = running_loss / num_batches
    avg_iou = running_iou / num_batches
    avg_dice = running_dice / num_batches

    return avg_loss, avg_iou, avg_dice

def train_model(model_name, data_dir, num_epochs=20, batch_size=8, lr=1e-4,
                save_dir='checkpoints_unet', resume=None):
    """
    Train segmentation model.

    Args:
        model_name: Name of the model ('unet', 'segformer', 'maskrcnn', 'segnet')
        data_dir: Path to COCO data directory
        num_epochs: Total number of epochs to train for (from epoch 0)
        batch_size: Batch size
        lr: Base learning rate
        save_dir: Directory to save checkpoints
        resume: Path to checkpoint to resume from (e.g. 'checkpoints_unet/unet_best.pth')
    """
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device = get_device()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} for Semantic Segmentation")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_loader, val_loader = get_segmentation_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=512,
        num_workers=4
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"\nInitializing {model_name} model...")
    if model_name == 'unet':
        model = ResNetUNet(num_classes=183, pretrained=True, freeze_encoder=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # ------------------------------------------------------------------
    # Loss, optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = SegmentationLoss(num_classes=183)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ------------------------------------------------------------------
    # Resume from checkpoint (if provided)
    # ------------------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')

    if resume is not None:
        print(f"\nResuming from checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device, weights_only=False)

        # Make sure your checkpoint uses these keys:
        # "model_state_dict", "optimizer_state_dict", "epoch", "loss", optionally "scheduler_state_dict", "best_val_loss"
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))

        print(f" → Starting at epoch {start_epoch}, previous val_loss = {checkpoint['loss']:.4f}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    train_losses = []
    val_losses = []

    print("\nStarting training...\n")

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 60)

        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, model_name
        )
        print(f"Train Loss: {train_loss:.4f} | Train mIoU: {train_iou:.4f}")

        # Validate
        val_loss, val_iou, val_dice = validate(
            model, val_loader, criterion, device, model_name
        )
        print(f"Val   Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}")

        # LR scheduling on val loss
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ---- Save checkpoint for this epoch ----
        checkpoint_path = save_dir / f'{model_name}_epoch_{epoch + 1}.pth'
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            filepath=checkpoint_path,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
        )

        # ---- Save best model ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_dir / f'{model_name}_best.pth'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_loss,
                filepath=best_path,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
            )
            print(f"Best model saved with val_loss: {val_loss:.4f}")

        print()

    # ------------------------------------------------------------------
    # Plot curves
    # ------------------------------------------------------------------
    plot_path = save_dir / f'{model_name}_training_curves.png'
    plot_training_curves(train_losses, val_losses, save_path=plot_path)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train semantic segmentation models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'segformer', 'maskrcnn', 'segnet'],
                       help='Model to train')
    parser.add_argument('--data_dir', type=str, default='../../dataset',
                       help='Path to COCO data directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints_unet',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        resume=args.resume
    )


if __name__ == '__main__':
    import sys

    sys.argv = [
        'script_name',
        '--model', 'unet',
        '--data_dir', '../../dataset',
        '--epochs', '15',
        '--batch_size', '4 ',
        '--lr', '1e-4',
        '--save_dir', 'checkpoints_unet',
        # '--resume', 'checkpoints_unet/unet_best.pth',

    ]
    main()
