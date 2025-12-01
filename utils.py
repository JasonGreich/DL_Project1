"""
Utility functions for visualization, metrics, and helpers.
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def get_device():
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def save_checkpoint(model, optimizer, epoch, loss, filepath, scheduler, best_val_loss):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scheduler': scheduler,
        'best_val_loss': best_val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_ckpt_pconv(ckpt_path, model, optimizer=None, for_predict=False, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Some checkpoints: {'model': state_dict}, others: just state_dict
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    # IGNORE missing buffers like mask_kernel_buffer
    load_info = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)

    if for_predict:
        model.eval()
        return model

    step = checkpoint.get('step', 0) if isinstance(checkpoint, dict) else 0
    if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, step
def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint

    Returns:
        epoch, loss
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return epoch, loss


# ============================================================================
# Visualization Functions
# ============================================================================

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor.

    Args:
        tensor: [C, H, W] or [B, C, H, W] normalized tensor
        mean: ImageNet mean
        std: ImageNet std

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


def visualize_segmentation(image, mask, pred_mask=None, save_path=None):
    """
    Visualize segmentation results.

    Args:
        image: [3, H, W] image tensor
        mask: [H, W] ground truth mask
        pred_mask: [H, W] predicted mask (optional)
        save_path: Path to save figure
    """
    image = denormalize(image).cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 1)

    mask = mask.cpu().numpy()

    if pred_mask is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
    else:
        pred_mask = pred_mask.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image)
        axes[0].set_title('Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='tab20')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')

        axes[2].imshow(pred_mask, cmap='tab20')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Segmentation visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_inpainting(original, masked, inpainted, mask, save_path=None):
    """
    Visualize inpainting results.

    Args:
        original: [3, H, W] original image tensor
        masked: [3, H, W] masked image tensor
        inpainted: [3, H, W] inpainted image tensor
        mask: [1, H, W] binary mask
        save_path: Path to save figure
    """
    # Denormalize if needed
    if original.max() > 2.0:
        # Assume already denormalized or in tanh range
        original = (original + 1) / 2
        inpainted = (inpainted + 1) / 2
        masked = (masked + 1) / 2
    else:
        original = denormalize(original)
        inpainted = denormalize(inpainted)
        masked = denormalize(masked)

    original = original.cpu().numpy().transpose(1, 2, 0)
    masked = masked.cpu().numpy().transpose(1, 2, 0)
    inpainted = inpainted.cpu().numpy().transpose(1, 2, 0)
    mask = mask.cpu().numpy().squeeze()

    original = np.clip(original, 0, 1)
    masked = np.clip(masked, 0, 1)
    inpainted = np.clip(inpainted, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    axes[2].imshow(masked)
    axes[2].set_title('Masked Input')
    axes[2].axis('off')

    axes[3].imshow(inpainted)
    axes[3].set_title('Inpainted')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Inpainting visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Mask Generation Functions
# ============================================================================

def generate_random_mask(size, min_size=32, max_size=128):
    """
    Generate random rectangular mask.

    Args:
        size: Image size (H, W)
        min_size: Minimum mask size
        max_size: Maximum mask size

    Returns:
        [1, H, W] binary mask
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.float32)

    # Random rectangle
    y = np.random.randint(0, h - max_size)
    x = np.random.randint(0, w - max_size)
    mask_h = np.random.randint(min_size, max_size)
    mask_w = np.random.randint(min_size, max_size)

    mask[y:y+mask_h, x:x+mask_w] = 1.0

    return torch.from_numpy(mask).unsqueeze(0)


def generate_freeform_mask(size, max_vertex=10, max_length=50, max_brush_width=20):
    """
    Generate free-form stroke mask.

    Args:
        size: Image size (H, W)
        max_vertex: Maximum number of vertices
        max_length: Maximum stroke length
        max_brush_width: Maximum brush width

    Returns:
        [1, H, W] binary mask
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.float32)

    num_vertex = np.random.randint(4, max_vertex)

    for _ in range(num_vertex):
        start_y = np.random.randint(0, h)
        start_x = np.random.randint(0, w)

        for _ in range(np.random.randint(1, 5)):
            angle = np.random.uniform(0, 2 * np.pi)
            length = np.random.randint(10, max_length)
            brush_width = np.random.randint(5, max_brush_width)

            end_y = int(start_y + length * np.sin(angle))
            end_x = int(start_x + length * np.cos(angle))

            end_y = np.clip(end_y, 0, h - 1)
            end_x = np.clip(end_x, 0, w - 1)

            try:
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 1.0, brush_width)
            except:
                # Fallback if cv2 not available
                y_min = max(0, min(start_y, end_y) - brush_width // 2)
                y_max = min(h, max(start_y, end_y) + brush_width // 2)
                x_min = max(0, min(start_x, end_x) - brush_width // 2)
                x_max = min(w, max(start_x, end_x) + brush_width // 2)
                mask[y_min:y_max, x_min:x_max] = 1.0

            start_y, start_x = end_y, end_x

    return torch.from_numpy(mask).unsqueeze(0)


# ============================================================================
# Metric Functions
# ============================================================================

def compute_iou(pred, target, num_classes, ignore_index=255):
    """
    Compute mean Intersection over Union (mIoU).

    Args:
        pred: [B, H, W] predicted class indices
        target: [B, H, W] ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        mIoU value
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        # Ignore pixels with ignore_index
        valid = (target != ignore_index)
        pred_cls = pred_cls & valid
        target_cls = target_cls & valid

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            continue

        iou = intersection / union
        ious.append(iou.item())

    return np.mean(ious) if len(ious) > 0 else 0.0


def compute_dice(pred, target, num_classes, ignore_index=255):
    """
    Compute Dice coefficient.

    Args:
        pred: [B, H, W] predicted class indices
        target: [B, H, W] ground truth class indices
        num_classes: Number of classes
        ignore_index: Index to ignore

    Returns:
        Dice value
    """
    dices = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        valid = (target != ignore_index)
        pred_cls = pred_cls & valid
        target_cls = target_cls & valid

        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum().float() + target_cls.sum().float()

        if total == 0:
            continue

        dice = (2.0 * intersection) / total
        dices.append(dice.item())

    return np.mean(dices) if len(dices) > 0 else 0.0


def compute_psnr(pred, target):
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: [B, C, H, W] predicted image
        target: [B, C, H, W] target image

    Returns:
        PSNR value
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def compute_ssim(pred, target, window_size=11):
    """
    Compute Structural Similarity Index (simplified version).

    Args:
        pred: [B, C, H, W] predicted image
        target: [B, C, H, W] target image
        window_size: Window size for SSIM

    Returns:
        SSIM value
    """
    # Simplified SSIM implementation
    # For production, use skimage.metrics.structural_similarity

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_y_sq
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    return ssim_map.mean().item()


if __name__ == '__main__':
    print("Testing utility functions...")

    # Test device
    device = get_device()

    # Test mask generation
    print("\nTesting mask generation...")
    random_mask = generate_random_mask((512, 512))
    print(f"Random mask shape: {random_mask.shape}")

    freeform_mask = generate_freeform_mask((512, 512))
    print(f"Freeform mask shape: {freeform_mask.shape}")

    # Test metrics
    print("\nTesting metrics...")
    pred = torch.randint(0, 10, (4, 256, 256))
    target = torch.randint(0, 10, (4, 256, 256))

    iou = compute_iou(pred, target, num_classes=10)
    print(f"mIoU: {iou:.4f}")

    dice = compute_dice(pred, target, num_classes=10)
    print(f"Dice: {dice:.4f}")

    pred_img = torch.randn(2, 3, 256, 256)
    target_img = torch.randn(2, 3, 256, 256)

    psnr = compute_psnr(pred_img, target_img)
    print(f"PSNR: {psnr:.2f}")

    ssim = compute_ssim(pred_img, target_img)
    print(f"SSIM: {ssim:.4f}")

    print("\nUtility functions test passed!")
