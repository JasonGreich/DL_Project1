"""
Preprocessing and data loading utilities for COCO dataset.
Supports both:
  1. Instance Segmentation (Mask R-CNN) - uses COCO-Instances JSON
  2. Semantic Segmentation (UNet/SegFormer/SegNet) - uses COCO-Stuff PNG masks

Assumes val-only setup:
  - Only val2017 images are present
  - Subset files:
      subset/train_subset.txt
      subset/val_subset.txt
    are created from val2017 and used to define splits.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
import random
import matplotlib.pyplot as plt
from .mask_generator import MaskGenerator
from glob import glob
import cv2

# ============================================================================
# 0. COMMONS
# ============================================================================

def collate_fn(batch):
    """Custom collate function for instance segmentation."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ============================================================================
# 1. INSTANCE SEGMENTATION DATASET (for Mask R-CNN)
# ============================================================================

class COCOInstanceDataset(Dataset):
    """
    COCO Instance Segmentation Dataset (val-only setup).

    Used for: Mask R-CNN
    Dataset: COCO-Instances (instances_val2017.json)

    We use:
      - val2017/ for images (for both splits)
      - subset/train_subset.txt, subset/val_subset.txt for split definition

    Args:
        data_dir: Path to COCO data directory
        split: 'train' or 'val'
        img_size: Image size (default 512x512)
        augment: Whether to apply augmentation (only used for split='train')
    """
    def __init__(self, data_dir, split='train', img_size=512, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        # Always use val2017 for images
        self.img_dir = self.data_dir / 'val2017'
        ann_file = self.data_dir / 'annotations' / 'instances_val2017.json'

        if not self.img_dir.exists():
            raise FileNotFoundError(f"val2017 directory not found at: {self.img_dir}")
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO instances annotation file not found at: {ann_file}")

        # Load COCO API
        print(f"Loading COCO-Instances annotations from {ann_file}...")
        self.coco = COCO(str(ann_file))

        # Load subset file to filter images
        subset_file = self.data_dir / 'subset' / f'{split}_subset.txt'
        if subset_file.exists():
            with open(subset_file, 'r') as f:
                self.image_names = [line.strip() for line in f]
        else:
            # Fallback: use all images (not ideal, but safe)
            self.image_names = [f.name for f in sorted(self.img_dir.glob('*.jpg'))]

        # Filter to only images that exist in both subset and COCO annotations
        self.image_ids = []
        for img_name in self.image_names:
            img_id = int(img_name.replace('.jpg', ''))
            if img_id in self.coco.imgs:
                self.image_ids.append(img_id)

        print(f"Loaded {len(self.image_ids)} images for instance segmentation ({split} split)")

        # Transforms
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']

        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Resize image
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)

        # Get annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        # Convert annotations to masks and boxes
        boxes = []
        masks = []
        labels = []

        for ann in anns:
            # Get bounding box [x, y, w, h] and convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']

            # Scale bbox to resized image
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            x1 = x * scale_x
            y1 = y * scale_y
            x2 = (x + w) * scale_x
            y2 = (y + h) * scale_y

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

            # Convert segmentation to binary mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                rle = coco_mask.frPyObjects(ann['segmentation'], orig_h, orig_w)
                mask = coco_mask.decode(rle)
                if len(mask.shape) == 3:
                    mask = mask.sum(axis=2) > 0
            else:
                # RLE format
                mask = coco_mask.decode(ann['segmentation'])

            # Resize mask to match image size
            mask = Image.fromarray(mask.astype(np.uint8))
            mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
            mask = np.array(mask, dtype=np.float32)
            masks.append(mask)

        # Convert to tensors
        image_tensor = transforms.ToTensor()(image)
        image_tensor = self.normalize(image_tensor)

        if len(boxes) == 0:
            # No objects in image, create dummy data
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, self.img_size, self.img_size), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([img_id])
        }

        return image_tensor, target


# ============================================================================
# 2. SEMANTIC SEGMENTATION DATASET (for UNet/SegFormer/SegNet)
# ============================================================================

class COCOSemanticDataset(Dataset):
    """
    COCO Semantic Segmentation Dataset (val-only setup).

    Used for: UNet, SegFormer, SegNet
    Dataset: COCO-Stuff (PNG masks)

    We use:
      - val2017/ for images (for both splits)
      - stuffthingmaps_trainval2017/val2017 for masks
      - subset/train_subset.txt, subset/val_subset.txt for split definition

    Returns single-channel class map (H×W) with class ID per pixel.
    """
    def __init__(self, data_dir, split='train', img_size=512, augment=True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

        # Always use val2017 for images and masks
        self.img_dir = self.data_dir / 'val2017'
        self.mask_dir = self.data_dir / 'stuffthingmaps_trainval2017' / 'val2017'

        if not self.img_dir.exists():
            raise FileNotFoundError(f"val2017 directory not found at: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"COCO-Stuff val2017 masks directory not found at: {self.mask_dir}")

        # Load image list from subset file
        subset_file = self.data_dir / 'subset' / f'{split}_subset.txt'
        if subset_file.exists():
            with open(subset_file, 'r') as f:
                self.image_names = [line.strip() for line in f]
        else:
            # Fallback: use all images in directory
            self.image_names = [f.name for f in sorted(self.img_dir.glob('*.jpg'))]

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        print(f"Loaded {len(self.image_names)} images for semantic segmentation ({split} split)")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_id = img_name.replace('.jpg', '')

        # Load image
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        # Load semantic mask (PNG)
        mask_path = self.mask_dir / f'{img_id}.png'
        if mask_path.exists():
            mask = Image.open(mask_path)
            mask = np.array(mask, dtype=np.int64)
        else:
            # Create dummy mask if annotation doesn't exist
            mask = np.zeros((orig_h, orig_w), dtype=np.int64)

        # Apply augmentation (IMPORTANT: same transform for both image and mask)
        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = np.fliplr(mask).copy()

            # Random rotation (±10 degrees)
            angle = np.random.uniform(-10, 10)
            image = image.rotate(angle, resample=Image.Resampling.BILINEAR)
            mask_pil = Image.fromarray(mask.astype(np.uint8))
            mask_pil = mask_pil.rotate(angle, resample=Image.Resampling.NEAREST)
            mask = np.array(mask_pil, dtype=np.int64)

        # Resize
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
        mask = np.array(mask, dtype=np.int64)

        # Apply color jitter to image only (not mask)
        if self.augment:
            color_jitter = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
            image = color_jitter(image)

        # Convert image to tensor and normalize
        image = transforms.ToTensor()(image)
        image = self.normalize(image)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).long()
        mask[mask == 183] = 0
        # Clip mask values to valid range (COCO-Stuff has 183 classes: 0-182)
        mask = torch.clamp(mask, 0, 182)

        return {
            'image': image,
            'mask': mask,
            'img_id': img_id
        }


# ============================================================================
# 3. INPAINTING DATASET (same for both instance and semantic modes)
# ============================================================================

class COCOInpaintingDataset(Dataset):
    """
        __getitem__ returns: masked_img, mask, img
    """
    def __init__(
        self,
        data_dir,
        img_size=256,
        split='train',       # 'train' or 'val'
        num_irregulars=10,
    ):
        super().__init__()
        self.data_dir    = Path(data_dir)
        self.img_size    = img_size
        self.split       = split

        # transforms: same as notebook
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),           # [0, 1], no ImageNet normalization
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),           # [0, 1], 3 channels
        ])

        # mask generator (notebook style)
        self.maskGenerator = MaskGenerator(
            height=img_size,
            width=img_size,
            channels=3,
            num_irregulars=num_irregulars,
        )

        # images: use your existing val2017 + subset files
        self.img_dir = self.data_dir / "val2017"
        if not self.img_dir.exists():
            raise FileNotFoundError(f"val2017 directory not found at: {self.img_dir}")

        subset_file = self.data_dir / "subset" / f"{split}_subset.txt"
        if subset_file.exists():
            with subset_file.open("r") as f:
                names = [line.strip() for line in f]
            self.paths = [str(self.img_dir / name) for name in names]
        else:
            # fallback: all jpgs in val2017
            self.paths = sorted(glob(str(self.img_dir / "*.jpg")))

        print(f"Loaded {len(self.paths)} images for inpainting ({split} split)")

    def __len__(self):
        return len(self.paths)

    def load_img(self, path: str):
        try:
            img = Image.open(path)
            return img
        except (FileNotFoundError, UnidentifiedImageError):
            raise FileNotFoundError(f"x Can't load image from: {path}")

    def __getitem__(self, index):
        img_path = self.paths[index]

        # ---- image (same as notebook) ----
        img = self.load_img(img_path)
        img = self.img_transform(img.convert("RGB"))      # (3, H, W), [0, 1]

        # ---- mask (same as notebook) ----
        mask_np = self.maskGenerator.generate() * 255     # (H, W, 3), 0/255
        mask_pil = Image.fromarray(mask_np.astype(np.uint8))
        mask = self.mask_transform(mask_pil.convert("RGB"))  # (3, H, W), [0, 1]

        # masked image (notebook: img * mask)
        masked_img = img * mask

        # return exactly like InitDataset: (img * mask, mask, img)
        return masked_img, mask, img


# ============================================================================
# 4. DATALOADER FACTORY FUNCTIONS
# ============================================================================

def get_instance_dataloaders(data_dir, batch_size=8, img_size=512, num_workers=4):
    """
    Create train and validation dataloaders for instance segmentation (Mask R-CNN).

    Uses val2017 images and subset txt files for splitting.
    """
    train_dataset = COCOInstanceDataset(
        data_dir=data_dir,
        split='train',
        img_size=img_size,
        augment=True
    )

    val_dataset = COCOInstanceDataset(
        data_dir=data_dir,
        split='val',
        img_size=img_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def get_semantic_dataloaders(data_dir, batch_size=8, img_size=512, num_workers=4):
    """
    Create train and validation dataloaders for semantic segmentation (UNet/SegFormer/SegNet).

    Uses val2017 images/masks and subset txt files for splitting.
    """
    train_dataset = COCOSemanticDataset(
        data_dir=data_dir,
        split='train',
        img_size=img_size,
        augment=True
    )

    val_dataset = COCOSemanticDataset(
        data_dir=data_dir,
        split='val',
        img_size=img_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# Backward compatibility (old function name)
get_segmentation_dataloaders = get_semantic_dataloaders



def _unnormalize_image(tensor):
    """
    Undo ImageNet normalization for visualization.
    Expects tensor in shape (C, H, W).
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.clone().cpu() * std + mean
    img = torch.clamp(img, 0.0, 1.0)
    return img.permute(1, 2, 0).numpy()  # (H, W, C)


def visualize_semantic_samples(dataset, n=3):
    """
    Visualize n random (image, mask) pairs from a COCOSemanticDataset.
    """
    assert len(dataset) > 0, "Dataset is empty!"
    n = min(n, len(dataset))

    indices = random.sample(range(len(dataset)), n)

    for idx in indices:
        sample = dataset[idx]
        image = sample["image"]      # (C, H, W)
        mask = sample["mask"]        # (H, W)
        img_id = sample["img_id"]

        img_vis = _unnormalize_image(image)
        mask_np = mask.cpu().numpy()

        plt.figure(figsize=(8, 4))
        plt.suptitle(f"img_id: {img_id}", fontsize=12)

        # Image
        plt.subplot(1, 2, 1)
        plt.imshow(img_vis)
        plt.axis("off")
        plt.title("Image")

        # Mask (use simple colormap, no legend)
        plt.subplot(1, 2, 2)
        plt.imshow(mask_np, cmap="tab20")
        plt.axis("off")
        plt.title("Mask (class ids)")

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("Testing dataloaders...")

    data_root = '../dataset'  # match your create_subset default

    # Test instance segmentation dataloader
    print("\n" + "="*60)
    print("Testing Instance Segmentation Dataloader (Mask R-CNN)")
    print("="*60)
    try:
        train_loader, val_loader = get_instance_dataloaders(
            data_dir=data_root,
            batch_size=2,
            num_workers=0
        )

        images, targets = next(iter(train_loader))
        print(f"✓ Batch size: {len(images)}")
        print(f"✓ Image tensor shape: {images[0].shape}")
        print(f"✓ Number of targets: {len(targets)}")
        print(f"✓ Example target keys: {targets[0].keys()}")
        if len(targets[0]['boxes']) > 0:
            print(f"✓ First image has {len(targets[0]['boxes'])} objects")
    except Exception as e:
        print(f"✗ Instance dataloader test failed: {e}")

    # Test semantic segmentation dataloader
    print("\n" + "="*60)
    print("Testing Semantic Segmentation Dataloader (UNet/SegFormer/SegNet)")
    print("="*60)
    try:
        train_loader, val_loader = get_semantic_dataloaders(
            data_dir=data_root,
            batch_size=2,
            num_workers=0
        )

        batch = next(iter(train_loader))
        print(f"✓ Image batch shape: {batch['image'].shape}")
        print(f"✓ Mask batch shape: {batch['mask'].shape}")
        print(f"✓ Mask unique values (first 10): {torch.unique(batch['mask'])[:10].tolist()}")
    except Exception as e:
        print(f"✗ Semantic dataloader test failed: {e}")

    # Test inpainting dataloader
    print("\n" + "="*60)
    print("Testing Inpainting Dataloader")
    print("="*60)
    # visually check a few semantic samples
    try:
        print("\nVisualizing a few semantic (image, mask) pairs...")
        sem_train_dataset = COCOSemanticDataset(
            data_dir=data_root,
            split='train',
            img_size=512,
            augment=False  # turn off aug for clean visualization
        )
        visualize_semantic_samples(sem_train_dataset, n=8)
    except Exception as e:
        print(f"✗ Visualization failed: {e}")


