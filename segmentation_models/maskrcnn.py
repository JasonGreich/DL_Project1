import torch
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)


class MaskRCNNPredictor:
    """
    Wrapper for Mask R-CNN (ResNet50-FPN) with COCO weights.

    Usage:
        predictor = MaskRCNNPredictor(device="cuda")
        outputs = predictor.predict(image_01)  # image_01: [3,H,W] in [0,1]
    """

    def __init__(self, device: str = None, score_thresh: float = 0.5):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.score_thresh = score_thresh

        # Load weights
        self.weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
        self.transforms = self.weights.transforms()
        self.categories = self.weights.meta.get("categories", [])
        # Load model with pretrained weights
        self.model = maskrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()

        # Class names (80 COCO classes, background is 0)
        self.categories = self.weights.meta.get("categories", None)

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor):
        """
        Args:
            image_tensor: [3, H, W] float tensor in [0, 1]
                (UN-normalized; just like torchvision's ToTensor output)

        Returns:
            A dict with filtered predictions:
                {
                    "boxes":  [N, 4]  (x1,y1,x2,y2),
                    "labels": [N],
                    "scores": [N],
                    "masks":  [N, 1, H, W] or None
                }
        """
        # Add batch dimension & move to device
        img = image_tensor.to(self.device)
        outputs = self.model([img])[0]  # dict for this single image

        scores = outputs["scores"]
        keep = scores >= self.score_thresh

        boxes = outputs["boxes"][keep].cpu()
        labels = outputs["labels"][keep].cpu()
        scores = outputs["scores"][keep].cpu()
        masks = outputs.get("masks", None)
        if masks is not None:
            masks = masks[keep].cpu()  # [N, 1, H, W]

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
        }


import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

from dataset_preprocess.preprocess import get_instance_dataloaders # adjust path as you like


def denormalize_imagenet(t: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization: your COCOInstanceDataset uses:
      mean = [0.485, 0.456, 0.406]
      std  = [0.229, 0.224, 0.225]

    Input:  t [3,H,W] (normalized)
    Output: [3,H,W] in [0,1]
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(3, 1, 1)
    img = t * std + mean
    return torch.clamp(img, 0.0, 1.0)


def main():
    # 1) Build predictor
    predictor = MaskRCNNPredictor(device="cpu", score_thresh=0.5)

    # 2) Load instance dataloader
    train_loader, val_loader = get_instance_dataloaders(
        data_dir="../dataset",
        batch_size=1,
        img_size=512,
        num_workers=0,
    )

    # 3) Get a single sample (images is a list of tensors, targets is list of dicts)
    images, targets = next(iter(val_loader))
    image = images[0]         # [3,H,W], *normalized*
    target = targets[0]       # dict with 'boxes', 'labels', 'masks', ...

    # 4) Denormalize for Mask R-CNN (expects [0,1] without normalization)
    image_01 = denormalize_imagenet(image)

    # 5) Run prediction
    preds = predictor.predict(image_01)
    boxes = preds["boxes"]
    labels = preds["labels"]
    scores = preds["scores"]
    masks = preds["masks"]

    print(f"Detected {len(boxes)} objects above threshold.")
    if predictor.categories is not None:
        for i in range(len(boxes)):
            cls_id = labels[i].item()
            cls_name = predictor.categories[cls_id] if cls_id < len(predictor.categories) else str(cls_id)
            print(f"  - {cls_name}: score={scores[i].item():.3f}")

    # 6) Quick visualization (image + predicted boxes)
    img_np = image_01.cpu()  # [3,H,W]
    img_pil = to_pil_image(img_np)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil)
    ax = plt.gca()

    # draw boxes
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

    plt.axis("off")
    plt.title("Mask R-CNN predictions")
    plt.show()


if __name__ == "__main__":
    main()
