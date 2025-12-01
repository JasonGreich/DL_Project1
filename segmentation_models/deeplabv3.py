import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabV3Predictor:
    """
    Wrapper class that:
      • loads DeepLabV3-ResNet50
      • loads pretrained VOC/COCO weights
      • moves model to device
      • exposes a .predict(tensor) method for inference
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

        # Load pretrained weights
        self.weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1

        # Load model with weights
        self.model = deeplabv3_resnet50(weights=self.weights)
        self.model.to(self.device)
        self.categories = self.weights.meta.get("categories", [])
        self.transforms = self.weights.transforms()
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensor: tensor [3, H, W] already normalized
                          (from your dataset loader)

        Returns:
            pred_mask: tensor [H, W] with class indices.
        """

        # Add batch dimension
        x = image_tensor.unsqueeze(0).to(self.device)

        # Run model
        output = self.model(x)["out"]  # [1, C, H, W]

        # Argmax per pixel → segmentation mask
        pred_mask = output.argmax(1).squeeze().cpu()  # [H, W]

        return pred_mask


from dataset_preprocess.preprocess import get_segmentation_dataloaders
from utils import visualize_segmentation

def main():

    # 1. Instantiate your new DeepLab class
    deeplab = DeepLabV3Predictor(device="cpu")

    # 2. Load validation dataset
    _, val_loader = get_segmentation_dataloaders(
        data_dir="../dataset",
        batch_size=1,
        img_size=512,
        num_workers=0
    )

    val_dataset = val_loader.dataset

    # 3. Pick a sample
    sample = val_dataset[200]
    image = sample["image"]    # [3, H, W]
    mask = sample["mask"]      # [H, W]

    # 4. Predict
    pred_mask = deeplab.predict(image)

    # 5. Visualize
    visualize_segmentation(
        image=image,
        mask=mask,
        pred_mask=pred_mask
    )


if __name__ == "__main__":
    main()
