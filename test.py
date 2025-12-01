import torch
import numpy as np
from segmentation_models.unet import ResNetUNet
from dataset_preprocess.preprocess import get_segmentation_dataloaders
from utils import visualize_segmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = ResNetUNet(num_classes=183, pretrained=False, freeze_encoder=True)
ckpt = torch.load("segmentation_models/training/checkpoints/unet_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

# Load val dataset
_, val_loader = get_segmentation_dataloaders(
    data_dir="./dataset",
    batch_size=1,
    img_size=512,
    num_workers=0
)
val_dataset = val_loader.dataset

# Select an image
idx = np.random.randint(0, len(val_dataset))
sample = val_dataset[idx]

img = sample["image"].unsqueeze(0).to(device)
mask = sample["mask"]

# Predict
with torch.no_grad():
    logits = model(img)
    pred_mask = logits.argmax(dim=1)[0].cpu()

# Visualize
visualize_segmentation(
    image=sample["image"],
    mask=mask,
    pred_mask=pred_mask
)
