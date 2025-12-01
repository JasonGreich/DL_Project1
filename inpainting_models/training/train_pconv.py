from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from inpainting_models.pconv.unet import PConvUNet
from inpainting_models.pconv.feature_extractor import VGG16FeatureExtractor
from inpainting_models.pconv.loss import InpaintingLoss
from dataset_preprocess.preprocess import COCOInpaintingDataset
import os

class Trainer:
    def __init__(self, step, device, model, dataset_train, dataset_val, criterion, optimizer):
        os.makedirs("outputs_pconv/val_vis", exist_ok=True)
        os.makedirs("outputs_pconv/models", exist_ok=True)
        self.stepped = step
        self.device = device
        self.model = model
        self.dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=4,
                                           pin_memory=True)
        self.dataset_val = dataset_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.writer = SummaryWriter(log_dir="outputs_pconv")
        self.best_loss = float('inf')
        # self.scheduler        = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    def iterate(self):
        print("Training ...")
        torch.autograd.set_detect_anomaly(True)
        progress_bar = tqdm(total=50000, desc="Progress", unit="step")

        for step, (input, mask, gt) in enumerate(self.dataloader_train):
            current_step = step + self.stepped
            loss_dict = self.train(current_step, input, mask, gt)

            if current_step % 100 == 0:
                self.report(current_step, loss_dict)
            if current_step % 5000 == 0:
                self.evaluate(current_step)
            if current_step % 5000 == 0:
                self.checkpoint(current_step)

            if step >= 50000:
                print("- Max iterations reached. Stop training!")
                break

            progress_bar.update(1)

        progress_bar.close()

    def train(self, step, input, mask, gt):
        self.model.train()

        input = input.to(self.device)
        mask = mask.to(self.device)
        gt = gt.to(self.device)

        output, _ = self.model(input, mask)
        loss_dict = self.criterion(input, mask, output, gt)
        coef = {
            "valid": 1.0,
            "hole": 6.0,
            "tv": 0.1,
            "perc": 0.05,
            "style": 120.0
        }

        loss = sum(coef[key] * val for key, val in loss_dict.items())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update scheduler
        # self.scheduler.step(loss.item())

        loss_dict['total'] = loss
        return {k: v.item() for k, v in loss_dict.items()}

    def report(self, step, loss_dict):
        print(f"[STEP: {step}] "
              f"\t| Valid Loss: {loss_dict['valid']:.6f} "
              f"\t| Hole Loss: {loss_dict['hole']:.6f} "
              f"\t| Total Loss: {loss_dict['total']:.6f}")

        for key, val in loss_dict.items():
            self.writer.add_scalar(f'Loss/{key}', val, step)

    def evaluate(self, step):
        print(f"- Evaluation at [STEP: {step}] ...")

        self.model.eval()
        samples = [self.dataset_val[i] for i in range(8)]
        image, mask, gt = zip(*samples)

        image = torch.stack(image).to(self.device)
        mask = torch.stack(mask).to(self.device)
        gt = torch.stack(gt).to(self.device)

        with torch.no_grad():
            output, _ = self.model(image, mask)
        output_comp = mask * image + (1 - mask) * output

        # Tính PSNR và SSIM
        psnr_total, ssim_total = 0, 0
        for i in range(image.size(0)):
            pred_img = output_comp[i].cpu().numpy().transpose(1, 2, 0)
            gt_img = gt[i].cpu().numpy().transpose(1, 2, 0)

            psnr_total += peak_signal_noise_ratio(gt_img, pred_img, data_range=1.0)
            ssim_total += structural_similarity(gt_img, pred_img, data_range=1.0, multichannel=True, win_size=3)
        avg_psnr = psnr_total / image.size(0)
        avg_ssim = ssim_total / image.size(0)
        print(f"> Average PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}")

        # Log PSNR và SSIM vào TensorBoard
        self.writer.add_scalar('Validation/PSNR', avg_psnr, step)
        self.writer.add_scalar('Validation/SSIM', avg_ssim, step)

        # Lưu kết quả visualize
        grid = make_grid(torch.cat([image.cpu(), mask.cpu(), output.cpu(), output_comp.cpu(), gt.cpu()], dim=0), nrow=8,
                         padding=2)
        save_image(grid, f'outputs_pconv/val_vis/{step}.png')

        coef = {
            "valid": 1.0,
            "hole": 6.0,
            "tv": 0.1,
            "perc": 0.05,
            "style": 120.0
        }

        # Tính loss validation
        val_loss_dict = self.criterion(image, mask, output, gt)
        val_loss = sum(coef[key] * val for key, val in val_loss_dict.items()).item()
        print(f"> Total Loss: {val_loss:.6f}")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save({
                'step': step,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
            }, f"outputs_pconv/models/best_model.pth")
            print(f"> New best model saved")

    def checkpoint(self, step):
        print(f"- Checkpoint at [STEP: {step}] ...")

        torch.save({
            'step': step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, f"outputs_pconv/models/{step}.pth")


def main():
    finetune= False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = PConvUNet(finetune=finetune, layer_size=7)
    model = model.to(device)

    trainer = Trainer(
        step=0,
        device=device,
        model=model,
        dataset_train=COCOInpaintingDataset("../../dataset/", 256, split="train"),
        dataset_val=COCOInpaintingDataset("../../dataset/", 256, split="val"),

        criterion=InpaintingLoss(VGG16FeatureExtractor(), tv_loss="mean").to(device),

        optimizer=torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.0005 if finetune else 0.0002,
            weight_decay=0
        )
    )
    trainer.iterate()


if __name__ == "__main__":
    main()