import os
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from inpainting_models.deepfill.generators import Generator
from inpainting_models.deepfill.discriminator import Discriminator
from dataset_preprocess.preprocess import COCOInpaintingDataset


# -----------------------------------------------------------
# Helper: build DeepFill generator input
# -----------------------------------------------------------

def build_generator_input(
    image: torch.Tensor,
    hole_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    image:     [B, 3, H, W] in [0,1]
    hole_mask: [B, 1, H, W] with 1 = hole, 0 = valid

    Returns:
        x           : [B, 5, H, W] (DeepFill input)
        image_n     : [B, 3, H, W] in [-1,1]
        hole_mask   : [B, 1, H, W]
        image_masked: [B, 3, H, W] in [-1,1]
    """
    image_n = image * 2.0 - 1.0              # [0,1] -> [-1,1]
    hole_mask = (hole_mask > 0.5).float()    # ensure {0,1} float

    image_masked = image_n * (1.0 - hole_mask)
    ones_x = torch.ones_like(image_masked[:, 0:1, :, :])  # [B,1,H,W]

    x = torch.cat([image_masked, ones_x, ones_x * hole_mask], dim=1)  # [B,5,H,W]

    return x, image_n, hole_mask, image_masked


# -----------------------------------------------------------
# DeepFill Trainer
# -----------------------------------------------------------

class DeepFillTrainer:
    def __init__(
        self,
        device: str,
        G: Generator,
        D: Discriminator,
        train_dataset,
        val_dataset=None,
        batch_size: int = 4,
        num_workers: int = 4,
        lr_G: float = 2e-4,
        lr_D: float = 2e-4,
        lambda_hole: float = 6.0,
        lambda_valid: float = 1.0,
        lambda_adv: float = 1e-3,
        log_dir: str = "outputs_deepfill",
    ):
        self.device = device
        self.G = G.to(device)
        self.D = D.to(device)

        self.lambda_hole = lambda_hole
        self.lambda_valid = lambda_valid
        self.lambda_adv = lambda_adv

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "vis"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "val_vis"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "models"), exist_ok=True)
        self.log_dir = log_dir

        # Train loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Val loader (optional)
        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

        self.opt_G = torch.optim.Adam(
            self.G.parameters(), lr=lr_G, betas=(0.5, 0.999)
        )
        self.opt_D = torch.optim.Adam(
            self.D.parameters(), lr=lr_D, betas=(0.5, 0.999)
        )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.best_val_loss = float('inf')

    # ----------------- single training step -----------------

    def train_step(self, batch):
        self.G.train()
        self.D.train()

        # dataset returns: masked_img, mask, img
        masked_img, mask, img = batch

        img = img.to(self.device)     # [B,3,H,W] in [0,1]
        mask = mask.to(self.device)   # [B,3,H,W] 1 = valid, 0 = hole

        # convert to 1-channel mask and invert: 1 = hole, 0 = valid
        mask_1ch = mask[:, :1, :, :]             # [B,1,H,W]
        hole_mask = (mask_1ch < 0.5).float()     # 1 where hole, 0 where valid

        # ---------- build DeepFill input ----------
        x, image_n, hole_mask, _ = build_generator_input(img, hole_mask)

        # ================ 1) UPDATE DISCRIMINATOR =================
        with torch.no_grad():
            x_stage1, x_stage2 = self.G(x, hole_mask)
            comp = image_n * (1.0 - hole_mask) + x_stage2 * hole_mask  # completed [-1,1]

        real_in = torch.cat([image_n, hole_mask], dim=1)    # [B,4,H,W]
        fake_in = torch.cat([comp, hole_mask], dim=1).detach()

        real_logits = self.D(real_in)
        fake_logits = self.D(fake_in)

        d_loss_real = F.relu(1.0 - real_logits).mean()
        d_loss_fake = F.relu(1.0 + fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake

        self.opt_D.zero_grad(set_to_none=True)
        d_loss.backward()
        self.opt_D.step()

        # ================ 2) UPDATE GENERATOR ======================
        x_stage1, x_stage2 = self.G(x, hole_mask)
        comp = image_n * (1.0 - hole_mask) + x_stage2 * hole_mask

        valid_region = 1.0 - hole_mask

        l1_hole = (torch.abs(image_n - x_stage2) * hole_mask).mean()
        l1_valid = (torch.abs(image_n - x_stage2) * valid_region).mean()

        fake_in_for_G = torch.cat([comp, hole_mask], dim=1)
        fake_logits_G = self.D(fake_in_for_G)
        g_adv = -fake_logits_G.mean()

        g_loss = (
            self.lambda_hole * l1_hole
            + self.lambda_valid * l1_valid
            + self.lambda_adv * g_adv
        )

        self.opt_G.zero_grad(set_to_none=True)
        g_loss.backward()
        self.opt_G.step()

        loss_dict = {
            "G_total": g_loss.item(),
            "G_hole": l1_hole.item(),
            "G_valid": l1_valid.item(),
            "G_adv": g_adv.item(),
            "D_total": d_loss.item(),
            "D_real": d_loss_real.item(),
            "D_fake": d_loss_fake.item(),
        }

        # return some tensors for visualization
        return loss_dict, image_n.detach(), hole_mask.detach(), comp.detach()

    # ----------------- validation -----------------

    @torch.no_grad()
    def evaluate(self, step: int):
        if self.val_loader is None:
            return

        print(f"[step {step}] Running DeepFill validation...")
        self.G.eval()

        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        n_images = 0

        for masked_img, mask, img in self.val_loader:
            img = img.to(self.device)     # [B,3,H,W] in [0,1]
            mask = mask.to(self.device)

            mask_1ch = mask[:, :1, :, :]
            hole_mask = (mask_1ch < 0.5).float()

            x, image_n, hole_mask, _ = build_generator_input(img, hole_mask)

            x_stage1, x_stage2 = self.G(x, hole_mask)
            comp = image_n * (1.0 - hole_mask) + x_stage2 * hole_mask  # [-1,1]

            # simple L1 val loss
            val_loss = torch.abs(image_n - x_stage2).mean()
            total_loss += val_loss.item()

            # convert to [0,1] and numpy for metrics
            gt_np = img.cpu().numpy().transpose(0, 2, 3, 1)             # [B,H,W,C] in [0,1]
            comp_np = ((comp + 1.0) / 2.0).cpu().numpy().transpose(0, 2, 3, 1)

            for g, p in zip(gt_np, comp_np):
                total_psnr += peak_signal_noise_ratio(g, p, data_range=1.0)
                total_ssim += structural_similarity(g, p, data_range=1.0, multichannel=True)
                n_images += 1

        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = total_psnr / n_images
        avg_ssim = total_ssim / n_images

        print(f"[VAL] loss={avg_loss:.4f} | PSNR={avg_psnr:.3f} | SSIM={avg_ssim:.3f}")

        # log to TensorBoard
        self.writer.add_scalar("Val/loss", avg_loss, step)
        self.writer.add_scalar("Val/PSNR", avg_psnr, step)
        self.writer.add_scalar("Val/SSIM", avg_ssim, step)

        # save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save(
                {
                    "step": step,
                    "G": self.G.state_dict(),
                    "D": self.D.state_dict(),
                    "best_val_loss": self.best_val_loss,
                },
                os.path.join(self.log_dir, "models", "best_model.pth"),
            )
            print("✓ New best DeepFill model saved.")

    # ----------------- full training loop (step-based) -----------------

    def train(
        self,
        max_steps: int = 100_000,
        log_every: int = 100,
        vis_every: int = 1000,
        val_every: int = 5000,
        ckpt_every: int = 5000,
    ):
        print("Starting DeepFill training...")
        pbar = tqdm(total=max_steps, desc="steps", unit="step")

        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break

                self.global_step += 1
                loss_dict, image_n, hole_mask, comp = self.train_step(batch)

                # logging to console & TensorBoard
                if self.global_step % log_every == 0:
                    self._log_losses(loss_dict)
                    self._print_losses(loss_dict)

                if self.global_step % vis_every == 0:
                    self._save_vis(image_n, hole_mask, comp, self.global_step)

                if self.global_step % val_every == 0:
                    self.evaluate(self.global_step)

                if self.global_step % ckpt_every == 0:
                    self._save_ckpt(self.global_step)

                pbar.update(1)

        pbar.close()
        print("Training finished.")

    # ----------------- helpers -----------------

    def _log_losses(self, loss_dict):
        for k, v in loss_dict.items():
            self.writer.add_scalar(f"Train/{k}", v, self.global_step)

    def _print_losses(self, loss_dict):
        msg = f"[step {self.global_step}] " + " | ".join(
            f"{k}: {v:.4f}" for k, v in loss_dict.items()
        )
        print(msg)

    def _save_vis(self, image_n, hole_mask, comp, step: int):
        # image_n, comp in [-1,1] -> [0,1]
        image = (image_n + 1.0) / 2.0
        comp = (comp + 1.0) / 2.0

        img_vis = image[:4].cpu()
        comp_vis = comp[:4].cpu()
        mask_vis = hole_mask[:4].repeat(1, 3, 1, 1).cpu()  # 1ch -> 3ch

        grid = make_grid(
            torch.cat([img_vis, mask_vis, comp_vis], dim=0),
            nrow=4,
            padding=2,
        )
        save_path = os.path.join(self.log_dir, "vis", f"{step:06d}.png")
        save_image(grid, save_path)

    def _save_ckpt(self, step: int):
        path = os.path.join(self.log_dir, "models", f"deepfill_{step:06d}.pth")
        torch.save(
            {
                "step": step,
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict(),
            },
            path,
        )
        print(f"[step {step}] checkpoint saved → {path}")


# -----------------------------------------------------------
# main()
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Generator: 5 input channels (image_masked + ones + ones*mask)
    G = Generator(cnum_in=5, cnum=48, return_flow=False)
    # Discriminator: 4 input channels (3 image + 1 mask)
    D = Discriminator(cnum_in=4, cnum=64)

    train_dataset = COCOInpaintingDataset(
        data_dir="../../dataset",
        split="train",
        img_size=256,
        num_irregulars=10,
    )

    val_dataset = COCOInpaintingDataset(
        data_dir="../../dataset",
        split="val",
        img_size=256,
        num_irregulars=10,
    )

    trainer = DeepFillTrainer(
        device=device,
        G=G,
        D=D,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_workers=4,
        lr_G=2e-4,
        lr_D=2e-4,
        lambda_hole=6.0,
        lambda_valid=1.0,
        lambda_adv=1e-3,
        log_dir="outputs_deepfill",
    )

    trainer.train(
        max_steps=100_000,
        log_every=100,
        vis_every=1_000,
        val_every=5_000,
        ckpt_every=5_000,
    )


if __name__ == "__main__":
    main()
