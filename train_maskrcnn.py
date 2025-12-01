import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from segmentation_models.maskrcnn import MaskRCNNPredictor
from dataset_preprocess.preprocess import get_instance_dataloaders


class MaskRCNNTrainer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.best_ap = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_ap': []}
        
    def load_model(self, score_thresh=0.5):
        print("[INFO] Loading pre-trained Mask R-CNN...")
        self.model = MaskRCNNPredictor(device=self.device, score_thresh=score_thresh)
        return self.model
    
    def setup_optimizer(self, lr=0.001, weight_decay=0.0005):
        params = [p for p in self.model.model.parameters() if p.requires_grad]
        self.optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        return self.optimizer
    
    def train_epoch(self, train_loader, epoch):
        self.model.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            loss_dict = self.model.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = total_loss / len(train_loader)
        return avg_epoch_loss
    
    def compute_iou(self, pred_boxes, pred_masks, target_boxes, target_masks):
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            return 0.0
        
        ious = []
        for pred_box, pred_mask in zip(pred_boxes, pred_masks):
            max_iou = 0.0
            for target_box, target_mask in zip(target_boxes, target_masks):
                intersection = (pred_mask * target_mask).sum()
                union = (pred_mask | target_mask).sum()
                if union > 0:
                    iou = intersection / union
                    max_iou = max(max_iou, iou)
            ious.append(max_iou)
        
        return np.mean(ious) if ious else 0.0
    
    def validate(self, val_loader):
        self.model.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                images = [img.to(self.device) for img in images]
                targets_device = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                  for k, v in t.items()} for t in targets]
                
                loss_dict = self.model.model(images, targets_device)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                
                preds = [self.model.model([img])[0] for img in images]
                
                for pred, target in zip(preds, targets):
                    pred_masks = pred['masks']
                    target_masks = target['masks']
                    
                    if len(pred_masks) > 0 and len(target_masks) > 0:
                        pred_masks_binary = (pred_masks > 0.5).float()
                        target_masks_binary = target_masks.float()
                        iou = self.compute_iou(pred['boxes'], pred_masks_binary, 
                                              target['boxes'], target_masks_binary)
                        total_iou += iou
                    
                    num_samples += 1
        
        avg_loss = total_loss / len(val_loader)
        avg_iou = total_iou / num_samples if num_samples > 0 else 0.0
        
        return avg_loss, avg_iou
    
    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_dir = Path('checkpoints/maskrcnn')
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        
        if is_best:
            path = checkpoint_dir / 'best_model.pth'
            print(f"[INFO] Saving best model to {path}")
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def train(self, data_dir, num_epochs=30, batch_size=4, lr=0.001, img_size=512):
        print(f"\n{'='*70}")
        print("FINE-TUNING MASK R-CNN ON COCO INSTANCE SEGMENTATION")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs} | Batch size: {batch_size} | LR: {lr} | Image size: {img_size}")
        print(f"{'='*70}\n")
        
        self.load_model()
        self.setup_optimizer(lr=lr)
        
        print("[INFO] Loading COCO instance segmentation dataloaders...")
        train_loader, val_loader = get_instance_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=4
        )
        
        print(f"[INFO] Training samples: {len(train_loader.dataset)}")
        print(f"[INFO] Validation samples: {len(val_loader.dataset)}\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 70)
            
            train_loss = self.train_epoch(train_loader, epoch)
            self.history['train_loss'].append(train_loss)
            
            val_loss, val_iou = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_ap'].append(val_iou)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
            
            is_best = val_iou > self.best_ap
            if is_best:
                self.best_ap = val_iou
                print(f"[INFO] Best validation IoU: {self.best_ap:.4f}")
            
            self.save_checkpoint(epoch, is_best=is_best)
        
        self.save_training_history()
        print(f"\n{'='*70}")
        print("TRAINING COMPLETED")
        print(f"Best validation IoU: {self.best_ap:.4f}")
        print(f"{'='*70}\n")
    
    def save_training_history(self):
        history_dir = Path('training_logs')
        history_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f'maskrcnn_history_{timestamp}.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"[INFO] Training history saved to {history_path}")


if __name__ == '__main__':
    trainer = MaskRCNNTrainer(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer.train(
        data_dir='dataset',
        num_epochs=30,
        batch_size=4,
        lr=0.001,
        img_size=512
    )
