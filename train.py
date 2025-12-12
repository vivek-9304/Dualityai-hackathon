import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import cv2

# Dataset Configuration
DATASET_PATH = r"D:\Desktop\Offroad_Segmentation_Training_Dataset"
TRAIN_COLOR_PATH = os.path.join(DATASET_PATH, "train", "Color_Images")
TRAIN_SEG_PATH = os.path.join(DATASET_PATH, "train", "Segmentation")
VAL_COLOR_PATH = os.path.join(DATASET_PATH, "val", "Color_Images")
VAL_SEG_PATH = os.path.join(DATASET_PATH, "val", "Segmentation")

# Training Configuration
BATCH_SIZE = 7
NUM_EPOCHS = 25
LEARNING_RATE = 0.00006
NUM_WORKERS = 0
DEVICE = torch.device("cuda")

# Class mapping based on your document
CLASS_MAPPING = {
    100: 0,   # Trees
    200: 1,   # Lush Bushes
    300: 2,   # Dry Grass
    500: 3,   # Dry Bushes
    550: 4,   # Ground Clutter
    600: 5,   # Flowers
    700: 6,   # Logs
    800: 7,   # Rocks
    7100: 8,  # Landscape
    10000: 9  # Sky
}
NUM_CLASSES = len(CLASS_MAPPING)

# Reverse mapping for visualization
ID2LABEL = {
    0: "Trees", 1: "Lush Bushes", 2: "Dry Grass", 3: "Dry Bushes",
    4: "Ground Clutter", 5: "Flowers", 6: "Logs", 7: "Rocks",
    8: "Landscape", 9: "Sky"
}

class OffRoadSegmentationDataset(Dataset):
    def __init__(self, color_dir, seg_dir, processor, is_train=True):
        self.color_dir = color_dir
        self.seg_dir = seg_dir
        self.processor = processor
        self.is_train = is_train
        
        # Get all image files
        self.color_images = sorted([f for f in os.listdir(color_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.seg_images = sorted([f for f in os.listdir(seg_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.color_images)} color images and {len(self.seg_images)} segmentation masks")
    
    def __len__(self):
        return len(self.color_images)
    
    def map_segmentation_ids(self, seg_array):
        """Map original class IDs to 0-9 range"""
        mapped = np.zeros_like(seg_array, dtype=np.int64)
        for original_id, new_id in CLASS_MAPPING.items():
            mapped[seg_array == original_id] = new_id
        return mapped
    
    def __getitem__(self, idx):
        # Load color image
        color_path = os.path.join(self.color_dir, self.color_images[idx])
        color_img = Image.open(color_path).convert("RGB")
        
        # Load segmentation mask
        seg_path = os.path.join(self.seg_dir, self.seg_images[idx])
        seg_img = Image.open(seg_path)
        seg_array = np.array(seg_img)
        
        # Map segmentation IDs to 0-9
        seg_array = self.map_segmentation_ids(seg_array)
        
        # Process with SegFormer processor
        encoded = self.processor(color_img, return_tensors="pt")
        encoded["pixel_values"] = encoded["pixel_values"].squeeze(0)
        
        # Add segmentation mask
        encoded["labels"] = torch.from_numpy(seg_array).long()
        
        return encoded

def compute_iou(pred, target, num_classes):
    """Compute IoU for each class and mean IoU"""
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    # Calculate mean IoU (ignoring NaN values)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0
    
    return mean_iou, ious

def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate IoU
        with torch.no_grad():
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            pred = upsampled_logits.argmax(dim=1)
            mean_iou, _ = compute_iou(pred, labels, NUM_CLASSES)
        
        total_loss += loss.item()
        total_iou += mean_iou
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "mIoU": f"{mean_iou:.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return avg_loss, avg_iou

def validate(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0
    total_iou = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Val]")
    with torch.no_grad():
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Calculate IoU
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            pred = upsampled_logits.argmax(dim=1)
            mean_iou, _ = compute_iou(pred, labels, NUM_CLASSES)
            
            total_loss += loss.item()
            total_iou += mean_iou
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "mIoU": f"{mean_iou:.4f}"})
    
    avg_loss = total_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)
    return avg_loss, avg_iou

def plot_training_history(train_losses, val_losses, train_ious, val_ious):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # IoU plot
    ax2.plot(train_ious, label='Train mIoU', marker='o')
    ax2.plot(val_ious, label='Val mIoU', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean IoU')
    ax2.set_title('Training and Validation mIoU')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history saved to training_history.png")

def main():
    print(f"Using device: {DEVICE}")
    print(f"Number of classes: {NUM_CLASSES}")
    
    # Create output directory
    os.makedirs("runs", exist_ok=True)
    
    # Initialize processor and model
    processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    processor.do_reduce_labels = False
    processor.do_resize = True
    processor.size = {"height": 512, "width": 512}
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)
    
    # Create datasets
    train_dataset = OffRoadSegmentationDataset(TRAIN_COLOR_PATH, TRAIN_SEG_PATH, processor, is_train=True)
    val_dataset = OffRoadSegmentationDataset(VAL_COLOR_PATH, VAL_SEG_PATH, processor, is_train=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training history
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    best_val_iou = 0
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
        train_losses.append(train_loss)
        train_ious.append(train_iou)
        
        # Validate
        val_loss, val_iou = validate(model, val_loader, DEVICE, epoch)
        val_losses.append(val_loss)
        val_ious.append(val_iou)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
        print("-" * 50)
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, 'runs/best_model.pth')
            print(f"✓ Best model saved! Val mIoU: {val_iou:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'runs/checkpoint_epoch_{epoch+1}.pth')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, train_ious, val_ious)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation mIoU: {best_val_iou:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()