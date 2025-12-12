import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# Configuration
TEST_COLOR_PATH = r"C:\Users\vivek\Desktop\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Color_Images"
TEST_SEG_PATH = r"C:\Users\vivek\Desktop\Offroad_Segmentation_testImages\Offroad_Segmentation_testImages\Segmentation"
MODEL_PATH = "runs/best_model.pth"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class mapping (same as training)
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
NUM_CLASSES = 10
ID2LABEL = {
    0: "Trees", 1: "Lush Bushes", 2: "Dry Grass", 3: "Dry Bushes",
    4: "Ground Clutter", 5: "Flowers", 6: "Logs", 7: "Rocks",
    8: "Landscape", 9: "Sky"
}

# Color map for visualization (high contrast colors)
COLOR_MAP = {
    0: [34, 139, 34],      # Forest Green - Trees
    1: [0, 255, 127],      # Spring Green - Lush Bushes
    2: [255, 215, 0],      # Gold - Dry Grass
    3: [139, 69, 19],      # Saddle Brown - Dry Bushes
    4: [210, 180, 140],    # Tan - Ground Clutter
    5: [255, 20, 147],     # Deep Pink - Flowers
    6: [160, 82, 45],      # Sienna - Logs
    7: [128, 128, 128],    # Gray - Rocks
    8: [222, 184, 135],    # Burlywood - Landscape
    9: [135, 206, 235]     # Sky Blue - Sky
}

class TestDatasetWithGT(Dataset):
    def __init__(self, color_dir, seg_dir, processor):
        self.color_dir = color_dir
        self.seg_dir = seg_dir
        self.processor = processor
        
        self.color_images = sorted([f for f in os.listdir(color_dir) 
                                   if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.seg_images = sorted([f for f in os.listdir(seg_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        print(f"Found {len(self.color_images)} test color images")
        print(f"Found {len(self.seg_images)} test segmentation masks")
    
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
        
        # Load ground truth segmentation
        seg_path = os.path.join(self.seg_dir, self.seg_images[idx])
        seg_img = Image.open(seg_path)
        seg_array = np.array(seg_img)
        seg_array = self.map_segmentation_ids(seg_array)
        
        # Process image
        encoded = self.processor(color_img, return_tensors="pt")
        encoded["pixel_values"] = encoded["pixel_values"].squeeze(0)
        encoded["labels"] = torch.from_numpy(seg_array).long()
        encoded["filename"] = self.color_images[idx]
        encoded["color_path"] = color_path
        
        return encoded

def compute_iou_per_class(pred, target, num_classes):
    """Compute IoU for each class"""
    ious = []
    pred = pred.cpu().numpy() if torch.is_tensor(pred) else pred
    target = target.cpu().numpy() if torch.is_tensor(target) else target
    
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return ious

def compute_mean_iou(ious_list):
    """Compute mean IoU across all classes, ignoring NaN"""
    valid_ious = [iou for iou in ious_list if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0

def colorize_mask(mask, color_map):
    """Convert class mask to RGB visualization"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        colored[mask == class_id] = color
    
    return colored

def create_comparison_viz(original, pred_mask, gt_mask, color_map, filename):
    """Create visualization comparing prediction with ground truth"""
    pred_colored = colorize_mask(pred_mask, color_map)
    gt_colored = colorize_mask(gt_mask, color_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(pred_colored)
    axes[1].set_title("Prediction", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(gt_colored)
    axes[2].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle(f"{filename}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def test_model_with_evaluation(model, dataloader, output_dir="test_results"):
    """Test model and compute IoU scores"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    
    print("\n" + "="*60)
    print("Running Evaluation on Test Dataset")
    print("="*60 + "\n")
    
    all_class_ious = {cls: [] for cls in range(NUM_CLASSES)}
    all_mean_ious = []
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            filenames = batch["filename"]
            color_paths = batch["color_path"]
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            # Upsample predictions
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            pred = upsampled_logits.argmax(dim=1)
            
            # Process each image in batch
            for i in range(pred.shape[0]):
                pred_mask = pred[i].cpu().numpy()
                gt_mask = labels[i].cpu().numpy()
                
                # Compute IoU per class
                class_ious = compute_iou_per_class(pred_mask, gt_mask, NUM_CLASSES)
                mean_iou = compute_mean_iou(class_ious)
                
                # Store results
                for cls_idx, iou in enumerate(class_ious):
                    if not np.isnan(iou):
                        all_class_ious[cls_idx].append(iou)
                all_mean_ious.append(mean_iou)
                
                # Save visualization for first 20 images
                if batch_idx * dataloader.batch_size + i < 20:
                    original_img = cv2.imread(color_paths[i])
                    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                    
                    fig = create_comparison_viz(original_img, pred_mask, gt_mask, 
                                               COLOR_MAP, filenames[i])
                    
                    viz_path = os.path.join(output_dir, "visualizations", 
                                           f"comparison_{filenames[i]}")
                    fig.savefig(viz_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    final_mean_iou = np.mean(all_mean_ious)
    
    # Calculate per-class IoU
    class_iou_avgs = {}
    for cls in range(NUM_CLASSES):
        if all_class_ious[cls]:
            class_iou_avgs[cls] = np.mean(all_class_ious[cls])
        else:
            class_iou_avgs[cls] = float('nan')
    
    # Print results
    print("\n" + "="*60)
    print("FINAL TEST RESULTS")
    print("="*60)
    print(f"\n{'Metric':<30} {'Value':<15}")
    print("-" * 60)
    print(f"{'Test Loss':<30} {avg_loss:.4f}")
    print(f"{'Mean IoU (mIoU)':<30} {final_mean_iou:.4f}")
    print(f"{'Mean IoU Percentage':<30} {final_mean_iou*100:.2f}%")
    
    print("\n" + "="*60)
    print("PER-CLASS IoU SCORES")
    print("="*60)
    print(f"\n{'Class':<20} {'IoU':<15} {'Percentage':<15}")
    print("-" * 60)
    
    for cls in range(NUM_CLASSES):
        class_name = ID2LABEL[cls]
        iou = class_iou_avgs[cls]
        if not np.isnan(iou):
            print(f"{class_name:<20} {iou:.4f} {'':<7} {iou*100:.2f}%")
        else:
            print(f"{class_name:<20} {'N/A':<15} {'N/A':<15}")
    
    # Create IoU bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    classes = [ID2LABEL[i] for i in range(NUM_CLASSES)]
    ious = [class_iou_avgs[i] if not np.isnan(class_iou_avgs[i]) else 0 
            for i in range(NUM_CLASSES)]
    
    bars = ax.bar(classes, ious, color='steelblue', edgecolor='black', linewidth=1.5)
    ax.axhline(y=final_mean_iou, color='red', linestyle='--', linewidth=2, 
               label=f'Mean IoU: {final_mean_iou:.4f}')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('IoU Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class IoU Scores on Test Set', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results to text file
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("FINAL TEST RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Mean IoU (mIoU): {final_mean_iou:.4f}\n")
        f.write(f"Mean IoU Percentage: {final_mean_iou*100:.2f}%\n\n")
        f.write("="*60 + "\n")
        f.write("PER-CLASS IoU SCORES\n")
        f.write("="*60 + "\n\n")
        for cls in range(NUM_CLASSES):
            class_name = ID2LABEL[cls]
            iou = class_iou_avgs[cls]
            if not np.isnan(iou):
                f.write(f"{class_name:<20} {iou:.4f} ({iou*100:.2f}%)\n")
            else:
                f.write(f"{class_name:<20} N/A\n")
    
    print("\n" + "="*60)
    print(f"✓ Results saved to '{output_dir}/' directory")
    print(f"  - Detailed results: {output_dir}/test_results.txt")
    print(f"  - IoU chart: {output_dir}/per_class_iou.png")
    print(f"  - Sample visualizations: {output_dir}/visualizations/")
    print("="*60 + "\n")
    
    return final_mean_iou, class_iou_avgs

def main():
    print(f"Using device: {DEVICE}")
    
    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
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
    
    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Trained for {checkpoint['epoch']+1} epochs")
    if 'val_iou' in checkpoint:
        print(f"  Best Validation mIoU: {checkpoint['val_iou']:.4f}")
    
    # Create test dataset
    test_dataset = TestDatasetWithGT(TEST_COLOR_PATH, TEST_SEG_PATH, processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Run evaluation
    final_miou, class_ious = test_model_with_evaluation(model, test_loader)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print(f"FINAL MEAN IoU: {final_miou:.4f} ({final_miou*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()