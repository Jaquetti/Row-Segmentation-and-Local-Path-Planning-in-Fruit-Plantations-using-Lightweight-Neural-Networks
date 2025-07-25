import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import json # ADDED: For saving results
from CreateModel import Model
from torch.optim.lr_scheduler import LambdaLR


# --- Assuming these are your custom files ---
from ProcessDataset import ProcessDataset

# --- Function to create DeepLabV3 (Unchanged) ---
def create_lightweight_deeplab(num_classes=1, pretrained=True):
    weights = torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, num_classes, kernel_size=1)
    return model

# --- Configuration (Unchanged) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 16
warmup_epochs = 0.1*EPOCHS
LEARNING_RATE = 0.001
CHECKPOINT_DIR = "checkpoints_cv_currentmodel_warmup_320"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def create_deeplabv3plus(num_classes=1, encoder_name="mobilenet_v2", pretrained="imagenet"):
    """
    Creates a DeepLabV3+ model using the segmentation-models-pytorch library.
    """
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,      # You can try other backbones like "resnet34"
        encoder_weights=pretrained,     # Use pre-trained weights from ImageNet
        in_channels=3,                  # Input channels (3 for RGB)
        classes=num_classes,            # Output classes (1 for binary segmentation)
    )
    return model
# ------------------------------------------------------------------------------------

# --- Loss and Metrics (Unchanged) ---
dice_loss = smp.losses.DiceLoss(mode='binary')
focal_loss = smp.losses.FocalLoss(mode='binary')
def total_loss_fn(preds, targets):
    return dice_loss(preds, targets) + 0.25 * focal_loss(preds, targets)

# --- Early Stopping Class (Unchanged) ---
class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.verbose = verbose
    def __call__(self, val_loss, model_to_save, path):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model_to_save.state_dict(), path)
            if self.verbose:
                print(f"✔️ Val loss improved. Best model for fold saved to {path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# --- Data Loading (Unchanged) ---
# train_dataset_part = ProcessDataset("../data/Fruticultura/train_folder/images", "../data/Fruticultura/train_folder/labels", fixsize=640)
# val_dataset_part = ProcessDataset("../data/Fruticultura/valid_folder/images", "../data/Fruticultura/valid_folder/labels", fixsize=640, do_train=False)
# full_dataset = ConcatDataset([train_dataset_part, val_dataset_part])
# kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
# --- Data Loading ---
# Point this to your single folder containing all images and masks.
full_dataset = ProcessDataset(
    "../data/Fruticultura/all/images", 
    "../data/Fruticultura/all/labels", 
    fixsize=320
)

kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

print(f"Full dataset size: {len(full_dataset)}")

# --- History Tracking (Unchanged) ---
fold_results = {'val_loss': [], 'val_f1': [], 'val_iou': []}
all_folds_iou_history = []


# --- Main K-Fold Cross-Validation Loop ---
for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f'=============== FOLD {fold+1}/{K_FOLDS} ===============')
    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=4, pin_memory=True)
    
    # model = create_lightweight_deeplab(num_classes=1).to(device)
    # model = nn.DataParallel(model)
    model = Model(num_classes=1, K=8).to(device)
    # model = create_deeplabv3plus().to(device)
    model = nn.DataParallel(model)
    # model  = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    def warmup_lambda(epoch):
        warmup_epochs=20
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 1.0  # after warmup, use main scheduler

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    best_fold_metrics = {'loss': float('inf'), 'f1': 0.0, 'iou': 0.0}
    this_fold_iou_history = []

    # --- Epoch Loop for the current fold ---
    for epoch in range(EPOCHS):
        train_loss = 0
        model.train()
        loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1} [T]", leave=False)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = total_loss_fn(outputs, masks)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)


        # Validation
        model.eval()
        val_loss, val_f1, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += total_loss_fn(outputs, masks).item()
                preds_sig = torch.sigmoid(outputs)
                val_f1 += BinaryF1Score(threshold=0.5).to(device)(preds_sig, masks.int()).item()
                val_iou += BinaryJaccardIndex(threshold=0.5).to(device)(preds_sig, masks.int()).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = val_f1 / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        this_fold_iou_history.append(avg_val_iou)

        if avg_val_loss < best_fold_metrics['loss']:
            best_fold_metrics['loss'] = avg_val_loss
            best_fold_metrics['f1'] = avg_val_f1
            best_fold_metrics['iou'] = avg_val_iou

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{EPOCHS} -> "
          f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Val F1: {avg_val_f1:.4f} | Val IoU: {avg_val_iou:.4f} | "
          f"LR: {current_lr:.6f}")
    

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(avg_val_loss)

        fold_model_path = os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold+1}.pth")
        early_stopping(avg_val_loss, model, fold_model_path)
        if early_stopping.early_stop:
            print(f"🛑 Fold {fold+1} early stopping at epoch {epoch+1}")
            break
            
    fold_results['val_loss'].append(best_fold_metrics['loss'])
    fold_results['val_f1'].append(best_fold_metrics['f1'])
    fold_results['val_iou'].append(best_fold_metrics['iou'])
    all_folds_iou_history.append(this_fold_iou_history)

# --- Plotting Function (Unchanged) ---
def plot_kfold_curves(iou_history, num_folds):
    plt.figure(figsize=(12, 8))
    max_epochs = max(len(history) for history in iou_history)
    padded_iou_history = []
    for history in iou_history:
        padded_history = history + [history[-1]] * (max_epochs - len(history))
        padded_iou_history.append(padded_history)
    
    iou_history_np = np.array(padded_iou_history)
    
    for i in range(num_folds):
        plt.plot(iou_history_np[i, :], label=f'Fold {i+1} IoU', alpha=0.4)
        
    mean_iou = iou_history_np.mean(axis=0)
    std_iou = iou_history_np.std(axis=0)
    
    plt.plot(mean_iou, 'b-', label='Average IoU', linewidth=2.5)
    plt.fill_between(range(max_epochs), mean_iou - std_iou, mean_iou + std_iou, color='b', alpha=0.2, label='Std. Dev.')

    plt.title('K-Fold Cross-Validation IoU Curves Proposed method', fontsize=16)
    # plt.title('K-Fold Cross-Validation: IoU Curves for DeepLabV3+ ', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Validation IoU', fontsize=12)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig('kfold_validation_iou_curves_currentmodel.png')
    plt.close() # Use close() instead of show() for scripts
    print("\n✅ K-Fold validation IoU plot saved to 'kfold_validation_iou_curves.png'")

# --- Final Report and Saving ---
plot_kfold_curves(all_folds_iou_history, K_FOLDS)

print('\n=============== K-FOLD CROSS-VALIDATION FINAL RESULTS ===============')
print(f'{K_FOLDS} folds validation results (based on best epoch of each fold):')
print(f"  - Average Val Loss: {np.mean(fold_results['val_loss']):.4f} ± {np.std(fold_results['val_loss']):.4f}")
print(f"  - Average Val F1:   {np.mean(fold_results['val_f1']):.4f} ± {np.std(fold_results['val_f1']):.4f}")
print(f"  - Average Val IoU:  {np.mean(fold_results['val_iou']):.4f} ± {np.std(fold_results['val_iou']):.4f}")
print('=======================================================================')

# --------------------------------------------------------------------------
# --- ADDED: Save the final results dictionary to a JSON file ---
# --------------------------------------------------------------------------
results_filepath = 'kfold_results.json'
with open(results_filepath, 'w') as f:
    json.dump(fold_results, f, indent=4)

print(f"\n✅ Final aggregated results saved to {results_filepath}")
# --------------------------------------------------------------------------