import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# -------------------------------
# 1. Configuration & Constants
# -------------------------------
DATA_PATH = r"C:\black_paper\FYP\deepfake-detection-project\data\deep-detect-2025\ddata\train"
SAVE_PATH = "best_deepfake_detector.pth"

IMG_SIZE = 512           # High resolution for better artifact detection
BATCH_SIZE = 4           # Optimized for 4GB T1200 VRAM
ACCUM_STEPS = 8          # Effective Batch Size = 32
EPOCHS = 20
SUBSET_FRACTION = 0.10 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    print(f"--- Configuration Check ---")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not found. Training will be extremely slow on CPU.")

    # -------------------------------
    # 2. Data Preparation
    # -------------------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
    all_indices = list(range(len(full_dataset)))
    all_labels = full_dataset.targets

    # Stratified subsetting to maintain Real/Fake balance
    subset_indices, _ = train_test_split(
        all_indices, train_size=SUBSET_FRACTION, stratify=all_labels, random_state=42
    )
    subset_labels = [all_labels[i] for i in subset_indices]

    train_idx, val_idx = train_test_split(
        subset_indices, test_size=0.2, stratify=subset_labels, random_state=42
    )

    # Note: num_workers is set to 2 to avoid overwhelming your 16GB RAM during multi-process spawning
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # -------------------------------
    # 3. Model Setup
    # -------------------------------
    # EfficientNet-V2-S is better for limited VRAM than heavier ViT models
    model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=torch.cuda.is_available()) # Only scale if CUDA is active

    best_f1 = 0.0

    # -------------------------------
    # 4. Training Loop
    # -------------------------------
    print(f"\nStarting training on {len(train_idx)} images...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / ACCUM_STEPS
                
            scaler.scale(loss).backward()
            running_loss += loss.item() * ACCUM_STEPS
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # -------------------------------
        # 5. Validation Phase
        # -------------------------------
        model.eval()
        all_preds, val_actuals = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                val_actuals.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        all_preds = np.array(all_preds)
        val_actuals = np.array(val_actuals)
        
        # Calculate Metrics
        acc = (all_preds == val_actuals).mean()
        f1 = f1_score(val_actuals, all_preds)
        prec = precision_score(val_actuals, all_preds, zero_division=0)
        rec = recall_score(val_actuals, all_preds, zero_division=0)
        cm = confusion_matrix(val_actuals, all_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(f"Metrics -> Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
        print(f"Confusion Matrix:\n{cm}\n")
        
        # Save checkpoint if it's the most accurate (F1)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'f1_score': best_f1,
                'img_size': IMG_SIZE
            }, SAVE_PATH)
            print(f"--- New Best Model Saved (F1: {best_f1:.4f}) ---")

# -------------------------------
# 5. The "Main Guard" (Required for Windows)
# -------------------------------
if __name__ == '__main__':
    train_model()