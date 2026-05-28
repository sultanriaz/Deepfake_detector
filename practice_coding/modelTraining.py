import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
import timm
DATASET_ROOT = r"C:\black_paper\FYP\deepfake-detection-project\data\FaceForensics++_C23_frames_v2"
CHECKPOINT_PATH = "best_deepfake_model.pth"

BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-5  # Safer LR to prevent NaN
IMAGE_SIZE = 224
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BinaryDeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.class_map = {"original": 0, "Deepfakes": 1}

        for cls_name, label in self.class_map.items():
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            for root, _, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(root, f), label))
        self.transform = transform
        print(f"[INFO] Dataset loaded: {len(self.samples)} images")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = np.clip(np.array(img), 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
def main():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    full_dataset = BinaryDeepfakeDataset(DATASET_ROOT)
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor([class_weights[l] for l in train_labels], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=2).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_seen = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_seen += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100*train_correct/train_seen:.2f}%")
        scheduler.step()
        model.eval()
        val_loss, val_correct, val_seen = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_seen += labels.size(0)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_seen
        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("✅ Best model updated")
    print(f"\n[SUCCESS] Training finished. Best model saved at {CHECKPOINT_PATH}")
if __name__ == "__main__":
    main()