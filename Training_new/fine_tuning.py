import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from deepfake_detector import EfficientNetV2WithAttention, Config
from torch import nn, optim

# ===============================
# Dataset class for fine-tuning
# ===============================
class FineTuneDataset(Dataset):
    def __init__(self, root_dir, classes=None, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.classes = classes or ['fake', 'real']
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    self.images.append(os.path.join(cls_dir, f))
                    self.labels.append(idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ===============================
# Fine-tuning function
# ===============================
def fine_tune_model(data_dir, checkpoint_path, epochs=5, batch_size=16, lr=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Transforms matching training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # Dataset & DataLoader
    dataset = FineTuneDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    print(f"Total images for fine-tuning: {len(dataset)}")

    # Load model
    model = EfficientNetV2WithAttention(num_classes=1, dropout_rate=Config.DROPOUT_RATE)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.train()
    print("Checkpoint loaded, starting fine-tuning...")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f}")

    # Save fine-tuned model
    fine_tuned_path = os.path.join(os.path.dirname(checkpoint_path), "fine_tuned_model.pth")
    torch.save({'model_state_dict': model.state_dict()}, fine_tuned_path)
    print(f"Fine-tuned model saved to: {fine_tuned_path}")
    return fine_tuned_path

# ===============================
# Run fine-tuning
# ===============================
if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # Folder structure:
    # data_dir/
    #    fake/   -> some fake images (optional, can include 0 or few for balance)
    #    real/   -> real images from new dataset (FaceForensics / Gemini)
    data_dir = r"C:\black_paper\FYP\deepfake-detection-project\data\Human"
    checkpoint_file = r"C:\black_paper\FYP\deepfake-detection-project\Training_new\checkpoints\deepfake_detector_best_f1_0.9966.pth"

    fine_tune_model(data_dir, checkpoint_file, epochs=5, batch_size=16, lr=1e-5)