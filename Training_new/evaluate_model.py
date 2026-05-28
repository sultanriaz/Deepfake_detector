import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from deepfake_detector import EfficientNetV2WithAttention, Config
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

# Add numpy scalar to PyTorch safe globals for loading checkpoints
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

class EvalDataset(Dataset):
    """Dataset for evaluating images split into 'real' and 'fake' subfolders"""
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.classes = ['fake', 'real']  # 1 = fake, 0 = real
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
        return image, label, path

def evaluate_model(eval_dir, checkpoint_path, batch_size=8):
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
    dataset = EvalDataset(eval_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Total images to evaluate: {len(dataset)}")

    # Load model
    model = EfficientNetV2WithAttention(num_classes=1, dropout_rate=Config.DROPOUT_RATE)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    all_preds = []
    all_probs = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save detailed results
    df = pd.DataFrame({
        'image_path': all_paths,
        'true_label': all_labels,
        'pred_label': all_preds,
        'confidence': all_probs
    })
    results_file = os.path.join(eval_dir, "evaluation_results.csv")
    df.to_csv(results_file, index=False)
    print(f"Detailed results saved to {results_file}")

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # Paths to your dataset and checkpoint
    eval_directory = r"C:\black_paper\FYP\deepfake-detection-project\data\deep-detect-2025\ddata\test"  # should contain 'fake' and 'real' subfolders
    checkpoint_file = r"C:\black_paper\FYP\deepfake-detection-project\Training_new\checkpoints\fine_tuned_model.pth"

    evaluate_model(eval_directory, checkpoint_file, batch_size=8)