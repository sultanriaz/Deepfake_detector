# deepfake_detector.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import random
import json
import csv
from datetime import datetime
import io
import hashlib

# ==================== REPRODUCIBILITY ====================
def set_seeds(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seeds(42)

# ==================== FOCAL LOSS IMPLEMENTATION ====================
class FocalLoss(nn.Module):
    """Focal Loss for binary classification"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# ==================== CONFIGURATION ====================
class Config:
    # Paths
    FAKE_PATH = r"C:\black_paper\FYP\deepfake-detection-project\data\deep-detect-2025\ddata\test\fake"
    REAL_PATH = r"C:\black_paper\FYP\deepfake-detection-project\data\deep-detect-2025\ddata\test\real"
    CHECKPOINT_DIR = "checkpoints"
    METRICS_FILE = "training_metrics.json"
    
    # Training parameters - Adjusted for T1200 (4GB VRAM)
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 64
    EPOCHS = 30
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2
    
    # Dataset split
    VALIDATION_SPLIT = 0.2
    SUBSET_FRACTION = 1.0
    
    # Model
    MODEL_NAME = 'efficientnet_v2_s'
    DROPOUT_RATE = 0.3
    
    # Loss function
    LOSS_TYPE = "bce"  # "bce" or "focal"
    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0
    
    # Mixed precision
    USE_AMP = True
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # Confidence scaling
    CONFIDENCE_SCALING = 1.0
    
    # Artifact augmentation probabilities
    JPEG_COMPRESSION_PROB = 0.3
    GAUSSIAN_NOISE_PROB = 0.3
    GAUSSIAN_BLUR_PROB = 0.2
    HIGH_PASS_PROB = 0.1

# ==================== ARTIFACT AUGMENTATIONS ====================
class ArtifactAugmentation:
    """Specialized augmentations to simulate deepfake artifacts"""
    
    @staticmethod
    def jpeg_compression(image, quality=None):
        """Simulate JPEG compression artifacts"""
        if quality is None:
            quality = random.randint(30, 80)
        
        # Save to JPEG with low quality and read back
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)
    
    @staticmethod
    def gaussian_noise(image, std_range=(0.01, 0.05)):
        """Add Gaussian noise to image"""
        img_array = np.array(image) / 255.0
        std = random.uniform(*std_range)
        noise = np.random.normal(0, std, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 1) * 255
        return Image.fromarray(noisy_img.astype(np.uint8))
    
    @staticmethod
    def gaussian_blur(image, radius_range=(0.5, 2.0)):
        """Apply Gaussian blur"""
        radius = random.uniform(*radius_range)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    @staticmethod
    def high_pass_filter(image):
        """High-pass filtering to emphasize artifacts"""
        # Apply edge enhancement
        enhanced = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        # Blend with original
        enhancer = ImageEnhance.Sharpness(enhanced)
        return enhancer.enhance(1.5)
    
    @staticmethod
    def random_resize_crop(image, scale_range=(0.8, 1.0)):
        """Random resize and crop to simulate resolution artifacts"""
        width, height = image.size
        scale = random.uniform(*scale_range)
        new_width, new_height = int(width * scale), int(height * scale)
        
        # Resize down then up to create artifacts
        small = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BILINEAR)

def apply_artifact_augmentation(image):
    """Apply artifact augmentations probabilistically"""
    
    # JPEG compression (common in downloaded images)
    if random.random() < Config.JPEG_COMPRESSION_PROB:
        image = ArtifactAugmentation.jpeg_compression(image)
    
    # Gaussian noise (common in GAN artifacts)
    if random.random() < Config.GAUSSIAN_NOISE_PROB:
        image = ArtifactAugmentation.gaussian_noise(image)
    
    # Gaussian blur (common in diffusion models)
    if random.random() < Config.GAUSSIAN_BLUR_PROB:
        image = ArtifactAugmentation.gaussian_blur(image)
    
    # High-pass filtering (emphasize artifacts)
    if random.random() < Config.HIGH_PASS_PROB:
        image = ArtifactAugmentation.high_pass_filter(image)
    
    # Random resize/crop (simulate upscaling artifacts)
    if random.random() < 0.2:  # 20% chance
        image = ArtifactAugmentation.random_resize_crop(image)
    
    return image

# ==================== CUSTOM DATASET ====================
class DeepfakeDataset(Dataset):
    def __init__(self, fake_path, real_path, transform=None, phase='train', apply_artifacts=False):
        self.images = []
        self.labels = []
        self.transform = transform
        self.phase = phase
        self.apply_artifacts = apply_artifacts and phase == 'train'
        
        # Load fake images (label = 1)
        print(f"Loading fake images from {fake_path}")
        fake_images = [os.path.join(fake_path, f) for f in os.listdir(fake_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        self.images.extend(fake_images)
        self.labels.extend([1] * len(fake_images))
        
        # Load real images (label = 0)
        print(f"Loading real images from {real_path}")
        real_images = [os.path.join(real_path, f) for f in os.listdir(real_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        self.images.extend(real_images)
        self.labels.extend([0] * len(real_images))
        
        print(f"Total images loaded: {len(self.images)}")
        print(f"Class distribution: Fake={len(fake_images)}, Real={len(real_images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Apply artifact augmentations for training
            if self.apply_artifacts:
                image = apply_artifact_augmentation(image)
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image and label
            return torch.zeros((3, 224, 224)), label

# ==================== DATA TRANSFORMS ====================
def get_transforms(phase='train'):
    """Get augmentation transforms based on phase"""
    
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Improved validation transforms
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# ==================== STRATIFIED DATA LOADING ====================
def load_data(config):
    """Load and split dataset with stratification"""
    
    # Create full dataset without transforms first to get labels
    temp_dataset = DeepfakeDataset(
        fake_path=config.FAKE_PATH,
        real_path=config.REAL_PATH,
        transform=None,
        phase='train',
        apply_artifacts=False
    )
    
    # Get all labels
    all_labels = temp_dataset.labels
    
    # Create train/val indices with stratification
    indices = list(range(len(temp_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.VALIDATION_SPLIT,
        stratify=all_labels,
        random_state=42
    )
    
    # Apply subset if specified
    if config.SUBSET_FRACTION < 1.0:
        subset_size = int(len(train_indices) * config.SUBSET_FRACTION)
        train_indices = train_indices[:subset_size]
        print(f"Using subset: {subset_size} training images")
    
    # Create datasets with appropriate transforms
    train_dataset = DeepfakeDataset(
        fake_path=config.FAKE_PATH,
        real_path=config.REAL_PATH,
        transform=get_transforms('train'),
        phase='train',
        apply_artifacts=True
    )
    
    val_dataset = DeepfakeDataset(
        fake_path=config.FAKE_PATH,
        real_path=config.REAL_PATH,
        transform=get_transforms('val'),
        phase='val',
        apply_artifacts=False
    )
    
    # Create subsets using indices
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    
    # Verify class distribution
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"\nClass distribution after stratification:")
    print(f"Train - Fake: {sum(train_labels)}, Real: {len(train_labels) - sum(train_labels)}")
    print(f"Val - Fake: {sum(val_labels)}, Real: {len(val_labels) - sum(val_labels)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=False
    )
    
    print(f"\nTrain size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    return train_loader, val_loader

# Build model function
def build_model(config, device):
    """Initialize and prepare model"""
    
    model = EfficientNetV2WithAttention(
        num_classes=1,
        dropout_rate=config.DROPOUT_RATE
    )
    
    model = model.to(device)
    
    return model

# ==================== MODEL ARCHITECTURE ====================
class EfficientNetV2WithAttention(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-V2-S
        if Config.MODEL_NAME == 'efficientnet_v2_s':
            self.backbone = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Get the feature dimension
        if 'efficientnet' in Config.MODEL_NAME:
            num_features = self.backbone.classifier[1].in_features
            # Remove the original classifier
            self.backbone.classifier = nn.Identity()
        
        # Simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_features, num_features // 16),
            nn.ReLU(),
            nn.Linear(num_features // 16, num_features),
            nn.Sigmoid()
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        output = self.classifier(attended_features)
        return output

# ==================== METRIC LOGGING ====================
class MetricLogger:
    def __init__(self, filename):
        self.filename = filename
        self.history = []
        
        # Create file with header if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                json.dump([], f)
    
    def log_epoch(self, epoch, train_loss, val_loss, metrics):
        """Log metrics for one epoch"""
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'accuracy': metrics['accuracy'],
            'f1_score': metrics['f1'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(log_entry)
        
        # Append to JSON file
        with open(self.filename, 'r+') as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=2)
        
        # Also save as CSV for easy viewing
        csv_filename = self.filename.replace('.json', '.csv')
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

# ==================== TRAINING FUNCTION ====================
def train_model(model, train_loader, val_loader, config, device):
    """Main training loop with memory optimization"""
    
    # Loss function
    if config.LOSS_TYPE == "focal":
        criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
        print(f"Using Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if config.USE_AMP else None
    
    # Metric logger
    logger = MetricLogger(config.METRICS_FILE)
    
    # Training metrics
    best_f1 = 0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Clear cache before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        optimizer.zero_grad()
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            # Mixed precision training
            if config.USE_AMP and scaler:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if batch_idx % 50 == 0:
                        torch.cuda.empty_cache()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            
            # Store predictions
            probs = torch.sigmoid(outputs.detach())
            preds = (probs > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': train_loss / (batch_idx + 1)})
            
            del images, labels, outputs, probs, preds
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        
        # Validation phase
        val_loss, val_metrics, val_conf_matrix = validate_model(
            model, val_loader, criterion, device, config
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log metrics
        logger.log_epoch(epoch + 1, train_loss, val_loss, val_metrics)
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}")
        
        print("\nConfusion Matrix (Validation):")
        print(val_conf_matrix)
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✓ New best model! F1: {best_f1:.4f}")
            
            save_model(model, optimizer, epoch, best_f1, config)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)
    
    return model, train_losses, val_losses

# ==================== VALIDATION FUNCTION ====================
def validate_model(model, val_loader, criterion, device, config):
    """Validate the model with memory optimization"""
    
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            
            if config.USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            del images, labels, outputs, probs, preds
    
    val_loss /= len(val_loader)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1': f1_score(all_targets, all_preds, zero_division=0),
        'precision': precision_score(all_targets, all_preds, zero_division=0),
        'recall': recall_score(all_targets, all_preds, zero_division=0)
    }
    
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return val_loss, metrics, conf_matrix

# Inference function with confidence scores
def predict_image(model, image_path, device, config):
    """Predict single image with confidence score"""
    
    transform = get_transforms('val')
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            if config.USE_AMP:
                with autocast():
                    output = model(image_tensor)
            else:
                output = model(image_tensor)
            
            probability = torch.sigmoid(output).item()
        
        # Apply confidence scaling
        probability = probability * config.CONFIDENCE_SCALING
        probability = min(max(probability, 0), 1)  # Clamp to [0, 1]
        
        confidence = probability * 100 if probability > 0.5 else (1 - probability) * 100
        
        result = {
            'prediction': 'FAKE' if probability > 0.5 else 'REAL',
            'confidence': confidence,
            'raw_probability': probability
        }
        
        return result
    
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None

# ==================== SAVE MODEL ====================
def save_model(model, optimizer, epoch, f1_score, config):
    """Save model checkpoint with improved naming"""
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Move model to CPU for saving
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': f1_score,
        'config': {k: v for k, v in config.__dict__.items() if not k.startswith('__')},
        'timestamp': datetime.now().isoformat()
    }
    
    # Create filename with F1 score
    filename = f'deepfake_detector_best_f1_{f1_score:.4f}.pth'
    path = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

# ==================== GRAD-CAM UTILITY (OPTIONAL) ====================
class GradCAM:
    """Grad-CAM visualization utility"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = 0 if output.item() < 0 else 1
        
        # Backward pass
        self.model.zero_grad()
        output[:, target_class].backward()
        
        # Get weights
        gradients = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (gradients * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()

def visualize_gradcam(model, image_path, device, config):
    """Optional function to visualize Grad-CAM"""
    try:
        from torchvision import transforms
        import matplotlib.pyplot as plt
        
        # Find last convolutional layer
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            print("Could not find convolutional layer for Grad-CAM")
            return
        
        # Load and preprocess image
        transform = get_transforms('val')
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Generate CAM
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate_cam(input_tensor)
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        ax2.imshow(image)
        ax2.imshow(cam[0, 0], cmap='jet', alpha=0.5)
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")


# ==================== MAIN FUNCTION ====================
def main():
    """Main execution function"""
    
    print("=" * 60)
    print("Deepfake Detector Training Pipeline")
    print("=" * 60)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data with stratification
    print("\n" + "=" * 60)
    print("Loading Data with Stratification")
    print("=" * 60)
    train_loader, val_loader = load_data(Config)
    
    # Build model
    print("\n" + "=" * 60)
    print("Building Model")
    print("=" * 60)
    model = build_model(Config, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, Config, device
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model

if __name__ == "__main__":
    model = main()