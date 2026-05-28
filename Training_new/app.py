# app.py
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from PIL import Image
import io
import base64
import os
import glob
from deepfake_detector import build_model, Config, predict_image, EfficientNetV2WithAttention
from torch.serialization import add_safe_globals

app = Flask(__name__)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

def load_model(checkpoint_path):
    global model

    print("=" * 60)
    print("Loading Deepfake Detection Model")
    print("=" * 60)

    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Device: {device}")

    try:
        # --------------------------------------------------
        # 1. Build model architecture
        # --------------------------------------------------
        print("\nBuilding model architecture...")

        model = EfficientNetV2WithAttention(
            num_classes=1,
            dropout_rate=Config.DROPOUT_RATE
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # --------------------------------------------------
        # 2. Load checkpoint
        # --------------------------------------------------
        print("\nLoading checkpoint...")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )

        print("✓ Checkpoint loaded successfully")

        # Show checkpoint keys (helps debugging)
        print("\nCheckpoint contents:")
        for key in checkpoint.keys():
            print(f"  - {key}")

        # --------------------------------------------------
        # 3. Load model weights
        # --------------------------------------------------
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("✓ Model weights loaded from 'model_state_dict'")
        else:
            # fallback if checkpoint saved directly
            model.load_state_dict(checkpoint)
            print("✓ Model weights loaded directly")

        # --------------------------------------------------
        # 4. Move model to device
        # --------------------------------------------------
        model = model.to(device)
        model.eval()

        print(f"\n✓ Model moved to device: {device}")
        print("✓ Model set to evaluation mode")

        # --------------------------------------------------
        # 5. Display training metrics if available
        # --------------------------------------------------
        f1_score = checkpoint.get("best_f1", checkpoint.get("f1_score", None))
        epoch = checkpoint.get("epoch", None)

        if f1_score is not None:
            print(f"✓ Best validation F1 score: {f1_score:.4f}")

        if epoch is not None:
            print(f"✓ Training epoch: {epoch}")

        print("\n✓ Model successfully initialized!")
        print("=" * 60)

        return True

    except Exception as e:
        print("\n✗ Model loading failed!")
        print("Error:", str(e))

        import traceback
        traceback.print_exc()

        return False
def get_latest_checkpoint():
    """Find the most recent checkpoint file"""
    if not os.path.exists(Config.CHECKPOINT_DIR):
        print(f"Checkpoint directory not found: {Config.CHECKPOINT_DIR}")
        return None
    
    checkpoint_files = glob.glob(os.path.join(Config.CHECKPOINT_DIR, "*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {Config.CHECKPOINT_DIR}")
        return None
    
    # Get the most recent checkpoint by creation time
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image"""
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, WEBP)'}), 400
    
    temp_path = None
    try:
        # Save temporarily with unique name to avoid conflicts
        temp_path = f'temp_image_{hash(file.filename)}.jpg'
        file.save(temp_path)
        
        # Predict
        result = predict_image(model, temp_path, device, Config)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Prediction failed - unable to process image'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    files = request.files.getlist('images')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No images selected'}), 400
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            # Validate file type
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                results.append({
                    'filename': file.filename,
                    'error': 'Invalid file type'
                })
                continue
            
            try:
                temp_path = f'temp_{hash(file.filename)}_{len(temp_files)}.jpg'
                file.save(temp_path)
                temp_files.append(temp_path)
                
                result = predict_image(model, temp_path, device, Config)
                
                if result:
                    result['filename'] = file.filename
                    results.append(result)
                else:
                    results.append({
                        'filename': file.filename,
                        'error': 'Prediction failed'
                    })
                    
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(results),
            'successful': sum(1 for r in results if 'error' not in r),
            'failed': sum(1 for r in results if 'error' in r),
            'results': results
        })
        
    finally:
        # Clean up all temporary files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'device': str(device),
        'model_type': Config.MODEL_NAME,
        'checkpoint_dir': Config.CHECKPOINT_DIR,
        'status': 'loaded'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Deepfake Detector API Server")
    print("=" * 60)
    
    # Find and load the latest model checkpoint
    latest_checkpoint = get_latest_checkpoint()
    
    if latest_checkpoint:
        success = load_model(latest_checkpoint)
        if success:
            print("\n✓ Server ready! Access at: http://localhost:5000")
            print("✓ API endpoints:")
            print("  - GET  /          : Web interface")
            print("  - POST /predict   : Single image prediction")
            print("  - POST /predict_batch : Batch prediction")
            print("  - GET  /model_info : Model information")
            print("  - GET  /health    : Health check")
            print("\n" + "=" * 60)
            
            # Run the Flask app
            app.run(debug=True, host='0.0.0.0', port=5000)
        else:
            print("✗ Failed to load model. Please check your checkpoint file.")
    else:
        print("✗ No checkpoints found. Please train the model first:")
        print("  python deepfake_detector.py")