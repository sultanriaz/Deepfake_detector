<<<<<<< HEAD
import os
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

INPUT_ROOT = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_frames"
OUTPUT_ROOT = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_faces_v2"

CLASSES = ["Deepfakes", "Original"]

IMAGE_SIZE = 224     # SAME as extracted frames
MARGIN = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=MARGIN,
    keep_all=False,
    post_process=False,   # 🔥 CRITICAL FIX
    device=DEVICE
)

def process_video(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(input_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        face = mtcnn(image)

        # Skip if no face detected
        if face is None:
            continue

        # Tensor -> PIL Image
        face = face.permute(1, 2, 0).cpu().numpy()
        face = (face * 255).astype("uint8")

        save_path = os.path.join(output_dir, img_name)
        Image.fromarray(face).save(save_path)


def main():
    for cls in CLASSES:
        class_input_dir = os.path.join(INPUT_ROOT, cls)
        class_output_dir = os.path.join(OUTPUT_ROOT, cls)

        os.makedirs(class_output_dir, exist_ok=True)

        videos = os.listdir(class_input_dir)

        for video in tqdm(videos, desc=f"Cropping faces | {cls}"):
            video_input_dir = os.path.join(class_input_dir, video)
            video_output_dir = os.path.join(class_output_dir, video)

            process_video(video_input_dir, video_output_dir)

    print("\n[SUCCESS] Face cropping completed (224×224).")


if __name__ == "__main__":
    main()
=======
import os
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

INPUT_ROOT = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_frames"
OUTPUT_ROOT = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_faces_v2"

CLASSES = ["Deepfakes", "Original"]

IMAGE_SIZE = 224     # SAME as extracted frames
MARGIN = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=MARGIN,
    keep_all=False,
    post_process=False,   # 🔥 CRITICAL FIX
    device=DEVICE
)

def process_video(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(input_dir, img_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        face = mtcnn(image)

        # Skip if no face detected
        if face is None:
            continue

        # Tensor -> PIL Image
        face = face.permute(1, 2, 0).cpu().numpy()
        face = (face * 255).astype("uint8")

        save_path = os.path.join(output_dir, img_name)
        Image.fromarray(face).save(save_path)


def main():
    for cls in CLASSES:
        class_input_dir = os.path.join(INPUT_ROOT, cls)
        class_output_dir = os.path.join(OUTPUT_ROOT, cls)

        os.makedirs(class_output_dir, exist_ok=True)

        videos = os.listdir(class_input_dir)

        for video in tqdm(videos, desc=f"Cropping faces | {cls}"):
            video_input_dir = os.path.join(class_input_dir, video)
            video_output_dir = os.path.join(class_output_dir, video)

            process_video(video_input_dir, video_output_dir)

    print("\n[SUCCESS] Face cropping completed (224×224).")


if __name__ == "__main__":
    main()
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
