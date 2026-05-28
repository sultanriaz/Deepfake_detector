<<<<<<< HEAD
import cv2
import os
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
INPUT_ROOT = r"C:\black_paper\FYP\deepfake-detection-project\data\FaceForensics++_C23"
OUTPUT_ROOT = r"C:\black_paper\FYP\deepfake-detection-project\data\FaceForensics++_C23_frames_v2"
CLASSES = ["Deepfakes", "original"]
TARGET_SIZE = (224, 224) 
FRAMES_PER_SECOND = 1      
VALID_EXTS = (".mp4", ".avi", ".mov", ".mkv")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def get_face_crop(frame, padding=0.25):
    temp_frame = frame.copy()
    gray = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    img_h, img_w = frame.shape[:2]
    x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
    x2, y2 = min(img_w, x + w + pad_w), min(img_h, h + y + pad_h)
    return frame[y1:y2, x1:x2]
def extract_frames(video_args):
    """Worker function for high-fidelity extraction."""
    video_path, video_output_dir = video_args
    if os.path.exists(video_output_dir) and len(os.listdir(video_output_dir)) > 0:
        return
    os.makedirs(video_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(int(fps // FRAMES_PER_SECOND), 1)
    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            face_img = get_face_crop(frame)  
            if face_img is None:
                h, w = frame.shape[:2]
                side = min(h, w)
                face_img = frame[(h-side)//2:(h+side)//2, (w-side)//2:(w+side)//2]
            face_img = cv2.resize(face_img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
            out_path = os.path.join(video_output_dir, f"frame_{saved_count:04d}.png")
            cv2.imwrite(out_path, face_img)
            saved_count += 1
        frame_idx += 1
    cap.release()
def main():
    tasks = []
    print(f"[INFO] Initializing Highest Quality Pipeline...")
    for cls in CLASSES:
        input_class_path = os.path.join(INPUT_ROOT, cls)
        search_pattern = os.path.join(input_class_path, "**", "*.*")
        all_files = glob.glob(search_pattern, recursive=True)
        videos = [f for f in all_files if f.lower().endswith(VALID_EXTS)]
        for v_path in videos:
            rel_path = os.path.relpath(v_path, input_class_path)
            v_folder_name = os.path.splitext(rel_path)[0].replace(os.sep, "_")
            v_out = os.path.join(OUTPUT_ROOT, cls, v_folder_name)
            tasks.append((v_path, v_out))
    if not tasks:
        print(f"[ERROR] Found 0 videos. Check folder paths and permissions.")
        return
    print(f"[INFO] Processing {len(tasks)} videos using {multiprocessing.cpu_count() - 1} cores.")
    with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count() - 1)) as executor:
        executor.map(extract_frames, tasks)
    print(f"\n[SUCCESS] Extraction complete. Frames stored in: {OUTPUT_ROOT}")
if __name__ == "__main__":
    main()
=======
import cv2
import os


INPUT_ROOT = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23"
OUTPUT_ROOT = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_frames"

CLASSES = ["Deepfakes", "Original"]

TARGET_SIZE = (224, 224)    
FRAMES_PER_SECOND = 1         
JPEG_QUALITY = 90            

VALID_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        video_fps = 30  # fallback

    frame_interval = max(int(video_fps // FRAMES_PER_SECOND), 1)

    frame_index = 0
    saved_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame = cv2.resize(frame, TARGET_SIZE)

            frame_name = f"frame_{saved_index:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)

            cv2.imwrite(
                frame_path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )

            saved_index += 1

        frame_index += 1

    cap.release()

def process_class(class_name):
    input_dir = os.path.join(INPUT_ROOT, class_name)
    output_dir = os.path.join(OUTPUT_ROOT, class_name)

    os.makedirs(output_dir, exist_ok=True)

    videos = sorted(os.listdir(input_dir))

    for video_file in videos:
        if not video_file.lower().endswith(VALID_VIDEO_EXTENSIONS):
            continue

        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        print(f"[INFO] {class_name} | Processing: {video_file}")
        extract_frames(video_path, video_output_dir)

def main():
    for cls in CLASSES:
        process_class(cls)

    print("\n[SUCCESS] Frame extraction completed.")


if __name__ == "__main__":
    main()
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
