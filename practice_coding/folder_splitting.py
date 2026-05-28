<<<<<<< HEAD
import os
import random
import shutil

random.seed(42)

BASE_DIR = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_frames"
OUT_DIR  = r"D:\XodeCerberus\deepfake-detection-project\data\dataset"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASSES = {
    "fake": "Deepfakes",
    "real": "Original"
}

for label, folder in CLASSES.items():
    src_root = os.path.join(BASE_DIR, folder)
    video_dirs = sorted(os.listdir(src_root))
    random.shuffle(video_dirs)

    n = len(video_dirs)
    train_end = int(SPLITS["train"] * n)
    val_end   = train_end + int(SPLITS["val"] * n)

    split_map = {
        "train": video_dirs[:train_end],
        "val":   video_dirs[train_end:val_end],
        "test":  video_dirs[val_end:]
    }

    for split, vids in split_map.items():
        for vid in vids:
            src = os.path.join(src_root, vid)
            dst = os.path.join(OUT_DIR, split, label, vid)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

print("✅ Dataset split completed successfully.")
=======
import os
import random
import shutil

random.seed(42)

BASE_DIR = r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23_frames"
OUT_DIR  = r"D:\XodeCerberus\deepfake-detection-project\data\dataset"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

CLASSES = {
    "fake": "Deepfakes",
    "real": "Original"
}

for label, folder in CLASSES.items():
    src_root = os.path.join(BASE_DIR, folder)
    video_dirs = sorted(os.listdir(src_root))
    random.shuffle(video_dirs)

    n = len(video_dirs)
    train_end = int(SPLITS["train"] * n)
    val_end   = train_end + int(SPLITS["val"] * n)

    split_map = {
        "train": video_dirs[:train_end],
        "val":   video_dirs[train_end:val_end],
        "test":  video_dirs[val_end:]
    }

    for split, vids in split_map.items():
        for vid in vids:
            src = os.path.join(src_root, vid)
            dst = os.path.join(OUT_DIR, split, label, vid)

            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)

print("✅ Dataset split completed successfully.")
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
