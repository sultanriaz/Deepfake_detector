<<<<<<< HEAD
# Write a script that:
# Traverses a directory recursively
# Prints file count per folder

# from pathlib import Path
# import os
# root_path= Path("D:\XodeCerberus\deepfake-detection-project\data")
# for root, dirs , files in os.walk(root_path):
#     file_count=len(files)
#     print(f"folder : {root}")
#     print(f"Contains : {file_count} files")




# Write a program that:
# Creates missing directories safely
# Does not crash if they already exist

# import os 
# dir=r"D:\XodeCerberus\deepfake-detection-project\data\testing_only"
# os.makedirs(dir, exist_ok=True)



# import random 
# def craete_random_file(filename, num_lines=20):
#     phrase= [
#         "A quick brown fox jumps over the lazy dog",
#         "he was running very fast in a speed of 110 km per second",
#         "Now we got it and found it dead as we wanted him to"
#     ]
#     sultan_pharase= [
#         "Sultan riaz is reading Pakistan",
#         "Sultan is ready to fight against millitants",
#         "the winner will be sultan "
#     ]
#     with open(filename, 'w', encoding='utf-8') as f:
#         for _ in range(num_lines):
#             if random.random() < 0.3:
#                 line=random.choice(sultan_pharase)
#             else:
#                 line= random.choice(phrase)
#         f.write(line + '\n')
# craete_random_file('newfile.txt',  num_lines=50)
# print(f"File newfile.txt created successfully ")
        
        
import cv2
from pathlib import Path

def getvideo_info(path_v):
    cap=cv2.VideoCapture(path_v)
    if cap.isOpened():
        fps=cap.get(cv2.CAP_PROP_FPS)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration=total_frames/fps
        frame_id=0
        save_pathr= Path(r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23\DeepFakeDetection")
        while True:
            ret , frame= cap.read()
            if not ret:
                break
            save_path=save_pathr / f"frame_{frame_id}.jpg"

            cv2.imwrite(str(save_path), frame)
            frame_id+=1
    print(f"Video Path: {path_v}")
    print(f"FPS:          {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration:     {duration:.2f} seconds")
    cap.release()
=======
# Write a script that:
# Traverses a directory recursively
# Prints file count per folder

# from pathlib import Path
# import os
# root_path= Path("D:\XodeCerberus\deepfake-detection-project\data")
# for root, dirs , files in os.walk(root_path):
#     file_count=len(files)
#     print(f"folder : {root}")
#     print(f"Contains : {file_count} files")




# Write a program that:
# Creates missing directories safely
# Does not crash if they already exist

# import os 
# dir=r"D:\XodeCerberus\deepfake-detection-project\data\testing_only"
# os.makedirs(dir, exist_ok=True)



# import random 
# def craete_random_file(filename, num_lines=20):
#     phrase= [
#         "A quick brown fox jumps over the lazy dog",
#         "he was running very fast in a speed of 110 km per second",
#         "Now we got it and found it dead as we wanted him to"
#     ]
#     sultan_pharase= [
#         "Sultan riaz is reading Pakistan",
#         "Sultan is ready to fight against millitants",
#         "the winner will be sultan "
#     ]
#     with open(filename, 'w', encoding='utf-8') as f:
#         for _ in range(num_lines):
#             if random.random() < 0.3:
#                 line=random.choice(sultan_pharase)
#             else:
#                 line= random.choice(phrase)
#         f.write(line + '\n')
# craete_random_file('newfile.txt',  num_lines=50)
# print(f"File newfile.txt created successfully ")
        
        
import cv2
from pathlib import Path

def getvideo_info(path_v):
    cap=cv2.VideoCapture(path_v)
    if cap.isOpened():
        fps=cap.get(cv2.CAP_PROP_FPS)
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration=total_frames/fps
        frame_id=0
        save_pathr= Path(r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23\DeepFakeDetection")
        while True:
            ret , frame= cap.read()
            if not ret:
                break
            save_path=save_pathr / f"frame_{frame_id}.jpg"

            cv2.imwrite(str(save_path), frame)
            frame_id+=1
    print(f"Video Path: {path_v}")
    print(f"FPS:          {fps:.2f}")
    print(f"Total Frames: {total_frames}")
    print(f"Duration:     {duration:.2f} seconds")
    cap.release()
>>>>>>> cd7429d9f936cca3505da743b7cc3b1f2102ea83
getvideo_info(r"D:\XodeCerberus\deepfake-detection-project\data\FaceForensics++_C23\DeepFakeDetection\01_02__meeting_serious__YVGY8LOK.mp4")