import cv2
import os

video_dir = "/home/s5726453/RT_Video_Stabilizer/training"
output_dir = os.path.join(video_dir, "frames")
os.makedirs(output_dir, exist_ok=True)

for i in range(1, 5):
    video_path = os.path.join(video_dir, f"{i:02d}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open {video_path}")
        continue

    vid_out_path = os.path.join(output_dir, f"video_{i:02d}")
    os.makedirs(vid_out_path, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (256, 256))
        out_path = os.path.join(vid_out_path, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame_resized)
        frame_idx += 1

    cap.release()
    print(f"✅ Extracted {frame_idx} frames from {video_path}")

