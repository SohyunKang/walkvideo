import cv2
import imageio
import mediapipe as mp
import numpy as np
from PIL import Image

def extract_frames_from_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_times = np.arange(0, total_frames / fps, 0.6602)
    frame_indices = (frame_times * fps).astype(int)
    frames = []

    for idx in frame_indices[:num_frames]:  # ⭐ 최소 프레임까지만 자르기
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (448,448))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


def draw_keypoints(frames):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    output_frames = []

    for frame in frames:
        results = pose.process(frame)

        # 흰 배경 생성 (frame과 동일한 크기)
        blank_image = np.ones_like(frame) * 0

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(blank_image, (x, y), 10, (0, 255, 0), -1)

        output_frames.append(blank_image)
    return output_frames


def extract_frames_from_gif(gif_path, num_frames=16):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGB").resize((448, 448))
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    selected = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    return [frames[i] for i in selected]

def save_concat_image(frames_1, frames_2, frames_3, output_path):
    combined_frames = []
    for f1, f2, f3 in zip(frames_1, frames_2, frames_3):
        combined = np.vstack([f1, f2, f3])  # 위-아래로 붙이기
        combined_frames.append(combined)
    final_image = np.hstack(combined_frames)
    Image.fromarray(final_image).save(output_path)
    print(f"Saved to: {output_path}")

# === 사용 예시 ===
video_path = '/home/fjqmfl5676/PycharmProjects/walkvideo/__GAIT_VIDEO_COMPLETE/SUBJ0001_GAIT_SIDE_preproc.mp4'
gif_path = '/home/fjqmfl5676/PycharmProjects/walkvideo/preprocessing/processed_video_seg_448_16/SUBJ0001_GAIT_SIDE_preproc.gif'
output_path = 'final_output.png'

video_frames = extract_frames_from_video(video_path)
keypoint_frames = draw_keypoints(video_frames)
seg_frames = extract_frames_from_gif(gif_path)

save_concat_image(video_frames, keypoint_frames, seg_frames, output_path)
