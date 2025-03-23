#%%
import os
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import imageio
from PIL import Image
import torchvision.transforms as transforms
from IPython.display import Image as IPyImage, display

# 기본 설정
video_dir = "./GAIT_VIDEO_COMPLETE"
save_dir = "./processed_tensor"
os.makedirs(save_dir, exist_ok=True)
frame_interval = 0.8125  # 1초마다 샘플링
frame_size = (224, 224)

# 라벨 준비
demo_path = './DEMO_ANNOY_250310_474PARTICIPANTS.xlsx'
demo = pd.read_excel(demo_path)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(demo['DX1'].astype(str))
label_dict = {f"{row['PTID_ANAOY']}": encoded_labels[idx] for idx, row in demo.iterrows()}

# 영상 샘플링 후 frame 개수 확인용
def get_sampled_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    sampled_frames = int(video_duration // frame_interval) + 1
    cap.release()
    return sampled_frames

# ⭐ Step 1: 가장 짧은 영상 찾기
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
min_sampled_frames = float('inf')
print("Finding minimum frame length across videos...")
for video_file in tqdm(video_files):
    video_path = os.path.join(video_dir, video_file)
    sampled_frames = get_sampled_frame_count(video_path)
    if sampled_frames < min_sampled_frames:
        min_sampled_frames = sampled_frames
        shortest_video_file = video_file  # 가장 짧은 영상 저장

print(f"✅ Minimum sampled frames: {min_sampled_frames}, Shortest video: {shortest_video_file}")

# ⭐ 전처리용 함수
def pad_or_truncate(tensor, target_frames):
    C, T, H, W = tensor.shape
    if T == target_frames:
        return tensor
    elif T < target_frames:
        pad = torch.zeros((C, target_frames - T, H, W), dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)
    else:
        return tensor[:, :target_frames, :, :]

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_times = np.arange(0, total_frames / fps, frame_interval)
    frame_indices = (frame_times * fps).astype(int)

    frames = []
    for idx in frame_indices[:min_sampled_frames]:  # ⭐ 최소 프레임까지만 자르기
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, frame_size)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
    frames_tensor = pad_or_truncate(frames_tensor, target_frames=min_sampled_frames)
    return frames_tensor

# ⭐ 전처리 후 바로 Tensor로 GIF 생성
def tensor_to_gif(tensor, gif_save_path="processed_shortest_video.gif", fps=5):
    frames = []
    T = tensor.shape[1]
    for t in range(T):
        frame = tensor[:, t, :, :]  # (C, H, W)
        frame = transforms.ToPILImage()(frame.cpu())
        frames.append(frame)

    # GIF 저장
    frames[0].save(
        gif_save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"✅ Processed video GIF saved at {gif_save_path}")
    
    # 주피터에서 바로 보기
    try:
        display(IPyImage(filename=gif_save_path))
    except:
        pass

# ⭐ Step 2: 가장 짧은 영상만 전처리하고 GIF로 바로 보여주기
shortest_video_path = os.path.join(video_dir, shortest_video_file)
print(f"Processing the shortest video: {shortest_video_file}")
processed_tensor = process_video(shortest_video_path)

# GIF로 저장 및 시각화
tensor_to_gif(processed_tensor, gif_save_path="processed_shortest_video.gif", fps=5)

# ⭐ Step 3: 전체 전처리 시작 (나머지 저장용)
print("✅ Starting preprocessing and saving tensors...")
for video_file in tqdm(video_files):
    video_path = os.path.join(video_dir, video_file)
    tensor = process_video(video_path)
    video_name = video_file.split('_')[0]
    torch.save({'video': tensor, 'label': label_dict.get(video_name, -1)}, os.path.join(save_dir, f"{video_name}.pt"))

print("✅ All videos processed and saved with uniform frame length.")
