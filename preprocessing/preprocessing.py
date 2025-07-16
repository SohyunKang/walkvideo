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

from ultralytics import YOLO

# 영상 샘플링 후 frame 개수 확인용
def get_sampled_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    sampled_frames = int(video_duration // frame_interval) + 1
    cap.release()
    return sampled_frames

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

def process_video(video_path, pre_type, frame_size, segment_length, gif_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_times = np.arange(0, total_frames / fps, frame_interval)
    frame_indices = (frame_times * fps).astype(int)

    model = YOLO("yolov8m-seg.pt") if pre_type == 'seg' else None
    frames = []
    segments = []

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    gif_save_base = os.path.join(gif_dir, f"{video_basename}") if gif_dir else None
    os.makedirs(gif_dir, exist_ok=True) if gif_dir else None

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        if pre_type == 'raw':
            frame_resized = cv2.resize(frame, frame_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        elif pre_type == 'seg':
            results = model(frame, verbose=False)
            masks = results[0].masks.data if results[0].masks is not None else []
            boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
            classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
            masks = masks.cpu().numpy() if len(masks) > 0 else []

            if len(boxes) > 0 and len(masks) > 0:
                person_indices = [i for i, cls in enumerate(classes) if int(cls) == 0]
                if person_indices:
                    leftmost_idx = min(person_indices, key=lambda i: boxes[i][0])
                    selected_mask = masks[leftmost_idx]
                    m_bin = (selected_mask > 0.5).astype(np.uint8) * 255
                    m_bin_resized = cv2.resize(m_bin, frame_size)
                    rgb_mask = cv2.cvtColor(m_bin_resized, cv2.COLOR_GRAY2RGB)
                    frames.append(rgb_mask)

        if len(frames) == segment_length:
            tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
            segments.append(tensor)

            # 🔹 GIF 저장
            if gif_save_base:
                gif_path = f"{gif_save_base}_{len(segments)-1}.gif"
                imageio.mimsave(gif_path, frames, fps=fps)
                print(gif_path)
            frames = []

    # 마지막 segment 처리 (padding)
    if len(frames) > 0:
        tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        C, T, H, W = tensor.shape
        if T < segment_length:
            pad = torch.zeros((C, segment_length - T, H, W), dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad], dim=1)
        segments.append(tensor)

        # 🔹 마지막 GIF 저장
        if gif_save_base:
            gif_path = f"{gif_save_base}_{len(segments)-1}.gif"
            imageio.mimsave(gif_path, frames, fps=fps)

    cap.release()
    return segments  # List[Tensor]


# def process_video(video_path, pre_type, frame_size, min_sampled_frames):
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     frame_times = np.arange(0, total_frames / fps, frame_interval)
#     frame_indices = (frame_times * fps).astype(int)

#     frames = []
#     if pre_type == 'raw':
#         for idx in frame_indices[:min_sampled_frames]:  # ⭐ 최소 프레임까지만 자르기
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame_resized = cv2.resize(frame, frame_size)
#             frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
#             frames.append(frame_rgb)
#         cap.release()

#         frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
#         frames_tensor = pad_or_truncate(frames_tensor, target_frames=min_sampled_frames)
    
#     elif pre_type == 'seg':
#         model = YOLO("yolov8m-seg.pt")
#         for idx in frame_indices[:min_sampled_frames]:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             # tic = time.time()

#             # YOLOv8 Segmentation 실행
#             results = model(frame, verbose=False)
#             masks = results[0].masks.data if results[0].masks is not None else []
#             boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
#             classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None else []
#             masks = masks.cpu().numpy() if len(masks) > 0 else []

#             combined_mask = np.zeros(frame_size, dtype=np.uint8)

#             if len(boxes) > 0 and len(masks) > 0:
#                 person_indices = [i for i, cls in enumerate(classes) if int(cls) == 0]
#                 if person_indices:
#                     leftmost_idx = min(person_indices, key=lambda i: boxes[i][0])
#                     selected_mask = masks[leftmost_idx]
#                     m_bin = (selected_mask > 0.5).astype(np.uint8) * 255
#                     m_bin_resized = cv2.resize(m_bin, frame_size)
#                     rgb_mask = cv2.cvtColor(m_bin_resized, cv2.COLOR_GRAY2RGB)
            
#                     frames.append(rgb_mask)

#             # toc = time.time()
#             # elapsed_times.append(toc - tic)

#         cap.release()

#         frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
#         frames_tensor = pad_or_truncate(frames_tensor, target_frames=min_sampled_frames)
#         # 저장
#         # print(video_path.split('/')[-1].split('.')[0]+'.gif')
#         gif_path = f'./preprocessing/processed_video_{pre_type}_{size_frame}/'
#         os.makedirs(gif_path, exist_ok=True)
#         imageio.mimsave(gif_path+video_path.split('/')[-1].split('.')[0]+'.gif', frames, fps=fps)

#     return frames_tensor

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

# 기본 설정
video_dir = "./GAIT_VIDEO_COMPLETE"
pre_type = 'seg'
size_frame = "224_16"
save_dir = f"./preprocessing/processed_tensor_{pre_type}_{size_frame}"
gif_path = f'./preprocessing/processed_video_{pre_type}_{size_frame}/'
os.makedirs(save_dir, exist_ok=True)
frame_interval = 0.6602  # 8 frames (1.3204), 16 frames (0.6602), 32 frames (0.3200)
frame_size = (224, 224)

# 엑셀 불러오기
# Step 0: 엑셀 로드 및 field dict 준비
demo_path = './preprocessing/DEMO_FINAL_EXTERNAL.xlsx'
demo = pd.read_excel(demo_path)

# 각 환자 정보를 ANONYID 기준으로 저장
patient_info_dict = {}
for _, row in demo.iterrows():
    ptid = str(row['ANONYID'])
    patient_info_dict[ptid] = {
        col: (row[col] if pd.notna(row[col]) else -1)
        for col in demo.columns
    }

# Step 1: 가장 짧은 영상 찾기
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
min_sampled_frames = float('inf')
print("Finding minimum frame length across videos...")
for video_file in tqdm(video_files):
    video_path = os.path.join(video_dir, video_file)
    sampled_frames = get_sampled_frame_count(video_path)
    if sampled_frames < min_sampled_frames:
        min_sampled_frames = sampled_frames
        shortest_video_file = video_file

print(f"✅ Minimum sampled frames: {min_sampled_frames}, Shortest video: {shortest_video_file}")

# Step 2: 가장 짧은 영상 처리 및 GIF 저장
shortest_video_path = os.path.join(video_dir, shortest_video_file)
print(f"Processing the shortest video: {shortest_video_file}")
processed_tensor = process_video(shortest_video_path, pre_type, frame_size, min_sampled_frames, None)
tensor_to_gif(processed_tensor[0], gif_save_path="processed_shortest_video.gif", fps=5)

# Step 3: 전체 전처리 및 저장
# print("✅ Starting preprocessing and saving tensors...")
# for video_file in tqdm(video_files):
#     video_path = os.path.join(video_dir, video_file)
#     tensor = process_video(video_path, pre_type, frame_size, min_sampled_frames)

#     video_name = video_file.split('_')[0]
#     patient_info = patient_info_dict.get(video_name)

#     if patient_info is not None:
#         save_data = {'video': tensor}
#         save_data.update(patient_info)  # 개별 field를 직접 삽입
#         save_path = os.path.join(save_dir, f"{video_name}.pt")
#         torch.save(save_data, save_path)
#     else:
#         print(f"⚠️ Patient info not found for {video_name}")

print("✅ Starting preprocessing and saving tensors...")

for video_file in tqdm(video_files):
    video_path = os.path.join(video_dir, video_file)
    video_name = video_file.split('_')[0]
    patient_info = patient_info_dict.get(video_name)

    if patient_info is None:
        print(f"⚠️ Patient info not found for {video_name}")
        continue

    # 🔹 process_video 함수는 이제 List[Tensor] 반환
    segment_tensors = process_video(video_path, pre_type, frame_size, min_sampled_frames, gif_path)

    # 🔹 각 segment를 개별 파일로 저장
    for i, tensor in enumerate(segment_tensors):
        save_data = {'video': tensor}
        save_data.update(patient_info)  # 메타데이터 추가
        save_path = os.path.join(save_dir, f"{video_name}_{i}.pt")
        torch.save(save_data, save_path)
