#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from transformers import TimesformerModel, TimesformerConfig
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import numpy as np
import cv2
import mediapipe as mp

# MediaPipe Pose 모델 로드 (관절 좌표 추출)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# OpenMP 및 MKL 관련 환경 변수 설정 (CPU dispatcher 오류 방지)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# GPU 정보 출력
def get_device_info():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    return device

# TimeSformer 기반 비디오 인코더 (비디오 특징 추출) pretrained
# class VideoEncoder(nn.Module):
#     def __init__(self, model_name="facebook/timesformer-base-finetuned-k400"):
#         super(VideoEncoder, self).__init__()
#         self.model = TimesformerModel.from_pretrained(model_name)

#     def forward(self, video_frames):
#         B, C, T, H, W = video_frames.shape
#         video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) → (B, T, C, H, W)
#         return self.model(video_frames).last_hidden_state.mean(dim=1)  # 최종 특징 벡터 반환

class VideoEncoder(nn.Module):
    def __init__(self, num_frames=16, image_size=224, patch_size=32, num_classes=768):
        super(VideoEncoder, self).__init__()
        config = TimesformerConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_channels=3,
            num_attention_heads=6,
            hidden_size=768,
            num_hidden_layers=6,
            intermediate_size=3072,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            num_classes=num_classes
        )
        self.model = TimesformerModel(config)  # 사전학습 없이 초기화된 모델 사용

    def forward(self, video_frames):
        B, C, T, H, W = video_frames.shape
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) → (B, T, C, H, W)
        return self.model(video_frames).last_hidden_state.mean(dim=1)  # 최종 특징 벡터 반환


# Pose 키포인트 인코더 (Pose 특징 추출)
class PoseEncoder(nn.Module):
    def __init__(self, input_dim=99, hidden_dim=256, output_dim=768):
        super(PoseEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, pose_data):
        return self.encoder(pose_data)


# 비디오-포즈 정렬을 위한 Self-Supervised Learning 모델
class SSLMultiModalModel(nn.Module):
    def __init__(self, video_dim=768, pose_dim=99, hidden_dim=512):
        super(SSLMultiModalModel, self).__init__()
        self.video_encoder = VideoEncoder()
        self.pose_encoder = PoseEncoder(input_dim=pose_dim, hidden_dim=256, output_dim=video_dim)

    def forward(self, video_frames, pose_data):
        video_feat = self.video_encoder(video_frames)
        pose_feat = self.pose_encoder(pose_data)
        return video_feat, pose_feat


# Contrastive Learning 손실 함수
def contrastive_video_pose_loss(video_feat, pose_feat, temperature=0.5):
    batch_size = video_feat.shape[0]

    # 특징 정규화 (L2 정규화)
    video_feat = F.normalize(video_feat, dim=-1)
    pose_feat = F.normalize(pose_feat, dim=-1)

    # 비디오 + 포즈 특징을 하나의 행렬로 결합 (2N, D)
    similarity_matrix = torch.cat([video_feat, pose_feat], dim=0)
    similarity_matrix = torch.matmul(similarity_matrix, similarity_matrix.T) / temperature  # 유사도 행렬 계산

    # 자기 자신을 제외한 유사도 행렬 생성
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=video_feat.device)  # 대각선 제거 (자기 자신 비교 X)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))  # 자기 자신과의 유사도 제거

    # 정확한 Positive Pair를 보장하기 위해 labels 설정 수정
    labels = torch.arange(batch_size, device=video_feat.device)  # [0, 1, 2, ..., batch_size-1]
    labels = torch.cat([labels + batch_size, labels], dim=0)  # [batch_size, 2*batch_size] 형태로 정렬하여 올바른 Positive Pair 매칭

    return F.cross_entropy(similarity_matrix, labels)  # 크로스 엔트로피 손실 적용


# CLIP-style Contrastive Loss (Video-to-Pose & Pose-to-Video)
def clip_style_contrastive_loss(video_feat, pose_feat, temperature=0.1):
    batch_size = video_feat.shape[0]

    # 특징 정규화 (L2 정규화)
    video_feat = F.normalize(video_feat, dim=-1)
    pose_feat = F.normalize(pose_feat, dim=-1)

    # Video-to-Pose Similarity Matrix
    video_to_pose_sim = torch.matmul(video_feat, pose_feat.T) / temperature
    pose_to_video_sim = video_to_pose_sim.T

    # 정답 레이블 생성
    labels = torch.arange(batch_size, device=video_feat.device)

    # Softmax & Cross Entropy 적용
    loss_video_to_pose = F.cross_entropy(video_to_pose_sim, labels)
    loss_pose_to_video = F.cross_entropy(pose_to_video_sim, labels)

    # 최종 Loss (평균)
    return (loss_video_to_pose + loss_pose_to_video) / 2

class VideoPoseDataset(Dataset):
    def __init__(self, video_dir, frame_interval=1.0):
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.frame_interval = frame_interval  # ✅ 1초 단위 프레임 추출 간격 설정

    def __len__(self):
        return len(self.video_files)

    def extract_keypoints(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return None

    def load_video(self, video_path, frame_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_times = np.arange(0, total_frames / fps, self.frame_interval)  # ✅ 1초 단위로 프레임 선택
        frame_indices = (frame_times * fps).astype(int)

        frames = []
        keypoints_list = []
        valid_frames = {}

        # 1초 간격으로 프레임 선택
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, frame_size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            keypoints = self.extract_keypoints(frame_rgb)
            
            if keypoints is not None:
                valid_frames[idx] = (frame_rgb, keypoints)
            frames.append((idx, frame_rgb, keypoints))
        
        # 키포인트가 없는 경우, 모든 프레임에서 가장 가까운 키포인트 포함된 프레임 탐색
        for i, (idx, frame_rgb, keypoints) in enumerate(frames):
            if keypoints is None:
                closest_idx = min(valid_frames.keys(), key=lambda x: abs(x - idx)) if valid_frames else None
                if closest_idx is not None:
                    frames[i] = (idx, valid_frames[closest_idx][0], valid_frames[closest_idx][1])

        cap.release()
        
        selected_frames = [frame for _, frame, _ in frames]
        selected_keypoints = [keypoints for _, _, keypoints in frames]
        
        frames = np.stack(selected_frames, axis=0)
        keypoints_array = np.stack(selected_keypoints, axis=0)

        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        keypoints_tensor = torch.tensor(keypoints_array, dtype=torch.float32)
        return frames, keypoints_tensor

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_tensor, pose_tensor = self.load_video(video_path)
        return video_tensor, pose_tensor
    def extract_keypoints(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return None

   
    # ... [existing methods] ...
    
    def visualize_frames_and_keypoints_separately(self, idx, frame_size=(128, 128)):
        """
        Visualize video frames and keypoints separately,
        with keypoints displayed on a black background.
        """
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_times = np.arange(0, total_frames / fps, self.frame_interval)
        frame_indices = (frame_times * fps).astype(int)

        frames = []
        valid_frames = {}

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = self.extract_keypoints(frame_rgb)
            
            if keypoints is not None:
                valid_frames[idx] = (frame_rgb, keypoints)
            frames.append((idx, frame_rgb, keypoints))
        
        # Handle missing keypoints by finding closest valid frame
        for i, (idx, frame_rgb, keypoints) in enumerate(frames):
            if keypoints is None:
                closest_idx = min(valid_frames.keys(), key=lambda x: abs(x - idx)) if valid_frames else None
                if closest_idx is not None:
                    frames[i] = (idx, frame_rgb, valid_frames[closest_idx][1])
        
        cap.release()

        selected_frames = [frame for _, frame, _ in frames]
        selected_keypoints = [keypoints for _, _, keypoints in frames]

        # Create black background images for keypoints
        keypoint_images = []
        for keypoints in selected_keypoints:
            black_bg = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
            if keypoints is not None:
                for kp in keypoints:
                    x, y = int(kp[0] * frame_size[1]), int(kp[1] * frame_size[0])
                    if 0 <= x < frame_size[1] and 0 <= y < frame_size[0]:
                        cv2.circle(black_bg, (x, y), 4, (0, 255, 0), -1)
            keypoint_images.append(black_bg)

        if selected_frames and keypoint_images:
            # Concatenate original frames
            concatenated_frames = np.hstack(selected_frames)
            
            # Concatenate keypoint images
            concatenated_keypoints = np.hstack(keypoint_images)
            
            # Create a figure with two subplots
            plt.figure(figsize=(14, 6))
            
            # Display frames in the first subplot
            plt.subplot(2, 1, 1)
            plt.imshow(concatenated_frames)
            plt.axis("off")
            plt.title("Original Video Frames")
            
            # Display keypoints in the second subplot
            plt.subplot(2, 1, 2)
            plt.imshow(concatenated_keypoints)
            plt.axis("off")
            plt.title("Keypoints on Black Background")
            
            plt.tight_layout()
            plt.show()
        else:
            print("Error: Unable to read frames from the video.")


   

# 학습 루프
if __name__ == "__main__":
    device = get_device_info()
    video_dir = "./GAIT_VIDEO_COMPLETE"
    dataset = VideoPoseDataset(video_dir)
    for j in range(5):
        dataset.visualize_frames_and_keypoints_separately(j)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    ssl_model = SSLMultiModalModel().to(device)
    optimizer = optim.Adam([
    {'params': ssl_model.video_encoder.parameters(), 'lr': 1e-5},  # 비디오 인코더는 낮은 학습률
    {'params': ssl_model.pose_encoder.parameters(), 'lr': 1e-4}   # 포즈 인코더는 기본 학습률
])
    num_epochs = 50
    loss_function = clip_style_contrastive_loss

    for epoch in range(num_epochs):
        ssl_model.train()
        total_loss = 0
        val_loss = 0
        all_video_feats, all_pose_feats = [], []

        for videos, poses in train_loader:
            videos, poses = videos.to(device), poses.to(device)

            optimizer.zero_grad()
            video_feat, pose_feat = ssl_model(videos, poses)
            loss = loss_function(video_feat, pose_feat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_video_feats.append(video_feat.cpu().detach())
            all_pose_feats.append(pose_feat.cpu().detach())

        print(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        visualize_embeddings(torch.cat(all_video_feats, dim=0), torch.cat(all_pose_feats, dim=0), epoch, "Train")

        ssl_model.eval()
        all_video_feats, all_pose_feats = [], []
        with torch.no_grad():
            for videos, poses in val_loader:
                videos, poses = videos.to(device), poses.to(device)
                video_feat, pose_feat = ssl_model(videos, poses)
                all_video_feats.append(video_feat.cpu().detach())
                all_pose_feats.append(pose_feat.cpu().detach())
                loss = loss_function(video_feat, pose_feat)
                val_loss += loss.item()

        print(f"Valid Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss / len(val_loader)}")
        visualize_embeddings(torch.cat(all_video_feats, dim=0), torch.cat(all_pose_feats, dim=0), epoch, "Validation")

    torch.save(ssl_model.state_dict(), "ssl_stage1_model.pth")

# %%
