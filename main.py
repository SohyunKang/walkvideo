import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from transformers import TimesformerModel
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
import numpy as np
import cv2
import mediapipe as mp
import os

# MediaPipe Pose 모델 로드 (관절 좌표 추출)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# GPU 정보 출력
def get_device_info():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    return device

# TimeSformer 기반 비디오 인코더 (비디오 특징 추출)
class VideoEncoder(nn.Module):
    def __init__(self, model_name="facebook/timesformer-base-finetuned-k400"):
        super(VideoEncoder, self).__init__()
        self.model = TimesformerModel.from_pretrained(model_name)

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
def clip_style_contrastive_loss(video_feat, pose_feat, temperature=0.5):
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

# 비디오 및 키포인트 데이터셋 클래스 (비디오 파일을 로드하여 데이터 생성)
class VideoPoseDataset(Dataset):
    def __init__(self, video_dir, max_samples=100):
        self.video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
        self.video_files = self.video_files[:max_samples]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_tensor = self.load_video(video_path)
        pose_tensor = self.extract_pose_from_video(video_path)
        return video_tensor, pose_tensor

    def load_video(self, video_path, frame_size=(224, 224), max_frames=16):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 개수 가져오기

        if total_frames > max_frames:
            frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)  # 균등 간격으로 선택
        else:
            frame_indices = np.arange(total_frames)  # 총 프레임 개수가 적으면 그냥 사용

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 특정 프레임으로 이동
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        while len(frames) < max_frames:  # 부족한 프레임을 0으로 채우기
            frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))

        frames = np.stack(frames, axis=0)
        frames = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0
        return frames

    def extract_pose_from_video(self, video_path):
        keypoints_list = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            keypoints = self.extract_keypoints(frame)
            keypoints_list.append(keypoints)
            if len(keypoints_list) >= 8:
                break
        cap.release()

        if not keypoints_list:
            keypoints_list = [np.zeros(99)] * 8
        keypoints_array = np.mean(np.array(keypoints_list), axis=0)
        pose_tensor = torch.tensor(keypoints_array, dtype=torch.float32)
        return pose_tensor

    def extract_keypoints(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = [landmark.x for landmark in results.pose_landmarks.landmark] + \
                        [landmark.y for landmark in results.pose_landmarks.landmark] + \
                        [landmark.z for landmark in results.pose_landmarks.landmark]
            return np.array(keypoints).flatten()
        else:
            return np.zeros(99)

    def visualize_video_frames(self, idx, frame_size=(32, 32), max_frames=16):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int) if total_frames > max_frames else np.arange(total_frames)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, frame_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        while len(frames) < max_frames:
            frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8))

        if frames:
            concatenated_image = np.hstack(frames)
            plt.figure(figsize=(12, 2))
            plt.imshow(concatenated_image)
            plt.axis("off")
            plt.title("Concatenated Frames from Video (Used in Dataset)")
            plt.show()
        else:
            print("Error: 비디오에서 프레임을 읽을 수 없습니다.")


# Feature Embedding 시각화 함수
def visualize_embeddings(video_feats, pose_feats, epoch, phase):
    reducer = umap.UMAP(n_components=2)
    combined_feats = torch.cat([video_feats.cpu().detach(), pose_feats.cpu().detach()], dim=0)
    reduced_feats = reducer.fit_transform(combined_feats.clone().detach().cpu().numpy().astype(np.float32))

    labels = ["Video"] * video_feats.shape[0] + ["Pose"] * pose_feats.shape[0]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_feats[:, 0], y=reduced_feats[:, 1], hue=labels, palette=["blue", "red"], alpha=0.7)
    plt.title(f"{phase} Feature Embeddings at Epoch {epoch}")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend()
    plt.show()

# 학습 루프
if __name__ == "__main__":
    device = get_device_info()
    video_dir = "./GAIT_VIDEO_COMPLETE"
    dataset = VideoPoseDataset(video_dir)
    dataset.visualize_video_frames(0)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    ssl_model = SSLMultiModalModel().to(device)
    optimizer = optim.Adam(ssl_model.parameters(), lr=1e-4)
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