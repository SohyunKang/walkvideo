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
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision.transforms as transforms

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
num_frames = 26
class VideoEncoder(nn.Module):
    def __init__(self, num_frames=num_frames, image_size=224, patch_size=32, hidden_dim=768):
        super(VideoEncoder, self).__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches_per_frame = (image_size // patch_size) ** 2

        config = TimesformerConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_channels=3,
            num_attention_heads=6,
            hidden_size=hidden_dim,
            num_hidden_layers=6,
            intermediate_size=3072,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            num_classes=hidden_dim
        )
        self.model = TimesformerModel(config)

    def forward(self, video_frames):
        B, C, T, H, W = video_frames.shape
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        # print(B,C,T,H,W)
        output = self.model(video_frames)
        # print(f"Timesformer output shape: {output.last_hidden_state.shape}")  # (B, num_tokens, hidden_dim)
        video_feature = output.last_hidden_state.mean(dim=1)  # (B, hidden_dim) - frame 평균

        return video_feature


# 비디오-포즈 정렬을 위한 Self-Supervised Learning 모델
class Videomodel(nn.Module):
    def __init__(self):
        super(Videomodel, self).__init__()
        self.video_encoder = VideoEncoder()
        
    def forward(self, video_frames):
        video_feat = self.video_encoder(video_frames)

        return video_feat

def pad_or_truncate(tensor, target_frames=60):
    """
    Args:
        tensor: torch.Tensor (C, T, H, W)
        target_frames: int
    """
    C, T, H, W = tensor.shape
    if T == target_frames:
        return tensor
    elif T < target_frames:
        pad = torch.zeros((C, target_frames - T, H, W), dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad], dim=1)
    else:
        return tensor[:, :target_frames, :, :]

class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir):
        self.tensor_files = sorted([f for f in os.listdir(tensor_dir) if f.endswith('.pt')])
        self.tensor_dir = tensor_dir

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.tensor_dir, self.tensor_files[idx]),weights_only=False)
        return data['video'], torch.tensor(data['label'], dtype=torch.long)

def visualize_embeddings(video_feats, labels, epoch, phase):
    """
    Args:
        video_feats: (N, D) Tensor
        labels: (N,) Tensor
        epoch: int
        phase: str ("Train" or "Validation")
    """
    # UMAP으로 2D 임베딩
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(video_feats.numpy())

    # 플롯 그리기
    plt.figure(figsize=(4, 3))
    scatter = sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels.numpy(),
        palette='tab10',
        s=60,
        alpha=0.7,
        legend="full"
    )
    plt.title(f"UMAP Visualization - {phase} Epoch {epoch + 1}")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def supervised_contrastive_loss(video_feat, labels, temperature=0.07):
    batch_size = video_feat.shape[0]

    # 특징 정규화 (L2 normalization)
    video_feat = F.normalize(video_feat, dim=-1)

    # 유사도 행렬 (N x N)
    similarity_matrix = torch.matmul(video_feat, video_feat.T) / temperature

    # 자기 자신을 제외한 마스크 생성
    self_mask = torch.eye(batch_size, device=video_feat.device).bool()

    # 라벨 비교를 통해 positive/negative 샘플 마스크 생성
    labels = labels.contiguous().view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(video_feat.device)

    # 자기 자신은 positive에서 제외
    positive_mask = positive_mask & (~self_mask)

    # positive 샘플에 대한 loss 계산
    exp_similarity = torch.exp(similarity_matrix) * (~self_mask)
    positive_exp_similarity = exp_similarity * positive_mask

    # loss 계산 (모든 positive pair에 대해 평균)
    numerator = positive_exp_similarity.sum(dim=1)
    denominator = exp_similarity.sum(dim=1)

    # numeric stability를 위해 작은 epsilon 추가
    epsilon = 1e-8
    loss = -torch.log((numerator + epsilon) / (denominator + epsilon))

    # 평균 loss 반환
    loss = loss.mean()

    return loss

def visualize_video_as_gif(dataset, sample_idx=0, save_path="video.gif", fps=5):
    video, label = dataset[sample_idx]  # video shape: [C, T, H, W]
    
    frames = []
    video = video.permute((1,0,2,3))
    for t in range(video.shape[0]):
        frame = video[t]  # shape: [C, H, W]
        # Convert torch tensor to PIL Image
        pil_img = transforms.ToPILImage()(frame.cpu())
        frames.append(pil_img)

    # Save frames as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )
    print(f"GIF saved at {save_path}")

    # Optional: Show GIF inline (if you're in a notebook)
    from IPython.display import Image as IPyImage, display
    display(IPyImage(filename=save_path))



# 첫 번째 비디오 샘플을 GIF로 시각화

# 학습 루프
if __name__ == "__main__":
    tensor_dir = "./processed_tensor"
    dataset = PreprocessedDataset(tensor_dir)
    visualize_video_as_gif(dataset, sample_idx=0, save_path="sample_video.gif", fps=5)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    device = get_device_info()

    ssl_model = Videomodel().to(device)
    optimizer = optim.Adam([
    {'params': ssl_model.video_encoder.parameters(), 'lr': 1e-4},  # 비디오 인코더는 낮은 학습률
])
    num_epochs = 50
    loss_function = supervised_contrastive_loss


    for epoch in range(num_epochs):
        ssl_model.train()
        total_loss = 0
        all_video_feats, all_labels = [], []

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()
            video_feat = ssl_model(videos)
            loss = loss_function(video_feat, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            all_video_feats.append(video_feat.cpu().detach())
            all_labels.append(labels.cpu().detach())

        print(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")
        visualize_embeddings(torch.cat(all_video_feats, dim=0), torch.cat(all_labels, dim=0), epoch, "Train")

        ssl_model.eval()
        val_loss = 0
        all_video_feats, all_labels = [], []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                video_feat = ssl_model(videos)
                loss = loss_function(video_feat, labels)
                val_loss += loss.item()
                all_video_feats.append(video_feat.cpu().detach())
                all_labels.append(labels.cpu().detach())

        print(f"Valid Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss / len(val_loader)}")
        visualize_embeddings(torch.cat(all_video_feats, dim=0), torch.cat(all_labels, dim=0), epoch, "Validation")
       

    torch.save(ssl_model.state_dict(), "ssl_stage1_model.pth")

# %%
