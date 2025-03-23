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
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# 변경: 사전 학습된 TimeSformer 사용
class VideoEncoder(nn.Module):
    def __init__(self, num_frames=16, image_size=224, patch_size=32, hidden_dim=768, pretrained=True):
        super(VideoEncoder, self).__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        if pretrained:
            try:
                # 사전 학습된 모델 로드 시도
                print("Loading pretrained TimeSformer model...")
                self.model = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k400")
                print("Pretrained model loaded successfully!")
            except:
                print("Failed to load pretrained model, using custom configuration instead.")
                self._init_custom_model()
        else:
            self._init_custom_model()
            
    def _init_custom_model(self):
        # 사전 학습된 모델을 사용하지 않을 경우 커스텀 구성 사용
        config = TimesformerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,  # 더 작은 패치 사이즈로 조정
            num_frames=self.num_frames,
            num_channels=3,
            num_attention_heads=12,  # 헤드 수 증가
            hidden_size=self.hidden_dim,
            num_hidden_layers=12,  # 레이어 수 증가
            intermediate_size=3072,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            num_classes=self.hidden_dim
        )
        self.model = TimesformerModel(config)

    def forward(self, video_frames):
        B, C, T, H, W = video_frames.shape
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        output = self.model(video_frames)
        video_feature = output.last_hidden_state[:, 0, :]  # [CLS] 토큰 사용 (첫 번째 토큰)
        
        return video_feature

# 비디오-포즈 정렬을 위한 Self-Supervised Learning 모델
class Videomodel(nn.Module):
    def __init__(self, projection_dim=256):
        super(Videomodel, self).__init__()
        self.video_encoder = VideoEncoder(pretrained=True)  # 사전 학습된 인코더 사용
        
        # 특징 투영 레이어 추가 (projection head)
        self.projection = nn.Sequential(
            nn.Linear(768, 512),  # TimeSformer의 hidden_dim이 일반적으로 768
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )
        
    def forward(self, video_frames):
        video_feat = self.video_encoder(video_frames)
        video_feat = self.projection(video_feat)  # 투영 레이어 적용
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
        data = torch.load(os.path.join(self.tensor_dir, self.tensor_files[idx]), weights_only=False)
        return data['video'], torch.tensor(data['label'], dtype=torch.long)

def visualize_embeddings(video_feats, labels, epoch, phase, save_path=None):
    """
    Args:
        video_feats: (N, D) Tensor
        labels: (N,) Tensor
        epoch: int
        phase: str ("Train" or "Validation")
        save_path: str or None
    """
    # UMAP으로 2D 임베딩
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(video_feats.numpy())

    # 플롯 그리기
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        hue=labels.numpy(),
        palette='tab10',
        s=100,
        alpha=0.7,
        legend="full"
    )
    plt.title(f"UMAP Visualization - {phase} Epoch {epoch + 1}")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/{phase}_epoch_{epoch+1}.png")
    
    plt.close()

# 수정된 대조 손실 함수
def improved_contrastive_loss(features, labels, temperature=0.07):
    """
    개선된 대조 손실 함수
    
    Args:
        features: 정규화된 특징 벡터 (B, D)
        labels: 레이블 (B,)
        temperature: 유사도 스케일링 파라미터
    """
    batch_size = features.shape[0]
    
    # 특징 정규화 (L2 normalization)
    features = F.normalize(features, dim=1)
    
    # 유사도 행렬 계산 (코사인 유사도)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Mask 생성
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    # 대각선(자기 자신) 제외
    identity_mask = torch.eye(batch_size, device=features.device)
    mask = mask - identity_mask
    
    # exp(sim) 계산을 위한 마스크
    exp_mask = (1 - identity_mask)
    
    # 유사도 행렬에서 로그-소프트맥스 계산
    exp_sim = torch.exp(similarity_matrix) * exp_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    
    # 각 샘플당 포지티브 샘플 수 계산
    num_positives = mask.sum(dim=1)
    
    # Positive 샘플이 있는 경우에만 대조 손실 계산
    loss = 0
    for i in range(batch_size):
        if num_positives[i] > 0:
            loss -= (log_prob[i] * mask[i]).sum() / num_positives[i]
    
    # 배치의 평균 손실 반환
    return loss / batch_size

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
    try:
        from IPython.display import Image as IPyImage, display
        display(IPyImage(filename=save_path))
    except ImportError:
        pass

# 학습 루프
if __name__ == "__main__":
    # 결과 저장 디렉토리 생성
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    tensor_dir = "./processed_tensor"
    dataset = PreprocessedDataset(tensor_dir)
    
    # 첫 번째 비디오 샘플 시각화
    visualize_video_as_gif(dataset, sample_idx=0, save_path="results/sample_video.gif", fps=5)
    
    # 데이터셋 분할 및 로더 설정
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 배치 크기 증가 (메모리 허용 범위 내에서)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
    
    device = get_device_info()
    
    # 모델 초기화
    ssl_model = Videomodel(projection_dim=128).to(device)
    
    # 옵티마이저 설정 - 다른 학습률 적용
    optimizer = optim.AdamW([
        {'params': ssl_model.video_encoder.parameters(), 'lr': 5e-5},  # 인코더는 더 낮은 학습률
        {'params': ssl_model.projection.parameters(), 'lr': 1e-3}      # 투영 레이어는 더 높은 학습률
    ], weight_decay=1e-4)
    
    # 코사인 스케줄러 추가
    num_epochs = 50
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 손실 함수
    loss_function = improved_contrastive_loss
    
    # 훈련 이력 추적
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 훈련 단계
        ssl_model.train()
        total_loss = 0
        all_video_feats, all_labels = [], []
        
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            video_feat = ssl_model(videos)
            loss = loss_function(video_feat, labels)
            loss.backward()
            
            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(ssl_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # 메모리 효율성을 위해 일부 샘플만 시각화에 사용
            if len(all_video_feats) < 500:  # 최대 500개 샘플만 시각화
                all_video_feats.append(video_feat.cpu().detach())
                all_labels.append(labels.cpu().detach())
        
        # 스케줄러 업데이트
        scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Train Epoch {epoch + 1}/{num_epochs}, Loss: {avg_train_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 훈련 세트 임베딩 시각화
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # 5 에포크마다 또는 마지막 에포크에 시각화
            visualize_embeddings(
                torch.cat(all_video_feats, dim=0), 
                torch.cat(all_labels, dim=0), 
                epoch, 
                "Train",
                save_path="results"
            )
        
        # 검증 단계
        ssl_model.eval()
        val_loss = 0
        all_video_feats, all_labels = [], []
        
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                video_feat = ssl_model(videos)
                loss = loss_function(video_feat, labels)
                val_loss += loss.item()
                
                # 메모리 효율성을 위해 일부 샘플만 시각화에 사용
                if len(all_video_feats) < 500:  # 최대 500개 샘플만 시각화
                    all_video_feats.append(video_feat.cpu().detach())
                    all_labels.append(labels.cpu().detach())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Valid Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss:.6f}")
        
        # 검증 세트 임베딩 시각화
        if epoch % 5 == 0 or epoch == num_epochs - 1:  # 5 에포크마다 또는 마지막 에포크에 시각화
            visualize_embeddings(
                torch.cat(all_video_feats, dim=0), 
                torch.cat(all_labels, dim=0), 
                epoch, 
                "Validation",
                save_path="results"
            )
        
        # 모델 저장 (검증 손실이 개선될 때만)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': ssl_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, "checkpoints/best_ssl_model.pth")
            print(f"Model saved at epoch {epoch + 1} with validation loss: {avg_val_loss:.6f}")
        
        # 주기적인 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ssl_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, f"checkpoints/ssl_model_epoch_{epoch+1}.pth")
    
    # 최종 모델 저장
    torch.save({
        'model_state_dict': ssl_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, "checkpoints/final_ssl_model.pth")
    
    # 손실 곡선 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_curve.png")
    plt.close()
    
    print("Training completed successfully!")
