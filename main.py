#%%
# train.py
import argparse
import os
import json
from pathlib import Path
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
# import umap.umap_ as umap
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Subset
import time
from collections import defaultdict

from sklearn.metrics import f1_score

from VideoDataset import VideoDataset
from VideoModel import VideoModel, CLIPStyleContrastiveModel, evaluate_linear_probe
from VideoLoss import improved_contrastive_loss, hierarchical_contrastive_loss, group_contrastive_loss, improved_multilabel_contrastive_loss
from VideoVisualization import visualize_embeddings, save_and_log_loss_curve, visualize_embeddings_joint, visualize_embeddings_joint_by_ptid, visualize_embeddings_multilabel_joint, visualize_embeddings_multilabel_joint_per_key

# ----------------- Config Loader -----------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
def train(config):
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    experiment_dir = os.path.join("experiments", config['experiment_name'])
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    config['save_dir']['results'] = os.path.join(experiment_dir, "results")
    config['save_dir']['checkpoints'] = os.path.join(experiment_dir, "checkpoints")

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    from torch.utils.data import Subset

    # 1. Dataset 로딩
    dataset = VideoDataset(tensor_dir=config['tensor_dir'], label_keys=config['label_type'])
    label_distributions = defaultdict(lambda: defaultdict(int))

    for _, label_tensor, _, _, _, _ in dataset:
        for i, key in enumerate(dataset.label_keys):
            label_val = label_tensor[i]
            
            # Multi-label인지 (vector인지) 확인
            if label_val.ndim == 0:  # Scalar (multi-class)
                label_distributions[key][label_val.item()] += 1
            else:  # Multi-label vector (e.g., [0,1,0,1])
                for class_idx, val in enumerate(label_val):
                    if val > 0:
                        label_distributions[key][class_idx] += 1

    # 결과 출력
    for key in dataset.label_keys:
        print(f"\nLabel distribution for '{key}':")
        for class_idx, count in sorted(label_distributions[key].items()):
            print(f"  Class {class_idx}: {count} samples")

    # 3. Split
    # 1) random
    total_indices = list(range(len(dataset)))
    np.random.seed(42)  # 재현성
    np.random.shuffle(total_indices)
    train_size = int(0.8 * len(dataset))
    train_indices = total_indices[:train_size]
    val_indices = total_indices[train_size:]
    # 2) setting
    total_indices = list(range(len(dataset)))
    print(len(dataset))
    train_indices = [i for i in total_indices if dataset[i][3]]
    val_indices   = [i for i in total_indices if not dataset[i][3]]

    # 4. Subset으로 dataset 나누기 (순서 고정)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # 5. Dataloader (shuffle=False, 이미 섞었음)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    model = VideoModel(config['model'])    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # print(torch.cuda.device_count())
    device_ids = [0,1]  # 원하는 GPU 인덱스
    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(f"cuda:{device_ids[0]}")
    
    optimizer = optim.AdamW([
    {'params': model.module.video_encoder.parameters(), 'lr': config['optimizer']['encoder_lr']},
    {'params': model.module.projection.parameters(), 'lr': config['optimizer']['proj_lr']}
    ], weight_decay=config['optimizer']['weight_decay'])

    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=config['scheduler']['eta_min'])
    
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    os.makedirs(config['save_dir']['results'], exist_ok=True)
    os.makedirs(config['save_dir']['checkpoints'], exist_ok=True)

    # wandb.init(project="gait-contrastive", name=config['experiment_name'], config=config)
    loss_fn = improved_contrastive_loss
    if len(config['label_type']) > 1:
        loss_fn = improved_multilabel_contrastive_loss
    else:
        loss_fn = improved_contrastive_loss
    # loss_fn = hierarchical_contrastive_loss
    # loss_fn = group_contrastive_loss
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
   
    for epoch in range(config['training']['epochs']):
        start_time = time.time()
         # 학습
        model.train()
        total_loss = 0
        all_feats, tr_encoder_feats, all_labels, all_anonyids, all_dxs, all_gabnormals = [], [], [], [], [], []

        for videos, labels, anonyids, _, dx, gabnormals in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            encoder_feats, feats = model(videos)
            loss = loss_fn(feats, labels, config['training']['temperature'])

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['max_grad_norm'])
            optimizer.step()
            total_loss += loss.item()
            
            all_feats.append(feats.cpu().detach())
            all_labels.append(labels.cpu().detach())
            tr_encoder_feats.append(encoder_feats.cpu().detach())
            all_anonyids.extend(anonyids)
            all_dxs.extend(dx)
            all_gabnormals.extend(gabnormals)

        scheduler.step()
        train_feats = torch.cat(all_feats, dim=0)
        train_labels_tensor = torch.cat(all_labels, dim=0)
        tr_encoder_feats = torch.cat(tr_encoder_feats, dim=0)

        norm_feats = F.normalize(train_feats, dim=1)
        sim_matrix = torch.matmul(norm_feats, norm_feats.T)
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool)
        sim_matrix.masked_fill_(mask, -1.0)

        is_multilabel = train_labels_tensor.ndim == 2 and train_labels_tensor.size(1) > 1

        # 🔹 Top-1 (가장 유사한 샘플 기준)
        nearest_indices = sim_matrix.argmax(dim=1)

        if not is_multilabel:
            tr_predicted_labels_top1 = train_labels_tensor[nearest_indices]
            tr_acc_top1 = (tr_predicted_labels_top1 == train_labels_tensor).float().mean().item()
            tr_f1_top1 = f1_score(train_labels_tensor.cpu(), tr_predicted_labels_top1.cpu(), average='macro')

            # 🔹 Top-5 (voting 방식)
            topk = 5
            topk_indices = torch.topk(sim_matrix, k=topk, dim=1).indices  # (N, K)
            topk_labels = train_labels_tensor[topk_indices]               # (N, K)
            tr_predicted_labels_top5 = torch.mode(topk_labels, dim=1).values
            tr_acc_top5 = (tr_predicted_labels_top5 == train_labels_tensor).float().mean().item()
            tr_f1_top5 = f1_score(train_labels_tensor.cpu(), tr_predicted_labels_top5.cpu(), average='macro')
            
        else:
        # ----- Multi-label -----
            label_norm = F.normalize(train_labels_tensor.float(), p=2, dim=1)
            tr_predicted_labels_top1 = train_labels_tensor[nearest_indices]
            sim_score = (label_norm * label_norm[nearest_indices]).sum(dim=1)
            tr_acc_top1 = (sim_score > 0.8).float().mean().item()
            tr_f1_top1 = f1_score(train_labels_tensor.cpu().int(),
                                tr_predicted_labels_top1.cpu().int(),
                                average='macro')

            # Top-5 (soft voting)
            topk = 5
            topk_indices = torch.topk(sim_matrix, k=topk, dim=1).indices
            topk_labels = train_labels_tensor[topk_indices]               # (N, K, C)
            tr_predicted_labels_top5 = (topk_labels.float().mean(dim=1) > 0.5).int()  # (N, C)

            acc_samplewise = (tr_predicted_labels_top5 == train_labels_tensor).float().mean(dim=1)
            tr_acc_top5 = acc_samplewise.mean().item()
            tr_f1_top5 = f1_score(train_labels_tensor.cpu().int(),
                                tr_predicted_labels_top5.cpu().int(),
                                average='macro')

        # 🔹 평균 Loss 저장
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)  

        # wandb.log({"train_loss": avg_train_loss}, step=epoch)

        # 검증 (validation)
        model.eval()
        total_val_loss = 0
        val_feats, val_encoder_feats, val_labels, val_anonyids, val_dxs, val_gabnormals = [], [], [], [], [], []
        with torch.no_grad():
            for videos, labels, anonyids, _ , dxs, gabnormals in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                encoder_feats, feats = model(videos)
                val_loss = loss_fn(feats, labels, config['training']['temperature'])
                total_val_loss += val_loss.item()
                
                val_feats.append(feats.cpu())
                val_labels.append(labels.cpu())
                val_encoder_feats.append(encoder_feats.cpu().detach())
                val_anonyids.extend(anonyids)
                val_dxs.extend(dxs)
                val_gabnormals.extend(gabnormals)

        valid_feats = torch.cat(val_feats, dim=0)
        valid_labels_tensor = torch.cat(val_labels, dim=0)
        val_encoder_feats = torch.cat(val_encoder_feats, dim=0)

        # norm_feats = F.normalize(valid_feats, dim=1)
        # sim_matrix = torch.matmul(norm_feats, norm_feats.T)
        valid_norm_feats = F.normalize(valid_feats, dim=1)
        train_norm_feats = F.normalize(train_feats.to(valid_feats.device), dim=1)
        
        sim_matrix = torch.matmul(valid_norm_feats, train_norm_feats.T)

        # mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        # sim_matrix.masked_fill_(mask, -1.0)

        is_multilabel = valid_labels_tensor.ndim == 2 and valid_labels_tensor.size(1) > 1

        # 🔹 Top-1 (가장 유사한 샘플 기준)
        nearest_indices = sim_matrix.argmax(dim=1)

        if not is_multilabel:
            val_predicted_labels_top1 = train_labels_tensor[nearest_indices]
            val_acc_top1 = (val_predicted_labels_top1 == valid_labels_tensor).float().mean().item()
            val_f1_top1 = f1_score(valid_labels_tensor.cpu(), val_predicted_labels_top1.cpu(), average='macro')

            # 🔹 Top-5 (voting 방식)
            topk = 5
            topk_indices = torch.topk(sim_matrix, k=topk, dim=1).indices  # (N, K)
            # topk_labels = valid_labels_tensor[topk_indices]               # (N, K)
            topk_labels = train_labels_tensor[topk_indices]
            val_predicted_labels_top5 = torch.mode(topk_labels, dim=1).values
            val_acc_top5 = (val_predicted_labels_top5 == valid_labels_tensor).float().mean().item()
            val_f1_top5 = f1_score(valid_labels_tensor.cpu(), val_predicted_labels_top5.cpu(), average='macro')

        else:
            # ----- Multi-label -----
            valid_label_norm = F.normalize(valid_labels_tensor.float(), p=2, dim=1)
            train_label_norm = F.normalize(train_labels_tensor.float(), p=2, dim=1)

            # Top-1
            val_predicted_labels_top1 = train_labels_tensor[nearest_indices]  # (N, C)
            nearest_train_label_norm = train_label_norm[nearest_indices]      # (N, C)

            # 코사인 유사도 기반 정확도 (sample-wise label vector similarity)
            sim_score = (valid_label_norm * nearest_train_label_norm).sum(dim=1)
            val_acc_top1 = (sim_score > 0.8).float().mean().item()

            val_f1_top1 = f1_score(
                valid_labels_tensor.cpu().int(),
                val_predicted_labels_top1.cpu().int(),
                average='macro'
            )

            # Top-5 (soft voting)
            topk = 5
            topk_indices = torch.topk(sim_matrix, k=topk, dim=1).indices
            topk_labels = train_labels_tensor[topk_indices]  # (N, 5, C)

            # 평균 후 thresholding
            val_predicted_labels_top5 = (topk_labels.float().mean(dim=1) > 0.5).int()

            # sample-wise 정확도
            acc_samplewise = (val_predicted_labels_top5 == valid_labels_tensor).float().mean(dim=1)
            val_acc_top5 = acc_samplewise.mean().item()

            val_f1_top5 = f1_score(
                valid_labels_tensor.cpu().int(),
                val_predicted_labels_top5.cpu().int(),
                average='macro'
            )

        # 🔹 평균 Loss 저장
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 🔹 로그 출력
        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Top-1 Acc: {tr_acc_top1:.4f}, F1: {tr_f1_top1:.4f} | "
              f"Top-5 Acc: {tr_acc_top5:.4f}, F1: {tr_f1_top5:.4f} | "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"Top-1 Acc: {val_acc_top1:.4f}, F1: {val_f1_top1:.4f} | "
              f"Top-5 Acc: {val_acc_top5:.4f}, F1: {val_f1_top5:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}, "
              f"Time: {time.time()-start_time:.2f}")


        if epoch % config['visualization']['visualize_every_n_epochs'] == 0:
            if len(config["label_type"]) == 1:
                visualize_embeddings(tr_encoder_feats, train_labels_tensor, tr_predicted_labels_top5, all_anonyids, epoch, "Train",
                                    config['save_dir']['results'], config['umap'], dataset.label_keys[0])
                visualize_embeddings(val_encoder_feats, valid_labels_tensor, val_predicted_labels_top5, val_anonyids, epoch, "Validation",
                                    config['save_dir']['results'], config['umap'], dataset.label_keys[0])
                visualize_embeddings_joint(
                    video_feats_list=[tr_encoder_feats, val_encoder_feats],
                    labels_list=[train_labels_tensor, valid_labels_tensor],
                    preds_list=[tr_predicted_labels_top5, val_predicted_labels_top5],
                    anonyids_list=[all_anonyids, val_anonyids],
                    phases=['Train', 'Valid'],
                    epoch=epoch,
                    save_path=config['save_dir']['results'],
                    umap_params=config['umap'],
                    label_key=dataset.label_keys[0]
                )
                visualize_embeddings_joint_by_ptid(
                    video_feats_list=[tr_encoder_feats, val_encoder_feats],
                    labels_list=[train_labels_tensor, valid_labels_tensor],
                    preds_list=[tr_predicted_labels_top5, val_predicted_labels_top5],
                    anonyids_list=[all_anonyids, val_anonyids],
                    phases=['Train', 'Valid'],
                    epoch=epoch,
                    save_path=config['save_dir']['results'],
                    umap_params=config['umap'],
                    label_key=dataset.label_keys[0]
                )
                save_and_log_loss_curve(train_losses, val_losses, epoch, config['save_dir']['results'])
        
                # 파일이 존재하는지 체크 후 log
                # train_umap_path = os.path.join(config['save_dir']['results'], f"Train_epoch_{epoch + 1}.png")
                # val_umap_path = os.path.join(config['save_dir']['results'], f"Validation_epoch_{epoch + 1}.png")
                # wandb.log({"Train_UMAP": wandb.Image(train_umap_path),
                #            "Validation_UMAP": wandb.Image(val_umap_path),
                #            "epoch": epoch})
                # if epoch == 0:
                #     umap_table = wandb.Table(columns=["Epoch", "Phase", "Image"])

                # 이미지 경로
                # train_img_path = os.path.join(config['save_dir']['results'], f"Train_epoch_{epoch + 1}.png")
                # val_img_path = os.path.join(config['save_dir']['results'], f"Validation_epoch_{epoch + 1}.png")

                # wandb.log({f"UMAP/Train": wandb.Image(train_img_path)}, step=epoch)
                # wandb.log({f"UMAP/Valid": wandb.Image(val_img_path)}, step=epoch)
                print("📊 Linear Probe Evaluation")
                probe_acc, probe_f1 = evaluate_linear_probe(
                    train_feats=tr_encoder_feats,
                    train_labels=train_labels_tensor,
                    val_feats=val_encoder_feats,
                    val_labels=valid_labels_tensor
                )

                print(f"▶️ Linear Probe Acc: {probe_acc:.4f}, F1: {probe_f1:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    model_path = os.path.join(config['save_dir']['checkpoints'], "best_model.pth")
                    torch.save(model.state_dict(), model_path)
                    # wandb.save(model_path)
            else:
                visualize_embeddings_multilabel_joint(
                    video_feats_list=[tr_encoder_feats, val_encoder_feats],
                    labels_list=[train_labels_tensor, valid_labels_tensor],
                    preds_list=[tr_predicted_labels_top5, val_predicted_labels_top5],
                    anonyids_list=[all_anonyids, val_anonyids],
                    splits_list=[['Train'] * len(all_anonyids), ['Valid'] * len(val_anonyids)],
                    epoch=epoch,
                    save_path=config['save_dir']['results'],
                    umap_params=config['umap'],
                    label_keys=dataset.label_keys
                )

                import scipy.io as sio
                # 저장할 데이터 구성
                mat_data = {
                    'train_feats': tr_encoder_feats.cpu().numpy(),
                    'valid_feats': val_encoder_feats.cpu().numpy(),
                    'train_labels': train_labels_tensor.cpu().numpy(),
                    'valid_labels': valid_labels_tensor.cpu().numpy(),
                    'train_preds': tr_predicted_labels_top5.cpu().numpy(),
                    'valid_preds': val_predicted_labels_top5.cpu().numpy(),
                    'train_anonyids': np.array(all_anonyids),
                    'valid_anonyids': np.array(val_anonyids),
                    'train_splits': np.array(['Train'] * len(all_anonyids)),
                    'valid_splits': np.array(['Valid'] * len(val_anonyids)),
                    'label_keys': np.array(dataset.label_keys)
                }

                # 저장
                mat_file_path = os.path.join(config['save_dir']['results'], f"epoch_{epoch}_visualization_data.mat")
                sio.savemat(mat_file_path, mat_data)

                # dx, gait_abnormality 가시화
                visualize_embeddings_multilabel_joint_per_key(
                    video_feats_list=[tr_encoder_feats, val_encoder_feats],
                    labels_list=[train_labels_tensor, valid_labels_tensor],
                    preds_list=[tr_predicted_labels_top5, val_predicted_labels_top5],
                    anonyids_list=[all_anonyids, val_anonyids],
                    splits_list=[['Train'] * len(all_anonyids), ['Valid'] * len(val_anonyids)],
                    epoch=epoch,
                    save_path=config['save_dir']['results'],
                    umap_params=config['umap'],
                    label_key="DX_CATEGORY",
                    dx_list=[all_dxs, val_dxs],
                )

                visualize_embeddings_multilabel_joint_per_key(
                    video_feats_list=[tr_encoder_feats, val_encoder_feats],
                    labels_list=[train_labels_tensor, valid_labels_tensor],
                    preds_list=[tr_predicted_labels_top5, val_predicted_labels_top5],
                    anonyids_list=[all_anonyids, val_anonyids],
                    splits_list=[['Train'] * len(all_anonyids), ['Valid'] * len(val_anonyids)],
                    epoch=epoch,
                    save_path=config['save_dir']['results'],
                    umap_params=config['umap'],
                    label_key="GAIT_DISTURBANCE",
                    dx_list=[all_gabnormals, val_gabnormals],
                )

                if (epoch + 1) % 20 == 0:
                    model_path = os.path.join(config['save_dir']['checkpoints'], f"best_model_{epoch}.pth")
                    torch.save(model.state_dict(), model_path)

                    # wandb.save(model_path)
                
    # wandb.finish()
    print("✅ Training completed!")


# ----------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if not config.get("experiment_name"):
        now = datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        config["experiment_name"] = now
    
    train(config)