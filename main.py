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
import umap.umap_ as umap
from datetime import datetime
import torchvision.transforms as transforms
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from VideoDataset import VideoDataset
from VideoModel import VideoModel
from VideoLoss import improved_contrastive_loss
from VideoVisualization import visualize_embeddings, save_and_log_loss_curve

# ----------------- Config Loader -----------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
def train(config):
    experiment_dir = os.path.join("experiments", config['experiment_name'])
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    config['save_dir']['results'] = os.path.join(experiment_dir, "results")
    config['save_dir']['checkpoints'] = os.path.join(experiment_dir, "checkpoints")

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    dataset = VideoDataset(config['tensor_dir'])
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model = VideoModel(config['model']).to(device)

    optimizer = optim.AdamW([
        {'params': model.video_encoder.parameters(), 'lr': config['optimizer']['encoder_lr']},
        {'params': model.projection.parameters(), 'lr': config['optimizer']['proj_lr']}
    ], weight_decay=config['optimizer']['weight_decay'])

    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'], eta_min=config['scheduler']['eta_min'])

    os.makedirs(config['save_dir']['results'], exist_ok=True)
    os.makedirs(config['save_dir']['checkpoints'], exist_ok=True)

    wandb.init(project="gait-contrastive", name=config['experiment_name'], config=config)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        all_feats, all_labels = [], []

        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)
            optimizer.zero_grad()
            feats = model(videos)
            loss = improved_contrastive_loss(feats, labels, config['training']['temperature'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['training']['max_grad_norm'])
            optimizer.step()
            total_loss += loss.item()
            if len(all_feats) < 500:
                all_feats.append(feats.cpu().detach())
                all_labels.append(labels.cpu().detach())

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        wandb.log({"train_loss": avg_train_loss}, step=epoch)

        model.eval()
        total_val_loss = 0
        val_feats, val_labels = [], []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device)
                feats = model(videos)
                val_loss = improved_contrastive_loss(feats, labels, config['training']['temperature'])
                total_val_loss += val_loss.item()
                if len(val_feats) < 500:
                    val_feats.append(feats.cpu())
                    val_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        wandb.log({"val_loss": avg_val_loss}, step= epoch)
        

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if epoch % config['visualization']['visualize_every_n_epochs'] == 0:
            visualize_embeddings(torch.cat(all_feats), torch.cat(all_labels), epoch, "Train",
                                 config['save_dir']['results'], config['umap'])
            visualize_embeddings(torch.cat(val_feats), torch.cat(val_labels), epoch, "Validation",
                                 config['save_dir']['results'], config['umap'])
            save_and_log_loss_curve(train_losses, val_losses, epoch, config['save_dir']['results'])
            
            # 파일이 존재하는지 체크 후 log
            train_umap_path = os.path.join(config['save_dir']['results'], f"Train_epoch_{epoch + 1}.png")
            val_umap_path = os.path.join(config['save_dir']['results'], f"Validation_epoch_{epoch + 1}.png")
            # wandb.log({"Train_UMAP": wandb.Image(train_umap_path),
            #            "Validation_UMAP": wandb.Image(val_umap_path),
            #            "epoch": epoch})
            if epoch == 0:
                umap_table = wandb.Table(columns=["Epoch", "Phase", "Image"])

            # 이미지 경로
            train_img_path = os.path.join(config['save_dir']['results'], f"Train_epoch_{epoch + 1}.png")
            val_img_path = os.path.join(config['save_dir']['results'], f"Validation_epoch_{epoch + 1}.png")

            wandb.log({f"UMAP/Train": wandb.Image(train_img_path)}, step=epoch)
            wandb.log({f"UMAP/Valid": wandb.Image(val_img_path)}, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(config['save_dir']['checkpoints'], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)

    wandb.finish()
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