import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os

def visualize_embeddings(video_feats, labels, epoch, phase, save_path, umap_config):
    label_map = {
        0: 'COGNITIVE DISORDER',
        1: 'STROKE',
        2: 'PARKINSONISM',
        3: 'HC',
        4: 'OTHERS'
    }
    label_order = list(label_map.values())
    labels_str = [label_map[int(i)] for i in labels.cpu().numpy()]

    reducer = umap.UMAP(n_neighbors=umap_config['n_neighbors'], min_dist=umap_config['min_dist'],
                        metric=umap_config['metric'], random_state=42)
    embedding_2d = reducer.fit_transform(video_feats.cpu().numpy())

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_str,
                    hue_order=label_order, palette='tab10', s=100, alpha=0.7)
    plt.title(f"UMAP - {phase} Epoch {epoch + 1}")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{phase}_epoch_{epoch + 1}.png"))
    plt.close()

# ----------------- Visualization Helpers -----------------
def save_and_log_loss_curve(train_losses, val_losses, epoch, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve (Live, Epoch {epoch})")
    plt.legend()
    path = os.path.join(save_dir, f"loss_curve.png")
    plt.savefig(path)
    plt.close()
    