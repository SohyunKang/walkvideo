import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os

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

import umap
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_embeddings(video_feats, labels, preds, anonyids, epoch, phase, save_path, umap_params, label_key):
    """
    Args:
        video_feats: (N, D) torch.Tensor
        labels: (N,) torch.Tensor (GT)
        preds: (N,) torch.Tensor (Predictions)
        label_map: dict (e.g., {'HC': 0, ...}) with string keys
    """
    if label_key == "GAIT_DISTURBANCE":
        label_map = {'NO': 0, 'POSSIBLE': 1, 'YES': 2}
    elif label_key == "DX_CATEGORY":
        label_map = {'HC': 0, 'OTHERS': 1, 'COGNITIVE DISORDER': 2, 'STROKE': 3, 'PARKINSONISM':4}
    else:
        label_map = {'Normal': 0, 'Abnormal': 1}

    # 인버스 맵: 숫자 → 문자
    
    inverse_label_map = {v: k for k, v in label_map.items()}
    label_order = [inverse_label_map[i] for i in sorted(inverse_label_map.keys())]

    # 문자열 라벨 + 정/오답 구분
    label_strs = [inverse_label_map[int(i)] for i in labels.cpu().numpy()]
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    correctness = ['Correct' if p == l else 'Wrong' for p, l in zip(preds_np, labels_np)]

    # UMAP 임베딩
    reducer = umap.UMAP(**umap_params, random_state=42)
    embedding_2d = reducer.fit_transform(video_feats.cpu().numpy())
    anonyids = [t.item() for t in anonyids]

    # 시각화
    import pandas as pd
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'Label': label_strs,
        'Correctness': correctness,
        'AnonyID': anonyids,
    })
    # print(df)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='Label',
        style='Correctness',  
        hue_order=sorted(label_map.keys()),  
        style_order=['Correct', 'Wrong'],  
        palette='tab10', 
        markers={'Correct': 'o', 'Wrong': 'X'},
        s=100,
        alpha=0.8
    )
    count = 0
    for i in range(len(df)):
        if df.loc[i, 'Correctness'] == 'Wrong' and count < 10:
            plt.text(df.loc[i, 'x'], df.loc[i, 'y'], df.loc[i, 'AnonyID'], fontsize=8)
            count+=1
    plt.title(f"UMAP Embedding - {phase} Epoch {epoch + 1}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/{phase}_epoch_{epoch+1}.png")

    plt.show()
    plt.close()

def visualize_embeddings_joint(video_feats_list, labels_list, preds_list, anonyids_list, phases, epoch, save_path, umap_params, label_key):
    import pandas as pd
    import umap
    import seaborn as sns
    import matplotlib.pyplot as plt
    import torch

    # 라벨 맵 정의
    if label_key == "GAIT_DISTURBANCE":
        label_map = {'NO': 0, 'POSSIBLE': 1, 'YES': 2}
    elif label_key == "DX_CATEGORY":
        label_map = {'HC': 0, 'OTHERS': 1, 'COGNITIVE DISORDER': 2, 'STROKE': 3, 'PARKINSONISM': 4}
    else:
        label_map = {'Normal': 0, 'Abnormal': 1}
    
    inverse_label_map = {v: k for k, v in label_map.items()}
    label_order = [inverse_label_map[i] for i in sorted(inverse_label_map.keys())]
    def to_tensor_list(tensor_or_list):
        return [torch.tensor(x) if isinstance(x, list) else x for x in tensor_or_list]

    # 리스트 내부 확인 및 변환
    video_feats_list = to_tensor_list(video_feats_list)
    labels_list = to_tensor_list(labels_list)
    preds_list = to_tensor_list(preds_list)
    anonyids_list = to_tensor_list(anonyids_list)

    # 전체 통합
    all_feats = torch.cat(video_feats_list, dim=0).cpu().numpy()
    all_labels = torch.cat(labels_list, dim=0).cpu().numpy()
    all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
    all_anonyids = torch.cat(anonyids_list, dim=0).cpu().numpy()
    all_splits = sum([[phase] * len(v) for phase, v in zip(phases, video_feats_list)], [])

    label_strs = [inverse_label_map[int(i)] for i in all_labels]
    correctness = ['Correct' if p == l else 'Wrong' for p, l in zip(all_preds, all_labels)]

    # UMAP
    reducer = umap.UMAP(**umap_params, random_state=42)
    embedding_2d = reducer.fit_transform(all_feats)

    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'Label': label_strs,
        'Correctness': correctness,
        'AnonyID': all_anonyids,
        'Split': all_splits
    })

    # Label + Split 조합으로 새로운 hue 설정 (예: HC_Train, HC_Valid)
    df['LabelSplit'] = df['Label'] + "_" + df['Split']
    label_split_order = sorted(df['LabelSplit'].unique())
    custom_palette = {
        'Normal_Train': '#9ecae1',     # 연한 파랑 (light blue)
        'Normal_Valid': '#08519c',     # 진한 파랑 (dark blue)
        'Abnormal_Train': '#fdd0a2',   # 연한 주황 (light orange)
        'Abnormal_Valid': '#e6550d'    # 진한 주황 (dark orange)
    }
    # 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='LabelSplit',
        style='Correctness',
        markers={'Correct': 'o', 'Wrong': 'X'},
        hue_order=list(custom_palette.keys()),
        palette=custom_palette,
        style_order=['Correct', 'Wrong'],
        s=100,
        alpha=0.8,
    )
    # 오답 10개 표시
    wrongs = df[df['Correctness'] == 'Wrong'].head(10)
    for _, row in wrongs.iterrows():
        plt.text(row['x'], row['y'], str(row['AnonyID']), fontsize=8)

    plt.title(f"UMAP Embedding (Epoch {epoch+1})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/joint_epoch_{epoch+1}.png")
    plt.show()
    plt.close()
