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
    # print(inverse_label_map)
    label_order = [inverse_label_map[i] for i in sorted(inverse_label_map.keys())]
    # print(labels)
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

    if label_key == "GAIT_DISTURBANCE":
        custom_palette = {
            'NO_Train': '#9ecae1',
            'NO_Valid': '#08519c',
            'POSSIBLE_Train': '#a1d99b',
            'POSSIBLE_Valid': '#31a354',
            'YES_Train': '#fcbba1',
            'YES_Valid': '#cb181d',
        }
    elif label_key == "DX_CATEGORY":
        custom_palette = {
            'HC_Train': '#c6dbef',          # 연한 파랑
            'HC_Valid': '#08519c',          # 진한 파랑

            'OTHERS_Train': '#fdd0a2',      # 연한 주황
            'OTHERS_Valid': '#e6550d',      # 진한 주황

            'COGNITIVE DISORDER_Train': '#d4b9da',  # 연한 보라
            'COGNITIVE DISORDER_Valid': '#756bb1',  # 진한 보라

            'STROKE_Train': '#a1d99b',      # 연한 초록
            'STROKE_Valid': '#31a354',      # 진한 초록

            'PARKINSONISM_Train': '#fcae91',  # 연한 붉은색
            'PARKINSONISM_Valid': '#cb181d',  # 진한 붉은색
        }
        # HC, OTHERS, COGNITIVE DISORDER, STROKE, PARKINSONISM 조합 정의 필요
        pass
    else:
        custom_palette = {
            'Normal_Train': '#9ecae1',
            'Normal_Valid': '#08519c',
            'Abnormal_Train': '#fdd0a2',
            'Abnormal_Valid': '#e6550d'
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

from collections import defaultdict, Counter
import numpy as np
import torch
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_embeddings_joint_by_ptid(
    video_feats_list, labels_list, preds_list, anonyids_list, phases,
    epoch, save_path, umap_params, label_key
):
    # 라벨 맵 정의
    if label_key == "GAIT_DISTURBANCE":
        label_map = {'NO': 0, 'POSSIBLE': 1, 'YES': 2}
    elif label_key == "DX_CATEGORY":
        label_map = {'HC': 0, 'OTHERS': 1, 'COGNITIVE DISORDER': 2, 'STROKE': 3, 'PARKINSONISM': 4}
    else:
        label_map = {'Normal': 0, 'Abnormal': 1}

    inverse_label_map = {v: k for k, v in label_map.items()}
    
    # Tensor 변환
    def to_tensor_list(tensor_or_list):
        return [torch.tensor(x) if isinstance(x, list) else x for x in tensor_or_list]
    
    video_feats_list = to_tensor_list(video_feats_list)
    labels_list = to_tensor_list(labels_list)
    preds_list = to_tensor_list(preds_list)
    anonyids_list = to_tensor_list(anonyids_list)

    all_feats = torch.cat(video_feats_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0).cpu().numpy()
    all_preds = torch.cat(preds_list, dim=0).cpu().numpy()
    all_anonyids = torch.cat(anonyids_list, dim=0).cpu().numpy()
    all_splits = sum([[phase] * len(v) for phase, v in zip(phases, video_feats_list)], [])

    
    cnt = Counter(all_anonyids)
    print(f"Average samples per PTID: {np.mean(list(cnt.values()))}")

    # 그룹별로 평균 및 대표값 생성
    ptid_to_feats = defaultdict(list)
    ptid_to_labels = defaultdict(list)
    ptid_to_preds = defaultdict(list)
    ptid_to_splits = defaultdict(list)

    for feat, label, pred, ptid, split in zip(all_feats, all_labels, all_preds, all_anonyids, all_splits):
        label = int(label) if isinstance(label, np.ndarray) else label
        pred = int(pred) if isinstance(pred, np.ndarray) else pred
        ptid_to_feats[ptid].append(feat)
        ptid_to_labels[ptid].append(label)
        ptid_to_preds[ptid].append(pred)
        ptid_to_splits[ptid].append(split)

    avg_feats = []
    avg_labels = []
    avg_preds = []
    avg_anonyids = []
    avg_correctness = []
    avg_splits = []

    for ptid in ptid_to_feats.keys():
        feats = torch.stack(ptid_to_feats[ptid], dim=0)
        avg_feat = feats.mean(dim=0).numpy()
        label = Counter(ptid_to_labels[ptid]).most_common(1)[0][0]
        pred = Counter(ptid_to_preds[ptid]).most_common(1)[0][0]
        correctness = 'Correct' if label == pred else 'Wrong'
        split = Counter(ptid_to_splits[ptid]).most_common(1)[0][0]

        avg_feats.append(avg_feat)
        avg_labels.append(label)
        avg_preds.append(pred)
        avg_anonyids.append(ptid)
        avg_correctness.append(correctness)
        avg_splits.append(split)
    # UMAP
    reducer = umap.UMAP(**umap_params, random_state=42)
    embedding_2d = reducer.fit_transform(np.stack(avg_feats))

    label_strs = [inverse_label_map[int(i)] for i in avg_labels]

    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'Label': label_strs,
        'Correctness': avg_correctness,
        'AnonyID': avg_anonyids,
        'Split': avg_splits
    })

    df['LabelSplit'] = df['Label'] + "_" + df['Split']
    label_split_order = sorted(df['LabelSplit'].unique())

    # 컬러 팔레트 정의
    if label_key == "DX_CATEGORY":
        custom_palette = {
            'HC_Train': '#c6dbef', 'HC_Valid': '#08519c',
            'OTHERS_Train': '#fdd0a2', 'OTHERS_Valid': '#e6550d',
            'COGNITIVE DISORDER_Train': '#d4b9da', 'COGNITIVE DISORDER_Valid': '#756bb1',
            'STROKE_Train': '#a1d99b', 'STROKE_Valid': '#31a354',
            'PARKINSONISM_Train': '#fcae91', 'PARKINSONISM_Valid': '#cb181d',
        }
    elif label_key == "GAIT_DISTURBANCE":
        custom_palette = {
            'NO_Train': '#9ecae1', 'NO_Valid': '#08519c',
            'POSSIBLE_Train': '#a1d99b', 'POSSIBLE_Valid': '#31a354',
            'YES_Train': '#fcbba1', 'YES_Valid': '#cb181d',
        }
    else:
        custom_palette = {
            'Normal_Train': '#9ecae1', 'Normal_Valid': '#08519c',
            'Abnormal_Train': '#fdd0a2', 'Abnormal_Valid': '#e6550d',
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

    # 오답 10개만 텍스트 표시
    wrongs = df[df['Correctness'] == 'Wrong'].head(10)
    for _, row in wrongs.iterrows():
        plt.text(row['x'], row['y'], str(row['AnonyID']), fontsize=8)

    plt.title(f"UMAP Embedding by PTID (Epoch {epoch+1})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/joint_epoch_{epoch+1}_by_ptid.png")
    plt.show()
    plt.close()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import umap
import torch

def get_custom_palette(label_key):
    if label_key == "DX_CATEGORY":
        return {
            'DX_CATEGORY_HC_Train': '#c6dbef', 'DX_CATEGORY_HC_Valid': '#08519c',
            'DX_CATEGORY_OTHERS_Train': '#fdd0a2', 'DX_CATEGORY_OTHERS_Valid': '#e6550d',
            'DX_CATEGORY_COGNITIVE DISORDER_Train': '#d4b9da', 'DX_CATEGORY_COGNITIVE DISORDER_Valid': '#756bb1',
            'DX_CATEGORY_STROKE_Train': '#a1d99b', 'DX_CATEGORY_STROKE_Valid': '#31a354',
            'DX_CATEGORY_PARKINSONISM_Train': '#fcae91', 'DX_CATEGORY_PARKINSONISM_Valid': '#cb181d',
        }
    elif label_key == "GAIT_DISTURBANCE":
        return {
            'GAIT_DISTURBANCE_NO_Train': '#9ecae1', 'GAIT_DISTURBANCE_NO_Valid': '#08519c',
            'GAIT_DISTURBANCE_POSSIBLE_Train': '#a1d99b', 'GAIT_DISTURBANCE_POSSIBLE_Valid': '#31a354',
            'GAIT_DISTURBANCE_YES_Train': '#fcbba1', 'GAIT_DISTURBANCE_YES_Valid': '#cb181d',
        }
    else:
        return {
            f'{label_key}_Normal_Train': '#9ecae1', f'{label_key}_Normal_Valid': '#08519c',
            f'{label_key}_Abnormal_Train': '#fdd0a2', f'{label_key}_Abnormal_Valid': '#e6550d',
        }

def visualize_embeddings_multilabel(
    video_feats, labels, preds, anonyids, splits,
    epoch, phase, save_path, umap_params, label_keys
):
    """
    Args:
        video_feats: (N, D) torch.Tensor
        labels, preds: (N, C) torch.Tensor
        anonyids: (N,) torch.Tensor or list
        splits: list of strings ['Train', 'Valid', ...] of length N
        label_keys: list of str, names of labels in order of C
    """

    # 라벨 맵 정의
    label_maps, inverse_label_maps = [], []
    for key in label_keys:
        if key == "GAIT_DISTURBANCE":
            m = {'NO': 0, 'POSSIBLE': 1, 'YES': 2}
        elif key == "DX_CATEGORY":
            m = {'HC': 0, 'OTHERS': 1, 'COGNITIVE DISORDER': 2, 'STROKE': 3, 'PARKINSONISM': 4}
        else:
            m = {'Normal': 0, 'Abnormal': 1}
        label_maps.append(m)
        inverse_label_maps.append({v: k for k, v in m.items()})

    # UMAP
    reducer = umap.UMAP(**umap_params, random_state=42)
    embedding_2d = reducer.fit_transform(video_feats.cpu().numpy())
    anonyids = [a.item() if isinstance(a, torch.Tensor) else a for a in anonyids]

    fig, axes = plt.subplots(1, len(label_keys), figsize=(6 * len(label_keys), 5))
    if len(label_keys) == 1:
        axes = [axes]

    for i, (label_key, label_map, inverse_map) in enumerate(zip(label_keys, label_maps, inverse_label_maps)):
        label_col = labels[:, i].cpu().numpy()
        pred_col = preds[:, i].cpu().numpy()
        split_col = splits  # list of strings: 'Train' or 'Valid'

        # 문자열 라벨 (예: DX_CATEGORY_HC_Train)
        label_strs = [f"{label_key}_{inverse_map[int(l)]}_{s}" for l, s in zip(label_col, split_col)]
        correctness = ['Correct' if p == l else 'Wrong' for p, l in zip(pred_col, label_col)]

        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'Label': label_strs,
            'Correctness': correctness,
            'AnonyID': anonyids,
        })

        palette = get_custom_palette(label_key)

        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='Label',
            style='Correctness',
            ax=axes[i],
            hue_order=sorted(palette.keys()),
            style_order=['Correct', 'Wrong'],
            markers={'Correct': 'o', 'Wrong': 'X'},
            s=100,
            alpha=0.8,
            palette=palette
        )

        # 오답 라벨 텍스트 출력
        wrongs = df[df['Correctness'] == 'Wrong'].head(10)
        for _, row in wrongs.iterrows():
            axes[i].text(row['x'], row['y'], str(row['AnonyID']), fontsize=7)

        axes[i].set_title(f"{label_key} (Epoch {epoch + 1})")
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{phase}_epoch_{epoch+1}_multilabel.png")
    plt.show()
    plt.close()

def visualize_embeddings_multilabel_joint(
    video_feats_list, labels_list, preds_list, anonyids_list, splits_list,
    epoch, save_path, umap_params, label_keys
):
    """
    Joint UMAP visualization across Train and Valid for multi-label setup.
    Args:
        *_list: list of (train_tensor, valid_tensor)
        label_keys: list of str
    """

    # 라벨 맵 정의
    label_maps, inverse_label_maps = [], []
    for key in label_keys:
        if key == "DX_CATEGORY":
            m = {'HC': 0, 'OTHERS': 1, 'COGNITIVE DISORDER': 2, 'STROKE': 3, 'PARKINSONISM': 4}
        elif key == "GAIT_DISTURBANCE":
            m = {'NO': 0, 'POSSIBLE': 1, 'YES': 2}
        else:
            m = {'Normal': 0, 'Abnormal': 1}
        label_maps.append(m)
        inverse_label_maps.append({v: k for k, v in m.items()})

    # 전체 concat
    feats = torch.cat(video_feats_list, dim=0).cpu().numpy()
    labels = torch.cat(labels_list, dim=0).cpu().numpy()
    preds = torch.cat(preds_list, dim=0).cpu().numpy()
    anonyids = np.concatenate([np.array(a) if isinstance(a, list) else a.cpu().numpy() for a in anonyids_list])
    splits = sum(splits_list, [])  # flatten list of ['Train', ..., 'Valid', ...]

    # UMAP
    reducer = umap.UMAP(**umap_params, random_state=42)
    embedding_2d = reducer.fit_transform(feats)

    fig, axes = plt.subplots(1, len(label_keys), figsize=(6 * len(label_keys), 5))
    if len(label_keys) == 1:
        axes = [axes]

    for i, (label_key, label_map, inverse_map) in enumerate(zip(label_keys, label_maps, inverse_label_maps)):
        label_col = labels[:, i]
        pred_col = preds[:, i]
        label_strs = [f"{label_key}_{inverse_map[int(v)]}_{split}" for v, split in zip(label_col, splits)]
        correctness = ['Correct' if p == l else 'Wrong' for p, l in zip(pred_col, label_col)]

        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'Label': label_strs,
            'Correctness': correctness,
            'AnonyID': anonyids,
        })

        palette = get_custom_palette(label_key)

        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            hue='Label',
            style='Correctness',
            ax=axes[i],
            hue_order=sorted(palette.keys()),
            style_order=['Correct', 'Wrong'],
            markers={'Correct': 'o', 'Wrong': 'X'},
            s=100,
            alpha=0.8,
            palette=palette
        )

        wrongs = df[df['Correctness'] == 'Wrong'].head(10)
        for _, row in wrongs.iterrows():
            axes[i].text(row['x'], row['y'], str(row['AnonyID']), fontsize=7)

        axes[i].set_title(f"{label_key} (Epoch {epoch + 1})")
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/joint_epoch_{epoch+1}_multilabel.png")
    plt.show()
    plt.close()


def visualize_embeddings_multilabel_joint_per_key(
    video_feats_list, labels_list, preds_list, anonyids_list, splits_list,
    epoch, save_path, umap_params, label_key, dx_list
):
    """
    UMAP 시각화: dx_list를 기반으로 Label 및 색상 지정
    Args:
        dx_list: ['HC', 'OTHERS', 'STROKE', ...]  ← Train + Valid 순서
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import torch
    import umap
    import os

    # 전체 concat
    feats = torch.cat(video_feats_list, dim=0).cpu().numpy()
    anonyids = np.concatenate([
        np.array(a) if isinstance(a, list) else a.cpu().numpy()
        for a in anonyids_list
    ])
    dx_list = np.concatenate([
        np.array(a) if isinstance(a, list) else a.cpu().numpy()
        for a in dx_list
    ])
    splits = sum(splits_list, [])  # flatten list

    assert len(dx_list) == len(feats), "❌ dx_list 길이와 feature 수가 일치해야 합니다."

    # 라벨: "DX_CATEGORY_<클래스>_<Split>" 형태
    dx_labels = [f"{label_key}_{dx}_{split}" for dx, split in zip(dx_list, splits)]

    # UMAP 임베딩
    reducer = umap.UMAP(**umap_params, random_state=42)
    embedding_2d = reducer.fit_transform(feats)

    # 데이터프레임 구성
    df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'Label': dx_labels,
        'AnonyID': anonyids
    })

    # 팔레트 가져오기
    palette = get_custom_palette(label_key)

    # Plot
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='Label',
        palette=palette,
        s=90,
        alpha=0.85,
        ax=ax
    )

    ax.set_title(f"UMAP by DX (Epoch {epoch + 1})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/joint_epoch_{epoch+1}_{label_key}.png")
    plt.show()
    plt.close()
