import os
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch

def is_static_gif(gif_path, threshold=0.005):
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGB")
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    if len(frames) < 2:
        return False  # 너무 적은 프레임 수

    frames_tensor = torch.tensor(frames, dtype=torch.float32).permute(3, 0, 1, 2) / 255.0  # (C, T, H, W)
    diffs = (frames_tensor[:, 1:] - frames_tensor[:, :-1]).abs().mean()
    return diffs.item() < threshold

# === 경로 설정 ===
gif_folder = "./preprocessing/processed_video_seg"  # 수정 가능
gif_files = [f for f in os.listdir(gif_folder) if f.endswith(".gif")]

# === 이상 탐지 실행 ===
static_results = []
for gif_file in tqdm(gif_files):
    gif_path = os.path.join(gif_folder, gif_file)
    is_static = is_static_gif(gif_path)
    if is_static:
        static_results.append({"GIF": gif_file, "Static": True})
        print("GIF:", gif_file, "Static:", True)

# === 결과 저장 ===
static_log_df = pd.DataFrame(static_results)
static_log_df.to_csv(os.path.join(gif_folder, "detected_static_gifs.csv"), index=False)
print(f"✅ Static GIFs saved to: {os.path.join(gif_folder, 'detected_static_gifs.csv')}")
