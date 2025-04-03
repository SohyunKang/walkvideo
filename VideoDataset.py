import torch
import os

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir):
        self.tensor_files = sorted([f for f in os.listdir(tensor_dir) if f.endswith('.pt')])
        self.tensor_dir = tensor_dir

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.tensor_dir, self.tensor_files[idx]), weights_only=False)
        return data['video'], torch.tensor(data['label'], dtype=torch.long)