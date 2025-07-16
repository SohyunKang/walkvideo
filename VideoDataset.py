import torch
import os

def label_encoder(raw_label, label_key):
    if label_key == "AGE":
        if raw_label<71:
            raw_label = 0
        else:
            raw_label = 1
    elif label_key == "HEIGHT":
        if raw_label<160:
            raw_label = 0
        else:
            raw_label = 1
    elif label_key == "GAIT_DISTURBANCE":
        if raw_label in ["NO"]:
            raw_label = 0
        elif raw_label in ["POSSIBLE"]:
            raw_label = 1
        elif raw_label in ["YES"]:
            raw_label = 2
    elif label_key == "DX_CATEGORY":
        if raw_label in ["HC"]:
            raw_label = 0
        elif raw_label in ["OTHERS"]:
            raw_label = 1
        elif raw_label in ["COGNITIVE DISORDER"]:
            raw_label = 2
        elif raw_label in ["STROKE"]:
            raw_label = 3
        elif raw_label in ["PARKINSONISM"]:
            raw_label = 4
            
    return raw_label

# class VideoDataset(torch.utils.data.Dataset):
#     def __init__(self, tensor_dir, label_keys):
#         self.tensor_files = sorted([f for f in os.listdir(tensor_dir) if f.endswith('.pt')])
#         self.tensor_dir = tensor_dir
#         self.label_key = label_keys

#     def __len__(self):
#         return len(self.tensor_files)

#     def __getitem__(self, idx):
#         data = torch.load(os.path.join(self.tensor_dir, self.tensor_files[idx]), weights_only=False)
#         video = data['video']

#         raw_label = data[self.label_key]
#         encoded_label = label_encoder(raw_label, self.label_key)
        
#         label_tensor = torch.tensor(encoded_label, dtype=torch.long)
#         filename = self.tensor_files[idx]
#         anonyid = int(filename.replace('SUBJ', '').replace('.pt', ''))

#         return video, label_tensor, anonyid
    

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, tensor_dir, label_keys):
        self.tensor_files = sorted([f for f in os.listdir(tensor_dir) if f.endswith('.pt')])
        self.tensor_dir = tensor_dir
        self.label_keys = label_keys  # Expects a list of keys

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.tensor_dir, self.tensor_files[idx]), weights_only=False)
        video = data['video']
        train = data['1_TRAINING']
        dx = data['DX_CATEGORY']
        gait_abnormality = data['GAIT_DISTURBANCE']

        # Multi-label encoding
        encoded_labels = []
        for key in self.label_keys:
            raw_label = data[key]
            encoded_label = label_encoder(raw_label, key)
            encoded_labels.append(encoded_label)
        label_tensor = torch.tensor(encoded_labels, dtype=torch.long)  # multi-label: shape (num_labels,)

        filename = self.tensor_files[idx]
        anonyid = int(filename.split("_")[0].replace("SUBJ", ""))

        return video, label_tensor, anonyid, train, dx, gait_abnormality
