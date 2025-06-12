from torch import nn
from VideoNetworks import VideoEncoder
from peft import get_peft_model, LoraConfig, PeftModel

class VideoModel(nn.Module):
    def __init__(self, model_config):
        super(VideoModel, self).__init__()
        self.video_encoder = VideoEncoder(**model_config)
        self.projection = nn.Linear(model_config['hidden_dim'], model_config['projection_dim'])
        # self.projection_mlp = nn.Sequential(
        #             nn.Linear(model_config['hidden_dim'], model_config['hidden_dim']),
        #             nn.LayerNorm(model_config['hidden_dim']),
        #             nn.GELU(),
        #             nn.Linear(model_config['hidden_dim'], model_config['projection_dim'])
        #         )

    def forward(self, video_frames):
        video_feats = self.video_encoder(video_frames)
        return video_feats, self.projection(video_feats)
    
    
class CLIPStyleContrastiveModel(nn.Module):
    def __init__(self, model_config, num_classes):
        super(CLIPStyleContrastiveModel, self).__init__()
        self.video_encoder = VideoEncoder(**model_config)
        self.projection = nn.Linear(model_config['hidden_dim'], model_config['projection_dim'])
        
        # 라벨 임베딩 (정수 class ID → vector)
        self.label_embedding = nn.Embedding(num_classes, model_config['projection_dim'])

        self.temperature = model_config.get('temperature', 0.07)

    def forward(self, video_frames):
        video_feat = self.video_encoder(video_frames)
        video_feat_proj = self.projection(video_feat)
        return video_feat, nn.functional.normalize(video_feat_proj, dim=1)  # (B, D)

    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def evaluate_linear_probe(train_feats, train_labels, val_feats, val_labels):

    # numpy 변환
    X_train = train_feats.numpy()
    y_train = train_labels.numpy()
    X_val = val_feats.numpy()
    y_val = val_labels.numpy()

    # 간단한 선형 분류기 학습
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # 예측 및 평가
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')

    return acc, f1
