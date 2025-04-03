from torch import nn
from VideoNetworks import VideoEncoder

class VideoModel(nn.Module):
    def __init__(self, model_config):
        super(VideoModel, self).__init__()
        self.video_encoder = VideoEncoder(model_config)
        self.projection = nn.Linear(model_config['hidden_dim'], model_config['projection_dim'])

    def forward(self, video_frames):
        return self.projection(self.video_encoder(video_frames))