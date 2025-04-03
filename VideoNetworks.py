from torch import nn
from transformers import TimesformerModel, TimesformerConfig


class VideoEncoder(nn.Module):
    def __init__(self, config):
        super(VideoEncoder, self).__init__()
        self.config = config
        if config['pretrained']:
            self.model = TimesformerModel.from_pretrained("facebook/timesformer-hr-finetuned-k400")
            if config['freeze_pretrained']:
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            self.model = TimesformerModel(TimesformerConfig(
                image_size=config['image_size'],
                patch_size=config['patch_size'],
                num_frames=config['num_frames'],
                num_channels=3,
                num_attention_heads=12,
                hidden_size=config['hidden_dim'],
                num_hidden_layers=12,
                intermediate_size=3072,
                attention_dropout=0.1,
                hidden_dropout=0.1,
                num_classes=config['hidden_dim']
            ))

    def forward(self, video_frames):
        B, C, T, H, W = video_frames.shape
        video_frames = video_frames.permute(0, 2, 1, 3, 4)
        output = self.model(video_frames)
        return output.last_hidden_state.mean(dim=1)