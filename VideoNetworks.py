from torch import nn
from transformers import TimesformerModel, TimesformerConfig, VideoMAEImageProcessor, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class VideoEncoder(nn.Module):
    def __init__(self, num_frames=16, image_size=224, patch_size=32, hidden_dim=768, projection_dim=128, pretrained=True, freeze_pretrained=True, encoder=''):
        super(VideoEncoder, self).__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        if pretrained:
            print("Loading pretrained TimeSformer model...")
            base_model = TimesformerModel.from_pretrained(encoder)
            # base_model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base',trust_remote_code=True)
            # self.processor = VideoMAEImageProcessor.from_pretrained("OpenGVLab/VideoMAEv2-Base")
            print("Pretrained model loaded successfully!")

            if freeze_pretrained == True:
                for param in base_model.parameters():
                    param.requires_grad = False
                print("Pretrained model parameters are frozen.")

            elif freeze_pretrained == False:
                for param in base_model.parameters():
                    param.requires_grad = True
                print("Pretrained model parameters will be learned.")

            elif freeze_pretrained == "partial":
                for name, param in base_model.named_parameters():
                    if "encoder.layer.8" in name or "encoder.layer.9" in name or "encoder.layer.10" in name or "encoder.layer.11" in name \
                            or name == "layernorm.weight" or name == "layernorm.bias":
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                print("Pretrained model parameters will be learned partially.")

            elif freeze_pretrained == "lora":
                from peft import get_peft_model, LoraConfig, TaskType
                print("Applying LoRA to TimeSformer...")
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["qkv"],  # 실제 Layer 이름 확인 필요
                    lora_dropout=0.1,
                    bias="none",
                )
                peft_model = get_peft_model(base_model, lora_config)

                # ✅ wrapper로 감싸서 forward(video_frames) 유지
                class TimeSformerWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, video_frames):
                        return self.model(pixel_values=video_frames)

                self.model = TimeSformerWrapper(peft_model)
                print("LoRA applied and wrapped for compatibility.")
                return  # LoRA이면 여기서 초기화 종료

            self.model = base_model

        else:
            self._init_custom_model()

    def _init_custom_model(self):
        config = TimesformerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            num_channels=3,
            num_attention_heads=12,
            hidden_size=self.hidden_dim,
            num_hidden_layers=12,
            intermediate_size=3072,
            attention_dropout=0.1,
            hidden_dropout=0.1,
            num_classes=self.hidden_dim
        )
        self.model = TimesformerModel(config)

    def forward(self, video_frames):
        B, C, T, H, W = video_frames.shape
        # print(B,C,T,H,W)
        video_frames = video_frames.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        output = self.model(video_frames)
        return output.last_hidden_state.mean(dim=1)  # (B, hidden_dim)

        # # (B, C, T, H, W) → (B, T, H, W, C)
        # video_frames = video_frames.permute(0, 2, 3, 4, 1)
         
        # # Convert to list of np.ndarray (T, H, W, C) in uint8
        # video_list = [v.cpu().numpy().astype(np.uint8) for v in video_frames]
        # print(video_list[0].shape)
        # print([v.shape for v in video_list])
        # print(video_list[0].dtype)
        # # Process input
        # inputs = self.processor(video_list, return_tensors="pt")
        # inputs = {k: v.to(video_frames.device) for k, v in inputs.items()}

        # outputs = self.model(**inputs)
        # return outputs.last_hidden_state.mean(dim=1)  # (B, hidden_dim)
    

