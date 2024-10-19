import torch
import torch.nn as nn
import clip
from model import ST_GCN_18

class ContrastiveModule(nn.Module):
    
    def __init__(self, args):
        super(ContrastiveModule, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        del model.visual
        self.model = model

        base_channel = 3
        base_channel = base_channel * 2 if args.gyro else base_channel
        base_channel = base_channel * 2 if args.stft else base_channel
        self.model.acc = ST_GCN_18(in_channels=base_channel)

        self.model = self.model.float()

        if args.stage == 'finetune':
            self.fc = nn.Linear(512, args.num_class)

    def encode_image(self, image):
        return self.model.acc(image.float()).squeeze(-1).squeeze(-1)
    
    def encode_text(self, text):
        x = self.model.token_embedding(text).float() # b,t,512
        x = x + self.model.positional_embedding.float()
        x = x.permute(1, 0, 2)  # b,t,512 -> t,b,512
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # t,b,512 -> b,t,512
        x = self.model.ln_final(x).float() # b,t,512

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection # b,512

        return x
    
    def classifier(self, image):
        # for fine-tuning
        imu_features = self.model.acc(image.float()).squeeze(-1).squeeze(-1)
        out = self.fc(imu_features)
        return out
    
    def forward(self, inputs_imu, inputs_text):

        imu_features = self.encode_image(inputs_imu)
        text_features = self.encode_text(inputs_text)

        # normalized features
        imu_features = imu_features / imu_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * imu_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text