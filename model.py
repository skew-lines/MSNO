import torch
import torch.nn as nn
import fairseq


class SSLModel(nn.Module):
    def __init__(self, device, checkpoint_path: str):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024

    def forward(self, input_data):
        first_param = next(self.model.parameters())
        if first_param.device != input_data.device or first_param.dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]

        return self.model(input_data, mask=False, features_only=True)["x"]


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args

        checkpoint_path = getattr(args, "ssl_checkpoint", "/root/autodl-fs/xlsr2_300m.pt")
        self.ssl_model = SSLModel(self.device, checkpoint_path)

        self.classifier = nn.Sequential(
            nn.Linear(self.ssl_model.out_dim * 2, 1024),
            nn.SELU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        frame_features = self.ssl_model(x.squeeze(-1))

        mean = frame_features.mean(dim=1)
        std = frame_features.std(dim=1, unbiased=False)
        pooled = torch.cat([mean, std], dim=1)

        return self.classifier(pooled)
