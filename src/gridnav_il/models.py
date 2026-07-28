import torch
import torch.nn as nn


class CNNPolicy(nn.Module):
    def __init__(self, input_channels=2, output_dim=2):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.mlp(x)
        return x

class CNNAttentionPolicy(nn.Module):
    def __init__(
        self,
        input_channels=2,
        output_dim=2,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # For 64x64 input with four stride-2 conv layers:
        # 64 -> 32 -> 16 -> 8 -> 4, so we get 4x4 = 16 tokens.
        self.num_tokens = 4 * 4
        self.d_model = d_model

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = self.conv(x)                 # [B, d_model, 4, 4]
        x = x.flatten(2).transpose(1, 2) # [B, 16, d_model]

        x = x + self.pos_embedding
        x = self.transformer(x)

        # Mean-pool spatial tokens.
        x = x.mean(dim=1)

        x = self.head(x)
        return x