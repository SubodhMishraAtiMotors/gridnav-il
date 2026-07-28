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


class WaypointQueryAttentionPolicy(nn.Module):
    def __init__(
        self,
        input_channels=2,
        output_dim=20,
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        if output_dim % 4 != 0:
            raise ValueError(
                "WaypointQueryAttentionPolicy expects output_dim to be "
                "4 * num_waypoints."
            )

        self.output_dim = int(output_dim)
        self.num_waypoints = output_dim // 4
        self.d_model = int(d_model)

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
        # 64 -> 32 -> 16 -> 8 -> 4, so 4x4 = 16 spatial tokens.
        self.num_spatial_tokens = 4 * 4

        self.spatial_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_spatial_tokens, d_model)
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

        self.spatial_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # One learnable query per future waypoint.
        self.waypoint_queries = nn.Parameter(
            torch.randn(1, self.num_waypoints, d_model) * 0.02
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.query_norm = nn.LayerNorm(d_model)
        self.token_norm = nn.LayerNorm(d_model)

        # Shared head applied independently to each waypoint query token.
        self.waypoint_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv(x)                    # [B, d_model, 4, 4]
        tokens = x.flatten(2).transpose(1, 2)  # [B, 16, d_model]

        tokens = tokens + self.spatial_pos_embedding
        tokens = self.spatial_encoder(tokens)

        queries = self.waypoint_queries.expand(batch_size, -1, -1)

        query_tokens, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=self.token_norm(tokens),
            value=tokens,
            need_weights=False,
        )

        # Residual update.
        query_tokens = queries + query_tokens

        waypoint_outputs = self.waypoint_head(query_tokens)  # [B, num_waypoints, 4]

        return waypoint_outputs.reshape(batch_size, self.output_dim)