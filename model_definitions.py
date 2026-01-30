import torch
import torch.nn as nn
import timm

# ================= SmallSpectralEncoder =================
class SmallSpectralEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int = 256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, emb_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.head(self.block(self.stem(x)))

# ================= CFG giả lập cho inference =================
class CFG:
    USE_RGB = True
    USE_MS  = True
    USE_HS  = True

# ================= MultiModalNet =================
class MultiModalNet(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        hs_in_ch=101,
        use_rgb=True,
        use_ms=True,
        use_hs=True,
        n_classes=3
    ):
        super().__init__()

        self.use_rgb = use_rgb
        self.use_ms  = use_ms
        self.use_hs  = use_hs

        feat_dims = []

        # ===== RGB =====
        if self.use_rgb:
            if backbone == "resnet":
                self.rgb_enc = timm.create_model(
                    "resnet18",
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg"
                )
            else:  # vit
                self.rgb_enc = timm.create_model(
                    "vit_base_patch16_224",
                    pretrained=False,
                    num_classes=0
                )

            self.rgb_norm = nn.LayerNorm(self.rgb_enc.num_features)
            feat_dims.append(self.rgb_enc.num_features)

        # ===== MS =====
        if self.use_ms:
            self.ms_enc = SmallSpectralEncoder(5, 256)
            self.ms_norm = nn.LayerNorm(256)
            feat_dims.append(256)

        # ===== HS =====
        if self.use_hs:
            self.hs_enc = SmallSpectralEncoder(hs_in_ch, 256)
            self.hs_norm = nn.LayerNorm(256)
            feat_dims.append(256)

        fusion_dim = sum(feat_dims)

        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

    def forward(self, rgb, ms, hs, mask):
        feats = []

        if self.use_rgb:
            feats.append(self.rgb_norm(self.rgb_enc(rgb)) * mask[:, 0:1])

        if self.use_ms:
            feats.append(self.ms_norm(self.ms_enc(ms)) * mask[:, 1:2])

        if self.use_hs:
            feats.append(self.hs_norm(self.hs_enc(hs)) * mask[:, 2:3])

        f = torch.cat(feats, dim=1)
        f = f * self.gate(f)
        return self.classifier(f)
