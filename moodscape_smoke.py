
---

## 2) `moodscape_smoke.py`
```python
# moodscape_smoke.py
# Minimal, runnable smoke test for the DH-CVAE-GAN generator (no dataset needed).
# Saves outputs/sample_random.png

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


class Generator(nn.Module):
    """
    Dual-Head CVAE generator (geometric conv encoder + contextual dense encoder).
    Decoder upsamples back to 128x128. Mirrors Sec. 4.3.2 high-level design:
      - Geometric encoder: 4 conv blocks (128->64->32->16->8)
      - Context encoder: one-hot class -> 2 FC layers
      - Latent heads: mu/logvar (dim=200)
      - Decoder: FC to 8x8x256 then 4 ConvTranspose blocks to 128x128 with
        LayerNorm + SELU in each upsampling stage.
    """
    def __init__(self, latent_dim=200, num_classes=4):
        super().__init__()
        self.latent_dim = latent_dim

        # ---- Geometric encoder (convolutional) ----
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # 128 -> 64
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 3, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_enc = nn.Linear(256 * 8 * 8, 128)  # compress conv features

        # ---- Contextual encoder (class/mood one-hot) ----
        self.fc_context1 = nn.Linear(num_classes, 64)
        self.fc_context2 = nn.Linear(64, 128)

        # ---- VAE latent heads ----
        self.fc_mu     = nn.Linear(128 + 128, latent_dim)
        self.fc_logvar = nn.Linear(128 + 128, latent_dim)

        # ---- Decoder: project z to 8x8x256, then upsample 4x ----
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)   # 8 -> 16
        self.ln1 = nn.LayerNorm([128, 16, 16]); self.act1 = nn.SELU(inplace=True)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)    # 16 -> 32
        self.ln2 = nn.LayerNorm([64, 32, 32]);  self.act2 = nn.SELU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)     # 32 -> 64
        self.ln3 = nn.LayerNorm([32, 64, 64]);  self.act3 = nn.SELU(inplace=True)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)

        self.deconv4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)     # 64 -> 128
        self.ln4 = nn.LayerNorm([16, 128, 128]); self.act4 = nn.SELU(inplace=True)
        self.conv4 = nn.Conv2d(16, 1, 3, 1, 1)                  # final 1-channel map

    # ----- Encoder side -----
    def encode(self, x, c_onehot):
        feat_map = self.enc_conv(x)                               # (B,256,8,8)
        geom_feat = self.fc_enc(feat_map.view(x.size(0), -1))     # (B,128)
        cont_feat = F.relu(self.fc_context1(c_onehot))
        cont_feat = self.fc_context2(cont_feat)                   # (B,128)
        fused = torch.cat([geom_feat, cont_feat], dim=1)          # (B,256)
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ----- Decoder side -----
    def decode(self, z):
        out = self.fc_dec(z).view(z.size(0), 256, 8, 8)
        out = self.deconv1(out); out = self.ln1(out); out = self.act1(out); out = self.conv1(out)
        out = self.deconv2(out); out = self.ln2(out); out = self.act2(out); out = self.conv2(out)
        out = self.deconv3(out); out = self.ln3(out); out = self.act3(out); out = self.conv3(out)
        out = self.deconv4(out); out = self.ln4(out); out = self.act4(out); out = self.conv4(out)
        return torch.sigmoid(out)  # normalize to [0,1]

    def forward(self, x, c_onehot):
        mu, logvar = self.encode(x, c_onehot)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    # Convenience for this smoke test: generate from random z (no dataset)
    def generate_from_noise(self, batch=1, device="cpu"):
        z = torch.randn((batch, self.latent_dim), device=device)
        with torch.no_grad():
            img = self.decode(z)
        return img


def save_png(tensor_img, path):
    """tensor_img: (1, 128, 128) in [0,1]"""
    arr = (tensor_img.detach().cpu().squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build generator and switch to eval mode
    gen = Generator(latent_dim=200, num_classes=4).to(device)
    gen.eval()

    # Create one random sample (no dataset needed)
    sample = gen.generate_from_noise(batch=1, device=device)[0]  # (1,128,128)

    out_path = os.path.join("outputs", "sample_random.png")
    save_png(sample, out_path)
    print(f"Saved: {out_path}")
