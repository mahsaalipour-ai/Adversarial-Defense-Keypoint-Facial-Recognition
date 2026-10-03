# steps/models_gan.py
import torch, torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

# ---------- Generator (COMPAT با G_mask.pth) ----------
# u2/u1: ConvTranspose2d  + align با interpolate
# c2/c1: DoubleConv
class Generator(nn.Module):
    def __init__(self, in_ch=4, base=32):
        super().__init__()
        # Encoder
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*2, base*4)

        # Decoder
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.c1 = DoubleConv(base*2, base)

        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.d1(x)                     # (B,b,H,W)
        x2 = self.d2(self.p1(x1))           # (B,2b,H/2,W/2)
        xb = self.b(self.p2(x2))            # (B,4b,H/4,W/4)

        x  = self.u2(xb)                    # ممکن است 1px اختلاف
        x  = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x  = self.c2(torch.cat([x, x2], dim=1))

        x  = self.u1(x)
        x  = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x  = self.c1(torch.cat([x, x1], dim=1))

        return self.out(x)                  # logits (B,1,H,W)

# ---------- Discriminator ----------
class Discriminator(nn.Module):
    def __init__(self, c=3, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, base, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base*4, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

# ---------- FaceMatcher ----------
class FaceMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = None
        try:
            from facenet_pytorch import InceptionResnetV1
            self.backbone = InceptionResnetV1(pretrained='vggface2').eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        except Exception:
            self.fallback = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((8,8)),
                nn.Flatten(),
                nn.Linear(32*8*8, 256)
            )
            for p in self.fallback.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        if self.backbone is not None:
            x = F.interpolate(x, size=(160,160), mode='bilinear', align_corners=False)
            emb = self.backbone(x)
            return F.normalize(emb, p=2, dim=1)
        x = F.interpolate(x, size=(128,128), mode='bilinear', align_corners=False)
        emb = self.fallback(x)
        return F.normalize(emb, p=2, dim=1)

def cosine_sim(a, b, eps=1e-8):
    a = F.normalize(a, p=2, dim=1)
    b = F.normalize(b, p=2, dim=1)
    return (a*b).sum(dim=1)
