# ========================
# BIBLIOTECAS NECESSÁRIOS
# ========================
import os
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 120
SAVE_DIR = "/content/resultados_gan_color"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========================
# EXTRAÇÃO DO ZIP
# ========================
zip_path = "/content/utk.zip"
extract_path = "/content/UTKFace"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# ========================
# DATASET UTKFACE
# ========================
class UTKFaceColorizationDataset(Dataset):
    def __init__(self, pasta, limite=10000):
        files = [os.path.join(pasta, f) for f in os.listdir(pasta) if f.endswith(".jpg")]
        self.files = files[:limite]
        self.pb_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.rgb_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.pb_transform(img), self.rgb_transform(img)

dataset = UTKFaceColorizationDataset("/content/UTKFace/crop_part1", limite=10000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========================
# GERADOR AJUSTADO (128x128)
# ========================
class GeneratorUNet(nn.Module):
    def __init__(self):
        super().__init__()
        def down_block(in_c, out_c): return nn.Sequential(nn.Conv2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())
        def up_block(in_c, out_c): return nn.Sequential(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1), nn.BatchNorm2d(out_c), nn.ReLU())

        self.down1 = down_block(1, 64)     # 128 -> 64
        self.down2 = down_block(64, 128)   # 64 -> 32
        self.down3 = down_block(128, 256)  # 32 -> 16
        self.down4 = down_block(256, 512)  # 16 -> 8

        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())  # 8 -> 4

        self.up1 = up_block(512, 512)      # 4 -> 8
        self.up2 = up_block(1024, 256)     # 8 -> 16
        self.up3 = up_block(512, 128)      # 16 -> 32
        self.up4 = up_block(256, 64)       # 32 -> 64

        self.final = nn.ConvTranspose2d(128, 3, 4, 2, 1)  # 64 -> 128
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bn = self.bottleneck(d4)

        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        out = self.final(torch.cat([u4, d1], dim=1))
        return self.output_act(out)

# ========================
# DISCRIMINADOR
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1)
        )
    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))

# ========================
# INSTANCIAMENTO
# ========================
G = GeneratorUNet().to(device)
D = Discriminator().to(device)
loss_GAN = nn.BCEWithLogitsLoss()
loss_L1 = nn.L1Loss()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# ========================
# TREINAMENTO
# ========================
loss_g_list = []
loss_d_list = []

print("Iniciando treinamento...")
for epoch in range(EPOCHS):
    G.train()
    total_G, total_D = 0, 0

    for pb, rgb in dataloader:
        pb, rgb = pb.to(device), rgb.to(device)

        fake_rgb = G(pb)
        real_valid = D(pb, rgb)
        fake_valid = D(pb, fake_rgb.detach())
        loss_real = loss_GAN(real_valid, torch.ones_like(real_valid))
        loss_fake = loss_GAN(fake_valid, torch.zeros_like(fake_valid))
        loss_D = 0.5 * (loss_real + loss_fake)
        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        fake_valid = D(pb, fake_rgb)
        loss_G_adv = loss_GAN(fake_valid, torch.ones_like(fake_valid))
        loss_G_recon = loss_L1(fake_rgb, rgb)
        loss_G = loss_G_adv + 100 * loss_G_recon
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        total_G += loss_G.item()
        total_D += loss_D.item()

    loss_g_list.append(total_G)
    loss_d_list.append(total_D)
    print(f"📉 Época {epoch+1}/{EPOCHS} - Loss G: {total_G:.4f} | Loss D: {total_D:.4f}")

# ========================
# INFERÊNCIA
# ========================
G.eval()
img_pb, _ = dataset[0]
with torch.no_grad():
    out = G(img_pb.unsqueeze(0).to(device)).squeeze().cpu().numpy()
output_img = np.transpose(out, (1, 2, 0))

def aplicar_filtros_rgb(img_rgb_tensor):
    img_rgb = (img_rgb_tensor * 255).astype(np.uint8)
    realcada = cv2.detailEnhance(img_rgb, sigma_s=10, sigma_r=0.15)
    suavizada = cv2.bilateralFilter(realcada, d=9, sigmaColor=75, sigmaSpace=75)
    return suavizada

filtrada = aplicar_filtros_rgb(output_img)

# ========================
# VISUALIZAÇÃO
# ========================
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(img_pb.squeeze(), cmap="gray")
plt.title("Entrada P&B")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(output_img)
plt.title("Colorizada (GAN)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(filtrada)
plt.title("Colorizada + Filtro")
plt.axis("off")
plt.tight_layout()
plt.show()

# ========================
# SALVAMENTO
# ========================
cv2.imwrite(f"{SAVE_DIR}/gan_colorida.png", cv2.cvtColor((output_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{SAVE_DIR}/gan_filtrada.png", cv2.cvtColor(filtrada, cv2.COLOR_RGB2BGR))

# ========================
# GRÁFICO DE LOSS
# ========================
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_g_list, label="Loss Gerador (G)")
plt.plot(range(1, EPOCHS + 1), loss_d_list, label="Loss Discriminador (D)")
plt.title("Curva de Perda por Época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{SAVE_DIR}/loss_gan.png")
plt.show()

print("Imagens e gráfico salvos em:", SAVE_DIR)
