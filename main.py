import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from config import *
from models.siren import Siren
from utils.image_utils import split_image, get_coords

torch.manual_seed(SEED)
device = DEVICE if torch.cuda.is_available() else "cpu"

os.makedirs(CKPT_DIR, exist_ok=True)

# 改进 SIREN 参数
HIDDEN_DIM = 512   # 隐藏层节点数
LAYERS = 4         # 隐藏层数量
W0 = 30            # 输入层频率参数
BATCH_SIZE = 4096  # mini-batch

# 读取图片
img = Image.open(IMG_PATH).convert("RGB")
img_t = transforms.ToTensor()(img).permute(1,2,0)
img_np = img_t.numpy()
h_total, w_total, _ = img_np.shape

# tile 切分 + overlap
OVERLAP = 16
tiles = split_image(img_np, TILE_SIZE, overlap=OVERLAP)
print(f"Total tiles: {len(tiles)}")

# 动态计算 tile 训练轮数
def compute_epochs(tile, min_epochs=1000, max_epochs=5000):
    std = np.std(tile)
    std_norm = np.clip(std / 0.3, 0, 1)
    epochs = int(min_epochs + std_norm * (max_epochs - min_epochs))
    return epochs

# tile 训练函数
def train_tile(tile, coords, epochs):
    target = torch.tensor(tile.reshape(-1, 3)).float().to(device)
    target = target * 2.0 - 1.0  # [-1,1]

    model = Siren(IN_DIM, OUT_DIM, HIDDEN_DIM, LAYERS, W0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    pbar = tqdm(range(epochs), desc="Training Tile")
    for _ in pbar:
        for i in range(0, coords.shape[0], BATCH_SIZE):
            batch_coords = coords[i:i+BATCH_SIZE]
            batch_target = target[i:i+BATCH_SIZE]
            pred = model(batch_coords)
            loss = loss_fn(pred, batch_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        pbar.set_postfix(loss=loss.item())

    return model

# tile 训练 + 保存模型
tile_models = []
for idx, (y, x, tile) in enumerate(tiles):
    h, w, _ = tile.shape
    coords = get_coords(h, w).to(device)
    epochs_tile = compute_epochs(tile)
    print(f"Tile {idx}: shape={h}x{w}, epochs={epochs_tile}")

    model = train_tile(tile, coords, epochs_tile)
    tile_models.append((model, (y, x), (h, w)))
    
    # 保存模型
    torch.save(
        {
            "state": model.state_dict(),
            "shape": (h, w),
            "pos": (y, x)
        },
        f"{CKPT_DIR}/tile_{idx}.pth"
    )

# 从 pth 文件加载模型并重建
def reconstruct_from_checkpoints(ckpt_dir, img_shape):
    """从保存的 pth 文件加载模型并重建图像"""
    import glob
    h_total, w_total, _ = img_shape
    recon = np.zeros(img_shape, dtype=np.float32)
    weight = np.zeros(img_shape, dtype=np.float32)
    
    # 加载所有 pth 文件
    pth_files = sorted(glob.glob(f"{ckpt_dir}/tile_*.pth"), 
                       key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    for pth_file in pth_files:
        ckpt = torch.load(pth_file, map_location='cpu')
        model = Siren(IN_DIM, OUT_DIM, HIDDEN_DIM, LAYERS, W0)
        model.load_state_dict(ckpt["state"])
        model.eval()
        
        y, x = ckpt["pos"]
        h, w = ckpt["shape"]
        
        coords = get_coords(h, w)
        with torch.no_grad():
            pred = model(coords).reshape(h, w, 3).cpu().numpy()
        pred = (pred + 1.0) / 2.0  # 反归一化

        # 边缘权重，线性平滑
        yy = np.linspace(0,1,h)[:,None]
        xx = np.linspace(0,1,w)[None,:]
        w_map = np.minimum(yy, 1-yy) * np.minimum(xx,1-xx)
        w_map = w_map[:,:,None]

        recon[y:y+h, x:x+w, :] += pred * w_map
        weight[y:y+h, x:x+w, :] += w_map

    # 避免除0
    recon = recon / np.maximum(weight, 1e-8)
    recon_img = Image.fromarray((np.clip(recon,0,1)*255).astype(np.uint8))
    return recon_img

# tile 拼接函数（线性平滑边缘）
def reconstruct_image(tile_models, img_shape):
    """从内存中的模型列表重建图像"""
    h_total, w_total, _ = img_shape
    recon = np.zeros(img_shape, dtype=np.float32)
    weight = np.zeros(img_shape, dtype=np.float32)

    for model, (y, x), (h, w) in tile_models:
        coords = get_coords(h, w)
        with torch.no_grad():
            pred = model(coords).reshape(h, w, 3).cpu().numpy()
        pred = (pred + 1.0) / 2.0  # 反归一化

        # 边缘权重，线性平滑
        yy = np.linspace(0,1,h)[:,None]
        xx = np.linspace(0,1,w)[None,:]
        w_map = np.minimum(yy, 1-yy) * np.minimum(xx,1-xx)
        w_map = w_map[:,:,None]

        recon[y:y+h, x:x+w, :] += pred * w_map
        weight[y:y+h, x:x+w, :] += w_map

    # 避免除0
    recon = recon / np.maximum(weight, 1e-8)
    recon_img = Image.fromarray((np.clip(recon,0,1)*255).astype(np.uint8))
    return recon_img

# 拼接生成最终重建图
if RECONSTRUCT_FROM_CKPT:
    # 从保存的 pth 文件重建
    recon_img = reconstruct_from_checkpoints(CKPT_DIR, img_np.shape)
    recon_img.save(os.path.join(CKPT_DIR, "reconstruction.png"))
    print("Reconstruction from checkpoints finished.")
else:
    # 从内存中的模型重建
    recon_img = reconstruct_image(tile_models, img_np.shape)
    recon_img.save(os.path.join(CKPT_DIR, "reconstruction.png"))
    print("Training and reconstruction finished.")
