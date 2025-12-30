import torch

def split_image(img, tile, overlap=0):
    h, w, _ = img.shape
    tiles = []
    step = tile - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            y_end = min(y + tile, h)
            x_end = min(x + tile, w)
            tiles.append((y, x, img[y:y_end, x:x_end]))
    return tiles

def get_coords(h, w):
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij"
    )
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)
