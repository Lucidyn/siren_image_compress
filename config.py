# 基础路径
IMG_PATH = "C:\\Users\\LoveL\\Desktop\\siren_image_compress\\test_image\\test.jpg"
CKPT_DIR = "ckpt"          # 压缩模型保存目录
RECON_PATH = "C:\\Users\\LoveL\\Desktop\\siren_image_compress\\result_image\\recon.png"   # 重建结果

# 训练参数
TILE_SIZE = 128            # tile 大小（64/128/256）
EPOCHS = 3000              # 增加训练轮数，提高重建质量
LR = 1e-4

# SIREN 网络结构
IN_DIM = 2
OUT_DIM = 3
HIDDEN_DIM = 512           # 控制压缩率（越大质量越好但模型越大）
LAYERS = 6                 # 可以增加到 6-8 层提高质量
W0 = 30.0

# 重建,超分
UPSCALE = 2                # 1=原尺寸，2/4=超分
CLAMP = True

DEVICE = "cuda"
SEED = 42
