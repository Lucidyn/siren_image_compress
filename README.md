# SIREN 图像压缩

使用 SIREN (Sinusoidal Representation Networks) 神经网络进行图像压缩和重建的项目。

## 功能

- 将图像分割成多个 tile，每个 tile 训练一个 SIREN 网络进行压缩
- 根据 tile 复杂度动态调整训练轮数（纹理复杂的 tile 训练更久，范围 1000-5000 轮）
- 使用 mini-batch 训练，提高训练效率
- 支持 tile 重叠（overlap）和边缘平滑拼接，减少拼接痕迹
- 自动重建图像并保存结果

## 安装

```bash
pip install -r requirements.txt
```

注意：PyTorch 需要单独安装，根据你的 CUDA 版本选择合适的版本。

## 使用方法

### 1. 配置参数

编辑 `config.py` 设置：
- `IMG_PATH`: 输入图像路径
- `TILE_SIZE`: tile 大小（建议 64/128/256）
- `EPOCHS`: 基础训练轮数（实际会根据 tile 复杂度动态调整，范围 1000-5000）
- `HIDDEN_DIM`: 隐藏层维度（控制压缩率和质量，建议 256-512）
- `LAYERS`: 网络层数（建议 4-6 层）
- `LR`: 学习率（默认 1e-4）
- `RECONSTRUCT_FROM_CKPT`: 是否从保存的 pth 文件重建（`True`=从文件加载重建，`False`=训练后从内存重建）

注意：`main.py` 中也可以直接修改 `HIDDEN_DIM`、`LAYERS`、`BATCH_SIZE` 和 `OVERLAP` 参数。

### 2. 运行

```bash
python main.py
```

程序会自动完成：
- 图像分割成 tiles（支持重叠切分，减少拼接痕迹）
- 训练每个 tile 的 SIREN 模型（根据复杂度动态调整训练轮数，使用 mini-batch 训练）
- 使用边缘平滑拼接重建完整图像

训练好的模型会保存在 `ckpt/` 目录下，重建结果会保存为 `ckpt/reconstruction.png`。

**重建模式说明：**
- `RECONSTRUCT_FROM_CKPT = False`（默认）：训练完成后直接从内存中的模型重建，速度更快
- `RECONSTRUCT_FROM_CKPT = True`：从保存的 `.pth` 文件加载模型重建，适合只重建不训练的场景

## 项目结构

```
.
├── main.py           # 主程序（训练+重建）
├── config.py         # 配置文件
├── models/
│   └── siren.py      # SIREN 网络模型
├── utils/
│   └── image_utils.py # 图像处理工具
├── ckpt/             # 模型检查点目录
└── requirements.txt  # 依赖包
```

