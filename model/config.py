import numpy as np

class Config:
    batch_size = 32
    lr = 3e-5
    weight_decay = 1e-6
    epochs = 200
    patience = 20
    warmup_epochs = 30
    diffusion_steps = 1000  # 扩散模型步数
    latent_dim = 512
    sparsity_range = [0.205, 1.0]
    sparsity_alpha = 8.0
    sparsity_beta = 2.0
    quad_weight = 1.8
    min_quad_ratio = 0.5
    max_quad_ratio = 0.75

config = Config()

BASE_VERTICES = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
], dtype=np.float32)

ORIGINAL_FACES = [
    [0, 1, 2, 3],       # 四边形
    [0, 1, 5, 4],       # 四边形
    [0, 1, 6, 7],       # 四边形
    [0, 2, 4, 6],       # 四边形
    [0, 5, 2, 2],       # 三角形 -> 重复最后一个顶点
    [0, 2, 7, 7],       # 三角形 -> 重复最后一个顶点
    [0, 4, 7, 3],       # 四边形
    [0, 3, 5, 6],       # 四边形
    [0, 5, 7, 7],       # 三角形 -> 重复最后一个顶点
    [1, 2, 4, 7],       # 四边形
    [1, 2, 5, 6],       # 四边形
    [1, 3, 4, 4],       # 三角形 -> 重复最后一个顶点
    [1, 3, 5, 7],       # 四边形
    [1, 3, 6, 6],       # 三角形 -> 重复最后一个顶点
    [1, 4, 6, 6],       # 三角形 -> 重复最后一个顶点
    [2, 3, 4, 5],       # 四边形
    [2, 3, 6, 7],       # 四边形
    [2, 5, 7, 7],       # 三角形 -> 重复最后一个顶点
    [3, 4, 6, 6],       # 三角形 -> 重复最后一个顶点
    [4, 5, 6, 7]        # 四边形
]