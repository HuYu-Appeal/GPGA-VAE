import numpy as np

def point_on_plane(point, normal, d):
    # 检查点是否在平面上，考虑浮点数精度
    dot_product = np.dot(normal, point) + d
    return np.isclose(dot_product, 0, atol=1e-6)