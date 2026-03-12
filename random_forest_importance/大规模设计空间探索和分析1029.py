# enhanced_design_space_analysis.py
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import pandas as pd
from dataset import SuperEnhancedDataset, custom_collate_fn
from model import SuperDiffusionVAE
from config import config, ORIGINAL_FACES, BASE_VERTICES
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib import cm
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import networkx as nx
from matplotlib.colors import Normalize
from scipy.spatial.transform import Rotation as R

# 设置绘图风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

# 定义20个面的颜色方案
FACE_COLORS = [
    '#9bbf8a', '#93baa5', '#8ab4bf', '#82afda', '#bda09a',
    '#f79059', '#efb696', '#e7dbd3', '#c2bdde', '#8dcec8',
    '#add3e2', '#71aacd', '#3480b8', '#6790a9', '#9a9f99',
    '#ccaf8a', '#ffbe7a', '#fa8878', '#e1564e', '#c82423'
]

# 定义三组正交对角面 - 更新为包含E2和E3对应的组
ORTHOGONAL_DIAGONAL_GROUPS = {
    'E1_Group': [10, 8],  # 对E1最重要的组 (X轴)
    'E2_Group': [4, 13],  # 对E2最重要的组 (Y轴)
    'E3_Group': [16, 3]  # 对E3最重要的组 (Z轴)
}

# 定义面的法向量与各轴的关系
FACE_AXIS_PROJECTIONS = {
    1: [0, 0, 1],
    2: [0, 1, 0],
    3: [0, 0.707, -0.707],
    4: [-0.707, 0.707, 0],
    5: [-0.577, 0.577, 0.577],
    6: [-0.577, 0.577, -0.577],
    7: [1, 0, 0],  # X轴
    8: [-0.707,0, 0.707],
    9: [0.577, 0.577, -0.577],
    10: [0.707, 0, 0.707],
    11: [1, 0, 0],
    12: [0.577, 0.577, 0.577],
    13: [0.707, 0.707, 0],
    14: [-0.577, -0.577, 0.577],
    15: [0.577, -0.577, 0.577],
    16: [0, 0.707, 0.707],
    17: [0, 1, 0],  # Y轴
    18: [0.577, 0.577, 0.577],
    19: [0.577, -0.577, -0.577],
    20: [0, 0, 1]  # Z轴
}


def calculate_anisotropy(E1, E2, E3):
    """计算各向异性指数"""
    numerator = (E1 - E2) ** 2 + (E2 - E3) ** 2 + (E3 - E1) ** 2
    denominator = 2 * (E1 ** 2 + E2 ** 2 + E3 ** 2) + 1e-10
    return np.sqrt(numerator / denominator)


def decode_configurations(model, latent_vectors, device, target_mean, target_std):
    """解码潜在向量并提取构型特征"""
    model.eval()
    all_config_data = []

    with torch.no_grad():
        for i, z in enumerate(tqdm(latent_vectors, desc="解码构型")):
            z_tensor = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)

            # 使用模型的decode方法
            decoded = model.decoder(z_tensor)
            face_probs = decoded['face_probs']
            face_mask = torch.where(face_probs > 0.5,
                                    torch.ones_like(face_probs),
                                    torch.zeros_like(face_probs))

            # 使用模型的predict_properties方法
            pred_properties = model.property_heads(z_tensor, face_mask).cpu().numpy()[0]

            # 反标准化 - 6个参数：E1, E2, E3, avg_shear, num_faces, relative_density
            normalized_E1 = pred_properties[0] * target_std[0] + target_mean[0]
            normalized_E2 = pred_properties[1] * target_std[1] + target_mean[1]
            normalized_E3 = pred_properties[2] * target_std[2] + target_mean[2]
            avg_shear = pred_properties[3] * target_std[3] + target_mean[3]
            num_faces = pred_properties[4] * target_std[4] + target_mean[4]
            relative_density = pred_properties[5] * target_std[5] + target_mean[5]  # 新增相对密度

            # 计算各向异性
            anisotropy = calculate_anisotropy(normalized_E1, normalized_E2, normalized_E3)

            adj_matrix = decoded['adj_matrix'].cpu().numpy()[0]
            face_mask_np = face_mask.cpu().numpy()[0]
            face_probs_np = face_probs.cpu().numpy()[0]

            # 计算连接密度（只考虑激活的面）
            active_faces = np.where(face_mask_np > 0.5)[0]
            if len(active_faces) > 0:
                active_adj = adj_matrix[active_faces][:, active_faces]
                connection_density = np.mean(active_adj)
            else:
                connection_density = 0

            config_data = {
                'latent_vector': z,
                'adj_matrix': adj_matrix,
                'face_mask': face_mask_np,
                'face_probs': face_probs_np,
                'normalized_E1': normalized_E1,
                'normalized_E2': normalized_E2,
                'normalized_E3': normalized_E3,
                'avg_shear': avg_shear,
                'num_faces': num_faces,
                'relative_density': relative_density,  # 新增相对密度
                'anisotropy': anisotropy,
                'connection_density': connection_density,
                'active_faces': active_faces
            }
            all_config_data.append(config_data)

    return all_config_data


def generate_configurations(model, num_samples, device, target_mean, target_std, batch_size=1000):
    """批量生成构型以避免内存问题"""
    print(f"生成 {num_samples} 个构型...")
    latent_vectors = np.random.randn(num_samples, config.latent_dim)
    all_configs = []

    for i in tqdm(range(0, num_samples, batch_size), desc="批量生成"):
        batch_vectors = latent_vectors[i:i + batch_size]
        batch_configs = decode_configurations(model, batch_vectors, device, target_mean, target_std)
        all_configs.extend(batch_configs)

    return all_configs


def find_extreme_configurations(all_config_data, e1_threshold=0.995, ani_threshold=0.995):
    """找出极端构型：高E1、低各向异性、高各向异性"""
    E1_vals = np.array([c['normalized_E1'] for c in all_config_data])
    E2_vals = np.array([c['normalized_E2'] for c in all_config_data])
    E3_vals = np.array([c['normalized_E3'] for c in all_config_data])
    anisotropy_vals = np.array([c['anisotropy'] for c in all_config_data])

    # 找出E1最高的构型 (前0.5%)
    e1_sorted_indices = np.argsort(E1_vals)[::-1]
    high_e1_indices = e1_sorted_indices[:int(len(E1_vals) * (1 - e1_threshold))]

    # 找出E2最高的构型 (前0.5%)
    e2_sorted_indices = np.argsort(E2_vals)[::-1]
    high_e2_indices = e2_sorted_indices[:int(len(E2_vals) * (1 - e1_threshold))]

    # 找出E3最高的构型 (前0.5%)
    e3_sorted_indices = np.argsort(E3_vals)[::-1]
    high_e3_indices = e3_sorted_indices[:int(len(E3_vals) * (1 - e1_threshold))]

    # 找出各向异性最低的构型 (最接近各向同性，前0.5%)
    ani_sorted_indices = np.argsort(anisotropy_vals)
    low_ani_indices = ani_sorted_indices[:int(len(anisotropy_vals) * (1 - ani_threshold))]

    # 找出各向异性最高的构型 (前0.5%)
    high_ani_indices = ani_sorted_indices[::-1][:int(len(anisotropy_vals) * (1 - ani_threshold))]

    # 找出E1最低的构型 (后0.5%)
    low_e1_indices = e1_sorted_indices[::-1][:int(len(E1_vals) * (1 - e1_threshold))]

    return {
        'high_e1': high_e1_indices,
        'high_e2': high_e2_indices,
        'high_e3': high_e3_indices,
        'low_e1': low_e1_indices,
        'low_ani': low_ani_indices,
        'high_ani': high_ani_indices
    }


def analyze_high_stiffness_designs(extreme_configs, all_config_data, stiffness_type='E1'):
    """深入分析高刚度设计模式"""
    print(f"\n=== {stiffness_type}高刚度设计深度分析 ===")

    if f'high_{stiffness_type.lower()}' not in extreme_configs or len(
            extreme_configs[f'high_{stiffness_type.lower()}']) == 0:
        print(f"没有{stiffness_type}高刚度设计样本")
        return

    high_stiffness_indices = extreme_configs[f'high_{stiffness_type.lower()}']
    high_stiffness_configs = [all_config_data[i] for i in high_stiffness_indices]

    # 提取面激活模式作为特征
    activation_patterns = []
    for config in high_stiffness_configs:
        pattern = [1 if i in config['active_faces'] else 0 for i in range(20)]
        activation_patterns.append(pattern)

    activation_patterns = np.array(activation_patterns)

    # 使用聚类算法识别不同的高刚度模式
    print(f"进行{stiffness_type}高刚度设计聚类分析...")
    kmeans = KMeans(n_clusters=min(5, len(high_stiffness_configs)), random_state=42)
    clusters = kmeans.fit_predict(activation_patterns)

    # 分析每个簇的特征
    cluster_results = {}
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_configs = [high_stiffness_configs[i] for i in cluster_indices]

        # 计算该簇中每个面的激活频率
        face_activation = np.zeros(20)
        for config in cluster_configs:
            for face_id in config['active_faces']:
                face_activation[face_id] += 1
        face_activation /= len(cluster_configs)

        # 计算性能统计
        e1_values = [c['normalized_E1'] for c in cluster_configs]
        e2_values = [c['normalized_E2'] for c in cluster_configs]
        e3_values = [c['normalized_E3'] for c in cluster_configs]
        ani_values = [c['anisotropy'] for c in cluster_configs]

        cluster_results[cluster_id] = {
            'size': len(cluster_configs),
            'face_activation': face_activation,
            f'avg_{stiffness_type}': np.mean(e1_values) if stiffness_type == 'E1' else
            (np.mean(e2_values) if stiffness_type == 'E2' else np.mean(e3_values)),
            'avg_anisotropy': np.mean(ani_values),
            'configs': cluster_configs
        }

    # 打印聚类结果
    for cluster_id, result in cluster_results.items():
        print(f"\n{stiffness_type}高刚度设计模式 {cluster_id + 1} (样本数: {result['size']})")
        print(
            f"平均{stiffness_type}: {result[f'avg_{stiffness_type}']:.3f}, 平均各向异性: {result['avg_anisotropy']:.3f}")

        # 找出关键面（激活频率 > 0.7）
        key_faces = [i + 1 for i, freq in enumerate(result['face_activation']) if freq > 0.7]
        print(f"关键激活面: {sorted(key_faces)}")

        # 打印所有面的激活频率
        print("面激活频率:")
        for i in range(20):
            if result['face_activation'][i] > 0.1:  # 只显示激活频率大于10%的面
                print(f"面 {i + 1}: {result['face_activation'][i]:.3f}")

    return cluster_results


def analyze_anisotropy_designs(extreme_configs, all_config_data, dataset):
    """分析各向异性设计模式"""
    print("\n=== 各向异性设计分析 ===")

    if 'high_ani' not in extreme_configs or len(extreme_configs['high_ani']) == 0:
        print("没有高各向异性设计样本")
        return

    high_ani_indices = extreme_configs['high_ani']
    high_ani_configs = [all_config_data[i] for i in high_ani_indices]

    # 分析高各向异性设计的特征
    activation_patterns = []
    for config in high_ani_configs:
        pattern = [1 if i in config['active_faces'] else 0 for i in range(20)]
        activation_patterns.append(pattern)

    activation_patterns = np.array(activation_patterns)

    # 计算每个面的激活频率
    face_activation = np.zeros(20)
    for config in high_ani_configs:
        for face_id in config['active_faces']:
            face_activation[face_id] += 1
    face_activation /= len(high_ani_configs)

    # 找出关键面
    key_faces = [i + 1 for i, freq in enumerate(face_activation) if freq > 0.3]  # 降低阈值到0.3
    print(f"高各向异性设计的关键激活面: {sorted(key_faces)}")

    return high_ani_configs


def find_sweet_spot_designs(all_config_data, e1_range=(0.4, 0.6), ani_range=(0.0, 0.3)):
    """寻找甜蜜点设计（中等刚度且低各向异性）"""
    print("\n=== 寻找甜蜜点设计 ===")

    sweet_spot_configs = []
    for config in all_config_data:
        if (e1_range[0] <= config['normalized_E1'] <= e1_range[1] and
                ani_range[0] <= config['anisotropy'] <= ani_range[1]):
            sweet_spot_configs.append(config)

    print(f"找到 {len(sweet_spot_configs)} 个甜蜜点设计")

    if sweet_spot_configs:
        # 分析甜蜜点设计的特征
        face_activation = np.zeros(20)
        for config in sweet_spot_configs:
            for face_id in config['active_faces']:
                face_activation[face_id] += 1
        face_activation /= len(sweet_spot_configs)

        key_faces = [i + 1 for i, freq in enumerate(face_activation) if freq > 0.3]  # 降低阈值到0.3
        print(f"甜蜜点设计的关键激活面: {sorted(key_faces)}")

    return sweet_spot_configs


def enhanced_interpolation_analysis(model, all_configs, extreme_configs, device, target_mean, target_std, steps=100):
    """增强的插值分析，探索设计空间的连续性"""
    print("\n=== 插值分析 ===")

    # 检查是否有足够的极端配置
    if (len(extreme_configs.get('high_e1', [])) == 0 or
            len(extreme_configs.get('low_ani', [])) == 0):
        print("缺少必要的极端配置，跳过插值分析")
        return []

    # 选择高E1和低各向异性的配置进行插值
    high_e1_idx = extreme_configs['high_e1'][0]
    low_ani_idx = extreme_configs['low_ani'][0]

    z_high_e1 = all_configs[high_e1_idx]['latent_vector']
    z_low_ani = all_configs[low_ani_idx]['latent_vector']

    # 生成插值路径
    interpolation_path = []
    for alpha in np.linspace(0, 1, steps):
        z_interp = (1 - alpha) * z_high_e1 + alpha * z_low_ani
        interpolation_path.append(z_interp)

    # 解码插值路径
    interp_configs = decode_configurations(model, interpolation_path, device, target_mean, target_std)

    # 分析性能变化
    e1_values = [c['normalized_E1'] for c in interp_configs]
    ani_values = [c['anisotropy'] for c in interp_configs]

    print(f"插值路径性能变化: E1 {min(e1_values):.3f}-{max(e1_values):.3f}, "
          f"各向异性 {min(ani_values):.3f}-{max(ani_values):.3f}")

    return interp_configs


def visualize_face_in_cube(dataset, face_ids, save_path="design_space_analysis/face_visualization"):
    """可视化指定面在立方体中的位置和方向"""
    os.makedirs(save_path, exist_ok=True)

    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制立方体
    cube_vertices = BASE_VERTICES
    for i in range(len(ORIGINAL_FACES)):
        face_vertices = [cube_vertices[vertex_idx] for vertex_idx in ORIGINAL_FACES[i][:3]]
        poly = Poly3DCollection([face_vertices], alpha=0.1, linewidths=1, edgecolor='gray')
        poly.set_facecolor('lightgray')
        ax.add_collection3d(poly)

    # 绘制指定的面
    for face_id in face_ids:
        if face_id <= len(ORIGINAL_FACES):
            face_vertices = [cube_vertices[vertex_idx] for vertex_idx in ORIGINAL_FACES[face_id - 1]]
            poly = Poly3DCollection([face_vertices], alpha=0.8, linewidths=2, edgecolor=FACE_COLORS[face_id - 1])
            poly.set_facecolor(FACE_COLORS[face_id - 1])
            ax.add_collection3d(poly)

            # 获取面的几何特征
            if face_id in dataset.face_id_to_features:
                face_features = dataset.face_id_to_features[face_id]
                normal_vector = face_features[:3]
                centroid = face_features[5:8]

                # 绘制法向量
                scale = 0.3
                ax.quiver(centroid[0], centroid[1], centroid[2],
                          normal_vector[0] * scale, normal_vector[1] * scale, normal_vector[2] * scale,
                          color=FACE_COLORS[face_id - 1], linewidth=3, label=f'Face {face_id} Normal')

    # 设置坐标轴
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])
    ax.set_zlim([-0.1, 1.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Faces {face_ids} in Unit Cube with Normal Vectors')
    ax.legend()

    # 保存图像
    face_ids_str = '_'.join(map(str, face_ids))
    plt.savefig(f"{save_path}/face_{face_ids_str}_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"面{face_ids}可视化已保存至 {save_path}/face_{face_ids_str}_visualization.png")


def analyze_face_connectivity(dataset, save_path="design_space_analysis/face_connectivity"):
    """分析面连接关系"""
    os.makedirs(save_path, exist_ok=True)

    # 创建DynamicDecoder实例来获取预计算的邻接矩阵
    from model import DynamicDecoder
    decoder = DynamicDecoder()
    adj_matrix = decoder.precomputed_adjacency.numpy()

    # 创建面连接图
    G = nx.Graph()

    # 添加节点
    for i in range(20):
        G.add_node(i + 1, pos=(np.random.rand(), np.random.rand()))

    # 添加边
    for i in range(20):
        for j in range(i + 1, 20):
            if adj_matrix[i, j] > 0.5:
                G.add_edge(i + 1, j + 1)

    # 绘制网络图
    plt.figure(figsize=(12, 10))

    # 设置节点颜色：使用指定的颜色方案
    node_colors = []
    for node in G.nodes():
        node_colors.append(FACE_COLORS[node - 1])

    # 设置节点大小：重要面更大
    node_sizes = []
    for node in G.nodes():
        if node in [8, 10, 11, 13, 5, 19]:  # 三组正交对角面
            node_sizes.append(800)
        else:
            node_sizes.append(300)

    # 绘制网络
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title('Face Connectivity Network')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{save_path}/face_connectivity_network.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 分析三组正交对角面的连接性
    connectivity_results = {}
    for group_name, face_ids in ORTHOGONAL_DIAGONAL_GROUPS.items():
        neighbors = set()
        for face_id in face_ids:
            face_neighbors = list(G.neighbors(face_id))
            neighbors.update(face_neighbors)

        # 移除组内面
        neighbors = neighbors - set(face_ids)
        connectivity_results[group_name] = {
            'faces': face_ids,
            'neighbors': sorted(neighbors),
            'common_neighbors': sorted(neighbors)  # 这里简化处理，实际应该找共同邻居
        }

    # 保存连接性分析结果
    with open(f"{save_path}/face_connectivity_analysis.txt", "w") as f:
        f.write("面连接性分析结果\n")
        f.write("================\n\n")

        for group_name, result in connectivity_results.items():
            f.write(f"{group_name} (面{result['faces']}):\n")
            f.write(f"  连接面: {result['neighbors']}\n")
            f.write(f"  共同连接面: {result['common_neighbors']}\n\n")

    print(f"面连接性分析已保存至 {save_path}/face_connectivity_analysis.txt")
    return connectivity_results


def extract_face_designs(all_config_data, face_ids, save_path="design_space_analysis/face_designs"):
    """提取指定面激活时的典型构型"""
    os.makedirs(save_path, exist_ok=True)

    face_ids_str = '_'.join(map(str, face_ids))
    save_path = f"{save_path}/face_{face_ids_str}"
    os.makedirs(save_path, exist_ok=True)

    # 找出同时激活指定面的构型
    face_configs = []
    for config in all_config_data:
        active = True
        for face_id in face_ids:
            if (face_id - 1) not in config['active_faces']:
                active = False
                break
        if active:
            face_configs.append(config)

    print(f"找到 {len(face_configs)} 个同时激活面{face_ids}的构型")

    if not face_configs:
        print(f"没有找到同时激活面{face_ids}的构型")
        return

    # 分析这些构型的性能
    e1_values = [c['normalized_E1'] for c in face_configs]
    e2_values = [c['normalized_E2'] for c in face_configs]
    e3_values = [c['normalized_E3'] for c in face_configs]
    ani_values = [c['anisotropy'] for c in face_configs]
    num_faces = [len(c['active_faces']) for c in face_configs]

    # 找出性能最好的5个构型
    if 'E1' in save_path:
        sorted_indices = np.argsort(e1_values)[::-1][:5]  # 按E1降序排列
    elif 'E2' in save_path:
        sorted_indices = np.argsort(e2_values)[::-1][:5]  # 按E2降序排列
    elif 'E3' in save_path:
        sorted_indices = np.argsort(e3_values)[::-1][:5]  # 按E3降序排列
    else:
        sorted_indices = np.argsort(e1_values)[::-1][:5]  # 默认按E1降序排列

    best_configs = [face_configs[i] for i in sorted_indices]

    # 保存构型信息
    with open(f"{save_path}/face_{face_ids_str}_configs_info.txt", "w") as f:
        f.write(f"同时激活面{face_ids}的构型分析\n")
        f.write("=========================\n\n")
        f.write(f"总构型数: {len(face_configs)}\n")
        f.write(f"平均E1: {np.mean(e1_values):.3f}\n")
        f.write(f"平均E2: {np.mean(e2_values):.3f}\n")
        f.write(f"平均E3: {np.mean(e3_values):.3f}\n")
        f.write(f"平均各向异性: {np.mean(ani_values):.3f}\n")
        f.write(f"平均面数: {np.mean(num_faces):.1f}\n\n")

        f.write("性能最好的5个构型:\n")
        for i, config in enumerate(best_configs):
            f.write(f"\n构型 {i + 1}:\n")
            f.write(f"  E1: {config['normalized_E1']:.3f}\n")
            f.write(f"  E2: {config['normalized_E2']:.3f}\n")
            f.write(f"  E3: {config['normalized_E3']:.3f}\n")
            f.write(f"  各向异性: {config['anisotropy']:.3f}\n")
            f.write(f"  面数: {len(config['active_faces'])}\n")
            f.write(f"  激活的面: {sorted([x + 1 for x in config['active_faces']])}\n")

    # 保存构型数据用于有限元仿真
    np.save(f"{save_path}/face_{face_ids_str}_configs.npy", best_configs)

    print(f"面{face_ids}构型分析已保存至 {save_path}/")
    print(f"提取了 {len(best_configs)} 个最佳构型用于有限元仿真")


def plot_face_activation_vs_performance(all_config_data, face_ids, stiffness_type='E1',
                                        save_path="design_space_analysis/face_activation_analysis"):
    """构建指定面的激活概率与性能的关系图"""
    os.makedirs(save_path, exist_ok=True)

    face_ids_str = '_'.join(map(str, face_ids))
    save_path = f"{save_path}/face_{face_ids_str}"
    os.makedirs(save_path, exist_ok=True)

    # 提取面的激活概率以及性能指标
    face_probs = {face_id: [] for face_id in face_ids}
    e1_values = []
    e2_values = []
    e3_values = []
    ani_values = []

    for config in all_config_data:
        for face_id in face_ids:
            face_probs[face_id].append(config['face_probs'][face_id - 1])
        e1_values.append(config['normalized_E1'])
        e2_values.append(config['normalized_E2'])
        e3_values.append(config['normalized_E3'])
        ani_values.append(config['anisotropy'])

    for face_id in face_ids:
        face_probs[face_id] = np.array(face_probs[face_id])

    e1_values = np.array(e1_values)
    e2_values = np.array(e2_values)
    e3_values = np.array(e3_values)
    ani_values = np.array(ani_values)

    # 选择要分析的刚度类型
    if stiffness_type == 'E1':
        stiffness_values = e1_values
    elif stiffness_type == 'E2':
        stiffness_values = e2_values
    elif stiffness_type == 'E3':
        stiffness_values = e3_values
    else:
        stiffness_values = e1_values

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 第一个面的激活概率与刚度的关系
    sc1 = axes[0, 0].scatter(face_probs[face_ids[0]], stiffness_values,
                             alpha=0.6, s=10, c=face_probs[face_ids[0]], cmap='Reds')
    axes[0, 0].set_xlabel(f'Face {face_ids[0]} Activation Probability')
    axes[0, 0].set_ylabel(f'Normalized {stiffness_type}')
    axes[0, 0].set_title(f'Face {face_ids[0]} Activation vs {stiffness_type}')
    plt.colorbar(sc1, ax=axes[0, 0], label=f'Face {face_ids[0]} Probability')

    # 第一个面的激活概率与各向异性的关系
    sc2 = axes[0, 1].scatter(face_probs[face_ids[0]], ani_values,
                             alpha=0.6, s=10, c=face_probs[face_ids[0]], cmap='Reds')
    axes[0, 1].set_xlabel(f'Face {face_ids[0]} Activation Probability')
    axes[0, 1].set_ylabel('Anisotropy Index')
    axes[0, 1].set_title(f'Face {face_ids[0]} Activation vs Anisotropy')
    plt.colorbar(sc2, ax=axes[0, 1], label=f'Face {face_ids[0]} Probability')

    # 第二个面的激活概率与刚度的关系
    sc3 = axes[1, 0].scatter(face_probs[face_ids[1]], stiffness_values,
                             alpha=0.6, s=10, c=face_probs[face_ids[1]], cmap='Blues')
    axes[1, 0].set_xlabel(f'Face {face_ids[1]} Activation Probability')
    axes[1, 0].set_ylabel(f'Normalized {stiffness_type}')
    axes[1, 0].set_title(f'Face {face_ids[1]} Activation vs {stiffness_type}')
    plt.colorbar(sc3, ax=axes[1, 0], label=f'Face {face_ids[1]} Probability')

    # 第二个面的激活概率与各向异性的关系
    sc4 = axes[1, 1].scatter(face_probs[face_ids[1]], ani_values,
                             alpha=0.6, s=10, c=face_probs[face_ids[1]], cmap='Blues')
    axes[1, 1].set_xlabel(f'Face {face_ids[1]} Activation Probability')
    axes[1, 1].set_ylabel('Anisotropy Index')
    axes[1, 1].set_title(f'Face {face_ids[1]} Activation vs Anisotropy')
    plt.colorbar(sc4, ax=axes[1, 1], label=f'Face {face_ids[1]} Probability')

    plt.tight_layout()
    plt.savefig(f"{save_path}/face_{face_ids_str}_activation_vs_{stiffness_type.lower()}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    # 计算相关系数
    corr_face1_stiffness = pearsonr(face_probs[face_ids[0]], stiffness_values)[0]
    corr_face1_ani = pearsonr(face_probs[face_ids[0]], ani_values)[0]
    corr_face2_stiffness = pearsonr(face_probs[face_ids[1]], stiffness_values)[0]
    corr_face2_ani = pearsonr(face_probs[face_ids[1]], ani_values)[0]

    # 保存相关系数
    with open(f"{save_path}/face_{face_ids_str}_activation_correlations.txt", "w") as f:
        f.write(f"面激活概率与{stiffness_type}性能指标的相关性分析\n")
        f.write("============================\n\n")
        f.write(f"面{face_ids[0]}激活概率与{stiffness_type}的相关系数: {corr_face1_stiffness:.3f}\n")
        f.write(f"面{face_ids[0]}激活概率与各向异性的相关系数: {corr_face1_ani:.3f}\n")
        f.write(f"面{face_ids[1]}激活概率与{stiffness_type}的相关系数: {corr_face2_stiffness:.3f}\n")
        f.write(f"面{face_ids[1]}激活概率与各向异性的相关系数: {corr_face2_ani:.3f}\n")

    print(f"面{face_ids}激活概率与性能关系图已保存至 {save_path}/")
    print(f"相关性分析结果已保存至 {save_path}/face_{face_ids_str}_activation_correlations.txt")


def analyze_top_e1_configurations(all_config_data, top_k=1000):
    """分析前top_k个E1最好构型的面激活模式和共现关系"""
    print(f"\n=== 分析前{top_k}个E1最好构型的面激活模式 ===")

    # 按E1值排序，取前top_k个
    e1_values = np.array([c['normalized_E1'] for c in all_config_data])
    top_indices = np.argsort(e1_values)[-top_k:]
    top_configs = [all_config_data[i] for i in top_indices]

    # 统计每个面的激活频率
    face_activation_freq = np.zeros(20)
    face_cooccurrence = np.zeros((20, 20))  # 共现矩阵

    for config in top_configs:
        active_faces = config['active_faces']
        for face_id in active_faces:
            face_activation_freq[face_id] += 1
            # 统计共现关系
            for other_face in active_faces:
                if face_id != other_face:
                    face_cooccurrence[face_id, other_face] += 1

    # 归一化
    face_activation_freq /= len(top_configs)

    # 计算平均面数
    avg_faces = np.mean([len(c['active_faces']) for c in top_configs])

    print(f"前{top_k}个高E1构型的分析结果:")
    print(f"平均激活面数: {avg_faces:.2f}")
    print(f"面激活频率 (降序排列):")

    # 按激活频率排序
    face_freq_sorted = sorted([(i + 1, freq) for i, freq in enumerate(face_activation_freq)],
                              key=lambda x: x[1], reverse=True)

    for face_id, freq in face_freq_sorted:
        if freq > 0.1:  # 只显示激活频率大于10%的面
            print(f"面 {face_id}: {freq:.3f}")

    # 分析最重要的连接关系
    print(f"\n最重要的面共现关系 (前20):")
    cooccurrence_pairs = []
    for i in range(20):
        for j in range(i + 1, 20):
            if face_cooccurrence[i, j] > 0:
                cooccurrence_pairs.append(((i + 1, j + 1), face_cooccurrence[i, j]))

    cooccurrence_pairs.sort(key=lambda x: x[1], reverse=True)
    for (face1, face2), count in cooccurrence_pairs[:20]:
        freq = count / len(top_configs)
        print(f"面 {face1}-面 {face2}: {freq:.3f}")

    return {
        'top_configs': top_configs,
        'face_activation_freq': face_activation_freq,
        'face_cooccurrence': face_cooccurrence,
        'avg_faces': avg_faces,
        'face_freq_sorted': face_freq_sorted,
        'cooccurrence_pairs': cooccurrence_pairs
    }


def extract_top_e1_density_data(all_config_data, top_k=10000, save_path="design_space_analysis"):
    """提取前top_k个高E1构型的E1和相对密度数据，用于Ashby图"""
    print(f"\n=== 提取前{top_k}个高E1构型的密度数据 ===")

    # 按E1值排序，取前top_k个
    e1_values = np.array([c['normalized_E1'] for c in all_config_data])
    top_indices = np.argsort(e1_values)[-top_k:]
    top_configs = [all_config_data[i] for i in top_indices]

    # 提取E1和相对密度数据
    data = []
    for i, config in enumerate(top_configs):
        data.append({
            'index': top_indices[i],
            'normalized_E1': config['normalized_E1'],
            'relative_density': config.get('relative_density', 0)  # 使用get方法避免KeyError
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 按E1值降序排列
    df = df.sort_values('normalized_E1', ascending=False).reset_index(drop=True)

    # 保存为CSV和Excel文件
    csv_path = f"{save_path}/top_{top_k}_e1_density_data.csv"
    excel_path = f"{save_path}/top_{top_k}_e1_density_data.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)

    print(f"已保存前{top_k}个高E1构型的密度数据:")
    print(f"CSV文件: {csv_path}")
    print(f"Excel文件: {excel_path}")
    print(f"数据统计 - E1范围: {df['normalized_E1'].min():.3f} - {df['normalized_E1'].max():.3f}")
    print(f"数据统计 - 相对密度范围: {df['relative_density'].min():.3f} - {df['relative_density'].max():.3f}")
    print(f"数据统计 - 平均E1: {df['normalized_E1'].mean():.3f}")
    print(f"数据统计 - 平均相对密度: {df['relative_density'].mean():.3f}")

    return df


def visualize_top_e1_network_analysis(analysis_results, top_k=1000,
                                      save_path="design_space_analysis/top_e1_network"):
    """可视化前 top_k 个 E1 构型的网络分析 - 专注于网络图"""
    import os
    os.makedirs(save_path, exist_ok=True)

    face_activation_freq = analysis_results['face_activation_freq']
    face_cooccurrence = analysis_results['face_cooccurrence']
    cooccurrence_pairs = analysis_results['cooccurrence_pairs']
    face_freq_sorted = analysis_results['face_freq_sorted']

    # 创建网络图
    G = nx.Graph()

    # 添加节点
    for i in range(20):
        face_id = i + 1
        G.add_node(face_id,
                   activation_freq=face_activation_freq[i],
                   importance=face_activation_freq[i])

    # 添加边 - 使用真实统计数据
    significant_edges = []

    for (face1, face2), count in cooccurrence_pairs:
        freq = count / top_k
        # 根据统计显著性调整阈值
        if freq > 0.001:  # 降低阈值以显示更多真实连接
            G.add_edge(face1, face2, weight=freq, count=count)
            significant_edges.append(((face1, face2), freq))

    # 按共现频率排序边
    significant_edges.sort(key=lambda x: x[1], reverse=True)

    # 创建图形 - 使用更大的画布
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))

    # === 网络图可视化 ===

    # 使用力导向布局，但优化参数以获得更好的分布
    pos = nx.spring_layout(G, k=3, iterations=200, seed=42, scale=2)

    # 特别调整核心面的位置使其更突出
    if 8 in pos and 10 in pos:
        # 将面8和10放在更中心的位置
        center_x = np.mean([pos[i][0] for i in pos])
        center_y = np.mean([pos[i][1] for i in pos])
        pos[8] = np.array([center_x - 0.8, center_y + 0.3])
        pos[10] = np.array([center_x + 0.8, center_y + 0.3])
        # 重新计算布局，固定核心面位置
        pos = nx.spring_layout(G, pos=pos, fixed=[8, 10], k=3, iterations=100, seed=42)

    # 节点大小基于激活频率 - 增强视觉效果
    node_sizes = []
    for i in range(20):
        face_id = i + 1
        base_size = 1200

        # 根据重要性调整大小
        if face_id in [8, 10]:  # E1核心面
            size_multiplier = 3.5
        elif face_id in [4, 13, 16, 3]:  # 其他方向的核心面
            size_multiplier = 2.2
        elif face_activation_freq[i] > 0.5:  # 高频面
            size_multiplier = 1.8
        elif face_activation_freq[i] > 0.3:  # 中频面
            size_multiplier = 1.3
        else:
            size_multiplier = 1.0

        scaled_size = base_size * (face_activation_freq[i] ** 0.6) * size_multiplier
        node_sizes.append(max(400, scaled_size))

    # 节点颜色统一为 #e7dbd3
    node_colors = ['#e7dbd3'] * 20

    # 绘制网络图 - 按重要性顺序绘制避免遮挡
    # 1. 先绘制低频边
    low_freq_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] <= 0.15]
    medium_freq_edges = [(u, v) for u, v in G.edges() if 0.15 < G[u][v]['weight'] <= 0.3]
    high_freq_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] > 0.3]
    core_edges = [(u, v) for u, v in G.edges() if set([u, v]) == {8, 10} or
                  (u in [8, 10] and v in [8, 10, 4, 13, 16, 3])]

    # 绘制边（从低频到高频）
    all_edges = low_freq_edges + medium_freq_edges + high_freq_edges + core_edges
    for u, v in all_edges:
        freq = G[u][v]['weight']
        edge_width = freq * 25 + 1

        # 修正边颜色逻辑
        if set([u, v]) == {8, 10}:  # 8和10之间的边只用红色
            edge_color = '#c82423'
            edge_alpha = 0.95
            style = 'solid'
        elif (u in [8, 10] and v in [4, 13, 16, 3]) or (v in [8, 10] and u in [4, 13, 16, 3]):
            edge_color = '#efb696'  # 橙色 - 核心面与其他核心面的连接
            edge_alpha = 0.8
            style = 'solid'
        elif u in [8, 10] or v in [8, 10]:  # 与核心面连接
            edge_color = '#add3e2'  # 橙色
            edge_alpha = 0.7
            style = 'solid'
        elif freq > 0.3:  # 高频连接 - 确保使用蓝色
            edge_color = '#add3e2'  # 蓝色
            edge_alpha = 0.6
            style = 'solid'
        elif freq > 0.15:  # 中频连接
            edge_color = '#c2bdde'  # 紫色
            edge_alpha = 0.5
            style = 'solid'
        else:  # 低频连接 - 灰色半透明
            edge_color = '#9a9f99'  # 灰色
            edge_alpha = 0.2  # 半透明
            style = 'dashed'

        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               width=edge_width,
                               edge_color=edge_color,
                               alpha=edge_alpha,
                               style=style,
                               ax=ax)

    # 2. 绘制节点（从次要到重要）
    nodes_by_importance = sorted(G.nodes(), key=lambda x: face_activation_freq[x - 1])
    for node in nodes_by_importance:
        face_id = node
        node_size = node_sizes[face_id - 1]
        node_color = node_colors[face_id - 1]

        nx.draw_networkx_nodes(G, pos, nodelist=[node],
                               node_size=node_size,
                               node_color=node_color,
                               alpha=0.9,
                               edgecolors='black',
                               linewidths=2,
                               ax=ax)

    # 3. 绘制数字标签 - 放大加粗
    labels = {}
    for face_id in G.nodes():
        labels[face_id] = f"{face_id}"

    nx.draw_networkx_labels(G, pos, labels,
                            font_size=16,  # 从11增加到14
                            font_weight='bold',
                            font_family='Arial',
                            ax=ax)

    # 设置标题和样式 - 进一步放大标题
    ax.set_title(f'Top {top_k} High-E1 Configurations: Face Activation Network',
                 fontsize=28, fontweight='bold', pad=30)  # 从22增加到26

    ax.axis('off')

    # 简化的图例 - 只保留边类型，放大字体
    legend_elements = [
        plt.Line2D([0], [0], color='#c82423', linewidth=4, label='Core Connection (F8-F10)'),
        plt.Line2D([0], [0], color='#efb696', linewidth=3, label='Connections to E1 Core'),
        plt.Line2D([0], [0], color='#add3e2', linewidth=2, label='High Frequency (>0.3)'),
        plt.Line2D([0], [0], color='#c2bdde', linewidth=1.5, label='Medium Frequency (0.15-0.3)'),
        plt.Line2D([0], [0], color='#9a9f99', linewidth=1, linestyle='dashed', alpha=0.3, label='Low Frequency (<0.15)'),
    ]

    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95,
              fontsize=15, title="Edge Types", title_fontsize=16)  # 从12增加到14，标题从13增加到15

    plt.tight_layout()
    plt.savefig(f"{save_path}/top_{top_k}_e1_network_detailed.png",
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Detailed high-E1 configuration network analysis saved to: {save_path}/top_{top_k}_e1_network_detailed.png")

    # 保存详细的连接分析报告
    with open(f"{save_path}/top_{top_k}_e1_connection_analysis.txt", "w") as f:
        f.write(f"Detailed Connection Analysis - Top {top_k} High-E1 Configurations\n")
        f.write("=" * 70 + "\n\n")

        f.write("FACE ACTIVATION FREQUENCIES:\n")
        f.write("-" * 40 + "\n")
        for i, (face_id, freq) in enumerate(face_freq_sorted):
            status = "CORE_E1" if face_id in [8, 10] else "CORE_OTHER" if face_id in [4, 13, 16, 3] else "AUXILIARY"
            f.write(f"Face {face_id:2d}: {freq:.3f} [{status}]\n")

        f.write(f"\nNETWORK TOPOLOGY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total nodes: {len(G.nodes())}\n")
        f.write(f"Total edges: {len(G.edges())}\n")
        f.write(f"Network density: {nx.density(G):.3f}\n")
        f.write(f"Average degree: {sum(dict(G.degree()).values()) / len(G):.2f}\n")

        f.write(f"\nCORE CONNECTIONS ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Face 8 connections: {list(G.neighbors(8))}\n")
        f.write(f"Face 10 connections: {list(G.neighbors(10))}\n")
        f.write(f"Common neighbors of 8 & 10: {list(nx.common_neighbors(G, 8, 10))}\n")

        f.write(f"\nDETAILED CO-OCCURRENCE MATRIX (Top 50):\n")
        f.write("-" * 40 + "\n")
        for i, ((face1, face2), freq) in enumerate(significant_edges[:50]):
            relationship_type = "CORE_E1" if set([face1, face2]) == {8, 10} else \
                "CORE_RELATED" if 8 in [face1, face2] or 10 in [face1, face2] else \
                    "CORE_CROSS" if (face1 in [4, 13, 16, 3] and face2 in [4, 13, 16, 3]) else \
                        "AUXILIARY"
            f.write(f"F{face1:2d}-F{face2:2d}: {freq:.3f} [{relationship_type}]\n")

        f.write(f"\nCLUSTER ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        # 计算连通分量
        components = list(nx.connected_components(G))
        f.write(f"Number of connected components: {len(components)}\n")
        for i, comp in enumerate(components):
            f.write(f"Component {i + 1}: {sorted(comp)} (size: {len(comp)})\n")

        f.write(f"\nCENTRALITY ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        f.write("Degree Centrality (Top 10):\n")
        for face, centrality in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  Face {face}: {centrality:.3f}\n")

        f.write("\nBetweenness Centrality (Top 10):\n")
        for face, centrality in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  Face {face}: {centrality:.3f}\n")

    print(f"Detailed connection analysis saved to: {save_path}/top_{top_k}_e1_connection_analysis.txt")

    return G
def analyze_face_geometry_and_importance(dataset, importance_e1, importance_e2, importance_e3, importance_ani,
                                         save_path="design_space_analysis/face_geometry_analysis"):
    """分析面的几何属性与其重要性之间的关系"""
    os.makedirs(save_path, exist_ok=True)

    # 获取每个面的几何特征
    face_features = {}
    for face_id in range(1, 21):
        if face_id in dataset.face_id_to_features:
            face_features[face_id] = dataset.face_id_to_features[face_id]

    # 提取法向量和计算在各轴上的投影
    face_normals = {}
    face_projections = {
        'x': {},
        'y': {},
        'z': {}
    }

    for face_id, features in face_features.items():
        normal_vector = features[:3]  # 单位法向量
        face_normals[face_id] = normal_vector
        # 计算在各轴上的投影分量（绝对值，因为方向不重要）
        face_projections['x'][face_id] = abs(normal_vector[0])
        face_projections['y'][face_id] = abs(normal_vector[1])
        face_projections['z'][face_id] = abs(normal_vector[2])

    # 创建DataFrame用于分析
    face_data = []
    for face_id in range(1, 21):
        face_data.append({
            'face_id': face_id,
            'importance_e1': importance_e1[face_id - 1],
            'importance_e2': importance_e2[face_id - 1],
            'importance_e3': importance_e3[face_id - 1],
            'importance_ani': importance_ani[face_id - 1],
            'x_projection': face_projections['x'].get(face_id, 0),
            'y_projection': face_projections['y'].get(face_id, 0),
            'z_projection': face_projections['z'].get(face_id, 0),
            'normal_x': face_normals.get(face_id, [0, 0, 0])[0],
            'normal_y': face_normals.get(face_id, [0, 0, 0])[1],
            'normal_z': face_normals.get(face_id, [0, 0, 0])[2],
        })

    df_faces = pd.DataFrame(face_data)

    # 计算相关性
    corr_proj_e1_x = pearsonr(df_faces['x_projection'], df_faces['importance_e1'])[0]
    corr_proj_e1_y = pearsonr(df_faces['y_projection'], df_faces['importance_e1'])[0]
    corr_proj_e1_z = pearsonr(df_faces['z_projection'], df_faces['importance_e1'])[0]

    corr_proj_e2_x = pearsonr(df_faces['x_projection'], df_faces['importance_e2'])[0]
    corr_proj_e2_y = pearsonr(df_faces['y_projection'], df_faces['importance_e2'])[0]
    corr_proj_e2_z = pearsonr(df_faces['z_projection'], df_faces['importance_e2'])[0]

    corr_proj_e3_x = pearsonr(df_faces['x_projection'], df_faces['importance_e3'])[0]
    corr_proj_e3_y = pearsonr(df_faces['y_projection'], df_faces['importance_e3'])[0]
    corr_proj_e3_z = pearsonr(df_faces['z_projection'], df_faces['importance_e3'])[0]

    corr_proj_ani_x = pearsonr(df_faces['x_projection'], df_faces['importance_ani'])[0]
    corr_proj_ani_y = pearsonr(df_faces['y_projection'], df_faces['importance_ani'])[0]
    corr_proj_ani_z = pearsonr(df_faces['z_projection'], df_faces['importance_ani'])[0]

    print(f"\nX轴投影与E1重要性的相关性: {corr_proj_e1_x:.3f}")
    print(f"Y轴投影与E2重要性的相关性: {corr_proj_e2_y:.3f}")
    print(f"Z轴投影与E3重要性的相关性: {corr_proj_e3_z:.3f}")

    # 特别分析三组正交对角面
    group_analysis = {}
    for group_name, face_ids in ORTHOGONAL_DIAGONAL_GROUPS.items():
        group_data = df_faces[df_faces['face_id'].isin(face_ids)]

        # 根据组名确定相关轴
        if group_name == 'E1_Group':
            projection_type = 'x_projection'
            importance_type = 'importance_e1'
        elif group_name == 'E2_Group':
            projection_type = 'y_projection'
            importance_type = 'importance_e2'
        elif group_name == 'E3_Group':
            projection_type = 'z_projection'
            importance_type = 'importance_e3'
        else:
            projection_type = 'x_projection'
            importance_type = 'importance_e1'

        group_analysis[group_name] = {
            'avg_projection': group_data[projection_type].mean(),
            'avg_importance': group_data[importance_type].mean(),
            'faces': group_data.to_dict('records')
        }
        print(f"\n{group_name} (面{face_ids}):")
        print(f"  平均投影: {group_analysis[group_name]['avg_projection']:.3f}")
        print(f"  平均重要性: {group_analysis[group_name]['avg_importance']:.3f}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # X轴投影与E1重要性的关系
    sc1 = axes[0, 0].scatter(df_faces['x_projection'], df_faces['importance_e1'],
                             c=df_faces['face_id'], cmap='tab20', s=100)
    axes[0, 0].set_xlabel('法向量在X轴上的投影分量 (绝对值)')
    axes[0, 0].set_ylabel('对E1的重要性')
    axes[0, 0].set_title('法向量X轴投影与E1重要性的关系')
    for i, row in df_faces.iterrows():
        axes[0, 0].annotate(row['face_id'], (row['x_projection'], row['importance_e1']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].text(0.05, 0.95, f'相关系数: {corr_proj_e1_x:.3f}',
                    transform=axes[0, 0].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Y轴投影与E2重要性的关系
    sc2 = axes[0, 1].scatter(df_faces['y_projection'], df_faces['importance_e2'],
                             c=df_faces['face_id'], cmap='tab20', s=100)
    axes[0, 1].set_xlabel('法向量在Y轴上的投影分量 (绝对值)')
    axes[0, 1].set_ylabel('对E2的重要性')
    axes[0, 1].set_title('法向量Y轴投影与E2重要性的关系')
    for i, row in df_faces.iterrows():
        axes[0, 1].annotate(row['face_id'], (row['y_projection'], row['importance_e2']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].text(0.05, 0.95, f'相关系数: {corr_proj_e2_y:.3f}',
                    transform=axes[0, 1].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Z轴投影与E3重要性的关系
    sc3 = axes[1, 0].scatter(df_faces['z_projection'], df_faces['importance_e3'],
                             c=df_faces['face_id'], cmap='tab20', s=100)
    axes[1, 0].set_xlabel('法向量在Z轴上的投影分量 (绝对值)')
    axes[1, 0].set_ylabel('对E3的重要性')
    axes[1, 0].set_title('法向量Z轴投影与E3重要性的关系')
    for i, row in df_faces.iterrows():
        axes[1, 0].annotate(row['face_id'], (row['z_projection'], row['importance_e3']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].text(0.05, 0.95, f'相关系数: {corr_proj_e3_z:.3f}',
                    transform=axes[1, 0].transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 三组正交对角面的比较
    group_colors = {'E1_Group': 'red', 'E2_Group': 'blue', 'E3_Group': 'green'}
    for group_name, group_info in group_analysis.items():
        if group_name == 'E1_Group':
            x = group_info['avg_projection']
            y = group_info['avg_importance']
        elif group_name == 'E2_Group':
            x = group_info['avg_projection']
            y = group_info['avg_importance']
        elif group_name == 'E3_Group':
            x = group_info['avg_projection']
            y = group_info['avg_importance']
        else:
            continue

        axes[1, 1].scatter(x, y, color=group_colors[group_name], s=200, label=group_name)
        axes[1, 1].annotate(group_name, (x, y), xytext=(10, 10),
                            textcoords='offset points', fontsize=12,
                            bbox=dict(boxstyle='round', facecolor=group_colors[group_name], alpha=0.3))

    axes[1, 1].set_xlabel('平均投影分量')
    axes[1, 1].set_ylabel('平均重要性')
    axes[1, 1].set_title('三组正交对角面的比较')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/face_geometry_vs_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 保存分析结果
    df_faces.to_csv(f"{save_path}/face_geometry_analysis.csv", index=False)

    with open(f"{save_path}/face_geometry_analysis_report.txt", "w") as f:
        f.write("面几何属性与重要性分析报告\n")
        f.write("==========================\n\n")
        f.write(f"法向量X轴投影与E1重要性的相关系数: {corr_proj_e1_x:.3f}\n")
        f.write(f"法向量Y轴投影与E2重要性的相关系数: {corr_proj_e2_y:.3f}\n")
        f.write(f"法向量Z轴投影与E3重要性的相关系数: {corr_proj_e3_z:.3f}\n")
        f.write(f"法向量X轴投影与各向异性重要性的相关系数: {corr_proj_ani_x:.3f}\n\n")

        f.write("三组正交对角面分析:\n")
        for group_name, group_info in group_analysis.items():
            f.write(f"{group_name}:\n")
            f.write(f"  平均投影: {group_info['avg_projection']:.3f}\n")
            f.write(f"  平均重要性: {group_info['avg_importance']:.3f}\n")
            for face in group_info['faces']:
                f.write(
                    f"    面{face['face_id']}: 投影={face['x_projection']:.3f}/{face['y_projection']:.3f}/{face['z_projection']:.3f}, "
                    f"E1重要性={face['importance_e1']:.3f}, E2重要性={face['importance_e2']:.3f}, E3重要性={face['importance_e3']:.3f}\n")
            f.write("\n")



    return df_faces, group_analysis


def analyze_force_transmission_paths(all_config_data, dataset, save_path="design_space_analysis/force_path_analysis"):
    """分析力传递路径，特别关注三组正交对角面的作用"""
    os.makedirs(save_path, exist_ok=True)

    # 获取高刚度配置（分别针对E1、E2、E3）
    e1_values = np.array([c['normalized_E1'] for c in all_config_data])
    e2_values = np.array([c['normalized_E2'] for c in all_config_data])
    e3_values = np.array([c['normalized_E3'] for c in all_config_data])

    high_e1_indices = np.argsort(e1_values)[-1000:]  # 取E1最高的1000个配置
    high_e2_indices = np.argsort(e2_values)[-1000:]  # 取E2最高的1000个配置
    high_e3_indices = np.argsort(e3_values)[-1000:]  # 取E3最高的1000个配置

    high_e1_configs = [all_config_data[i] for i in high_e1_indices]
    high_e2_configs = [all_config_data[i] for i in high_e2_indices]
    high_e3_configs = [all_config_data[i] for i in high_e3_indices]

    # 分析这些配置中面的共现模式
    face_cooccurrence_e1 = np.zeros((20, 20))
    face_cooccurrence_e2 = np.zeros((20, 20))
    face_cooccurrence_e3 = np.zeros((20, 20))

    face_activation_freq_e1 = np.zeros(20)
    face_activation_freq_e2 = np.zeros(20)
    face_activation_freq_e3 = np.zeros(20)

    for config in high_e1_configs:
        active_faces = config['active_faces']
        for i in active_faces:
            face_activation_freq_e1[i] += 1
            for j in active_faces:
                if i != j:
                    face_cooccurrence_e1[i, j] += 1

    for config in high_e2_configs:
        active_faces = config['active_faces']
        for i in active_faces:
            face_activation_freq_e2[i] += 1
            for j in active_faces:
                if i != j:
                    face_cooccurrence_e2[i, j] += 1

    for config in high_e3_configs:
        active_faces = config['active_faces']
        for i in active_faces:
            face_activation_freq_e3[i] += 1
            for j in active_faces:
                if i != j:
                    face_cooccurrence_e3[i, j] += 1

    # 归一化
    face_activation_freq_e1 /= len(high_e1_configs)
    face_activation_freq_e2 /= len(high_e2_configs)
    face_activation_freq_e3 /= len(high_e3_configs)

    for i in range(20):
        if face_activation_freq_e1[i] > 0:
            face_cooccurrence_e1[i, :] /= face_activation_freq_e1[i]
        if face_activation_freq_e2[i] > 0:
            face_cooccurrence_e2[i, :] /= face_activation_freq_e2[i]
        if face_activation_freq_e3[i] > 0:
            face_cooccurrence_e3[i, :] /= face_activation_freq_e3[i]

    # 特别关注三组正交对角面的共现模式
    cooccurrence_results = {}
    for group_name, face_ids in ORTHOGONAL_DIAGONAL_GROUPS.items():
        if group_name == 'E1_Group':
            cooccurrence_matrix = face_cooccurrence_e1
        elif group_name == 'E2_Group':
            cooccurrence_matrix = face_cooccurrence_e2
        elif group_name == 'E3_Group':
            cooccurrence_matrix = face_cooccurrence_e3
        else:
            cooccurrence_matrix = face_cooccurrence_e1

        top_partners = {}
        for face_id in face_ids:
            face_idx = face_id - 1
            face_top_partners = np.argsort(cooccurrence_matrix[face_idx, :])[::-1][:6]  # 前6个
            top_partners[face_id] = [x + 1 for x in face_top_partners if x != face_idx]

        cooccurrence_results[group_name] = top_partners

    # 分析这些面的几何特性
    face_features = {}
    for face_id in range(1, 21):
        if face_id in dataset.face_id_to_features:
            face_features[face_id] = dataset.face_id_to_features[face_id]

    # 计算这些面的法向量与各轴的夹角
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])

    angle_with_axes = {}
    for face_id, features in face_features.items():
        normal_vector = features[:3]

        # 计算与X轴的夹角
        dot_product_x = np.dot(normal_vector, x_axis)
        magnitude_x = np.linalg.norm(normal_vector) * np.linalg.norm(x_axis)
        angle_x_rad = np.arccos(np.clip(dot_product_x / magnitude_x, -1, 1))

        # 计算与Y轴的夹角
        dot_product_y = np.dot(normal_vector, y_axis)
        magnitude_y = np.linalg.norm(normal_vector) * np.linalg.norm(y_axis)
        angle_y_rad = np.arccos(np.clip(dot_product_y / magnitude_y, -1, 1))

        # 计算与Z轴的夹角
        dot_product_z = np.dot(normal_vector, z_axis)
        magnitude_z = np.linalg.norm(normal_vector) * np.linalg.norm(z_axis)
        angle_z_rad = np.arccos(np.clip(dot_product_z / magnitude_z, -1, 1))

        angle_with_axes[face_id] = {
            'x': np.degrees(angle_x_rad),
            'y': np.degrees(angle_y_rad),
            'z': np.degrees(angle_z_rad)
        }

    # 创建力传递路径分析图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # E1相关的面与X轴的夹角
    angles_x = [angle_with_axes.get(i, {}).get('x', 0) for i in range(1, 21)]
    bars1 = axes[0].bar(range(1, 21), angles_x, alpha=0.7)
    for i, bar in enumerate(bars1):
        bar.set_color(FACE_COLORS[i])
    axes[0].set_xlabel('面ID')
    axes[0].set_ylabel('与X轴的夹角（度）')
    axes[0].set_title('面的法向量与X轴的夹角 (E1相关)')
    axes[0].set_xticks(range(1, 21))
    axes[0].grid(True, alpha=0.3)

    # E2相关的面与Y轴的夹角
    angles_y = [angle_with_axes.get(i, {}).get('y', 0) for i in range(1, 21)]
    bars2 = axes[1].bar(range(1, 21), angles_y, alpha=0.7)
    for i, bar in enumerate(bars2):
        bar.set_color(FACE_COLORS[i])
    axes[1].set_xlabel('面ID')
    axes[1].set_ylabel('与Y轴的夹角（度）')
    axes[1].set_title('面的法向量与Y轴的夹角 (E2相关)')
    axes[1].set_xticks(range(1, 21))
    axes[1].grid(True, alpha=0.3)

    # E3相关的面与Z轴的夹角
    angles_z = [angle_with_axes.get(i, {}).get('z', 0) for i in range(1, 21)]
    bars3 = axes[2].bar(range(1, 21), angles_z, alpha=0.7)
    for i, bar in enumerate(bars3):
        bar.set_color(FACE_COLORS[i])
    axes[2].set_xlabel('面ID')
    axes[2].set_ylabel('与Z轴的夹角（度）')
    axes[2].set_title('面的法向量与Z轴的夹角 (E3相关)')
    axes[2].set_xticks(range(1, 21))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/force_transmission_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 保存分析结果
    with open(f"{save_path}/force_path_analysis_report.txt", "w") as f:
        f.write("力传递路径分析报告\n")
        f.write("==================\n\n")
        f.write("基于E1、E2、E3最高的1000个配置的分析\n\n")

        for group_name, partners in cooccurrence_results.items():
            f.write(f"{group_name}:\n")
            for face_id, top_partners in partners.items():
                f.write(f"  与面{face_id}最常共现的面: {top_partners}\n")
            f.write("\n")

        f.write("这些面与各轴的夹角:\n")
        for face_id in range(1, 21):
            angles = angle_with_axes.get(face_id, {})
            f.write(
                f"面{face_id}: X={angles.get('x', 0):.1f}°, Y={angles.get('y', 0):.1f}°, Z={angles.get('z', 0):.1f}°\n")


    return cooccurrence_results, angle_with_axes


def perform_design_space_mapping(all_config_data, dataset):
    """执行设计空间映射，识别性能与拓扑之间的关系"""
    print("\n=== 设计空间映射分析 ===")

    # 准备数据
    X = []  # 面激活模式
    y_e1 = []  # E1值
    y_e2 = []  # E2值
    y_e3 = []  # E3值
    y_ani = []  # 各向异性值

    for config in all_config_data:
        activation_pattern = [1 if i in config['active_faces'] else 0 for i in range(20)]
        X.append(activation_pattern)
        y_e1.append(config['normalized_E1'])
        y_e2.append(config['normalized_E2'])
        y_e3.append(config['normalized_E3'])
        y_ani.append(config['anisotropy'])

    X = np.array(X)
    y_e1 = np.array(y_e1)
    y_e2 = np.array(y_e2)
    y_e3 = np.array(y_e3)
    y_ani = np.array(y_ani)

    # 使用随机森林分析特征重要性
    print("训练随机森林模型分析面重要性...")
    rf_e1 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_e1.fit(X, y_e1)

    rf_e2 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_e2.fit(X, y_e2)

    rf_e3 = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_e3.fit(X, y_e3)

    rf_ani = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_ani.fit(X, y_ani)

    # 获取特征重要性
    importance_e1 = rf_e1.feature_importances_
    importance_e2 = rf_e2.feature_importances_
    importance_e3 = rf_e3.feature_importances_
    importance_ani = rf_ani.feature_importances_

    # 分析面的几何属性与重要性的关系
    df_faces, group_analysis = analyze_face_geometry_and_importance(
        dataset, importance_e1, importance_e2, importance_e3, importance_ani
    )

    # 分析力传递路径
    face_cooccurrence, angle_with_axes = analyze_force_transmission_paths(
        all_config_data, dataset
    )

    # 打印最重要的面
    print("\n对E1最重要的面 (前10):")
    e1_important_faces = np.argsort(importance_e1)[::-1][:10]
    for i, face_idx in enumerate(e1_important_faces):
        print(f"{i + 1}. 面 {face_idx + 1}: {importance_e1[face_idx]:.4f}")

    print("\n对E2最重要的面 (前10):")
    e2_important_faces = np.argsort(importance_e2)[::-1][:10]
    for i, face_idx in enumerate(e2_important_faces):
        print(f"{i + 1}. 面 {face_idx + 1}: {importance_e2[face_idx]:.4f}")

    print("\n对E3最重要的面 (前10):")
    e3_important_faces = np.argsort(importance_e3)[::-1][:10]
    for i, face_idx in enumerate(e3_important_faces):
        print(f"{i + 1}. 面 {face_idx + 1}: {importance_e3[face_idx]:.4f}")

    print("\n对各向异性最重要的面 (前10):")
    ani_important_faces = np.argsort(importance_ani)[::-1][:10]
    for i, face_idx in enumerate(ani_important_faces):
        print(f"{i + 1}. 面 {face_idx + 1}: {importance_ani[face_idx]:.4f}")

    # 计算面之间的相关性
    print("\n面激活与性能的相关性分析:")
    correlations_e1 = []
    correlations_e2 = []
    correlations_e3 = []
    correlations_ani = []

    for i in range(20):
        face_activation = X[:, i]
        corr_e1 = pearsonr(face_activation, y_e1)[0]
        corr_e2 = pearsonr(face_activation, y_e2)[0]
        corr_e3 = pearsonr(face_activation, y_e3)[0]
        corr_ani = pearsonr(face_activation, y_ani)[0]
        correlations_e1.append((i + 1, corr_e1))
        correlations_e2.append((i + 1, corr_e2))
        correlations_e3.append((i + 1, corr_e3))
        correlations_ani.append((i + 1, corr_ani))

    # 按相关性排序
    correlations_e1.sort(key=lambda x: abs(x[1]), reverse=True)
    correlations_e2.sort(key=lambda x: abs(x[1]), reverse=True)
    correlations_e3.sort(key=lambda x: abs(x[1]), reverse=True)
    correlations_ani.sort(key=lambda x: abs(x[1]), reverse=True)

    print("与E1最相关的面 (前10):")
    for i, (face, corr) in enumerate(correlations_e1[:10]):
        print(f"{i + 1}. 面 {face}: {corr:.3f}")

    print("与E2最相关的面 (前10):")
    for i, (face, corr) in enumerate(correlations_e2[:10]):
        print(f"{i + 1}. 面 {face}: {corr:.3f}")

    print("与E3最相关的面 (前10):")
    for i, (face, corr) in enumerate(correlations_e3[:10]):
        print(f"{i + 1}. 面 {face}: {corr:.3f}")

    print("与各向异性最相关的面 (前10):")
    for i, (face, corr) in enumerate(correlations_ani[:10]):
        print(f"{i + 1}. 面 {face}: {corr:.3f}")

    return {
        'importance_e1': importance_e1,
        'importance_e2': importance_e2,
        'importance_e3': importance_e3,
        'importance_ani': importance_ani,
        'correlations_e1': correlations_e1,
        'correlations_e2': correlations_e2,
        'correlations_e3': correlations_e3,
        'correlations_ani': correlations_ani,
        'face_geometry_analysis': df_faces,
        'group_analysis': group_analysis,
        'face_cooccurrence': face_cooccurrence,
        'angle_with_axes': angle_with_axes
    }


def visualize_design_space(all_config_data, analysis_results, save_path="design_space_analysis"):
    """可视化设计空间和分析结果"""
    print("\n=== 生成设计空间可视化 ===")

    # 创建输出目录
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/detailed_analysis", exist_ok=True)

    # 1. E1 vs E2 vs E3 3D散点图
    e1_vals = [c['normalized_E1'] for c in all_config_data]
    e2_vals = [c['normalized_E2'] for c in all_config_data]
    e3_vals = [c['normalized_E3'] for c in all_config_data]
    ani_vals = [c['anisotropy'] for c in all_config_data]
    num_faces = [len(c['active_faces']) for c in all_config_data]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(e1_vals, e2_vals, e3_vals, c=ani_vals,
                         cmap='viridis', alpha=0.6, s=20)

    ax.set_xlabel('Normalized E1')
    ax.set_ylabel('Normalized E2')
    ax.set_zlabel('Normalized E3')
    ax.set_title('3D Design Space: E1 vs E2 vs E3 (Colored by Anisotropy)')

    cbar = plt.colorbar(scatter)
    cbar.set_label('Anisotropy Index')

    plt.tight_layout()
    plt.savefig(f"{save_path}/e1_vs_e2_vs_e3_3d.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 面重要性可视化
    importance_e1 = analysis_results['importance_e1']
    importance_e2 = analysis_results['importance_e2']
    importance_e3 = analysis_results['importance_e3']
    importance_ani = analysis_results['importance_ani']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # E1重要性 - 使用指定的颜色方案
    bars1 = axes[0, 0].bar(range(1, 21), importance_e1)
    for i, bar in enumerate(bars1):
        bar.set_color(FACE_COLORS[i])
    axes[0, 0].set_title('Face Importance for E1')
    axes[0, 0].set_xlabel('Face ID')
    axes[0, 0].set_ylabel('Importance')

    # E2重要性 - 使用指定的颜色方案
    bars2 = axes[0, 1].bar(range(1, 21), importance_e2)
    for i, bar in enumerate(bars2):
        bar.set_color(FACE_COLORS[i])
    axes[0, 1].set_title('Face Importance for E2')
    axes[0, 1].set_xlabel('Face ID')
    axes[0, 1].set_ylabel('Importance')

    # E3重要性 - 使用指定的颜色方案
    bars3 = axes[1, 0].bar(range(1, 21), importance_e3)
    for i, bar in enumerate(bars3):
        bar.set_color(FACE_COLORS[i])
    axes[1, 0].set_title('Face Importance for E3')
    axes[1, 0].set_xlabel('Face ID')
    axes[1, 0].set_ylabel('Importance')

    # 各向异性重要性 - 使用指定的颜色方案
    bars4 = axes[1, 1].bar(range(1, 21), importance_ani)
    for i, bar in enumerate(bars4):
        bar.set_color(FACE_COLORS[i])
    axes[1, 1].set_title('Face Importance for Anisotropy')
    axes[1, 1].set_xlabel('Face ID')
    axes[1, 1].set_ylabel('Importance')

    plt.tight_layout()
    plt.savefig(f"{save_path}/face_importance_all.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 性能分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # E1分布
    axes[0, 0].hist(e1_vals, bins=50, alpha=0.7)
    axes[0, 0].set_title('E1 Distribution')
    axes[0, 0].set_xlabel('Normalized E1')
    axes[0, 0].set_ylabel('Frequency')

    # E2分布
    axes[0, 1].hist(e2_vals, bins=50, alpha=0.7)
    axes[0, 1].set_title('E2 Distribution')
    axes[0, 1].set_xlabel('Normalized E2')
    axes[0, 1].set_ylabel('Frequency')

    # E3分布
    axes[1, 0].hist(e3_vals, bins=50, alpha=0.7)
    axes[1, 0].set_title('E3 Distribution')
    axes[1, 0].set_xlabel('Normalized E3')
    axes[1, 0].set_ylabel('Frequency')

    # 各向异性分布
    axes[1, 1].hist(ani_vals, bins=50, alpha=0.7)
    axes[1, 1].set_title('Anisotropy Distribution')
    axes[1, 1].set_xlabel('Anisotropy Index')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"{save_path}/performance_distribution_all.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"可视化结果已保存至 {save_path}/ 目录")
def create_additional_scatter_plots(all_config_data, save_path="design_space_analysis"):
    """创建额外的散点图来展示设计空间的不同维度"""
    print("\n=== 生成额外散点图 ===")

    # 提取数据
    e1_vals = [c['normalized_E1'] for c in all_config_data]
    e2_vals = [c['normalized_E2'] for c in all_config_data]
    e3_vals = [c['normalized_E3'] for c in all_config_data]
    ani_vals = [c['anisotropy'] for c in all_config_data]
    num_faces = [len(c['active_faces']) for c in all_config_data]
    shear_vals = [c['avg_shear'] for c in all_config_data]

    # 创建图形网格
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # 1. 各向异性 vs E3 (颜色用面数)
    ax1 = fig.add_subplot(gs[0, 0])
    scatter1 = ax1.scatter(ani_vals, e3_vals, c=num_faces,
                           cmap='viridis', alpha=0.6, s=15)
    ax1.set_xlabel('Anisotropy Index')
    ax1.set_ylabel('Normalized E3')
    ax1.set_title('Anisotropy vs E3 (Colored by Number of Faces)')
    plt.colorbar(scatter1, ax=ax1, label='Number of Faces')
    ax1.grid(True, alpha=0.3)

    # 2. 各向异性 vs 面数 (颜色用E1)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter2 = ax2.scatter(ani_vals, num_faces, c=e1_vals,
                           cmap='viridis', alpha=0.6, s=15)
    ax2.set_xlabel('Anisotropy Index')
    ax2.set_ylabel('Number of Faces')
    ax2.set_title('Anisotropy vs Number of Faces (Colored by E1)')
    plt.colorbar(scatter2, ax=ax2, label='Normalized E1')
    ax2.grid(True, alpha=0.3)

    # 3. E1 vs E3 (颜色用各向异性)
    ax3 = fig.add_subplot(gs[0, 2])
    scatter3 = ax3.scatter(e1_vals, e3_vals, c=ani_vals,
                           cmap='coolwarm', alpha=0.6, s=15)
    ax3.set_xlabel('Normalized E1')
    ax3.set_ylabel('Normalized E3')
    ax3.set_title('E1 vs E3 (Colored by Anisotropy)')
    plt.colorbar(scatter3, ax=ax3, label='Anisotropy Index')
    ax3.grid(True, alpha=0.3)

    # 4. 剪切模量 vs 各向异性 (颜色用E1)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter4 = ax4.scatter(shear_vals, ani_vals, c=e1_vals,
                           cmap='viridis', alpha=0.6, s=15)
    ax4.set_xlabel('Average Shear Modulus')
    ax4.set_ylabel('Anisotropy Index')
    ax4.set_title('Shear Modulus vs Anisotropy (Colored by E1)')
    plt.colorbar(scatter4, ax=ax4, label='Normalized E1')
    ax4.grid(True, alpha=0.3)

    # 5. E1 vs 剪切模量 (颜色用各向异性)
    ax5 = fig.add_subplot(gs[1, 1])
    scatter5 = ax5.scatter(e1_vals, shear_vals, c=ani_vals,
                           cmap='coolwarm', alpha=0.6, s=15)
    ax5.set_xlabel('Normalized E1')
    ax5.set_ylabel('Average Shear Modulus')
    ax5.set_title('E1 vs Shear Modulus (Colored by Anisotropy)')
    plt.colorbar(scatter5, ax=ax5, label='Anisotropy Index')
    ax5.grid(True, alpha=0.3)

    # 6. 面数 vs E1 (颜色用各向异性)
    ax6 = fig.add_subplot(gs[1, 2])
    scatter6 = ax6.scatter(num_faces, e1_vals, c=ani_vals,
                           cmap='coolwarm', alpha=0.6, s=15)
    ax6.set_xlabel('Number of Faces')
    ax6.set_ylabel('Normalized E1')
    ax6.set_title('Number of Faces vs E1 (Colored by Anisotropy)')
    plt.colorbar(scatter6, ax=ax6, label='Anisotropy Index')
    ax6.grid(True, alpha=0.3)

    # 7. 面数 vs E3 (颜色用各向异性)
    ax7 = fig.add_subplot(gs[2, 0])
    scatter7 = ax7.scatter(num_faces, e3_vals, c=ani_vals,
                           cmap='coolwarm', alpha=0.6, s=15)
    ax7.set_xlabel('Number of Faces')
    ax7.set_ylabel('Normalized E3')
    ax7.set_title('Number of Faces vs E3 (Colored by Anisotropy)')
    plt.colorbar(scatter7, ax=ax7, label='Anisotropy Index')
    ax7.grid(True, alpha=0.3)

    # 8. E2 vs E3 (颜色用各向异性)
    ax8 = fig.add_subplot(gs[2, 1])
    scatter8 = ax8.scatter(e2_vals, e3_vals, c=ani_vals,
                           cmap='coolwarm', alpha=0.6, s=15)
    ax8.set_xlabel('Normalized E2')
    ax8.set_ylabel('Normalized E3')
    ax8.set_title('E2 vs E3 (Colored by Anisotropy)')
    plt.colorbar(scatter8, ax=ax8, label='Anisotropy Index')
    ax8.grid(True, alpha=0.3)

    # 9. E1 vs E2 (颜色用面数)
    ax9 = fig.add_subplot(gs[2, 2])
    scatter9 = ax9.scatter(e1_vals, e2_vals, c=num_faces,
                           cmap='plasma', alpha=0.6, s=15)
    ax9.set_xlabel('Normalized E1')
    ax9.set_ylabel('Normalized E2')
    ax9.set_title('E1 vs E2 (Colored by Number of Faces)')
    plt.colorbar(scatter9, ax=ax9, label='Number of Faces')
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/comprehensive_scatter_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 单独保存一些重要的散点图
    create_individual_plots(all_config_data, save_path)

    print(f"额外散点图已保存至 {save_path}/ 目录")


def create_individual_plots(all_config_data, save_path):
    """创建单独的散点图以便更详细地查看"""

    e1_vals = [c['normalized_E1'] for c in all_config_data]
    e2_vals = [c['normalized_E2'] for c in all_config_data]
    e3_vals = [c['normalized_E3'] for c in all_config_data]
    ani_vals = [c['anisotropy'] for c in all_config_data]
    num_faces = [len(c['active_faces']) for c in all_config_data]
    shear_vals = [c['avg_shear'] for c in all_config_data]

    # 1. 各向异性 vs E3
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(ani_vals, e3_vals, c=num_faces,
                          cmap='viridis', alpha=0.6, s=20)
    plt.xlabel('Anisotropy Index')
    plt.ylabel('Normalized E3')
    plt.title('Anisotropy vs E3 (Colored by Number of Faces)')
    plt.colorbar(scatter, label='Number of Faces')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/anisotropy_vs_e3.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 各向异性 vs 面数
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(ani_vals, num_faces, c=e1_vals,
                          cmap='viridis', alpha=0.6, s=20)
    plt.xlabel('Anisotropy Index')
    plt.ylabel('Number of Faces')
    plt.title('Anisotropy vs Number of Faces (Colored by E1)')
    plt.colorbar(scatter, label='Normalized E1')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/anisotropy_vs_num_faces.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3. E1 vs E3
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(e1_vals, e3_vals, c=ani_vals,
                          cmap='viridis', alpha=0.6, s=20)
    plt.xlabel('Normalized E1')
    plt.ylabel('Normalized E3')
    plt.title('E1 vs E3 (Colored by Anisotropy)')
    plt.colorbar(scatter, label='Anisotropy Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/e1_vs_e3.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. E1 vs E2
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(e1_vals, e2_vals, c=ani_vals,
                          cmap='viridis', alpha=0.6, s=20)
    plt.xlabel('Normalized E1')
    plt.ylabel('Normalized E2')
    plt.title('E1 vs E2 (Colored by Anisotropy)')
    plt.colorbar(scatter, label='Anisotropy Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/e1_vs_e2.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建输出目录 - 确保先创建主目录
    os.makedirs("design_space_analysis", exist_ok=True)
    os.makedirs("design_space_analysis/detailed_analysis", exist_ok=True)

    # 加载数据集（用于获取标准化参数和面几何信息）
    dataset = SuperEnhancedDataset(
        adj_h5_path="sampled_adjacency_matrice1.h5",
        node_feature_excel_path="output_8-1_with_results.xlsx",
        csv_path="enhanced_homogenized_results_8001_with_density.csv"
    )

    # 加载模型
    model = SuperDiffusionVAE().to(device)
    model.load_state_dict(torch.load("./models10.21/best_model.pt", map_location=device, weights_only=True))
    model.eval()

    # 大规模生成构型
    num_samples = 1000000  # 生成20万个构型
    all_configs = generate_configurations(
        model, num_samples, device, dataset.target_mean, dataset.target_std
    )

    print(f"成功生成 {len(all_configs)} 个构型")

    # 保存所有生成的数据 - 确保目录存在
    os.makedirs("design_space_analysis", exist_ok=True)
    np.save("design_space_analysis/all_configs.npy", all_configs)
    # 新增：提取前10000个高E1构型的密度数据
    top_e1_density_df = extract_top_e1_density_data(all_configs, top_k=10000)

    # 新增：分析前1000个E1最好构型的面激活网络
    print("\n=== 开始分析前1000个高E1构型的面激活网络 ===")
    top_e1_analysis = analyze_top_e1_configurations(all_configs, top_k=1000)
    top_e1_network = visualize_top_e1_network_analysis(top_e1_analysis, top_k=1000)

    # 找出极端构型
    extreme_configs = find_extreme_configurations(all_configs)

    # 执行设计空间映射分析
    analysis_results = perform_design_space_mapping(all_configs, dataset)

    # 深入分析高刚度设计 (E1, E2, E3)
    high_e1_stiffness_clusters = analyze_high_stiffness_designs(extreme_configs, all_configs, 'E1')
    high_e2_stiffness_clusters = analyze_high_stiffness_designs(extreme_configs, all_configs, 'E2')
    high_e3_stiffness_clusters = analyze_high_stiffness_designs(extreme_configs, all_configs, 'E3')

    # 深入分析各向异性设计
    anisotropy_configs = analyze_anisotropy_designs(extreme_configs, all_configs, dataset)

    # 寻找甜蜜点设计
    sweet_spot_configs = find_sweet_spot_designs(all_configs)

    # 高分辨率插值分析
    interpolation_paths = enhanced_interpolation_analysis(
        model, all_configs, extreme_configs, device,
        dataset.target_mean, dataset.target_std, steps=500
    )

    # 新增的面分析 - 三组正交对角面
    print("\n=== 三组正交对角面深度分析 ===")

    # 可视化三组正交对角面在立方体中的位置和方向
    for group_name, face_ids in ORTHOGONAL_DIAGONAL_GROUPS.items():
        visualize_face_in_cube(dataset, face_ids)

    # 分析面连接关系
    connectivity_results = analyze_face_connectivity(dataset)

    # 提取三组正交对角面激活时的典型构型
    for group_name, face_ids in ORTHOGONAL_DIAGONAL_GROUPS.items():
        extract_face_designs(all_configs, face_ids)

    # 构建三组正交对角面的激活概率与性能的关系图
    for group_name, face_ids in ORTHOGONAL_DIAGONAL_GROUPS.items():
        stiffness_type = group_name.split('_')[0]  # 提取E1, E2, E3
        plot_face_activation_vs_performance(all_configs, face_ids, stiffness_type)

    # 生成可视化
    visualize_design_space(all_configs, analysis_results)
    # 添加额外的散点图
    create_additional_scatter_plots(all_configs)

    # 输出详细分析报告
    with open("design_space_analysis/detailed_analysis_report.txt", "w") as f:
        f.write("=== 超材料设计空间详细分析报告 ===\n\n")
        f.write(f"基于 {len(all_configs)} 个生成构型的分析结果\n\n")

        # 高刚度设计分析
        f.write("1. 高刚度设计分析:\n")

        # E1高刚度设计
        if high_e1_stiffness_clusters:
            f.write("   E1高刚度设计:\n")
            for cluster_id, result in high_e1_stiffness_clusters.items():
                f.write(f"     模式 {cluster_id + 1} (样本数: {result['size']}):\n")
                f.write(f"       平均E1: {result['avg_E1']:.3f}, 平均各向异性: {result['avg_anisotropy']:.3f}\n")
                key_faces = [i + 1 for i, freq in enumerate(result['face_activation']) if freq > 0.7]
                f.write(f"       关键激活面: {sorted(key_faces)}\n")

        # E2高刚度设计
        if high_e2_stiffness_clusters:
            f.write("   E2高刚度设计:\n")
            for cluster_id, result in high_e2_stiffness_clusters.items():
                f.write(f"     模式 {cluster_id + 1} (样本数: {result['size']}):\n")
                f.write(f"       平均E2: {result['avg_E2']:.3f}, 平均各向异性: {result['avg_anisotropy']:.3f}\n")
                key_faces = [i + 1 for i, freq in enumerate(result['face_activation']) if freq > 0.7]
                f.write(f"       关键激活面: {sorted(key_faces)}\n")

        # E3高刚度设计
        if high_e3_stiffness_clusters:
            f.write("   E3高刚度设计:\n")
            for cluster_id, result in high_e3_stiffness_clusters.items():
                f.write(f"     模式 {cluster_id + 1} (样本数: {result['size']}):\n")
                f.write(f"       平均E3: {result['avg_E3']:.3f}, 平均各向异性: {result['avg_anisotropy']:.3f}\n")
                key_faces = [i + 1 for i, freq in enumerate(result['face_activation']) if freq > 0.7]
                f.write(f"       关键激活面: {sorted(key_faces)}\n")

        f.write("\n")

        # 甜蜜点设计分析
        f.write("2. 甜蜜点设计分析 (中等刚度且低各向异性):\n")
        if sweet_spot_configs:
            face_activation = np.zeros(20)
            for config in sweet_spot_configs:
                for face_id in config['active_faces']:
                    face_activation[face_id] += 1
            face_activation /= len(sweet_spot_configs)

            key_faces = [i + 1 for i, freq in enumerate(face_activation) if freq > 0.3]
            f.write(f"   关键激活面: {sorted(key_faces)}\n")

            e1_values = [c['normalized_E1'] for c in sweet_spot_configs]
            e2_values = [c['normalized_E2'] for c in sweet_spot_configs]
            e3_values = [c['normalized_E3'] for c in sweet_spot_configs]
            ani_values = [c['anisotropy'] for c in sweet_spot_configs]
            num_faces = [len(c['active_faces']) for c in sweet_spot_configs]

            f.write(f"   平均E1: {np.mean(e1_values):.3f}\n")
            f.write(f"   平均E2: {np.mean(e2_values):.3f}\n")
            f.write(f"   平均E3: {np.mean(e3_values):.3f}\n")
            f.write(f"   平均各向异性: {np.mean(ani_values):.3f}\n")
            f.write(f"   平均面数: {np.mean(num_faces):.1f}\n")
        f.write("\n")

        # 面重要性分析
        f.write("3. 面重要性分析:\n")
        importance_e1 = analysis_results['importance_e1']
        importance_e2 = analysis_results['importance_e2']
        importance_e3 = analysis_results['importance_e3']
        importance_ani = analysis_results['importance_ani']

        f.write("   对E1最重要的面 (前5):\n")
        e1_important_faces = np.argsort(importance_e1)[::-1][:5]
        for i, face_idx in enumerate(e1_important_faces):
            f.write(f"     {i + 1}. 面 {face_idx + 1}: {importance_e1[face_idx]:.4f}\n")

        f.write("   对E2最重要的面 (前5):\n")
        e2_important_faces = np.argsort(importance_e2)[::-1][:5]
        for i, face_idx in enumerate(e2_important_faces):
            f.write(f"     {i + 1}. 面 {face_idx + 1}: {importance_e2[face_idx]:.4f}\n")

        f.write("   对E3最重要的面 (前5):\n")
        e3_important_faces = np.argsort(importance_e3)[::-1][:5]
        for i, face_idx in enumerate(e3_important_faces):
            f.write(f"     {i + 1}. 面 {face_idx + 1}: {importance_e3[face_idx]:.4f}\n")

        f.write("   对各向异性最重要的面 (前5):\n")
        ani_important_faces = np.argsort(importance_ani)[::-1][:5]
        for i, face_idx in enumerate(ani_important_faces):
            f.write(f"     {i + 1}. 面 {face_idx + 1}: {importance_ani[face_idx]:.4f}\n")
        f.write("\n")

        # 几何属性与重要性关系分析
        if 'face_geometry_analysis' in analysis_results:
            df_faces = analysis_results['face_geometry_analysis']
            corr_proj_e1_x = pearsonr(df_faces['x_projection'], df_faces['importance_e1'])[0]
            corr_proj_e2_y = pearsonr(df_faces['y_projection'], df_faces['importance_e2'])[0]
            corr_proj_e3_z = pearsonr(df_faces['z_projection'], df_faces['importance_e3'])[0]

            f.write("4. 几何属性与重要性关系分析:\n")
            f.write(f"   法向量X轴投影与E1重要性的相关系数: {corr_proj_e1_x:.3f}\n")
            f.write(f"   法向量Y轴投影与E2重要性的相关系数: {corr_proj_e2_y:.3f}\n")
            f.write(f"   法向量Z轴投影与E3重要性的相关系数: {corr_proj_e3_z:.3f}\n")

            f.write("   三组正交对角面比较:\n")
            for group_name, group_info in analysis_results.get('group_analysis', {}).items():
                f.write(
                    f"     {group_name}: 平均投影={group_info['avg_projection']:.3f}, 平均重要性={group_info['avg_importance']:.3f}\n")
            f.write("\n")

        # 性能关系分析
        f.write("5. 性能关系分析:\n")
        e1_vals = [c['normalized_E1'] for c in all_configs]
        e2_vals = [c['normalized_E2'] for c in all_configs]
        e3_vals = [c['normalized_E3'] for c in all_configs]
        ani_vals = [c['anisotropy'] for c in all_configs]

        correlation_e1_e2 = pearsonr(e1_vals, e2_vals)[0]
        correlation_e1_e3 = pearsonr(e1_vals, e3_vals)[0]
        correlation_e2_e3 = pearsonr(e2_vals, e3_vals)[0]
        correlation_e1_ani = pearsonr(e1_vals, ani_vals)[0]

        f.write(f"   E1与E2相关系数: {correlation_e1_e2:.3f}\n")
        f.write(f"   E1与E3相关系数: {correlation_e1_e3:.3f}\n")
        f.write(f"   E2与E3相关系数: {correlation_e2_e3:.3f}\n")
        f.write(f"   E1与各向异性相关系数: {correlation_e1_ani:.3f}\n")

        # 性能范围分析
        f.write(f"   E1范围: {min(e1_vals):.3f} - {max(e1_vals):.3f}\n")
        f.write(f"   E2范围: {min(e2_vals):.3f} - {max(e2_vals):.3f}\n")
        f.write(f"   E3范围: {min(e3_vals):.3f} - {max(e3_vals):.3f}\n")
        f.write(f"   各向异性范围: {min(ani_vals):.3f} - {max(ani_vals):.3f}\n")
        f.write("\n")






    print("分析完成! 所有结果已保存至 design_space_analysis1029/ 目录")
    print("详细分析报告已保存至: design_space_analysis/detailed_analysis_report.txt")


if __name__ == "__main__":
    main()