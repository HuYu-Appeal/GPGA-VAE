import torch
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from config import config


def extract_node_features_from_df(df):
    """直接从DataFrame中提取特征，不需要重新计算"""
    features = []

    # 直接使用Excel中的特征值
    for col in ['单位法向量_X', '单位法向量_Y', '单位法向量_Z',
                '平面到（0，0，0）的距离', 'Area',
                'Centroid_X', 'Centroid_Y', 'Centroid_Z']:
        feat = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.float64)
        features.append(feat.values.reshape(-1, 1))

    vertex_counts = []
    for _, row in df.iterrows():
        count = 0
        for v_col in ['顶点1', '顶点2', '顶点3', '顶点4']:
            if isinstance(row[v_col], str) and row[v_col].startswith('['):
                count += 1
        vertex_counts.append(count)
    features.append(np.array(vertex_counts).reshape(-1, 1))

    return np.hstack(features)


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate_fn(batch):
    if isinstance(batch[0], dict):
        node_counts = [len(d['x']) for d in batch]
        node_offsets = torch.cumsum(torch.cat([
            torch.tensor([0], dtype=torch.long),
            torch.tensor(node_counts[:-1], dtype=torch.long)
        ]), dim=0)

        x = torch.cat([d['x'] for d in batch], dim=0)
        y = torch.stack([d['y'] for d in batch])
        edge_indices = []
        for i, d in enumerate(batch):
            if d['edge_index'].numel() > 0:
                edge_indices.append(d['edge_index'] + node_offsets[i])
        edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long)

        batch_idx = torch.cat([
            torch.full((len(d['x']),), i, dtype=torch.long)
            for i, d in enumerate(batch)
        ])

        face_mask = torch.stack([d['face_mask'] for d in batch]).view(len(batch), 20)
        adj_matrix = torch.stack([d['adj_matrix'] for d in batch]).view(len(batch), 20, 20)
        face_ids = [d['face_ids'] for d in batch]
        node_counts_tensor = torch.tensor(node_counts, dtype=torch.long)

    else:
        x = torch.cat([d.x for d in batch], dim=0)
        node_offsets = torch.cumsum(torch.tensor([0] + [d.num_nodes for d in batch[:-1]]), dim=0)
        edge_indices = [
            d.edge_index + offset
            for d, offset in zip(batch, node_offsets)
        ]
        edge_index = torch.cat(edge_indices, dim=1)
        y = torch.stack([d.y for d in batch])
        batch_idx = torch.cat([
            torch.full((d.num_nodes,), i, dtype=torch.long)
            for i, d in enumerate(batch)
        ])
        face_mask = torch.stack([d.face_mask for d in batch]).view(len(batch), 20)
        adj_matrix = torch.stack([d.adj_matrix for d in batch]).view(len(batch), 20, 20)
        face_ids = [d.face_ids.flatten() for d in batch]
        node_counts_tensor = torch.tensor([d.num_nodes for d in batch], dtype=torch.long)

    return {
        'x': x,
        'edge_index': edge_index,
        'y': y,
        'batch': batch_idx,
        'face_mask': face_mask,
        'adj_matrix': adj_matrix,
        'face_ids': face_ids,
        'node_counts': node_counts_tensor
    }


class SuperEnhancedDataset:
    def __init__(self, adj_h5_path, node_feature_excel_path, csv_path):
        self.adj_h5_path = adj_h5_path
        self.node_feature_excel_path = node_feature_excel_path
        self.csv_path = csv_path

        # 加载CSV文件
        self.df_csv = pd.read_csv(csv_path)
        print(f"CSV文件加载成功，包含 {len(self.df_csv)} 行")
        print(f"CSV列名: {list(self.df_csv.columns)}")

        # 创建key到目标的映射（6个参数，包括相对密度）
        self.key_to_target = {}
        missing_density_count = 0

        for idx, row in self.df_csv.iterrows():
            key = row['key']
            try:
                # 检查相对密度是否存在且有效
                if 'relative_density' not in row or pd.isna(row['relative_density']):
                    missing_density_count += 1
                    # 跳过没有相对密度的样本
                    continue

                # 确保所有6个目标值都存在且有效
                targets = np.array([
                    float(row['normalized_E1']),
                    float(row['normalized_E2']),
                    float(row['normalized_E3']),
                    float(row['avg_shear_modulus']),
                    float(row['num_faces']),
                    float(row['relative_density'])  # 相对密度
                ], dtype=np.float32)

                # 检查是否有无效值
                if np.any(np.isnan(targets)) or np.any(np.isinf(targets)):
                    print(f"跳过键 {key}: 包含无效目标值 {targets}")
                    continue

                self.key_to_target[key] = targets
            except Exception as e:
                print(f"处理键 {key} 的目标值时出错: {e}")
                if 'relative_density' in row:
                    print(f"相对密度值: {row['relative_density']} (类型: {type(row['relative_density'])})")
                continue

        print(f"成功映射 {len(self.key_to_target)} 个键到目标值")
        if missing_density_count > 0:
            print(f"跳过 {missing_density_count} 个缺少相对密度的样本")

        # 加载Excel文件中的预计算特征
        self.df_node_features = pd.read_excel(node_feature_excel_path)
        print(f"Excel文件加载成功，包含 {len(self.df_node_features)} 行")

        # 创建面编号到预计算特征的映射
        self.face_id_to_features = {}
        for _, row in self.df_node_features.iterrows():
            face_id = int(row['平面编号'])

            # 直接从Excel中提取预计算的特征
            features = np.array([
                row['单位法向量_X'], row['单位法向量_Y'], row['单位法向量_Z'],
                row['平面到（0，0，0）的距离'],
                row['Area'],
                row['Centroid_X'], row['Centroid_Y'], row['Centroid_Z']
            ], dtype=np.float32)

            # 计算顶点数
            vertex_count = 0
            for v_col in ['顶点1', '顶点2', '顶点3', '顶点4']:
                if isinstance(row[v_col], str) and row[v_col].startswith('['):
                    vertex_count += 1

            # 添加顶点数作为第9个特征
            full_features = np.append(features, vertex_count)
            self.face_id_to_features[face_id] = full_features

        print(f"已加载的面编号数量: {len(self.face_id_to_features)}")

        # 处理数据样本
        self.data_list = []
        self.skipped_keys = []

        with h5py.File(adj_h5_path, 'r') as h5_file:
            for key in tqdm(self.key_to_target.keys(), desc="处理样本"):
                try:
                    # 解析key获取结构ID和面索引
                    parts = key.split('_')
                    structure_id = int(parts[2])
                    face_indices = [int(x) for x in parts[3:]]

                    # 检查是否所有面都有预计算特征
                    missing_faces = [f for f in face_indices if f not in self.face_id_to_features]
                    if missing_faces:
                        print(f"键 {key} 缺少面数据: {missing_faces}")
                        self.skipped_keys.append(key)
                        continue

                    # 获取目标值（6个参数）
                    y = self.key_to_target[key]

                    # 获取邻接矩阵
                    adj_matrix = h5_file[key][:]

                    # 检查邻接矩阵大小
                    if adj_matrix.shape[0] > 20 or adj_matrix.shape[1] > 20:
                        print(f"键 {key} 邻接矩阵过大: {adj_matrix.shape}")
                        self.skipped_keys.append(key)
                        continue

                    # 从预计算特征中获取节点特征
                    node_features = []
                    for f_id in face_indices:
                        if f_id in self.face_id_to_features:
                            feat = self.face_id_to_features[f_id]
                            node_features.append(feat)

                    node_features = np.array(node_features) if node_features else np.zeros((0, 9))

                    # 创建边索引（基于邻接矩阵）
                    edge_index = []
                    num_faces = len(face_indices)
                    for i in range(num_faces):
                        for j in range(num_faces):
                            if adj_matrix[i, j] > 0:
                                edge_index.append([i, j])

                    # 创建面掩码（指示哪些面被激活）
                    face_mask = torch.zeros(20, dtype=torch.float32)
                    for f_id in face_indices:
                        if 1 <= f_id <= 20:  # 确保面编号在有效范围内
                            face_mask[f_id - 1] = 1

                    # 创建数据样本
                    data = {
                        'x': torch.tensor(node_features, dtype=torch.float32),
                        'edge_index': torch.tensor(edge_index).t().contiguous() if edge_index else torch.zeros((2, 0),
                                                                                                               dtype=torch.long),
                        'y': torch.tensor(y, dtype=torch.float32),
                        'face_mask': face_mask,
                        'adj_matrix': torch.tensor(adj_matrix, dtype=torch.float32),
                        'num_nodes': num_faces,
                        'face_ids': torch.tensor(face_indices, dtype=torch.long)
                    }

                    self.data_list.append(data)

                except Exception as e:
                    print(f"处理键 {key} 时出错: {str(e)}")
                    self.skipped_keys.append(key)
                    continue

        # 数据清洗：移除无效样本
        if not self.data_list:
            raise ValueError("没有有效的样本数据")

        valid_indices = []
        for i, sample in enumerate(self.data_list):
            y = sample['y'].numpy()
            x = sample['x'].numpy()
            # 检查目标值和特征是否有效
            if not (np.isnan(y).any() or (np.abs(y) > 1e6).any() or
                    np.isnan(x).any() or (np.abs(x) > 1e6).any()):
                valid_indices.append(i)

        self.data_list = [self.data_list[i] for i in valid_indices]
        print(f"数据清洗后保留 {len(self.data_list)} 个样本")

        # 标准化特征数据（使用预计算的特征）
        all_features = np.vstack([
            sample['x'].numpy()
            for sample in self.data_list
            if sample['x'].numel() > 0
        ])

        self.scaler = StandardScaler()
        self.scaler.fit(all_features.astype(np.float32))

        # 应用特征标准化
        for sample in self.data_list:
            if sample['x'].numel() > 0:
                sample['x'] = torch.tensor(
                    self.scaler.transform(sample['x'].numpy()),
                    dtype=torch.float32
                )

        # 获取目标值并标准化（6个参数）
        all_targets = np.array([sample['y'].numpy() for sample in self.data_list])

        # 检查目标值维度
        print(f"目标值矩阵形状: {all_targets.shape}")
        if all_targets.shape[1] != 6:
            print(f"错误: 目标值应该有6列，但实际有{all_targets.shape[1]}列")
            print("前5个样本的目标值:")
            for i in range(min(5, len(all_targets))):
                print(f"样本{i}: {all_targets[i]}")
            raise ValueError(f"目标值维度不正确，期望6列但得到{all_targets.shape[1]}列")

        # 计算并保存目标值的均值和标准差
        self.target_mean = all_targets.mean(axis=0)
        self.target_std = all_targets.std(axis=0)

        print(f"目标值均值 (6个参数):")
        param_names = ['E1', 'E2', 'E3', '剪切模量', '面数', '相对密度']
        for i, name in enumerate(param_names):
            print(f"  {name}: {self.target_mean[i]:.6f} ± {self.target_std[i]:.6f}")

        # 应用目标值标准化
        for i, sample in enumerate(self.data_list):
            normalized_targets = (all_targets[i] - self.target_mean) / (self.target_std + 1e-6)
            sample['y'] = torch.tensor(normalized_targets, dtype=torch.float32)

        print(f"成功加载 {len(self.data_list)} 个样本, 跳过 {len(self.skipped_keys)} 个样本")

        # 验证相对密度范围
        relative_densities = all_targets[:, 5]  # 第6列是相对密度
        print(f"相对密度统计:")
        print(f"  最小值: {relative_densities.min():.6f}")
        print(f"  最大值: {relative_densities.max():.6f}")
        print(f"  平均值: {relative_densities.mean():.6f}")
        print(f"  中位数: {np.median(relative_densities):.6f}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]