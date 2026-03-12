import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from config import config, BASE_VERTICES, ORIGINAL_FACES




class DynamicDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 确保所有面都有4个顶点
        self.original_faces = torch.tensor(ORIGINAL_FACES, dtype=torch.long)
        self.base_vertices = torch.tensor(BASE_VERTICES, dtype=torch.float32)

        # 增强的面激活网络
        self.face_activation = nn.Sequential(
            nn.Linear(config.latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20),
            nn.Sigmoid()
        )

        # 拓扑重建损失
        self.reconstruction_loss = nn.BCELoss()

        # 预计算面之间的邻接关系（提高效率）
        self.precomputed_adjacency = self.precompute_face_adjacency()
        # 添加dropout层
        self.dropout = nn.Dropout(0.3)  # 将dropout移到这里

    def precompute_face_adjacency(self):
        """预计算面之间的邻接关系"""
        adjacency_matrix = torch.zeros(20, 20, dtype=torch.float32)

        base_vertices_np = self.base_vertices.numpy()
        original_faces_np = self.original_faces.numpy()

        for i in range(20):
            for j in range(i + 1, 20):
                face_i_vertices = [base_vertices_np[idx] for idx in original_faces_np[i]]
                face_j_vertices = [base_vertices_np[idx] for idx in original_faces_np[j]]

                if self.are_faces_adjacent(face_i_vertices, face_j_vertices):
                    adjacency_matrix[i, j] = 1.0
                    adjacency_matrix[j, i] = 1.0

        return adjacency_matrix

    def are_faces_adjacent(self, face1_vertices, face2_vertices):
        """判断两个面是否相邻（共享边）"""
        # 转换为numpy数组
        face1_vertices = np.array(face1_vertices)
        face2_vertices = np.array(face2_vertices)

        # 检查是否有共享的边
        for i in range(len(face1_vertices)):
            for j in range(len(face2_vertices)):
                # 检查边是否相同（考虑顺序和反向）
                edge1 = tuple(sorted([tuple(face1_vertices[i]), tuple(face1_vertices[(i + 1) % len(face1_vertices)])]))
                edge2 = tuple(sorted([tuple(face2_vertices[j]), tuple(face2_vertices[(j + 1) % len(face2_vertices)])]))

                if edge1 == edge2:
                    return True

        return False


    def forward(self, z, true_face_mask=None):
        z = self.dropout(z)
        # 1. 面激活概率预测
        face_probs = self.face_activation(z)

        # 始终使用预测的概率，但在训练时计算重建损失
        if self.training and true_face_mask is not None:
            # 训练时计算重建损失
            recon_loss = self.reconstruction_loss(face_probs, true_face_mask)
            face_mask = true_face_mask  # 教师强制
        else:
            # 推理时使用确定性阈值
            face_mask = torch.where(face_probs > 0.5,
                                    torch.ones_like(face_probs),
                                    torch.zeros_like(face_probs))
            recon_loss = torch.tensor(0.0).to(z.device)



        # 3. 根据激活的面计算邻接矩阵
        adj_matrix = self.compute_adjacency_from_mask(face_mask)

        return {
            'adj_matrix': adj_matrix,
            'face_mask': face_mask,
            'face_probs': face_probs,
            'recon_loss': recon_loss
        }

    def compute_adjacency_from_mask(self, face_mask):
        """根据面激活掩码计算邻接矩阵"""
        batch_size = face_mask.shape[0]
        device = face_mask.device

        # 使用预计算的邻接关系
        precomputed_adj = self.precomputed_adjacency.to(device)

        # 扩展预计算邻接矩阵到批次维度
        adj_matrix = torch.zeros(batch_size, 20, 20, device=device)

        for b in range(batch_size):
            # 获取激活的面索引
            active_faces = torch.where(face_mask[b] > 0.5)[0]

            if len(active_faces) > 0:
                # 使用预计算的邻接关系
                for i in active_faces:
                    for j in active_faces:
                        if i != j and precomputed_adj[i, j] > 0.5:
                            adj_matrix[b, i, j] = 1.0

        return adj_matrix

    def calculate_normal_and_distance(self, vertices):
        """计算面的法向量和距离"""
        if len(vertices) < 3:
            return None, None

        # 使用前3个点计算法向量
        v1 = np.array(vertices[1]) - np.array(vertices[0])
        v2 = np.array(vertices[2]) - np.array(vertices[0])
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) == 0:
            return None, None

        normal = normal / np.linalg.norm(normal)  # 归一化
        d = -np.dot(normal, np.array(vertices[0]))
        return normal, d

    def point_on_plane(self, point, normal, d, atol=1e-6):
        """检查点是否在平面上"""
        dot_product = np.dot(normal, point) + d
        return np.isclose(dot_product, 0, atol=atol)

    def are_faces_intersecting(self, face1_vertices, face2_vertices):
        """判断两个面是否相交（共享至少2个点）"""
        # 计算两个面的法向量和距离
        normal1, d1 = self.calculate_normal_and_distance(face1_vertices)
        normal2, d2 = self.calculate_normal_and_distance(face2_vertices)

        if normal1 is None or normal2 is None:
            return False

        # 检查两个面的所有顶点组合
        shared_points_count = 0

        # 检查face1的顶点是否在face2的平面上
        for vertex in face1_vertices:
            if self.point_on_plane(vertex, normal2, d2):
                shared_points_count += 1
                if shared_points_count >= 2:
                    return True

        # 检查face2的顶点是否在face1的平面上
        for vertex in face2_vertices:
            if self.point_on_plane(vertex, normal1, d1):
                shared_points_count += 1
                if shared_points_count >= 2:
                    return True

        return shared_points_count >= 2


class PhysicsValidator(nn.Module):
    def __init__(self):
        super().__init__()
        # 简化物理验证，主要检查面激活的合理性
        self.target_face_count = 12  # 目标激活面数
        self.min_face_count = 8  # 最小激活面数
        self.max_face_count = 16  # 最大激活面数

    def forward(self, face_mask):
        batch_size = face_mask.size(0)

        # 1. 激活面数量约束
        active_face_count = face_mask.sum(dim=1)

        # 鼓励面数量在合理范围内
        count_loss = torch.mean(
            F.relu(self.min_face_count - active_face_count) +  # 太少的面
            F.relu(active_face_count - self.max_face_count)  # 太多的面
        )

        # 2. 面激活概率的稀疏性约束（鼓励概率接近0或1）
        sparsity_loss = torch.mean(face_mask * (1 - face_mask))

        # 3. 对称性约束（鼓励对称的面激活模式）
        symmetry_loss = self.calculate_symmetry_loss(face_mask)

        total_loss = count_loss + 0.1 * sparsity_loss + 0.05 * symmetry_loss
        return total_loss

    def calculate_symmetry_loss(self, face_mask):
        """计算对称性损失"""
        # 这里可以定义面的对称关系
        # 例如：面1和面2对称，面3和面4对称等
        symmetric_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

        symmetry_loss = 0
        for pair in symmetric_pairs:
            if pair[0] < 20 and pair[1] < 20:
                diff = torch.abs(face_mask[:, pair[0]] - face_mask[:, pair[1]])
                symmetry_loss += torch.mean(diff)

        return symmetry_loss / len(symmetric_pairs)


class PropertyHeads(nn.Module):
    def __init__(self):
        super().__init__()
        # 主要基于面激活模式进行预测
        self.shared = nn.Sequential(
            nn.Linear(532, 256),  # 输入是20个面的激活概率 + 潜在向量
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )

        # 6个独立的输出头（增加相对密度）
        self.e1_head = nn.Linear(128, 1)
        self.e2_head = nn.Linear(128, 1)
        self.e3_head = nn.Linear(128, 1)
        self.avg_shear_head = nn.Linear(128, 1)
        self.face_head = nn.Linear(128, 1)
        self.density_head = nn.Linear(128, 1)  # 新增相对密度预测头

    def forward(self, z, face_mask):
        # 主要使用面激活模式进行预测，只轻微使用潜在向量
        combined = torch.cat([face_mask, z], dim=1)
        shared = self.shared(combined)
        e1 = self.e1_head(shared)
        e2 = self.e2_head(shared)
        e3 = self.e3_head(shared)
        avg_shear = self.avg_shear_head(shared)
        face_output = self.face_head(shared)
        density_output = self.density_head(shared)  # 新增密度输出

        return torch.cat([e1, e2, e3, avg_shear, face_output, density_output], dim=1)


class SuperDiffusionVAE(nn.Module):
    def __init__(self):
        super(SuperDiffusionVAE, self).__init__()
        self.config = config

        # Encoder components
        self.gat1 = GATConv(9, 256, heads=4, concat=True)
        self.ln1 = nn.LayerNorm(256 * 4)
        self.gat2 = GATConv(256 * 4, 512, heads=2, concat=True)
        self.ln2 = nn.LayerNorm(512 * 2)
        self.gcn = GCNConv(512 * 2, config.latent_dim)

        # Core components
        self.decoder = DynamicDecoder()  # 使用增强的解码器
        self.validator = PhysicsValidator()
        self.property_heads = PropertyHeads()

        # VAE parameters
        self.fc_mu = nn.Linear(config.latent_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.latent_dim, config.latent_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = self.ln1(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.ln2(x)
        x = F.elu(self.gcn(x, edge_index))

        graph_feat = global_mean_pool(x, batch)
        mu = self.fc_mu(graph_feat)
        logvar = self.fc_logvar(graph_feat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, true_face_mask=None):  # 修改这里，添加可选参数
        """解码方法，支持传入真实面掩码用于教师强制训练"""
        return self.decoder(z, true_face_mask)

    def predict_properties(self, z, face_mask):  # 添加参数
        return self.property_heads(z, face_mask)

    def forward(self, batch):
        x = batch['x']
        edge_index = batch['edge_index']
        batch_idx = batch['batch']

        # Encode
        mu, logvar = self.encode(x, edge_index, batch_idx)
        z_0 = self.reparameterize(mu, logvar)

        # Decode - 传入真实面掩码用于训练
        true_face_mask = batch.get('face_mask', None)
        decoded = self.decode(z_0, true_face_mask if self.training else None)

        # Compute physics loss
        physics_loss = self.validator(decoded['face_mask'])

        # Property prediction: 传入z_0和decoded['face_mask']作为两个单独的参数
        pred = self.predict_properties(z_0, decoded['face_mask'])  # 修改这里

        return {
            'pred': pred,
            'mu': mu,
            'logvar': logvar,
            'z_0': z_0,
            'physics_loss': physics_loss,
            **decoded
        }