import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class PhysicsGuidedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.current_epoch = 0
        # 调整属性权重，增加相对密度权重（6个参数）
        self.property_weights = torch.tensor([1.2, 1.2, 1.2, 1.0, 0.8, 1.0], dtype=torch.float32)
        self.sparsity_loss_weight = 0.1
        self.concentration = 8.0  # 降低浓度参数

        # 拓扑重建损失权重
        self.recon_weight = 0.8
        # 邻接矩阵重建损失权重
        self.adj_recon_weight = 0.3

        # 四边形权重
        self.quad_weight = config.quad_weight
        self.register_buffer('quad_weight_tensor', torch.tensor(config.quad_weight))

        # 各向异性指数损失权重
        self.anisotropy_weight = 0.2

        # 物理约束权重
        self.physics_weight = 0.15

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.huber_loss = nn.HuberLoss(delta=1.0)

    def set_epoch(self, epoch):
        """设置当前epoch，用于动态调整损失权重"""
        self.current_epoch = epoch
        # 动态调整重建损失权重（随着训练进行逐渐减少）
        self.recon_weight = max(0.3, 0.8 * (1 - epoch / config.epochs * 0.5))

    def get_quad_weight(self):
        """获取四边形权重"""
        return self.quad_weight_tensor.item()

    def set_quad_weight(self, weight):
        """设置四边形权重"""
        self.quad_weight_tensor.fill_(weight)
        self.quad_weight = weight

    def calculate_anisotropy(self, E1, E2, E3):
        """计算各向异性指数"""
        numerator = (E1 - E2) ** 2 + (E2 - E3) ** 2 + (E3 - E1) ** 2
        denominator = 2 * (E1 ** 2 + E2 ** 2 + E3 ** 2) + 1e-10
        return torch.sqrt(numerator / denominator)

    def calculate_adjacency_reconstruction_loss(self, pred_adj, true_adj, face_mask):
        """计算邻接矩阵重建损失"""
        batch_size = pred_adj.shape[0]
        adj_loss = 0.0

        for b in range(batch_size):
            # 只计算激活面的邻接关系
            active_faces = torch.where(face_mask[b] > 0.5)[0]
            if len(active_faces) > 0:
                # 提取激活面的邻接子矩阵
                pred_sub_adj = pred_adj[b][active_faces][:, active_faces]
                true_sub_adj = true_adj[b][active_faces][:, active_faces]

                # 计算二元交叉熵损失
                adj_loss += F.binary_cross_entropy(pred_sub_adj, true_sub_adj)

        return adj_loss / batch_size if batch_size > 0 else torch.tensor(0.0)

    def forward(self, outputs, batch):
        # 获取预测和目标值
        pred = outputs['pred']
        targets = batch['y'].view(-1, 6)  # 修改为6个目标

        # 获取面掩码和邻接矩阵
        pred_face_probs = outputs.get('face_probs', None)
        pred_face_mask = outputs.get('face_mask', None)
        pred_adj = outputs.get('adj_matrix', None)

        true_face_mask = batch.get('face_mask', None)
        true_adj = batch.get('adj_matrix', None)

        # 1. 计算6个参数的属性损失
        property_losses = [
            self.hubert_loss(pred[:, 0], targets[:, 0]) * self.property_weights[0],
            self.hubert_loss(pred[:, 1], targets[:, 1]) * self.property_weights[1],
            self.hubert_loss(pred[:, 2], targets[:, 2]) * self.property_weights[2],
            self.hubert_loss(pred[:, 3], targets[:, 3]) * self.property_weights[3],
            self.mse_loss(pred[:, 4], targets[:, 4]) * self.property_weights[4],
            self.hubert_loss(pred[:, 5], targets[:, 5]) * self.property_weights[5]  # 新增密度损失
        ]

        property_loss = sum(property_losses)

        # 2. 计算派生指标损失（各向异性指数）
        pred_anisotropy = self.calculate_anisotropy(pred[:, 0], pred[:, 1], pred[:, 2])
        true_anisotropy = self.calculate_anisotropy(targets[:, 0], targets[:, 1], targets[:, 2])
        anisotropy_loss = self.hubert_loss(pred_anisotropy, true_anisotropy) * self.anisotropy_weight

        # 3. 物理损失
        physics_loss = outputs.get('physics_loss', torch.tensor(0.0).to(property_loss.device))

        # 4. 稀疏性损失
        sparsity_loss = torch.tensor(0.0).to(property_loss.device)
        if pred_adj is not None:
            actual_sparsity = 1 - pred_adj.mean(dim=[1, 2])
            actual_sparsity_clamped = actual_sparsity.clamp(1e-3, 1 - 1e-3)
            alpha_actual = actual_sparsity_clamped * self.concentration
            beta_actual = (1 - actual_sparsity_clamped) * self.concentration

            # 使用更稳定的KL散度计算
            try:
                actual_dist = torch.distributions.Beta(alpha_actual, beta_actual)
                target_dist = torch.distributions.Beta(config.sparsity_alpha, config.sparsity_beta)
                sparsity_loss = torch.distributions.kl_divergence(actual_dist, target_dist).mean()
            except:
                # 如果KL散度计算失败，使用MSE作为备用
                target_sparsity = config.sparsity_alpha / (config.sparsity_alpha + config.sparsity_beta)
                sparsity_loss = self.mse_loss(actual_sparsity, torch.full_like(actual_sparsity, target_sparsity))

        # 5. 拓扑重建损失
        recon_loss = torch.tensor(0.0).to(property_loss.device)
        if pred_face_probs is not None and true_face_mask is not None:
            recon_loss = self.bce_loss(pred_face_probs, true_face_mask)

        # 6. 邻接矩阵重建损失
        adj_recon_loss = torch.tensor(0.0).to(property_loss.device)
        if pred_adj is not None and true_adj is not None and pred_face_mask is not None:
            adj_recon_loss = self.calculate_adjacency_reconstruction_loss(pred_adj, true_adj, pred_face_mask)

        # 根据epoch动态调整损失权重
        epoch_factor = min(1.0, self.current_epoch / config.warmup_epochs)
        sparsity_weight = self.sparsity_loss_weight * epoch_factor
        current_recon_weight = self.recon_weight * (1.0 - epoch_factor * 0.5)  # 逐渐减少重建权重

        # 总损失计算
        total_loss = (
                property_loss +
                anisotropy_loss +
                self.physics_weight * physics_loss +
                sparsity_weight * sparsity_loss +
                current_recon_weight * recon_loss +
                self.adj_recon_weight * adj_recon_loss
        )

        # 返回详细的损失信息（用于监控）
        return {
            'total': total_loss,
            'property': property_loss,
            'anisotropy': anisotropy_loss,
            'physics': physics_loss,
            'sparsity': sparsity_loss,
            'recon': recon_loss,
            'adj_recon': adj_recon_loss,
            'property_details': {
                'E1': property_losses[0],
                'E2': property_losses[1],
                'E3': property_losses[2],
                'shear': property_losses[3],
                'faces': property_losses[4],
                'density': property_losses[5]  # 新增密度损失详情
            }
        }

    def hubert_loss(self, pred, target):
        """Huber损失函数封装"""
        return F.huber_loss(pred, target, delta=1.0, reduction='mean')

    def _get_target_features(self, batch):
        """获取目标特征（用于几何特征损失）"""
        batch_size = len(batch['node_counts'])
        features = torch.zeros(batch_size, 20, 9, device=batch['x'].device)

        current_idx = 0
        for i in range(batch_size):
            sample_face_ids = batch['face_ids'][i]
            num_nodes = batch['node_counts'][i].item()
            sample_features = batch['x'][current_idx:current_idx + num_nodes]
            current_idx += num_nodes

            for j, f_id in enumerate(sample_face_ids):
                if j < len(sample_features):
                    features[i, f_id - 1] = sample_features[j]

        return features