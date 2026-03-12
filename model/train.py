import numpy as np
import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dataset import SuperEnhancedDataset, CustomDataset, custom_collate_fn
from model import SuperDiffusionVAE
from loss import PhysicsGuidedLoss
from utils1 import compute_metrics, evaluate, plot_predictions_vs_true, plot_error_distribution, plot_loss_curve
import torch
from config import config

# 导入可视化模块
import sys

sys.path.append('.')
try:
    from visualization import LatentSpaceAnalyzer, visualize_invalid_cases
    VISUALIZATION_AVAILABLE = True
    print("✅ 可视化模块导入成功")
except ImportError as e:
    print(f"❌ 可视化模块导入失败: {e}")
    VISUALIZATION_AVAILABLE = False
def train_epoch(model, loader, optimizer, loss_fn, scheduler, scaler, device, epoch):
    model.train()
    total_train_loss = 0.0
    total_recon_loss = 0.0
    active_ratio_history = []

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()

        # 使用新的autocast API
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            outputs = model(batch)

            # 监控激活面比例
            face_mask = outputs['face_mask']

            # 确保维度正确
            if face_mask.dim() == 3:
                face_mask = face_mask.squeeze(-1)

            # 计算激活面比例
            active_faces = face_mask.sum(dim=1)
            total_faces = face_mask.shape[1]
            active_ratio = active_faces.float() / total_faces
            active_ratio_mean = active_ratio.mean().item()
            active_ratio_history.append(active_ratio_mean)

            # 动态调整四边形权重
            momentum = 0.7
            current_quad_weight = loss_fn.get_quad_weight()

            # 根据激活面比例调整权重
            if active_ratio_mean < 0.4:  # 激活面太少
                new_weight = min(2.0, current_quad_weight + 0.15)
                smoothed_weight = momentum * current_quad_weight + (1 - momentum) * new_weight
                loss_fn.set_quad_weight(smoothed_weight)
            elif active_ratio_mean > 0.8:  # 激活面太多
                new_weight = max(0.8, current_quad_weight - 0.1)
                smoothed_weight = momentum * current_quad_weight + (1 - momentum) * new_weight
                loss_fn.set_quad_weight(smoothed_weight)

            # 计算损失
            losses = loss_fn(outputs, batch)
            total_train_loss += losses['total'].item() * len(batch['y'])
            total_recon_loss += losses.get('recon', torch.tensor(0.0)).item() * len(batch['y'])

        # 混合精度训练
        if scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    # 计算平均激活面比例
    avg_active_ratio = sum(active_ratio_history) / len(active_ratio_history) if active_ratio_history else 0.0

    # 计算平均训练损失
    avg_train_loss = total_train_loss / len(loader.dataset)
    avg_recon_loss = total_recon_loss / len(loader.dataset)

    return avg_train_loss, avg_recon_loss, avg_active_ratio


def validate_epoch(model, loader, loss_fn, device, target_mean, target_std):
    model.eval()
    total_val_loss = 0.0
    predictions = []
    true_values = []

    with torch.no_grad():
        for batch in loader:
            # 将批次数据移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 模型前向传播
            outputs = model(batch)

            # 计算邻接矩阵（根据激活的面）
            face_mask = outputs['face_mask']
            new_adj = model.decoder.compute_adjacency_from_mask(face_mask)
            outputs['adj_matrix'] = new_adj

            # 计算损失
            losses = loss_fn(outputs, batch)
            total_val_loss += losses['total'].item() * batch['y'].size(0)

            # 反标准化预测值和真实值 (6个参数)
            pred_denorm = outputs['pred'].detach().cpu().numpy() * target_std + target_mean
            true_denorm = batch['y'].detach().cpu().numpy() * target_std + target_mean

            # 计算各向异性指数（只使用前3个参数E1,E2,E3）
            def calculate_anisotropy(E1, E2, E3):
                numerator = (E1 - E2) ** 2 + (E2 - E3) ** 2 + (E3 - E1) ** 2
                denominator = 2 * (E1 ** 2 + E2 ** 2 + E3 ** 2) + 1e-10
                return np.sqrt(numerator / denominator)

            true_anisotropy = calculate_anisotropy(true_denorm[:, 0], true_denorm[:, 1], true_denorm[:, 2])
            pred_anisotropy = calculate_anisotropy(pred_denorm[:, 0], pred_denorm[:, 1], pred_denorm[:, 2])

            # 合并数据用于评估 (6个参数 + 1个派生指标)
            true_with_ani = np.column_stack((true_denorm, true_anisotropy))
            pred_with_ani = np.column_stack((pred_denorm, pred_anisotropy))

            predictions.extend(pred_with_ani)
            true_values.extend(true_with_ani)

    # 计算平均验证损失
    avg_val_loss = total_val_loss / len(loader.dataset)

    # 将列表转换为numpy数组
    pred_array = np.array(predictions)
    true_array = np.array(true_values)

    # 计算评估指标
    metrics = compute_metrics(true_array, pred_array)

    return avg_val_loss, metrics, pred_array, true_array


def create_vanilla_vae():
    """创建没有物理约束的Vanilla VAE用于对比"""
    return SuperDiffusionVAE(use_physics=False)


def train_model():
    adj_h5_path = "sampled_adjacency_matrice1.h5"
    node_feature_excel_path = "output_8-1_with_results.xlsx"
    csv_path = "enhanced_homogenized_results_8001_with_density.csv"

    print(f"加载数据集...")
    dataset_generator = SuperEnhancedDataset(
        adj_h5_path=adj_h5_path,
        node_feature_excel_path=node_feature_excel_path,
        csv_path=csv_path
    )

    full_dataset = CustomDataset(dataset_generator.data_list)

    if len(full_dataset) == 0:
        print("错误: 数据集为空")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print(f"数据集包含 {len(full_dataset)} 个样本")

    os.makedirs('./models11.17', exist_ok=True)
    os.makedirs('./visualizations', exist_ok=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []
    best_overall_metrics = None
    best_overall_pred = None
    best_overall_true = None
    best_train_losses = []
    best_val_losses = []

    # 初始化分析器（如果可用）
    if VISUALIZATION_AVAILABLE:
        gpga_analyzer = LatentSpaceAnalyzer()
        vanilla_analyzer = LatentSpaceAnalyzer()

    # 创建Vanilla VAE模型用于对比
    vanilla_model = create_vanilla_vae().to(device)

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n=== 训练折叠 {fold + 1}/5 ===")
        # 初始化早停相关变量
        best_val_loss = float('inf')
        best_composite_metric = float('-inf')
        best_loss_epoch = 0
        best_composite_epoch = 0
        patience_counter_loss = 0
        patience_counter_metric = 0
        train_losses = []
        val_losses = []
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        )

        # 初始化GPGA-VAE模型
        model = SuperDiffusionVAE(use_physics=True).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=config.warmup_epochs * len(train_loader)
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(config.epochs - config.warmup_epochs) * len(train_loader),
            eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs * len(train_loader)]
        )

        # 为GPGA-VAE和Vanilla VAE分别创建损失函数
        gpga_loss_fn = PhysicsGuidedLoss().to(device)
        vanilla_loss_fn = PhysicsGuidedLoss().to(device)

        scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(config.epochs):
            gpga_loss_fn.set_epoch(epoch)
            avg_train_loss, avg_recon_loss, avg_active_ratio = train_epoch(
                model, train_loader, optimizer, gpga_loss_fn, scheduler, scaler, device, epoch
            )
            train_losses.append(avg_train_loss)

            avg_val_loss, val_metrics, val_pred, val_true = validate_epoch(
                model, val_loader, gpga_loss_fn, device,
                dataset_generator.target_mean, dataset_generator.target_std
            )
            val_losses.append(avg_val_loss)

            # 每5个epoch收集一次潜在空间数据（如果可视化可用）
            if VISUALIZATION_AVAILABLE and epoch % 5 == 0:
                gpga_analyzer.collect_samples(model, val_loader, device, epoch)
                vanilla_analyzer.collect_samples(vanilla_model, val_loader, device, epoch)

            # 简洁的epoch报告
            print(f"\nEpoch {epoch + 1}/{config.epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"  Recon Loss: {avg_recon_loss:.4f} | Active Ratio: {avg_active_ratio:.2f}")
            print(f"  Quad Weight: {gpga_loss_fn.get_quad_weight():.2f}")

            # 打印6个参数的R²指标
            print("\n  Validation R²:")
            labels = [
                'normalized_E1', 'normalized_E2', 'normalized_E3',
                'avg_shear_modulus', 'num_faces', 'relative_density', 'anisotropy_index'
            ]
            for i, label in enumerate(labels):
                r2 = val_metrics[label]['R²']

                if label == 'num_faces':
                    acc = val_metrics[label]['Accuracy']
                    print(f"    {label:<20}: R²={r2:.4f}, Acc={acc:.4f}")
                elif label == 'anisotropy_index':
                    mae = val_metrics[label]['MAE']
                    print(f"    {label:<20}: R²={r2:.4f}, MAE={mae:.4f}")
                else:  # 其他连续属性
                    mape = val_metrics[label]['MAPE']
                    smape = val_metrics[label]['sMAPE']
                    print(f"    {label:<20}: R²={r2:.4f}, MAPE={mape:.2f}%, sMAPE={smape:.2f}%")

            # ==================== 综合指标早停策略 ====================
            # 计算综合指标（加权平均）
            composite_metric = (
                    0.25 * val_metrics['normalized_E1']['R²'] +  # E1预测重要性
                    0.20 * val_metrics['normalized_E2']['R²'] +  # E2预测重要性
                    0.20 * val_metrics['normalized_E3']['R²'] +  # E3预测重要性
                    0.15 * val_metrics['num_faces']['Accuracy'] +  # 面数准确率重要性
                    0.10 * val_metrics['anisotropy_index']['R²'] +  # 各向异性预测重要性
                    0.10 * (1 - avg_val_loss / max(1, val_losses[0]))  # 损失相对改进
            )

            print(f"  Composite Metric: {composite_metric:.4f}")

            # 保存最佳综合指标模型
            if epoch == 0 or composite_metric > best_composite_metric:
                best_composite_metric = composite_metric
                best_composite_epoch = epoch
                patience_counter_metric = 0
                torch.save(model.state_dict(), f"./models11.17/model_fold{fold + 1}_best_metric.pt")
                current_best_metrics = val_metrics
                best_val_pred = val_pred
                best_val_true = val_true
                print(f"  ★ 保存最佳综合指标模型 (指标: {best_composite_metric:.4f})")
            else:
                patience_counter_metric += 1

            # 同时保存最佳损失模型（用于对比）
            if epoch == 0 or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_loss_epoch = epoch
                patience_counter_loss = 0
                torch.save(model.state_dict(), f"./models11.17/model_fold{fold + 1}_best_loss.pt")
                print(f"  💾 保存最佳损失模型 (损失: {best_val_loss:.4f})")
            else:
                patience_counter_loss += 1

            # 双重早停条件：综合指标或损失都无改善
            if (patience_counter_metric >= config.patience and
                    patience_counter_loss >= config.patience):
                print(f"  ! 早停触发 (连续 {config.patience} 个epoch无改善)")
                print(f"     最佳综合指标: {best_composite_metric:.4f} @ epoch {best_composite_epoch + 1}")
                print(f"     最佳验证损失: {best_val_loss:.4f} @ epoch {best_loss_epoch + 1}")
                break

            # 仅损失无改善但综合指标还在提升，继续训练
            elif patience_counter_loss >= config.patience:
                print(f"  ⚠️  损失无改善但综合指标可能还在提升，继续训练...")

            # 仅综合指标无改善但损失还在下降，继续训练
            elif patience_counter_metric >= config.patience:
                print(f"  ⚠️  综合指标无改善但损失还在下降，继续训练...")

        all_metrics.append(current_best_metrics)

        # 为每个fold绘制误差分布图
        labels = [
            'normalized_E1', 'normalized_E2', 'normalized_E3',
            'avg_shear_modulus', 'num_faces'
        ]
        for i, label in enumerate(labels):
            errors = np.abs(best_val_true[:, i] - best_val_pred[:, i])
            plot_error_distribution(errors, f"fold{fold + 1}_{label}")

        # 单独处理anisotropy_index（需要重新计算）
        true_E1 = best_val_true[:, 0]
        true_E2 = best_val_true[:, 1]
        true_E3 = best_val_true[:, 2]
        true_anisotropy = ((true_E1 - true_E2) ** 2 + (true_E2 - true_E3) ** 2 + (true_E3 - true_E1) ** 2) / \
                          (2 * (true_E1 ** 2 + true_E2 ** 2 + true_E3 ** 2) + 1e-10)
        true_anisotropy = np.sqrt(true_anisotropy)

        pred_E1 = best_val_pred[:, 0]
        pred_E2 = best_val_pred[:, 1]
        pred_E3 = best_val_pred[:, 2]
        pred_anisotropy = ((pred_E1 - pred_E2) ** 2 + (pred_E2 - pred_E3) ** 2 + (pred_E3 - pred_E1) ** 2) / \
                          (2 * (pred_E1 ** 2 + pred_E2 ** 2 + pred_E3 ** 2) + 1e-10)
        pred_anisotropy = np.sqrt(pred_anisotropy)

        anisotropy_errors = np.abs(true_anisotropy - pred_anisotropy)
        plot_error_distribution(anisotropy_errors, f"fold{fold + 1}_anisotropy_index")

        if best_overall_metrics is None or best_val_loss < best_overall_metrics['normalized_E1']['MSE']:
            best_overall_metrics = current_best_metrics
            best_overall_pred = best_val_pred
            best_overall_true = best_val_true
            best_train_losses = train_losses.copy()
            best_val_losses = val_losses.copy()

        # 每个fold结束后保存可视化（只在第一个fold进行详细可视化，且可视化可用时）
        if fold == 0 and VISUALIZATION_AVAILABLE:
            # 潜在空间对比
            gpga_analyzer.visualize_latent_space_comparison(
                vanilla_analyzer,
                f"./visualizations/latent_space_comparison_fold{fold + 1}.png"
            )

            # 物理约束演化
            gpga_analyzer.plot_physical_constraint_evolution(
                model.validator,
                f"./visualizations/constraint_evolution_fold{fold + 1}.png"
            )

            # 违规案例对比
            visualize_invalid_cases(
                model, vanilla_model, val_loader, device,
                f"./visualizations/invalid_cases_comparison_fold{fold + 1}.png"
            )

    print("\n=== 交叉验证结果汇总 ===")
    for i, fold_metrics in enumerate(all_metrics):
        print(f"\n折叠 {i + 1} 指标:")
        labels = [
            'normalized_E1', 'normalized_E2', 'normalized_E3',
            'avg_shear_modulus', 'num_faces', 'relative_density', 'anisotropy_index'
        ]
        for label in labels:
            metric = fold_metrics[label]
            if label == 'num_faces':
                print(f"  {label}: R²={metric['R²']:.4f}, Acc={metric['Accuracy']:.4f}")
            elif label == 'anisotropy_index':
                # 各向异性指数没有sMAPE，使用MAPE
                print(f"  {label}: R²={metric['R²']:.4f}, MAE={metric['MAE']:.4f}, MAPE={metric['MAPE']:.2f}%")
            else:
                print(f"  {label}: R²={metric['R²']:.4f}, MAPE={metric['MAPE']:.2f}%, sMAPE={metric['sMAPE']:.2f}%")

    print("\n=== 最佳模型指标 ===")
    labels = [
        'normalized_E1', 'normalized_E2', 'normalized_E3',
        'avg_shear_modulus', 'num_faces', 'relative_density', 'anisotropy_index'
    ]
    for label in labels:
        metric = best_overall_metrics[label]
        if label == 'num_faces':
            print(f"  {label}: R²={metric['R²']:.4f}, Acc={metric['Accuracy']:.4f}")
        elif label == 'anisotropy_index':
            # 各向异性指数没有sMAPE，使用MAPE
            print(f"  {label}: R²={metric['R²']:.4f}, MAE={metric['MAE']:.4f}, MAPE={metric['MAPE']:.2f}%")
        else:
            print(f"  {label}: R²={metric['R²']:.4f}, MAPE={metric['MAPE']:.2f}%, sMAPE={metric['sMAPE']:.2f}%")

    # 保存最佳模型
    torch.save(model.state_dict(), "./models11.17/best_model.pt")
    print("\n最佳模型已保存至: ./models11.17/best_model.pt")

    # 绘制预测与真实值对比图
    plot_predictions_vs_true(best_overall_true, best_overall_pred, best_overall_metrics, "best_predictions_vs_true")

    # 绘制损失曲线
    plot_loss_curve(best_train_losses, best_val_losses)

    print("\n训练完成!")


if __name__ == "__main__":
    train_model()