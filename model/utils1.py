import os
import numpy as np
import matplotlib
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


def compute_metrics(true_vals, pred_vals):
    """
    计算评估指标，包括6个直接预测参数和1个派生指标（各向异性指数）
    """
    # 确保输入形状一致
    if true_vals.shape != pred_vals.shape:
        raise ValueError(f"真实值和预测值形状不一致: true_vals={true_vals.shape}, pred_vals={pred_vals.shape}")

    # 计算各向异性指数（使用前3个参数E1, E2, E3）
    def calculate_anisotropy(E1, E2, E3):
        numerator = (E1 - E2) ** 2 + (E2 - E3) ** 2 + (E3 - E1) ** 2
        denominator = 2 * (E1 ** 2 + E2 ** 2 + E3 ** 2) + 1e-10
        return np.sqrt(numerator / denominator)

    # 从真实值中计算anisotropy_index
    true_E1 = true_vals[:, 0]
    true_E2 = true_vals[:, 1]
    true_E3 = true_vals[:, 2]
    true_anisotropy = calculate_anisotropy(true_E1, true_E2, true_E3)

    # 从预测值中计算anisotropy_index
    pred_E1 = pred_vals[:, 0]
    pred_E2 = pred_vals[:, 1]
    pred_E3 = pred_vals[:, 2]
    pred_anisotropy = calculate_anisotropy(pred_E1, pred_E2, pred_E3)

    # 评估6个直接预测的参数
    metrics = {}
    output_labels = ['normalized_E1', 'normalized_E2', 'normalized_E3',
                     'avg_shear_modulus', 'num_faces', 'relative_density']  # 增加相对密度

    for i, label in enumerate(output_labels):
        true_col = true_vals[:, i]
        pred_col = pred_vals[:, i]

        if label == 'num_faces':
            # 处理面数（离散值）
            metrics[label] = {
                'R²': r2_score(true_col, pred_col),
                'MSE': mean_squared_error(true_col, pred_col),
                'Accuracy': np.mean(np.abs(true_col - pred_col) < 0.5)
            }
        else:
            # 处理连续值属性
            ape = np.abs((true_col - pred_col) / (np.abs(true_col) + 1e-6)) * 100
            mape = np.mean(ape)
            smape = 200 * np.mean(
                np.abs(pred_col - true_col) / (np.abs(pred_col) + np.abs(true_col) + 1e-6))

            metrics[label] = {
                'MAPE': mape,
                'sMAPE': smape,
                'R²': r2_score(true_col, pred_col),
                'MSE': mean_squared_error(true_col, pred_col)
            }

    # 添加计算得到的各向异性指数指标
    metrics['anisotropy_index'] = {
        'MAE': np.mean(np.abs(true_anisotropy - pred_anisotropy)),
        'MedAE': np.median(np.abs(true_anisotropy - pred_anisotropy)),
        'R²': r2_score(true_anisotropy, pred_anisotropy),
        'MSE': mean_squared_error(true_anisotropy, pred_anisotropy),
        'MAPE': np.mean(np.abs((true_anisotropy - pred_anisotropy) / (np.abs(true_anisotropy) + 1e-6)) * 100)
    }

    return metrics



def evaluate(model, dataloader, device, target_mean, target_std):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)

            vertices = outputs['vertices']
            face_mask = outputs['face_mask']
            vertex_counts = outputs['vertex_counts']
            new_adj = model.decoder.compute_adjacency(vertices, vertex_counts, face_mask, outputs['adj_matrix'])
            outputs['adj_matrix'] = new_adj

            pred_denorm = outputs[
                              'pred'].detach().cpu().numpy() * target_std + target_mean  # 直接使用 target_std 和 target_mean
            # 保持现有的anisotropy_index计算逻辑不变（从E1,E2,E3计算得出）
            # 但确保在评估时正确传递和处理6个参数

            # 在evaluate函数中修改：
            true_denorm = batch['y'].detach().cpu().numpy() * np.concatenate(
                [target_std[:3], [target_std[3]], target_std[4:]]) + np.concatenate(
                [target_mean[:3], [target_mean[3]], target_mean[4:]])
            predictions.extend(pred_denorm)
            true_values.extend(true_denorm)

    true_array = np.array(true_values)
    pred_array = np.array(predictions)
    metrics = compute_metrics(true_array, pred_array)

    labels = ['normalized_E1', 'normalized_E2', 'normalized_E3',
              'avg_shear_modulus', 'num_faces', 'anisotropy_index']

    print("\n=== 评估指标 ===")
    for label in labels:
        metric = metrics[label]
        print(f"{label}:")
        if label == 'num_faces':
            print(f"  R²: {metric['R²']:.4f}")
            print(f"  MSE: {metric['MSE']:.4f}")
            print(f"  准确率: {metric['Accuracy']:.4f}")
        else:
            print(f"  R²: {metric['R²']:.4f}")
            print(f"  MSE: {metric['MSE']:.4f}")
            print(f"  MAPE: {metric['MAPE']:.2f}%")
            print(f"  sMAPE: {metric['sMAPE']:.2f}%")

    return metrics, pred_array, true_array


def evaluate_topology_reconstruction(model, dataloader, device):
    model.eval()
    jaccard_scores = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)

            # 获取真实和预测的面掩码
            true_mask = batch['face_mask'].cpu().numpy()
            pred_mask = outputs['face_mask'].cpu().numpy()

            # 计算Jaccard相似度
            for i in range(len(true_mask)):
                true_set = set(np.where(true_mask[i] > 0.5)[0])
                pred_set = set(np.where(pred_mask[i] > 0.5)[0])

                if len(true_set | pred_set) > 0:
                    jaccard = len(true_set & pred_set) / len(true_set | pred_set)
                    jaccard_scores.append(jaccard)

    return np.mean(jaccard_scores) if jaccard_scores else 0.0

def plot_predictions_vs_true(true_vals, pred_vals, metrics, save_name="predictions_vs_true"):
    labels = ['normalized_E1', 'normalized_E2', 'normalized_E3',
              'avg_shear_modulus', 'num_faces', 'relative_density', 'anisotropy_index']  # 增加相对密度

    plt.figure(figsize=(21, 14))  # 调整图形大小以适应7个子图
    for i, label in enumerate(labels):
        plt.subplot(3, 3, i+1)  # 改为3x3布局
        plt.scatter(true_vals[:, i], pred_vals[:, i], alpha=0.5)

        min_val = min(true_vals[:, i].min(), pred_vals[:, i].min())
        max_val = max(true_vals[:, i].max(), pred_vals[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        metric = metrics[label]
        if label == 'num_faces':
            title = f"{label}\nR²={metric['R²']:.3f}, MSE={metric['MSE']:.4f}, Acc={metric['Accuracy']:.4f}"
        elif label == 'anisotropy_index':
            title = f"{label}\nR²={metric['R²']:.3f}, MAE={metric['MAE']:.4f}"
        else:
            title = f"{label}\nR²={metric['R²']:.3f}, MAPE={metric['MAPE']:.2f}%"

        plt.title(title)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True)

    plt.tight_layout()
    save_path = os.path.abspath(f"{save_name}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"预测对比图已保存至: {save_path}")

def plot_error_distribution(errors, property_name):
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f"{property_name} Prediction Error Distribution")
    plt.xlabel("Absolute Error")
    save_path = os.path.abspath(f"{property_name}_error_dist.png")
    plt.savefig(save_path)
    plt.close()
    print(f"{property_name}误差分布图已保存至: {save_path}")

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Total Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.abspath('loss_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"损失曲线保存至: {save_path}")