import h5py
import numpy as np
import random

def load_adjacency_matrices_from_hdf5(file_path):
    adjacency_matrices = {}
    with h5py.File(file_path, 'r') as hdf5_file:
        for key in hdf5_file.keys():
            adjacency_matrices[key] = hdf5_file[key][:]
    return adjacency_matrices

def save_sampled_adjacency_matrices(sampled_matrices, hdf5_file_path):
    with h5py.File(hdf5_file_path, 'w') as hdf5_file:
        for key in sampled_matrices:
            hdf5_file.create_dataset(key, data=sampled_matrices[key])
    print(f"Sampled adjacency matrices saved to {hdf5_file_path}")

def main():
    hdf5_file_path = 'adjacency_matrice1.h5'  # 原始HDF5文件路径
    sampled_hdf5_file_path = 'sampled_adjacency_matrice.h5'  # 采样后的HDF5文件路径

    # 从HDF5文件中加载所有邻接矩阵
    adjacency_matrices = load_adjacency_matrices_from_hdf5(hdf5_file_path)

    # 按k值分组
    matrices_by_k = {}
    for key in adjacency_matrices:
        parts = key.split('_')
        k = int(parts[2])  # 提取k值
        if k not in matrices_by_k:
            matrices_by_k[k] = []
        matrices_by_k[k].append((key, adjacency_matrices[key]))

    # 计算总组合数
    total_combinations = sum(len(matrices_by_k[k]) for k in matrices_by_k)
    print(f"Total combinations: {total_combinations}")

    # 确定采样数量（总数的1%）
    sample_size_total = max(1, int(total_combinations * 0.01))
    print(f"Total sample size: {sample_size_total}")

    # 按k值分配采样数量
    sampled_matrices = {}
    remaining_sample_size = sample_size_total

    # 确保20个面全部参与的结构被选中
    full_participation_key = None
    for key in adjacency_matrices:
        parts = key.split('_')
        if len(parts) >= 4 and parts[2] == '20':
            full_participation_key = key
            break
    if full_participation_key is not None:
        sampled_matrices[full_participation_key] = adjacency_matrices[full_participation_key]
        remaining_sample_size -= 1

    # 按k值分配剩余的采样数量
    for k in matrices_by_k:
        if remaining_sample_size <= 0:
            break
        k_count = len(matrices_by_k[k])
        k_ratio = k_count / total_combinations
        k_sample_size = max(1, int(k_ratio * sample_size_total))
        k_sample_size = min(k_sample_size, remaining_sample_size)

        if k_sample_size > 0:
            sampled = random.sample(matrices_by_k[k], k_sample_size)
            for item in sampled:
                key, matrix = item
                if key not in sampled_matrices:
                    sampled_matrices[key] = matrix
                    remaining_sample_size -= 1
                    if remaining_sample_size <= 0:
                        break

    # 保存采样后的邻接矩阵到新的HDF5文件
    save_sampled_adjacency_matrices(sampled_matrices, sampled_hdf5_file_path)

    print("Sampling completed successfully.")

if __name__ == "__main__":
    main()