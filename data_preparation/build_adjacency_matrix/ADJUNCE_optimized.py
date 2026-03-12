import pandas as pd
import numpy as np
from itertools import combinations
import h5py

from adjacency_utils import are_faces_intersecting

def read_faces_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def load_points(file_path):
    try:
        points = set()
        with open(file_path, 'r') as file:
            for line in file:
                points.add(tuple(map(float, line.strip().split(','))))
        return points
    except Exception as e:
        print(f"Error loading points: {e}")
        return set()

def generate_adjacency_matrix(faces, points, combination):
    # 初始化一个全零的20x20矩阵
    adjacency_matrix = np.zeros((20, 20))

    # 将组合中的面编号转换为0-based索引
    sub_indices = list(combination)  # 直接使用1-based index

    # 处理当前组合中的面之间的相交关系
    for i in range(len(sub_indices)):
        for j in range(i + 1, len(sub_indices)):
            if are_faces_intersecting(faces.iloc[sub_indices[i]-1], faces.iloc[sub_indices[j]-1], points):
                adjacency_matrix[sub_indices[i]-1, sub_indices[j]-1] = 1
                adjacency_matrix[sub_indices[j]-1, sub_indices[i]-1] = 1

    return adjacency_matrix

def save_adjacency_matrix_to_hdf5(adjacency_matrix, hdf5_file_path, key):
    with h5py.File(hdf5_file_path, 'a') as hdf5_file:  # 使用'a'模式以追加方式打开文件
        if key in hdf5_file:
            del hdf5_file[key]  # 删除旧的数据集
            print(f"Existing key '{key}' found and removed.")
        hdf5_file.create_dataset(key, data=adjacency_matrix)
    print(f"Adjacency matrix saved to {hdf5_file_path} with key {key}")

def main():
    file_path = 'output_8-1_with_results.xlsx'  # Excel文件路径
    points_file = 'nodes1.csv.txt'  # 点数据文件路径
    faces = read_faces_from_excel(file_path)
    points = load_points(points_file)

    if faces is None or len(faces) < 2:
        print("Not enough faces data available.")
        return None

    num_faces = min(len(faces), 20)  # 只考虑前20个面

    print(f"Number of faces read: {num_faces}")  # 打印面的数量

    # 生成不同面组合的邻接矩阵并保存到HDF5文件
    hdf5_file_path = 'adjacency_matrice1.h5'
    for r in range(2, num_faces + 1):
        for combination in combinations(range(1, num_faces + 1), r):
            adjacency_matrix = generate_adjacency_matrix(faces, points, combination)
            key = f'adjacency_matrix_{r}_{"_".join(map(str, combination))}'
            save_adjacency_matrix_to_hdf5(adjacency_matrix, hdf5_file_path, key)

    print("Adjacency matrix generation and saving completed successfully.")

if __name__ == "__main__":
    main()