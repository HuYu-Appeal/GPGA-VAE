import numpy as np
import pandas as pd


# 读取节点数据
def read_nodes(file_name):
    nodes = []
    with open(file_name, 'r') as file:
        for line in file:
            node = tuple(map(float, line.strip().split(',')))
            nodes.append(node)
    return np.array(nodes)


# 读取面数据
def read_faces(file_name):
    df = pd.read_excel(file_name)
    return df


# 计算法向量和点到原点的距离
def calculate_normal_and_distance(vertices):
    v1 = np.array(vertices[1]) - np.array(vertices[0])
    v2 = np.array(vertices[2]) - np.array(vertices[0])
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # 归一化
    d = -np.dot(normal, np.array(vertices[0]))
    return normal, d


# 检查点是否在平面上
def point_on_plane(point, normal, d, atol=1e-6):
    dot_product = np.dot(normal, point) + d
    return np.isclose(dot_product, 0, atol=atol)


# 确定面上的点并进行标号
def label_points_on_faces(nodes, faces):
    labeled_nodes = {i: [] for i in range(len(faces))}  # 初始化每个面的节点列表
    for face_index, face in faces.iterrows():
        # 提取顶点坐标并转换为浮点数
        vertices = [
            [float(coord) for coord in face['顶点1'].strip('[]').split(', ')],
            [float(coord) for coord in face['顶点2'].strip('[]').split(', ')],
            [float(coord) for coord in face['顶点3'].strip('[]').split(', ')]
        ]
        normal, d = calculate_normal_and_distance(vertices)
        for node_index, node in enumerate(nodes):
            if point_on_plane(node, normal, d):
                labeled_nodes[face_index].append(node_index)
    return labeled_nodes


# 主程序
def main():
    # 读取节点和面数据
    nodes = read_nodes('nodes3.csv.txt')
    faces = read_faces('output_8-1_with_results.xlsx')

    # 确定面上的点并进行标号
    labeled_nodes = label_points_on_faces(nodes, faces)

    # 打印结果
    for face_index, points in labeled_nodes.items():
        print(f"Face {face_index + 1} contains points: {points}")


if __name__ == '__main__':
    main()