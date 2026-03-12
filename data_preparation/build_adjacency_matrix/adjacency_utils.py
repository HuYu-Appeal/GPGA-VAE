# adjacency_utils.py
import numpy as np
from utils import point_on_plane

def calculate_normal_and_distance(vertices):
    v1 = np.array(list(map(float, vertices[1].strip('[]').split(', ')))) - np.array(list(map(float, vertices[0].strip('[]').split(', '))))
    v2 = np.array(list(map(float, vertices[2].strip('[]').split(', ')))) - np.array(list(map(float, vertices[0].strip('[]').split(', '))))
    normal = np.cross(v1, v2)
    if np.linalg.norm(normal) == 0:
        return None, None
    normal = normal / np.linalg.norm(normal)  # 归一化
    d = -np.dot(normal, np.array(list(map(float, vertices[0].strip('[]').split(',')))))
    return normal, d

def are_faces_intersecting(face1, face2, points):
    p1_1, p1_2, p1_3 = face1['顶点1'], face1['顶点2'], face1['顶点3']
    p2_1, p2_2, p2_3 = face2['顶点1'], face2['顶点2'], face2['顶点3']

    normal1, d1 = calculate_normal_and_distance([p1_1, p1_2, p1_3])
    normal2, d2 = calculate_normal_and_distance([p2_1, p2_2, p2_3])

    shared_points_count = 0
    for point in points:
        if point_on_plane(point, normal1, d1) and point_on_plane(point, normal2, d2):
            shared_points_count += 1

    return shared_points_count >= 2