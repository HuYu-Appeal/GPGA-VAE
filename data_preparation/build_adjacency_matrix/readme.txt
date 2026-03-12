Data Generation and Preprocessing Tools
Utility scripts for generating adjacency matrices, computing face features, and processing node data.

Files
ADJUNCE_optimized.py - Generate adjacency matrices from Excel face data and save to HDF5

adjacency_utils.py - Utility functions for face intersection判断 (depends on utils.py)

point_on_plane.py - Compute node indices lying on each face

utils.py - Point-on-plane判断 function

Features
Adjacency Matrix Generation: Create 20×20 adjacency matrices based on face intersection criteria

Face Intersection判断: Implementation of the intersection criterion from the paper (Eq. 1)

Node Labeling: Determine which nodes belong to each face

Point-Plane Relationship: High-precision判断 of points on planes (tolerance 1e-6)

Usage
Generate Adjacency Matrices
bash
python ADJUNCE_optimized.py
Input:

output_8-1_with_results.xlsx - Geometric features of 20 faces

nodes1.csv.txt - Node coordinates file

Output:

adjacency_matrice1.h5 - Adjacency matrices for all face combinations

Compute Node Labels on Faces
bash
python point_on_plane.py
Input:

nodes3.csv.txt - Node coordinates file

output_8-1_with_results.xlsx - Face data

Output:

Console print of node indices for each face

Input File Format
Face Data Excel (output_8-1_with_results.xlsx)
Required columns:

顶点1, 顶点2, 顶点3, 顶点4 - Format as string "[x, y, z]"

Other geometric feature columns (normal vectors, area, centroid, etc.)

Node File (nodes*.csv.txt)
One node coordinate per line:

text
x, y, z
Output File Format
HDF5 Adjacency Matrix (adjacency_matrice1.h5)
Key format: adjacency_matrix_r_faceID_list

Dataset: 20×20 binary matrix (1 indicates faces intersect)

Dependencies
bash
pip install numpy pandas h5py openpyxl
Core Functions
utils.py
python
point_on_plane(point, normal, d)
Checks if a point lies on a plane with tolerance 1e-6.

adjacency_utils.py
python
are_faces_intersecting(face1, face2, points)
Determines if two faces intersect (share at least 2 points).

