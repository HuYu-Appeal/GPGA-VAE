SuperDiffusionVAE
A physics-guided VAE for predicting 3D microstructure properties.

Overview
This model predicts 5 properties of 3D polyhedral cells:

E₁, E₂, E₃ (elastic moduli)

Shear modulus

Number of active faces


Requirements
bash
pip install torch torch-geometric numpy pandas h5py scikit-learn matplotlib seaborn tqdm openpyxl
Data Files Needed
Place these in the project folder:

sampled_adjacency_matrice1.h5 - Adjacency matrices

output_8-1_with_results.xlsx - Face features

enhanced_homogenized_results_8001_with_density.csv - Target properties

Quick Start
bash
python train.py
Key Features
GNN encoder (GAT + GCN layers)

Physics-constrained decoder

5-fold cross-validation

Mixed precision training

Automatic model saving in ./models11.17/

Configuration
Edit config.py to adjust:

Batch size, learning rate, epochs

Latent dimension (default: 512)

Loss weights

Output
Best models saved in ./models11.17/

Plots saved as PNG files

Console shows R², MAPE for each property

