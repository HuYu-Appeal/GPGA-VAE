Homogenization Code for 3D Plate-Lattice Structures
MATLAB implementation of numerical homogenization for computing effective elastic properties of periodic plate-lattice structures.

Files
homo3D.m - Core homogenization function (based on Dong et al. 2018)

main_homogenization.m - Batch processing script for HDF5 voxel models

visual.m - 3D Young's modulus surface visualization

Features
Periodic boundary conditions with reduced degrees of freedom

Efficient PCG solver for large-scale systems

Batch processing of multiple voxel models from HDF5 files

Automatic connectivity check to skip invalid configurations

Relative density normalization for fair comparison

Extracts 6 key properties: E₁, E₂, E₃, G₁₂, G₂₃, G₁₃

UsageBatch Processing
matlab
main_homogenization
Configure parameters in the script:

h5File: Input HDF5 file with voxel models

outputCSV: Output CSV file for results

lambda, mu: Lamé parameters (E=200 GPa, ν=0.3 → λ=115.4, μ=76.9)

lx, ly, lz: Unit cell dimensions

Single Model Processing
matlab
% Load voxel model
voxel = h5read('voxel_models.h5', '/model_1/voxel');
voxel = logical(voxel);

% Compute homogenized properties
result = homo3D(lx, ly, lz, lambda, mu, voxel);

% Access results
E1 = result.E1;  % Normalized Young's modulus in X-direction
G12 = result.G12; % Normalized shear modulus
Visualization
matlab
visual(result.CH); % Plot 3D Young's modulus surface
Input Format
HDF5 file should contain:

Datasets: /model_N/voxel - 3D logical array (1 = solid, 0 = void)

Attributes: density, active_faces (optional)

Output
CSV file with columns:

key: Model identifier

E1, E2, E3: Normalized Young's moduli

G12, G23, G13: Normalized shear moduli

Reference
Dong G, Tang Y, Zhao Y. A 149 Line Homogenization Code for Three-Dimensional Cellular Materials Written in MATLAB. ASME J. Eng. Mater. Technol. 2018;141(1):011005. doi:10.1115/1.4040555

Notes
Unit cell dimensions should match those used in voxel generation

Results are normalized by relative density for fair comparison

PCG solver parameters can be adjusted for convergence

