Large-Scale Design Space Exploration and Analysis
Tools for analyzing 1,000,000 generated plate-lattice configurations.

Files
大规模设计空间探索和分析10.30.py - Main analysis script (latest version)

大规模设计空间探索和分析1029.py - Version with relative density support

大规模设计空间探索和分析0922.py - Base version

寻找极端构型.py - Extreme configuration finder

寻找极端构型(1).py - Updated version with data export

Features
Generate 1M configurations from latent space

Extreme configuration search (high E1/E2/E3, high/low anisotropy)

Face importance analysis using Random Forest

Co-occurrence networks for high-performance designs

Geometry-property correlation analysis

Data export to CSV/Excel (5 files × 200k samples)

Key Outputsall_configs_key_data_part_1-5.csv - All 1M configurations with properties

top_10000_e1_density_data.csv - Top E1 designs for Ashby plots

face_importance_all.png - Face importance bar charts

top_1000_e1_network.png - Co-occurrence network visualization

e1_vs_anisotropy.png - Design space scatter plots

detailed_analysis_report.txt - Complete analysis report

Data Fields
Each configuration includes:

normalized_E1, E2, E3 - Directional stiffness

relative_density - Relative density

E1, E2, E3 - Actual stiffness (normalized × density)

anisotropy - Anisotropy index

active_faces - List of activated faces (1-20)

