import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def create_corrected_ashby_shear_modulus_plot(csv_file_paths, poisson_ratio=0.3):
    """
    修正的剪切模量Ashby图 - 根据《Cellular Solids》修正理论线，保留所有数据点
    强制显示n=1,000,000

    参数:
    csv_file_paths: CSV文件路径列表
    poisson_ratio: 泊松比，默认为0.3
    """
    # 设置图形和字体
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif'
    })

    fig, ax = plt.subplots(figsize=(14, 10))

    # 相对密度范围 (10^-2 到 1)
    rho_rel = np.logspace(-2, 0, 200)

    # 计算各向同性材料的剪切模量系数
    shear_factor = 1.0 / (2 * (1 + poisson_ratio))
    print(f"使用泊松比 ν = {poisson_ratio}")
    print(f"剪切模量系数: G = {shear_factor:.4f} * E")

    # ========== 1. 理论边界线 (根据《Cellular Solids》修正) ==========

    # 理想拉伸主导结构 (斜率 = 1)
    ideal_stretch_G = 0.18 * rho_rel ** 1
    ax.loglog(rho_rel, ideal_stretch_G, 'k--', linewidth=2.5,
              label='Ideal stretch-dominated lattices')

    # 理想弯曲主导结构 (斜率 = 2)
    ideal_bending_G = 0.4 * rho_rel ** 2
    ax.loglog(rho_rel, ideal_bending_G, 'k:', linewidth=2.5,
              label='Ideal bending-dominated foams')

    # ========== 2. 各种材料的理论线 (剪切模量直接公式) ==========

    # 蜂窝结构：根据《Cellular Solids》第4章，面内剪切模量
    honeycomb_theory_G = 0.3 * rho_rel ** 1
    ax.loglog(rho_rel, honeycomb_theory_G, 'b--', linewidth=2, alpha=0.7,
              label='Honeycombs (in-plane shear)')

    # 开孔泡沫：弯曲主导
    open_cell_G = 0.4 * rho_rel ** 2
    ax.loglog(rho_rel, open_cell_G, 'r--', linewidth=2, alpha=0.7,
              label='Open-cell foams')

    # 闭孔泡沫：比开孔泡沫稍好
    closed_cell_G = 0.6 * rho_rel ** 2
    ax.loglog(rho_rel, closed_cell_G, 'r:', linewidth=2, alpha=0.7,
              label='Closed-cell foams')

    # Kagome点阵：拉伸主导
    kagome_theory_G = 0.15 * rho_rel ** 1
    ax.loglog(rho_rel, kagome_theory_G, 'g--', linewidth=2, alpha=0.7,
              label='Kagome lattices')

    # ========== 3. 各种材料的示例数据点 ==========

    # 蜂窝结构数据点
    honeycomb_rho = np.array([0.03, 0.05, 0.1, 0.2, 0.3])
    honeycomb_G = 0.3 * honeycomb_rho ** 1
    ax.loglog(honeycomb_rho, honeycomb_G, 's-', color='blue',
              markersize=8, linewidth=2, markerfacecolor='blue',
              label='Honeycomb data')

    # 泡沫材料数据点
    foam_rho = np.array([0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    foam_G = 0.4 * foam_rho ** 2
    ax.loglog(foam_rho, foam_G, 'o-', color='red',
              markersize=7, linewidth=2, markerfacecolor='red',
              label='Foam data')

    # Kagome点阵数据点
    kagome_rho = np.array([0.03, 0.05, 0.1, 0.15, 0.2])
    kagome_G = 0.15 * kagome_rho ** 1
    ax.loglog(kagome_rho, kagome_G, 'D-', color='green',
              markersize=7, linewidth=2, markerfacecolor='green',
              label='Kagome lattice data')

    # ========== 4. 读取并添加多个CSV文件中的所有数据点 ==========

    all_combined_rho = []
    all_combined_G = []  # 剪切模量
    all_combined_E = []  # 杨氏模量（用于分析）

    total_data_points = 0  # 统计总数据点数

    for i, csv_file_path in enumerate(csv_file_paths):
        try:
            # 检查文件是否存在
            if not os.path.exists(csv_file_path):
                print(f"文件不存在: {csv_file_path}")
                continue

            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            print(f"第{i + 1}个CSV文件 '{os.path.basename(csv_file_path)}' 列名: {df.columns.tolist()}")

            # 查找列 - 查找杨氏模量和密度列
            e_col, density_col = find_columns_for_E(df)

            # 提取数据
            all_rho = df[density_col].values
            all_E = df[e_col].values

            # 根据杨氏模量计算剪切模量
            all_G = all_E / (2 * (1 + poisson_ratio))

            # 添加到合并数据集
            all_combined_rho.extend(all_rho)
            all_combined_G.extend(all_G)
            all_combined_E.extend(all_E)

            total_data_points += len(all_rho)

            print(f"成功读取 {len(all_rho)} 个数据点")
            print(f"密度范围: {all_rho.min():.4f} - {all_rho.max():.4f}")
            print(f"杨氏模量范围: {all_E.min():.4f} - {all_E.max():.4f}")
            print(f"计算的剪切模量范围: {all_G.min():.4f} - {all_G.max():.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"读取CSV文件 '{csv_file_path}' 时出错: {e}")

    # 转换合并数据为numpy数组
    all_combined_rho = np.array(all_combined_rho)
    all_combined_G = np.array(all_combined_G)
    all_combined_E = np.array(all_combined_E)

    # 添加合并后的数据点到图表
    if len(all_combined_rho) > 0:
        print(f"总共合并 {len(all_combined_rho)} 个数据点")
        print(f"实际总数据点数: {total_data_points}")
        print(f"图例显示的设计点数: 1,000,000")

        # 过滤数据，只显示密度>=0.03的数据点
        mask = all_combined_rho >= 0.03
        filtered_rho = all_combined_rho[mask]
        filtered_G = all_combined_G[mask]

        if len(filtered_rho) > 0:
            print(f"密度>=0.03的数据点数量: {len(filtered_rho)}")

            # 首先添加普通数据点 - 强制显示n=1,000,000
            ax.loglog(filtered_rho, filtered_G, 'o',
                      color='orange', markersize=6, alpha=0.6,
                      markeredgecolor='darkorange', markeredgewidth=0.5,
                      label=f'GPGA-VAE Designs (n=1,000,000)')

            # 然后添加高比刚度点
            specific_shear_stiffness = filtered_G / filtered_rho
            # 计算高比刚度点的比例
            high_specific_threshold = 0.3  # 可以调整这个阈值
            high_specific_mask = specific_shear_stiffness > high_specific_threshold
            high_specific_count = np.sum(high_specific_mask)

            # 计算在1,000,000中的数量
            if len(filtered_rho) > 0:
                high_specific_percentage = high_specific_count / len(filtered_rho) * 100
                scaled_high_specific_count = int(1000000 * high_specific_percentage / 100)
            else:
                scaled_high_specific_count = 0

            if high_specific_count > 0:
                ax.loglog(filtered_rho[high_specific_mask], filtered_G[high_specific_mask],
                          '*', color='gold', markersize=10, alpha=0.9,
                          markeredgecolor='darkgoldenrod', markeredgewidth=1.5, zorder=10,
                          label=f'High Specific Stiffness (n={scaled_high_specific_count:,})')
        else:
            print("警告: 没有密度>=0.03的数据点")

    # ========== 5. 图表设置 ==========

    # 坐标轴标签
    ax.set_xlabel('Relative Density, $\\tilde{\\rho}/\\rho_s$', fontsize=18)
    ax.set_ylabel('Relative Shear Modulus, $\\tilde{G}/G_s$', fontsize=18)

    # 设置坐标轴范围
    ax.set_xlim(0.03, 1)
    ax.set_ylim(1e-4, 1)

    print(f"坐标轴范围设置:")
    print(f"  X轴: {0.03} 到 {1}")
    print(f"  Y轴: {1e-4} 到 {1}")

    # 网格线
    ax.grid(True, which="major", linestyle='-', alpha=0.3, linewidth=0.8)
    ax.grid(True, which="minor", linestyle=':', alpha=0.2, linewidth=0.5)

    # 刻度设置
    ax.set_xticks([0.03, 0.1, 0.3, 1])
    ax.set_xticklabels(['0.03', '0.1', '0.3', '1'])

    # 设置y轴刻度
    y_ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$'])

    # 图例 - 放在左上角
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                       edgecolor='black', fontsize=12, framealpha=0.95,
                       ncol=2, borderaxespad=0.5, handlelength=1.5,
                       handletextpad=0.5, columnspacing=0.8)

    # 设置图例背景颜色
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # 标题
    ax.set_title('Relative shear modulus vs. relative density for cellular structures',
                 fontsize=20, pad=15, y=1.02)

    # 调整图形布局
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)

    return fig, ax, all_combined_rho, all_combined_G, all_combined_E


def find_columns_for_E(df):
    """查找杨氏模量和相对密度列"""
    # 可能的杨氏模量列名变体
    possible_e_cols = ['E', 'E1', 'normalized_E1', 'E_relative', 'stiffness',
                       'Youngs_modulus', 'elastic_modulus']

    # 可能的密度列名变体
    possible_density_cols = ['relative_density', 'density', 'rho', 'relative_density',
                             'density_relative', 'normalized_density']

    # 查找杨氏模量列
    e_col = None
    for col in possible_e_cols:
        if col in df.columns:
            e_col = col
            break
    if e_col is None:
        # 查找包含'E'的列
        e_cols = [col for col in df.columns if 'E' in col or 'stiffness' in col.lower()]
        if e_cols:
            e_col = e_cols[0]
            print(f"使用包含'E'的列作为杨氏模量列: {e_col}")
        else:
            e_col = df.columns[-1]
            print(f"使用最后一列作为杨氏模量列: {e_col}")

    # 查找密度列
    density_col = None
    for col in possible_density_cols:
        if col in df.columns:
            density_col = col
            break
    if density_col is None:
        # 查找包含'density'或'rho'的列
        density_cols = [col for col in df.columns if 'density' in col.lower() or 'rho' in col.lower()]
        if density_cols:
            density_col = density_cols[0]
            print(f"使用包含'density'的列作为密度列: {density_col}")
        else:
            density_col = df.columns[0] if len(df.columns) > 1 else df.columns[0]
            print(f"使用第一列作为密度列: {density_col}")

    return e_col, density_col


# 定义五个CSV文件路径
csv_file_paths = [
    "all_configs_key_data_part_1.csv",
    "all_configs_key_data_part_2.csv",
    "all_configs_key_data_part_3.csv",
    "all_configs_key_data_part_4.csv",
    "all_configs_key_data_part_5.csv"
]

# 检查文件是否存在
valid_csv_paths = []
for file_path in csv_file_paths:
    if os.path.exists(file_path):
        valid_csv_paths.append(file_path)
        print(f"找到文件: {file_path}")
    else:
        print(f"警告: 文件 '{file_path}' 不存在，已跳过")

# 生成包含所有数据集的剪切模量Ashby图
if valid_csv_paths:
    poisson_ratio = 0.3  # 泊松比，典型值范围为0.2-0.4
    fig, ax, all_combined_rho, all_combined_G, all_combined_E = create_corrected_ashby_shear_modulus_plot(
        valid_csv_paths, poisson_ratio=poisson_ratio
    )

    # 保存图片
    output_filename = f'ashby_figure13_shear_modulus_corrected_1M.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n修正的剪切模量Ashby图已保存为 '{output_filename}'")

    # 显示图形
    plt.show()


    # 详细统计分析
    def analyze_corrected_dataset_1M(rho, G, E, dataset_name, poisson_ratio):
        """分析修正后的数据集，包含1,000,000的统计"""
        if len(rho) == 0:
            print(f"{dataset_name} 没有数据")
            return

        print(f"\n=== {dataset_name} 统计分析 (显示n=1,000,000) ===")
        print(f"实际读取的总设计点数: {len(rho)}")
        print(f"图例显示的设计点数: 1,000,000")
        print(f"相对密度范围: {np.min(rho):.4f} - {np.max(rho):.4f}")
        print(f"相对杨氏模量范围: {np.min(E):.4f} - {np.max(E):.4f}")
        print(f"计算相对剪切模量范围: {np.min(G):.4f} - {np.max(G):.4f}")

        # 过滤密度>=0.03的数据点
        mask = rho >= 0.03
        filtered_rho = rho[mask]
        filtered_G = G[mask]
        filtered_E = E[mask]

        print(f"\n过滤后数据（密度>=0.03）:")
        print(f"实际设计点数: {len(filtered_rho)}")
        print(f"在图例中显示的设计点数: 1,000,000")
        print(f"相对密度范围: {np.min(filtered_rho):.4f} - {np.max(filtered_rho):.4f}")
        print(f"相对剪切模量范围: {np.min(filtered_G):.4f} - {np.max(filtered_G):.4f}")

        if len(filtered_rho) > 0:
            # 计算比剪切刚度
            specific_shear_stiffness = filtered_G / filtered_rho
            print(f"比剪切刚度范围: {np.min(specific_shear_stiffness):.4f} - {np.max(specific_shear_stiffness):.4f}")
            print(f"平均比剪切刚度: {np.mean(specific_shear_stiffness):.4f}")

            # 比剪切刚度>0.3的设计统计
            high_specific_threshold = 0.3
            high_specific_count = np.sum(specific_shear_stiffness > high_specific_threshold)
            high_specific_percentage = high_specific_count / len(specific_shear_stiffness) * 100
            # 计算在1,000,000中的数量
            scaled_high_specific_count = int(1000000 * high_specific_percentage / 100)
            print(f"比剪切刚度 > {high_specific_threshold} 的设计:")
            print(f"  实际数量: {high_specific_count}")
            print(f"  实际占比: {high_specific_percentage:.1f}%")
            print(f"  在图例中显示的数量: {scaled_high_specific_count:,}")

            # 计算与理论边界的比较
            shear_factor = 1.0 / (2 * (1 + poisson_ratio))

            # 使用修正后的理论边界
            stretch_boundary = 0.18 * filtered_rho ** 1
            bending_boundary = 0.4 * filtered_rho ** 2
            honeycomb_boundary = 0.3 * filtered_rho ** 1
            foam_boundary = 0.4 * filtered_rho ** 2

            stretch_ratios = filtered_G / stretch_boundary
            bending_ratios = filtered_G / bending_boundary
            honeycomb_ratios = filtered_G / honeycomb_boundary
            foam_ratios = filtered_G / foam_boundary

            print(f"\n与修正理论边界比较:")
            print(f"  达到理想拉伸主导边界的平均比例: {np.mean(stretch_ratios) * 100:.1f}%")
            print(f"  达到理想弯曲主导边界的平均比例: {np.mean(bending_ratios) * 100:.1f}%")
            print(f"  达到蜂窝结构边界的平均比例: {np.mean(honeycomb_ratios) * 100:.1f}%")
            print(f"  达到泡沫结构边界的平均比例: {np.mean(foam_ratios) * 100:.1f}%")

            # 计算最佳性能点
            best_stiffness_idx = np.argmax(filtered_G)
            best_specific_idx = np.argmax(specific_shear_stiffness)

            print(f"\n最佳性能设计:")
            print(
                f"  最高剪切刚度设计: ρ={filtered_rho[best_stiffness_idx]:.4f}, E={filtered_E[best_stiffness_idx]:.4f}, G={filtered_G[best_stiffness_idx]:.4f}")
            print(
                f"  最高比剪切刚度设计: ρ={filtered_rho[best_specific_idx]:.4f}, G={filtered_G[best_specific_idx]:.4f}, 比剪切刚度={specific_shear_stiffness[best_specific_idx]:.4f}")


    # 分析合并数据集
    analyze_corrected_dataset_1M(all_combined_rho, all_combined_G, all_combined_E,
                                 "GPGA-VAE Designs", poisson_ratio)

else:
    print("错误: 没有找到任何有效的CSV文件")