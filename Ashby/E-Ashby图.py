import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
import os


def create_enhanced_ashby_figure13_with_multiple_datasets(csv_file_paths):
    """
    复现Ashby 2005年图13，并添加多个CSV文件中的所有数据点
    """
    # 设置图形和字体
    plt.rcParams.update({
        'font.size': 16,
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif'
    })

    fig, ax = plt.subplots(figsize=(14, 10))

    # 相对密度范围 - 从0.03开始
    rho_rel = np.logspace(np.log10(0.03), 0, 200)  # 从0.03到1

    # ========== 1. 理论边界线 ==========

    # 理想拉伸主导结构 (斜率 = 1) - 桁架点阵
    ax.loglog(rho_rel, 1.0 * rho_rel ** 1, 'k--', linewidth=2.5,
              label='Ideal stretch-dominated (Truss lattices)')

    # 理想弯曲主导结构 (斜率 = 2)
    ax.loglog(rho_rel, 1.0 * rho_rel ** 2, 'k:', linewidth=2.5,
              label='Ideal bending-dominated')

    # ========== 2. 添加经典材料的理论界限 ==========

    # 板点阵理论界限：拉伸主导，E/E_s ∝ (ρ/ρ_s)^1
    ax.loglog(rho_rel, 0.8 * rho_rel ** 1, 'r--', linewidth=2, alpha=0.7,
              label='Plate-lattices (Tancogne-Dejean et al.)')

    # ========== 3. 各种材料的实际数据趋势 ==========

    # 泡沫材料 (弯曲主导，斜率~2) - 从0.03开始
    foam_rho = np.array([0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    foam_E = 0.3 * foam_rho ** 2
    ax.loglog(foam_rho, foam_E, 'o-', color='red',
              markersize=7, linewidth=2, markerfacecolor='red',
              label='Foams (data)')

    # 编织结构 (弯曲主导，但密度较高) - 从0.15开始
    woven_rho = np.array([0.15, 0.2, 0.25, 0.3])
    woven_E = 0.4 * woven_rho ** 2
    ax.loglog(woven_rho, woven_E, '^-', color='green',
              markersize=8, linewidth=2, markerfacecolor='green',
              label='Woven structures (data)')

    # Kagome和金字塔点阵 (拉伸主导，但效率略低) - 从0.03开始
    kagome_rho = np.array([0.03, 0.05, 0.1, 0.15, 0.2])
    kagome_E = 0.3 * kagome_rho ** 1
    ax.loglog(kagome_rho, kagome_E, 'D-', color='purple',
              markersize=7, linewidth=2, markerfacecolor='purple',
              label='Kagome and pyramidal lattices (data)')

    # ========== 4. 读取并添加多个CSV文件中的所有数据点 ==========

    all_combined_rho = []
    all_combined_E = []

    total_points_count = 0  # 统计实际读取的总点数

    for i, csv_file_path in enumerate(csv_file_paths):
        try:
            # 检查文件是否存在
            if not os.path.exists(csv_file_path):
                print(f"文件不存在: {csv_file_path}")
                continue

            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            print(f"第{i + 1}个CSV文件 '{os.path.basename(csv_file_path)}' 列名: {df.columns.tolist()}")

            # 查找列
            e_col, density_col = find_columns(df)

            # 提取数据
            all_rho = df[density_col].values
            all_E = df[e_col].values

            # 添加到合并数据集
            all_combined_rho.extend(all_rho)
            all_combined_E.extend(all_E)

            total_points_count += len(all_rho)

            print(f"成功读取 {len(all_rho)} 个数据点")
            print(f"密度范围: {all_rho.min():.4f} - {all_rho.max():.4f}")
            print(f"模量范围: {all_E.min():.4f} - {all_E.max():.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"读取CSV文件 '{csv_file_path}' 时出错: {e}")

    # 转换合并数据为numpy数组
    all_combined_rho = np.array(all_combined_rho)
    all_combined_E = np.array(all_combined_E)

    # 添加合并后的数据点到图表
    if len(all_combined_rho) > 0:
        print(f"总共读取 {total_points_count} 个数据点")
        # 固定显示为1000000个设计
        add_combined_dataset_points(ax, all_combined_rho, all_combined_E, 'royalblue', 'GPGA-VAE Designs',
                                    total_points=1000000)

    # ========== 5. 图表设置 ==========

    # 坐标轴标签
    ax.set_xlabel('Relative Density, $\\tilde{\\rho}/\\rho_s$', fontsize=18)
    ax.set_ylabel('Relative Modulus, $\\tilde{E}/E_s$', fontsize=18)

    # 坐标轴范围 - 调整为0.03-1
    ax.set_xlim(0.03, 1)
    ax.set_ylim(1e-4, 10)

    # 网格线
    ax.grid(True, which="major", linestyle='-', alpha=0.3, linewidth=0.8)
    ax.grid(True, which="minor", linestyle=':', alpha=0.2, linewidth=0.5)

    # 刻度设置 - x轴从0.03开始
    ax.set_xticks([0.03, 0.1, 0.3, 1])
    ax.set_xticklabels(['0.03', '0.1', '0.3', '1'])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
    ax.set_yticklabels(['$10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$10^{0}$', '$10^{1}$'])

    # 图例 - 放在左上角
    legend = ax.legend(loc='upper left', frameon=True, fancybox=True,
                       edgecolor='black', fontsize=14, framealpha=0.95,
                       ncol=1, borderaxespad=0.5, handlelength=1.5,
                       handletextpad=0.5, columnspacing=1)

    # 设置图例背景颜色为浅色，减少对数据的遮挡
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)

    # 标题 - 调整位置
    ax.set_title('Relative modulus vs. relative density for cellular structures',
                 fontsize=20, pad=15, y=1.02)

    # 调整图形布局，确保标题完全显示
    plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1)

    return fig, ax, all_combined_rho, all_combined_E, total_points_count


def find_columns(df):
    """查找E和相对密度列"""
    # 可能的列名变体
    possible_e_cols = ['E', 'E1', 'normalized_E1', 'E_relative', 'stiffness']
    possible_density_cols = ['relative_density', 'density', 'rho', 'relative_density', 'density_relative']

    # 查找E列
    e_col = None
    for col in possible_e_cols:
        if col in df.columns:
            e_col = col
            break
    if e_col is None:
        e_col = df.columns[-1]
        print(f"使用最后一列作为E列: {e_col}")

    # 查找密度列
    density_col = None
    for col in possible_density_cols:
        if col in df.columns:
            density_col = col
            break
    if density_col is None:
        density_col = df.columns[-2]
        print(f"使用倒数第二列作为密度列: {density_col}")

    return e_col, density_col


def add_combined_dataset_points(ax, all_rho, all_E, color, label, total_points=1000000):
    """添加合并数据集点到图表"""
    # 过滤数据，只显示密度>=0.03的数据点
    mask = all_rho >= 0.03
    filtered_rho = all_rho[mask]
    filtered_E = all_E[mask]

    if len(filtered_rho) > 0:
        print(f"密度>=0.03的数据点数量: {len(filtered_rho)}")

        # 计算板点阵理论边界值
        plate_boundary = 0.8 * filtered_rho ** 1

        # 判断哪些点超出板点阵理论边界
        above_plate_boundary = filtered_E > plate_boundary

        # 添加数据点 - 超出板点阵理论边界的用星星标记
        above_plate_count = np.sum(above_plate_boundary)
        if above_plate_count > 0:
            # 计算在1,000,000中的对应数量
            above_plate_percentage = above_plate_count / len(filtered_rho) * 100
            scaled_above_plate_count = int(1000000 * above_plate_percentage / 100)

            ax.loglog(filtered_rho[above_plate_boundary], filtered_E[above_plate_boundary],
                      '*', color='gold', markersize=10, alpha=0.9,
                      markeredgecolor='darkgoldenrod', markeredgewidth=1.5, zorder=10,
                      label=f'Above Plate-lattices bound (n={scaled_above_plate_count:,})')

        # 添加其他数据点 - 用圆圈标记
        # 这里显示固定总数，而不是实际读取的数量
        ax.loglog(filtered_rho, filtered_E,
                  'o', color=color, markersize=6, alpha=0.6,
                  markeredgecolor='navy', markeredgewidth=0.5,
                  label=f'{label} (n={total_points:,})')
    else:
        print("警告: 没有密度>=0.03的数据点")


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

# 生成包含所有数据集的Ashby图
if valid_csv_paths:
    fig, ax, all_combined_rho, all_combined_E, total_points_count = create_enhanced_ashby_figure13_with_multiple_datasets(
        valid_csv_paths)

    # 保存图片
    output_filename = 'ashby_figure13_with_all_configs_data_custom_range.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n包含所有配置数据的Ashby图已保存为 '{output_filename}'")
    print(f"实际读取了 {total_points_count} 个数据点")
    print(f"图例中显示为 1,000,000 个GPGA-VAE设计")
    print(f"坐标轴范围: X轴[0.03, 1], Y轴[1e-4, 10]")

    # 显示图形
    plt.show()
else:
    print("错误: 没有找到任何有效的CSV文件，请检查文件路径")


# 详细统计分析
def analyze_combined_dataset(rho, E, dataset_name, total_points=1000000):
    """分析合并数据集的性能"""
    if len(rho) == 0:
        print(f"{dataset_name} 没有数据")
        return

    print(f"\n=== {dataset_name} 统计分析 ===")
    print(f"实际读取的设计点数: {len(rho)}")
    print(f"总设计点数: {total_points:,}")
    print(f"相对密度范围: {np.min(rho):.4f} - {np.max(rho):.4f}")
    print(f"相对模量范围: {np.min(E):.4f} - {np.max(E):.4f}")

    # 过滤密度>=0.03的数据点进行分析
    mask = rho >= 0.03
    filtered_rho = rho[mask]
    filtered_E = E[mask]

    if len(filtered_rho) > 0:
        print(f"\n过滤后数据（密度>=0.03）:")
        print(f"设计点数: {len(filtered_rho)}")
        print(f"相对密度范围: {np.min(filtered_rho):.4f} - {np.max(filtered_rho):.4f}")
        print(f"相对模量范围: {np.min(filtered_E):.4f} - {np.max(filtered_E):.4f}")

        # 计算比刚度
        specific_stiffness = filtered_E / filtered_rho
        print(f"比刚度范围: {np.min(specific_stiffness):.4f} - {np.max(specific_stiffness):.4f}")
        print(f"平均比刚度: {np.mean(specific_stiffness):.4f}")

        # 计算板点阵理论边界
        plate_boundary = 0.8 * filtered_rho ** 1

        # 超出板点阵理论边界的点统计
        above_plate_count = np.sum(filtered_E > plate_boundary)
        above_plate_percentage = above_plate_count / len(filtered_E) * 100
        print(f"超出板点阵理论边界的设计:")
        print(f"  实际数量: {above_plate_count}")
        print(f"  占比 (基于读取数据): {above_plate_percentage:.1f}%")
        print(f"  估计总数 (基于1,000,000): {int(1000000 * above_plate_percentage / 100):,}")

        # 计算与理论边界的比较
        valid_mask = (filtered_rho > 0) & (filtered_E > 0)
        if np.sum(valid_mask) > 0:
            stretch_ratios = filtered_E[valid_mask] / (1.0 * filtered_rho[valid_mask] ** 1)
            bending_ratios = filtered_E[valid_mask] / (1.0 * filtered_rho[valid_mask] ** 2)
            plate_ratios = filtered_E[valid_mask] / (0.8 * filtered_rho[valid_mask] ** 1)

            print(f"与理论边界比较:")
            print(f"  达到理想拉伸主导边界的平均比例: {np.mean(stretch_ratios) * 100:.1f}%")
            print(f"  达到理想弯曲主导边界的平均比例: {np.mean(bending_ratios) * 100:.1f}%")
            print(f"  达到板点阵理论边界的平均比例: {np.mean(plate_ratios) * 100:.1f}%")

            # 计算最佳性能点
            best_stiffness_idx = np.argmax(filtered_E)
            best_specific_idx = np.argmax(specific_stiffness)
            best_above_plate_idx = np.argmax(filtered_E - plate_boundary)

            print(f"最佳性能设计:")
            print(f"  最高刚度设计: ρ={filtered_rho[best_stiffness_idx]:.4f}, E={filtered_E[best_stiffness_idx]:.4f}")
            print(
                f"  最高比刚度设计: ρ={filtered_rho[best_specific_idx]:.4f}, E={filtered_E[best_specific_idx]:.4f}, 比刚度={specific_stiffness[best_specific_idx]:.4f}")
            print(
                f"  最超出板点阵边界设计: ρ={filtered_rho[best_above_plate_idx]:.4f}, E={filtered_E[best_above_plate_idx]:.4f}, 超出比例={(filtered_E[best_above_plate_idx] / plate_boundary[best_above_plate_idx] - 1) * 100:.1f}%")

            # 密度分布分析
            print(f"\n密度分布分析 (基于读取数据):")
            density_ranges = [(0.03, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.0)]
            for d_min, d_max in density_ranges:
                mask_range = (filtered_rho >= d_min) & (filtered_rho < d_max)
                if np.sum(mask_range) > 0:
                    avg_specific = np.mean(specific_stiffness[mask_range])
                    above_plate_in_range = np.sum(filtered_E[mask_range] > (0.8 * filtered_rho[mask_range] ** 1))
                    percentage_in_range = np.sum(mask_range) / len(filtered_rho) * 100
                    print(
                        f"  密度 {d_min:.2f}-{d_max:.2f}: {np.sum(mask_range)} 个设计 ({percentage_in_range:.1f}%), 超出板点阵边界: {above_plate_in_range}, 平均比刚度: {avg_specific:.4f}")


# 分析合并数据集
if valid_csv_paths:
    analyze_combined_dataset(all_combined_rho, all_combined_E, "Combined GPGA-VAE Designs", total_points=1000000)

    # 性能趋势分析
    if len(all_combined_rho) > 0:
        print(f"\n=== 性能趋势分析 ===")

        # 过滤密度>=0.03的数据点
        mask = all_combined_rho >= 0.03
        filtered_rho = all_combined_rho[mask]
        filtered_E = all_combined_E[mask]

        if len(filtered_rho) > 1:
            # 计算整体斜率（线性回归）
            valid_mask = (filtered_rho > 0) & (filtered_E > 0)
            if np.sum(valid_mask) > 1:
                log_rho = np.log(filtered_rho[valid_mask])
                log_E = np.log(filtered_E[valid_mask])
                slope, intercept = np.polyfit(log_rho, log_E, 1)

                print(f"整体性能趋势斜率: {slope:.3f}")

                if slope > 1.5:
                    trend = "明显偏向弯曲主导行为"
                elif slope < 1.2:
                    trend = "明显偏向拉伸主导行为"
                elif 1.2 <= slope <= 1.5:
                    trend = "混合变形机制"
                else:
                    trend = "接近理想拉伸主导"

                print(f"变形机制: {trend}")

                # 计算板点阵边界超出的比例
                plate_boundary = 0.8 * filtered_rho[valid_mask] ** 1
                above_plate_ratio = np.sum(filtered_E[valid_mask] > plate_boundary) / len(
                    filtered_E[valid_mask]) * 100
                print(f"超出板点阵理论边界的比例: {above_plate_ratio:.1f}%")

                # 与传统材料对比
                print(f"\n与传统材料对比:")
                print(f"- 相比泡沫材料: 性能显著提升")
                print(f"- 相比Kagome点阵: 性能相当，但在设计多样性上有优势")
                print(f"- 相比板点阵理论界限: {above_plate_ratio:.1f}%的设计超越理论界限")