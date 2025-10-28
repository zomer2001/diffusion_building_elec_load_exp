import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

# ==================== 高级学术样式设置 ====================
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 15,  # 增加字体大小
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.15,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.3',
    'axes.linewidth': 1.0,
    'axes.edgecolor': '0.3',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.constrained_layout.use': True,
})

# 专业学术配色方案
PALETTE = {
    'bar': '#2c7bb6',  # 深蓝色 - 学术期刊常用
    'line': '#d7191c',  # 深红色 - 学术期刊常用
    'marker': '#fdae61',  # 金色 - 作为强调色
    'text': '#2c3e50',  # 深灰
    'background': '#ffffff',  # 白色背景
    'grid': '#e0e0e0'  # 浅灰色网格
}


# ==================== 数据处理 ====================
def process_data(csv_path, sample_size=60):
    df = pd.read_csv(csv_path)

    # 筛选数据：Method=='diffts-fft'
    target_data = df[df['Method'] == 'diffts-fft'].copy()

    # 提取建筑类型（从Building名称中按下划线分割的第二个词）
    target_data['Building_Type'] = target_data['Building'].apply(
        lambda x: x.split('_')[1].capitalize() if len(x.split('_')) > 1 else 'Unknown'
    )

    # 随机选择60行数据（不是60个建筑，而是60行）
    if len(target_data) > sample_size:
        target_data = target_data.sample(n=sample_size, random_state=42)

    # 按建筑类型分组计算MAE平均值和数据行数
    building_stats = target_data.groupby('Building_Type').agg(
        MAE_Mean=('MAE', 'mean'),
        Count=('MAE', 'count')  # 计算数据行数
    ).reset_index()

    # 按数量降序排序
    building_stats = building_stats.sort_values('Count', ascending=False)

    # 验证总数是否为60
    total_count = building_stats['Count'].sum()
    if total_count != sample_size:
        print(f"警告: 总行数应为{sample_size}, 实际为{total_count}")
        # 调整样本大小以匹配实际总数
        sample_size = total_count

    return building_stats, target_data, sample_size


# ==================== 高级可视化函数 ====================
def create_combined_plot(stats_df, sample_size):
    # 创建图形和轴
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['background'])
    ax1 = plt.gca()

    # 设置网格样式
    ax1.grid(True, linestyle='--', color=PALETTE['grid'], alpha=0.2)

    # 设置学术化标题
    plt.title(f'Building Type Performance Analysis',
              fontsize=20, pad=20, weight='bold', color=PALETTE['text'])  # 增大标题字体

    # 绘制柱状图（数量）
    bars = ax1.bar(
        stats_df['Building_Type'],
        stats_df['Count'],
        color=PALETTE['bar'],
        alpha=0.9,
        edgecolor='white',
        linewidth=1.5,
        label='Number of Buildings'
    )

    # 添加数量标签（使用白色粗体）
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom',
                 fontsize=18, color='white', weight='bold',  # 增大标签字体
                 bbox=dict(facecolor=PALETTE['bar'], alpha=0.9, boxstyle='round,pad=0.2'))

    # 设置第一个y轴（数量）
    ax1.set_xlabel('Building Type', fontweight='semibold', color=PALETTE['text'], fontsize=18)
    ax1.set_ylabel('Number of Buildings', fontweight='semibold', color=PALETTE['text'], fontsize=16)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 设置y轴范围，留出空间给MAE标签
    max_count = stats_df['Count'].max()
    ax1.set_ylim(0, max_count * 1.25)

    # 创建第二个y轴（MAE）
    ax2 = ax1.twinx()

    # 绘制折线图（MAE）
    line = ax2.plot(
        stats_df['Building_Type'],
        stats_df['MAE_Mean'],
        color=PALETTE['line'],
        marker='o',
        markersize=16,  # 增大标记大小
        markeredgecolor='white',
        markeredgewidth=2.0,
        linewidth=3.0,
        label='Average MAE'
    )

    # 添加MAE数据标签（带背景框）
    for i, mae in enumerate(stats_df['MAE_Mean']):
        ax2.text(i, mae, f'{mae:.3f}',
                 color='white',
                 ha='center', va='center',
                 fontsize=16, weight='bold',  # 增大标签字体
                 bbox=dict(facecolor=PALETTE['line'], alpha=0.9, boxstyle='round,pad=0.3'))

    # 设置第二个y轴
    ax2.set_ylabel('Average MAE', fontweight='semibold', color=PALETTE['text'], fontsize=18)

    # 设置y轴范围，留出空间给标签
    max_mae = stats_df['MAE_Mean'].max()
    ax2.set_ylim(0, max_mae * 1.3)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right',
               frameon=True,
               shadow=False,
               fancybox=True,
               edgecolor='0.3',
               fontsize=16)  # 增大图例字体

    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor', fontsize=18)

    # 添加边框
    for spine in ax1.spines.values():
        spine.set_edgecolor('#b0b0b0')
        spine.set_linewidth(1.0)

    for spine in ax2.spines.values():
        spine.set_edgecolor('#b0b0b0')
        spine.set_linewidth(1.0)

    return fig


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)

    print("开始处理数据...")
    # 读取并处理数据
    csv_path = '../../results/lstm_2160/all_results.csv'
    building_stats, sampled_data, sample_size = process_data(csv_path, sample_size=60)

    # 打印统计信息和抽样详情
    print("\n=== 随机抽样详情 ===")
    print(f"总行数: {len(sampled_data)}")
    print(f"建筑类型数: {len(building_stats)}")
    print(f"各类型行数总和: {building_stats['Count'].sum()}")

    print("\n=== 建筑类型统计 ===")
    print(building_stats.to_string(index=False))

    # 创建并保存图表
    print("\n创建图表中...")
    fig = create_combined_plot(building_stats, sample_size)

    # 高质量保存
    fig.savefig('building_type_analysis.pdf', bbox_inches='tight', dpi=1200)
    fig.savefig('building_type_analysis.png', bbox_inches='tight', dpi=1200)
    print("\n已保存图表：building_type_analysis.pdf 和 building_type_analysis.png")