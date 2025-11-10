import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

# ==================== 高级学术样式设置 ====================
rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 19,  # 全局字体 19
    'axes.titlesize': 21,
    'axes.labelsize': 19,
    'xtick.labelsize': 19,
    'ytick.labelsize': 19,
    'legend.fontsize': 16,  # 图例保持稍小
    'figure.dpi': 600,
    'savefig.dpi': 600,
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
    'bar': '#2c7bb6',   # 深蓝色
    'line': '#d7191c',  # 深红色
    'marker': '#fdae61',  # 金色
    'text': '#2c3e50',
    'background': '#ffffff',
    'grid': '#e0e0e0'
}


# ==================== 数据处理 ====================
def process_data(csv_path, sample_size=60):
    df = pd.read_csv(csv_path)

    # 筛选数据：Method == 'diffts-fft'
    target_data = df[df['Method'] == 'diffts-fft'].copy()

    # 提取建筑类型
    target_data['Building_Type'] = target_data['Building'].apply(
        lambda x: x.split('_')[1].capitalize() if len(x.split('_')) > 1 else 'Unknown'
    )

    # 随机选择60行
    if len(target_data) > sample_size:
        target_data = target_data.sample(n=sample_size, random_state=42)

    # 按建筑类型分组
    building_stats = target_data.groupby('Building_Type').agg(
        MAE_Mean=('MAE', 'mean'),
        Count=('MAE', 'count')
    ).reset_index()

    building_stats = building_stats.sort_values('Count', ascending=False)

    total_count = building_stats['Count'].sum()
    if total_count != sample_size:
        print(f"警告: 总行数应为{sample_size}, 实际为{total_count}")
        sample_size = total_count

    return building_stats, target_data, sample_size


# ==================== 高级可视化函数 ====================
def create_combined_plot(stats_df, sample_size):
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['background'])
    ax1 = plt.gca()

    ax1.grid(True, linestyle='--', color=PALETTE['grid'], alpha=0.2)

    # 主标题（21号）
    plt.title(f'Building Type Performance Analysis',
              fontsize=25, pad=20, weight='bold', color=PALETTE['text'])

    # 柱状图（数量）
    bars = ax1.bar(
        stats_df['Building_Type'],
        stats_df['Count'],
        color=PALETTE['bar'],
        alpha=0.9,
        edgecolor='white',
        linewidth=1.5,
        label='Number of Buildings'
    )

    # 数量标签（21号）
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom',
                 fontsize=21, color='white', weight='bold',
                 bbox=dict(facecolor=PALETTE['bar'], alpha=0.9, boxstyle='round,pad=0.2'))

    # x, y轴标签（19号）
    ax1.set_xlabel('Building Type', fontweight='semibold', color=PALETTE['text'], fontsize=24)
    ax1.set_ylabel('Number of Buildings', fontweight='semibold', color=PALETTE['text'], fontsize=21)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    max_count = stats_df['Count'].max()
    ax1.set_ylim(0, max_count * 1.25)

    # 第二个y轴（MAE）
    ax2 = ax1.twinx()
    ax2.plot(
        stats_df['Building_Type'],
        stats_df['MAE_Mean'],
        color=PALETTE['line'],
        marker='o',
        markersize=14,
        markeredgecolor='white',
        markeredgewidth=2.0,
        linewidth=3.0,
        label='Average MAE'
    )

    # MAE标签（21号）
    for i, mae in enumerate(stats_df['MAE_Mean']):
        ax2.text(i, mae, f'{mae:.3f}',
                 color='white',
                 ha='center', va='center',
                 fontsize=21, weight='bold',
                 bbox=dict(facecolor=PALETTE['line'], alpha=0.9, boxstyle='round,pad=0.3'))

    ax2.set_ylabel('Average MAE', fontweight='semibold', color=PALETTE['text'], fontsize=19)
    max_mae = stats_df['MAE_Mean'].max()
    ax2.set_ylim(0, max_mae * 1.3)

    # 图例（保持16号）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right',
               frameon=True,
               shadow=False,
               fancybox=True,
               edgecolor='0.3',
               fontsize=18)

    # x轴标签旋转（19号）
    ax1.set_xticklabels(
        stats_df['Building_Type'],  # x轴标签内容


        fontsize=21  # 这里修改为你需要的字号（例如20）
    )
    plt.yticks(fontsize=21)

    # 边框样式
    for spine in ax1.spines.values():
        spine.set_edgecolor('#b0b0b0')
        spine.set_linewidth(1.0)

    for spine in ax2.spines.values():
        spine.set_edgecolor('#b0b0b0')
        spine.set_linewidth(1.0)

    return fig


# ==================== 主程序 ====================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("开始处理数据...")
    csv_path = '../../results/lstm_2160/all_results.csv'
    building_stats, sampled_data, sample_size = process_data(csv_path, sample_size=60)

    print("\n=== 随机抽样详情 ===")
    print(f"总行数: {len(sampled_data)}")
    print(f"建筑类型数: {len(building_stats)}")
    print(f"各类型行数总和: {building_stats['Count'].sum()}")

    print("\n=== 建筑类型统计 ===")
    print(building_stats.to_string(index=False))

    print("\n创建图表中...")
    fig = create_combined_plot(building_stats, sample_size)

    fig.savefig('building_type_analysis.pdf', bbox_inches='tight', dpi=600)
    fig.savefig('building_type_analysis.png', bbox_inches='tight', dpi=600)
    print("\n已保存图表：building_type_analysis.pdf 和 building_type_analysis.png")
