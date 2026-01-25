import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

# ==================== 高级学术样式设置 ====================
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun', 'Microsoft YaHei'],
    'axes.unicode_minus': False,

    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 15,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,

    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'mathtext.fontset': 'stix',

    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.4,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
})
# 专业学术配色方案
PALETTE = {
    'bar': '#2c7bb6',
    'line': '#d7191c',
    'marker': '#fdae61',
    'text': '#2c3e50',
    'background': '#ffffff',
    'grid': '#e0e0e0'
}

# ==================== 数据处理 ====================
def process_data(csv_path, sample_size=60):
    df = pd.read_csv(csv_path)

    target_data = df[df['Method'] == 'diffts-fft'].copy()

    target_data['Building_Type'] = target_data['Building'].apply(
        lambda x: x.split('_')[1].capitalize() if len(x.split('_')) > 1 else 'Unknown'
    )

    if len(target_data) > sample_size:
        target_data = target_data.sample(n=sample_size, random_state=42)

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

    # 主标题
    plt.title(
        '不同建筑类型的模型性能分析',
        fontsize=25,
        pad=20,
        weight='bold',
        color=PALETTE['text']
    )

    bars = ax1.bar(
        stats_df['Building_Type'],
        stats_df['Count'],
        color=PALETTE['bar'],
        alpha=0.9,
        edgecolor='white',
        linewidth=1.5,
        label='建筑数量'
    )

    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=21,
            color='white',
            weight='bold',
            bbox=dict(
                facecolor=PALETTE['bar'],
                alpha=0.9,
                boxstyle='round,pad=0.2'
            )
        )

    ax1.set_xlabel('建筑类型', fontweight='semibold', color=PALETTE['text'], fontsize=24)
    ax1.set_ylabel('建筑数量', fontweight='semibold', color=PALETTE['text'], fontsize=21)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylim(0, stats_df['Count'].max() * 1.25)

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
        label='平均 MAE'
    )

    for i, mae in enumerate(stats_df['MAE_Mean']):
        ax2.text(
            i,
            mae,
            f'{mae:.3f}',
            color='white',
            ha='center',
            va='center',
            fontsize=21,
            weight='bold',
            bbox=dict(
                facecolor=PALETTE['line'],
                alpha=0.9,
                boxstyle='round,pad=0.3'
            )
        )

    ax2.set_ylabel('平均 MAE', fontweight='semibold', color=PALETTE['text'], fontsize=19)
    ax2.set_ylim(0, stats_df['MAE_Mean'].max() * 1.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper right',
        frameon=True,
        fancybox=True,
        edgecolor='0.3',
        fontsize=18
    )

    ax1.set_xticklabels(stats_df['Building_Type'], fontsize=21)
    plt.yticks(fontsize=21)

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

    csv_path = '../../results/lstm_2160/all_results.csv'
    building_stats, sampled_data, sample_size = process_data(csv_path, sample_size=60)

    fig = create_combined_plot(building_stats, sample_size)
    fig.savefig('building_type_analysis_CH.pdf', bbox_inches='tight', dpi=600)
    fig.savefig('building_type_analysis_CH.png', bbox_inches='tight', dpi=600)
