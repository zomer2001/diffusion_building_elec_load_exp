import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# ==================== 全局样式设置 ====================
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
})

# ==================== 配色 ====================
PALETTE = sns.color_palette([
    '#4C72B0',
    '#55A868',
    '#C44E52',
    '#8172B2',
    '#CCB974',
    '#64B5CD'
])

# =========================================================
# Step 1: 读取原始CSV并生成“精简绘图CSV”
# =========================================================

raw_csv_path = '../../results/lstm_2160/all_results.csv'

df = pd.read_csv(raw_csv_path)

method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'wgan': 'VAEGAN',
    'diffts': 'DDPM',
    'diffts-fft': 'OURS',
    'oridata': 'Real Data'
}

df['Method'] = df['Method'].map(method_mapping)

# 仅保留需要的 sparsity
df = df[df['Sparsity'].isin([30, 50, 70, 90])]

# MAPE处理
df['MAPE'] = df['MAPE'] / 8.0

# 仅保留画图需要的列
df = df[[
    'Sparsity',
    'Method',
    'MAE',
    'MSE',
    'RMSE',
    'MAPE'
]]

# groupby得到最终绘图数据
plot_df = df.groupby(
    ['Sparsity', 'Method']
)[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().reset_index()

# 保存精简绘图CSV
plot_csv_path = '../../results/lstm_2160/plot_ready_data.csv'
plot_csv_path2 = '../../results/lstm_2160/plot_ready_data2.csv'
plot_df.to_csv(plot_csv_path, index=False)

print(f'精简绘图CSV已保存: {plot_csv_path}')

# =========================================================
# Step 2: 后续所有画图只读取精简CSV
# =========================================================

grouped = pd.read_csv(plot_csv_path2)

methods_order = [
    'TIMEGAN',
    'CGAN',
    'VAEGAN',
    'DDPM',
    'Real Data',
    'OURS'
]

# ==================== 柱状图 ====================

metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']

metric_names = [
    'MAE',
    'MSE',
    'RMSE',
    'MAPE (%)'
]

titles = [
    '(a) MAE',
    '(b) MSE',
    '(c) RMSE',
    '(d) MAPE'
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

hatch_pattern = '////'

for ax, metric, name, title in zip(
        axes.flatten(),
        metrics,
        metric_names,
        titles):

    sns.barplot(
        data=grouped,
        x='Sparsity',
        y=metric,
        hue='Method',
        hue_order=methods_order,
        ax=ax,
        palette=PALETTE,
        edgecolor='black',
        linewidth=0.8,
        saturation=0.9
    )

    # OURS 高亮
    for j, bar in enumerate(ax.patches):

        method_idx = j // len(grouped['Sparsity'].unique())
        method = methods_order[method_idx]

        if method == 'OURS':
            bar.set_hatch(hatch_pattern)
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax.set_title(title, fontsize=20, pad=10)

    ax.set_xlabel(
        'Training Data Ratio(%)',
        fontsize=20
    )

    ax.set_ylabel(name, fontsize=19)

    ax.set_xticklabels(
        ['30%', '50%', '70%', '90%'],
        fontsize=18
    )

    ax.tick_params(axis='y', labelsize=14)

    if ax.get_legend():
        ax.get_legend().remove()

# ==================== 全局图例 ====================

legend_handles = []

for j, label in enumerate(methods_order):

    color = PALETTE[j % len(PALETTE)]

    if label == 'OURS':

        patch = Rectangle(
            (0, 0),
            1.2,
            1.2,
            facecolor=color,
            edgecolor='black',
            hatch=hatch_pattern,
            linewidth=1.5
        )

    else:

        patch = Rectangle(
            (0, 0),
            1.2,
            1.2,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )

    legend_handles.append(patch)

legend = fig.legend(
    legend_handles,
    methods_order,
    title='Methods',
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=6,
    fontsize=19,
    title_fontsize=19,
    frameon=True
)

for text in legend.get_texts():

    if text.get_text() == 'OURS':
        text.set_fontweight('bold')

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig(
    'performance_comparison_en.pdf',
    bbox_inches='tight',
    dpi=600
)

plt.savefig(
    'performance_comparison_en.png',
    bbox_inches='tight',
    dpi=600
)

# ==================== 折线图 ====================

plt.figure(figsize=(10, 8))

ax = plt.gca()

sns.lineplot(
    data=grouped,
    x='Sparsity',
    y='MAE',
    hue='Method',
    hue_order=methods_order,
    style='Method',
    markers=True,
    dashes=False,
    markersize=10,
    linewidth=3,
    palette=PALETTE,
    ax=ax
)

plt.xlabel(
    'Training Data Ratio(%)',
    fontsize=18
)

plt.ylabel(
    'MAE',
    fontsize=17
)

plt.xticks(
    [30, 50, 70, 90],
    ['30%', '50%', '70%', '90%']
)

ax.get_legend().remove()

legend = plt.gcf().legend(
    methods_order,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    fontsize=13,
    frameon=True
)

plt.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.93])

plt.savefig(
    'mae_trends_en.pdf',
    bbox_inches='tight',
    dpi=600
)

plt.savefig(
    'mae_trends_en.png',
    bbox_inches='tight',
    dpi=600
)

print("全部图已生成")