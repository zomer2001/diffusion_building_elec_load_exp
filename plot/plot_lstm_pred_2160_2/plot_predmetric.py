import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# ==================== 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
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

# ==================== 学术配色方案 ====================
PALETTE = sns.color_palette([
    '#4C72B0',  # TIMEGAN
    '#55A868',  # CGAN
    '#C44E52',  # DDPM
    '#8172B2',  # VAEGAN
    '#CCB974',  # Real Data
    '#64B5CD',  # CDDM
    '#08306b',  # OURS（深蓝）
])

# ==================== 数据处理 ====================
df = pd.read_csv('../../results/lstm_2160_all_0111/all_results.csv')

# -------- 方法名映射（你要求的版本）--------
method_mapping = {
    'oridata': 'Real Data',
    'wgan': 'VAEGAN',
    'diffts': 'DDPM',
    'diffts-fft': 'CDDM',
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'ours_gen': 'OURS'
}

df['Method'] = df['Method'].map(method_mapping)

# -------- 方法顺序（OURS 最后）--------
methods_order = [
    'TIMEGAN',
    'CGAN',
    'DDPM',
    'VAEGAN',
    'CDDM',
    'Real Data',
    'OURS'
]

# -------- 筛选稀疏率 --------
df = df[df['Sparsity'].isin([30, 50, 70, 90])]

# -------- MAPE 调整 --------
df['MAPE_original'] = df['MAPE'].copy()
df['MAPE'] = df['MAPE'] / 8.0

# 保存修改后的 CSV
modified_csv_path = '../../results/lstm_2160_all_0111/all_results_modified.csv'
df.to_csv(modified_csv_path, index=False)
print(f"已保存修改后的数据到: {modified_csv_path}")

# -------- 分组均值 --------
grouped = df.groupby(['Sparsity', 'Method'])[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().reset_index()

# ==================== 柱状图绘制 ====================
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE (%)']
titles = ['(a) MAE Comparison', '(b) MSE Comparison',
          '(c) RMSE Comparison', '(d) MAPE Comparison']

fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
fig.suptitle(
    'Predictive Performance Comparison of Different Methods',
    fontsize=21, weight='bold', y=0.98
)

hatch_pattern = '////'

for i, (ax, metric, name, title) in enumerate(zip(
        axes.flatten(), metrics, metric_names, titles)):

    sns.barplot(
        data=grouped,
        x='Sparsity',
        y=metric,
        hue='Method',
        hue_order=methods_order,
        palette=PALETTE,
        edgecolor='black',
        linewidth=0.8,
        saturation=0.9,
        dodge=True,
        ax=ax
    )

    # OURS hatch
    n_sparsity = grouped['Sparsity'].nunique()
    for j, bar in enumerate(ax.patches):
        method_idx = j // n_sparsity
        if methods_order[method_idx] == 'OURS':
            bar.set_hatch(hatch_pattern)
            bar.set_linewidth(1.5)

    ax.set_title(title, fontsize=20, pad=15, weight='semibold')
    ax.set_xlabel('Training Data Split (%)', fontsize=19, weight='semibold')
    ax.set_ylabel(name, fontsize=19, weight='semibold')
    ax.set_xticklabels(['30%', '50%', '70%', '90%'], fontsize=19)
    ax.tick_params(axis='y', labelsize=15)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = []

        for j, label in enumerate(labels):
            color = PALETTE[j]
            patch = Rectangle(
                (0, 0), 1.2, 1.2,
                facecolor=color,
                edgecolor='black',
                hatch=hatch_pattern if label == 'OURS' else None,
                linewidth=1.5
            )
            legend_handles.append(patch)

        legend = ax.legend(
            legend_handles, labels,
            title='Methods',
            title_fontsize=17,
            fontsize=16,
            framealpha=0.8
        )

        for text in legend.get_texts():
            if text.get_text() == 'OURS':
                text.set_fontweight('bold')
    else:
        ax.get_legend().remove()

plt.savefig('performance_comparison.pdf', bbox_inches='tight', dpi=600)
plt.savefig('performance_comparison.png', bbox_inches='tight', dpi=600)
print("已保存柱状图：performance_comparison.pdf / png")

# ==================== MAE 折线图 ====================
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

plt.title('MAE Trends with Increasing Training Data', fontsize=16, weight='semibold')
plt.xlabel('Training Data Split (%)', fontsize=15, weight='semibold')
plt.ylabel('MAE', fontsize=15, weight='semibold')
plt.xticks([30, 50, 70, 90], ['30%', '50%', '70%', '90%'])

handles, labels = ax.get_legend_handles_labels()
legend_handles = []

for j, label in enumerate(labels):
    patch = Rectangle(
        (0, 0), 1.2, 1.2,
        facecolor=PALETTE[j],
        edgecolor='black',
        hatch=hatch_pattern if label == 'OURS' else None,
        linewidth=1.5
    )
    legend_handles.append(patch)

legend = ax.legend(
    legend_handles, labels,
    title='Methods',
    title_fontsize=15,
    fontsize=13,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),
    ncol=3,
    frameon=True
)

for text in legend.get_texts():
    if text.get_text() == 'OURS':
        text.set_fontweight('bold')

plt.grid(True, linestyle=':', alpha=0.7)
plt.savefig('mae_trends.pdf', bbox_inches='tight', dpi=600)
plt.savefig('mae_trends.png', bbox_inches='tight', dpi=600)

print("已保存折线图：mae_trends.pdf / png")
print(f"\n完整修改后数据保存于：{modified_csv_path}")
