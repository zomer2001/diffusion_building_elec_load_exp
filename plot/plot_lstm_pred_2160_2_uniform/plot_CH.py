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
    'legend.loc': 'best',
})

# 专业配色方案
PALETTE = sns.color_palette([
    '#4C72B0',
    '#55A868',
    '#C44E52',
    '#8172B2',
    '#CCB974',
    '#64B5CD'
])

# ==================== 数据处理 ====================
df = pd.read_csv('../../results/lstm_2160_all_0111/all_results.csv')

method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'ours_gen': 'OURS',
    'diffts-fft': 'CDDM',
    'oridata': 'Real Data'
}
df['Method'] = df['Method'].map(method_mapping)

methods_order = ['TIMEGAN', 'CGAN', 'DDPM', 'CDDM', 'Real Data', 'OURS']

# 仅保留指定稀疏率
df = df[df['Sparsity'].isin([20, 40, 80])]

# MAPE 修正
df['MAPE_original'] = df['MAPE'].copy()
df['MAPE'] = df['MAPE'] / 8.0

modified_csv_path = '../../results/lstm_2160_all_0111/all_results_modified.csv'
df.to_csv(modified_csv_path, index=False)

grouped = df.groupby(['Sparsity', 'Method'])[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().reset_index()

# ==================== 柱状图绘制 ====================
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE (%)']
titles = ['(a) MAE 对比结果', '(b) MSE 对比结果',
          '(c) RMSE 对比结果', '(d) MAPE 对比结果']

fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
fig.suptitle('不同合成方法的预测表现对比',
             fontsize=21, weight='bold', y=0.98)

hatch_pattern = '////'

# <<< MOD >>> 指定 Sparsity 的绘图顺序
sparsity_order = [40, 80, 20]

for i, (ax, metric, name, title) in enumerate(zip(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        metrics, metric_names, titles)):

    sns.barplot(
        data=grouped,
        x='Sparsity',
        y=metric,
        hue='Method',
        hue_order=methods_order,
        order=sparsity_order,   # <<< MOD >>>
        ax=ax,
        palette=PALETTE,
        edgecolor='black',
        linewidth=0.8,
        saturation=0.9,
        dodge=True
    )

    # OURS 特殊填充
    for j, bar in enumerate(ax.patches):
        method_idx = j // len(sparsity_order)
        if methods_order[method_idx] == 'OURS':
            bar.set_hatch(hatch_pattern)
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    ax.set_title(title, fontsize=20, pad=15, weight='semibold')
    ax.set_xlabel('训练数据均匀度', fontsize=19, weight='semibold')
    ax.set_ylabel(name, fontsize=19, weight='semibold')

    # <<< MOD >>> x 轴标签语义映射
    ax.set_xticklabels(['1/2', '1/4', '1/8'], fontsize=19)
    ax.tick_params(axis='y', labelsize=15)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = []

        for j, label in enumerate(labels):
            color = PALETTE[j % len(PALETTE)]
            if label == 'OURS':
                patch = Rectangle((0, 0), 1.2, 1.2,
                                  facecolor=color,
                                  edgecolor='black',
                                  hatch=hatch_pattern,
                                  linewidth=1.5)
            else:
                patch = Rectangle((0, 0), 1.2, 1.2,
                                  facecolor=color,
                                  edgecolor='black',
                                  linewidth=1.5)
            legend_handles.append(patch)

        legend = ax.legend(
            legend_handles, labels,
            title='Methods',
            title_fontsize=17,
            fontsize=16,
            frameon=True,
            framealpha=0.8
        )

        for text in legend.get_texts():
            if text.get_text() == 'OURS':
                text.set_fontweight('bold')
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig('performance_comparison_CH.pdf', bbox_inches='tight', dpi=600)
plt.savefig('performance_comparison_CH.png', bbox_inches='tight', dpi=600)

print("已保存柱状图：performance_comparison_CH.pdf 和 performance_comparison.png")
print(f"修改后数据保存至: {modified_csv_path}")
