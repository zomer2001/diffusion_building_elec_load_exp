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
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.4,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.loc': 'best',
})

# 专业配色方案 - 增加至7种颜色，确保不循环
PALETTE = [
    '#4C72B0',  # TIMEGAN (Blue)
    '#55A868',  # CGAN (Green)
    '#C44E52',  # DDPM (Red)
    '#8172B2',  # cddpm (Purple)
    '#CCB974',  # VAEGAN (Gold)
    '#64B5CD',  # Real Data (Cyan)
    '#E6855E'  # OURS (Orange/Coral - 突出显示)
]

# ==================== 数据处理 ====================
# 读取数据
# 注意：请确保路径正确，或根据需要修改
try:
    df = pd.read_csv('../../results/lstm_2160_all_0111/all_results.csv')
except FileNotFoundError:
    print("错误：未找到数据文件。请检查路径。")
    exit()

# 1. 方法名称映射和排序
method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'wgan': 'VAEGAN',
    'ours_gen': 'OURS',
    'oridata': 'Real Data',
    'diffts-fft': 'CDDM'
}
df['Method'] = df['Method'].map(method_mapping)

# 确保方法顺序
methods_order = ['TIMEGAN', 'CGAN', 'DDPM', 'CDDM', 'VAEGAN', 'Real Data', 'OURS']

# 2. 筛选稀疏率
df = df[df['Sparsity'].isin([30, 50, 70, 90])]

# 3. MAPE值调整
if 'MAPE' in df.columns:
    df['MAPE_original'] = df['MAPE'].copy()
    df['MAPE'] = df['MAPE'] / 8.0

# 保存修改后的数据
modified_csv_path = '../../results/lstm_2160_all_0111/all_results_modified.csv'
df.to_csv(modified_csv_path, index=False)
print(f"已保存修改后的数据到: {modified_csv_path}")

# 分组计算平均值
grouped = df.groupby(['Sparsity', 'Method'])[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().reset_index()

# ==================== 柱状图绘制 ====================
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE (%)']
titles = ['(a) MAE Comparison', '(b) MSE Comparison',
          '(c) RMSE Comparison', '(d) MAPE Comparison']

fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
fig.suptitle('Predictive Performance Comparison of Different Methods', fontsize=21, weight='bold', y=0.98)

hatch_pattern = '////'

for i, (ax, metric, name, title) in enumerate(zip(
        axes.flatten(),
        metrics,
        metric_names,
        titles
)):
    # 绘制柱状图
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
        saturation=0.9,
        dodge=True
    )

    # 【修复重点】：通过 get_label 匹配 OURS 样式，避开索引越界问题
    for bar in ax.patches:
        if bar.get_label() == 'OURS':
            bar.set_hatch(hatch_pattern)
            bar.set_edgecolor('black')
            bar.set_linewidth(1.5)

    # 坐标轴设置
    ax.set_title(title, fontsize=20, pad=15, weight='semibold')
    ax.set_xlabel('Training Data Split (%)', weight='semibold', fontsize=19)
    ax.set_ylabel(name, weight='semibold', fontsize=19)

    # 修正坐标轴刻度警告
    ax.set_xticks(range(len(grouped['Sparsity'].unique())))
    ax.set_xticklabels(['30%', '50%', '70%', '90%'], fontsize=19)
    ax.tick_params(axis='y', labelsize=15)

    # 图例处理：仅在左上角子图保留图例并定制
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        legend_handles = []
        for j, label in enumerate(labels):
            color = PALETTE[j % len(PALETTE)]
            if label == 'OURS':
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', hatch=hatch_pattern, linewidth=1.5)
            else:
                patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=1.5)
            legend_handles.append(patch)

        legend = ax.legend(
            legend_handles, labels,
            title='Methods', title_fontsize=17,
            frameon=True, loc='best', fontsize=16, framealpha=0.8
        )
        for text in legend.get_texts():
            if text.get_text() == 'OURS':
                text.set_fontweight('bold')
    else:
        ax.get_legend().remove()

# 保存柱状图
plt.savefig('performance_comparison.pdf', bbox_inches='tight')
plt.savefig('performance_comparison.png', bbox_inches='tight')
print("已保存柱状图：performance_comparison.pdf 和 .png")

# ==================== MAE折线图绘制 ====================
plt.figure(figsize=(10, 8))
ax_line = plt.gca()

sns.lineplot(
    data=grouped,
    x='Sparsity',
    y='MAE',
    hue='Method',
    hue_order=methods_order,
    style='Method',
    markers=True,
    dashes=False,
    markersize=12,
    linewidth=3,
    palette=PALETTE,
    ax=ax_line
)

plt.title('MAE Trends with Increasing Training Data', fontsize=18, pad=15, weight='semibold')
plt.xlabel('Training Data Split (%)', weight='semibold', fontsize=16)
plt.ylabel('MAE', weight='semibold', fontsize=16)
plt.xticks([30, 50, 70, 90], ['30%', '50%', '70%', '90%'], fontsize=14)
plt.grid(True, linestyle=':', alpha=0.7)

# 定制化折线图图例
handles, labels = ax_line.get_legend_handles_labels()
line_legend_handles = []
for j, label in enumerate(labels):
    color = PALETTE[j % len(PALETTE)]
    if label == 'OURS':
        patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', hatch=hatch_pattern, linewidth=1.5)
    else:
        patch = Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=1.5)
    line_legend_handles.append(patch)

ax_line.legend(
    line_legend_handles, labels,
    title='Methods', title_fontsize=15,
    loc='upper center', bbox_to_anchor=(0.5, 0.98),
    ncol=3, fontsize=13, frameon=True,
    columnspacing=1.5, handletextpad=0.5
)

for text in ax_line.get_legend().get_texts():
    if text.get_text() == 'OURS':
        text.set_fontweight('bold')

plt.savefig('mae_trends.pdf', bbox_inches='tight')
plt.savefig('mae_trends.png', bbox_inches='tight')
print("已保存折线图：mae_trends.pdf 和 .png")

print("\n=== 任务完成 ===")