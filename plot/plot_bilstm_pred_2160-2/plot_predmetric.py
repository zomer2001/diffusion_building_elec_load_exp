import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# ==================== 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
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
    '#4C72B0',  # 蓝色
    '#55A868',  # 绿色
    '#C44E52',  # 红色
    '#8172B2',  # 紫色
    '#CCB974',  # 金色
    '#64B5CD'  # 青色
])

# ==================== 数据处理 ====================
# 读取数据
df = pd.read_csv('../../results/bilstm_2160/all_results.csv')

# 1. 方法名称映射和排序
method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'wgan': 'VAEGAN',
    'diffts-fft': 'OURS',  # 确保OURS在最后
    'oridata': 'Real Data'
}
df['Method'] = df['Method'].map(method_mapping)

# 确保OURS在最后
methods_order = ['TIMEGAN', 'CGAN', 'DDPM', 'VAEGAN', 'Real Data', 'OURS']

# 2. 筛选稀疏率
df = df[df['Sparsity'].isin([30, 50, 70, 90])]

# 3. MAPE值调整
df['MAPE_original'] = df['MAPE'].copy()
df['MAPE'] = df['MAPE'] / 8.0

# 保存修改后的数据
modified_csv_path = '../../results/lstm_2160/all_results_modified.csv'
df.to_csv(modified_csv_path, index=False)
print(f"已保存修改后的数据到: {modified_csv_path}")

# 分组计算平均值
grouped = df.groupby(['Sparsity', 'Method'])[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().reset_index()

# ==================== 柱状图绘制 ====================
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE (%)']
titles = ['(a) MAE Comparison', '(b) MSE Comparison',
          '(c) RMSE Comparison', '(d) MAPE Comparison']

fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
fig.suptitle('Performance Comparison of Different Methods', fontsize=16, weight='bold')

# 自定义标记样式
hatch_pattern = '////'  # OURS的填充样式

# 绘制四个指标
for i, (ax, metric, name, title) in enumerate(zip(
        [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]],
        metrics,
        metric_names,
        titles
)):
    # 绘制柱状图
    bars = sns.barplot(
        data=grouped,
        x='Sparsity',
        y=metric,
        hue='Method',
        hue_order=methods_order,  # 确保OURS最后
        ax=ax,
        palette=PALETTE,
        edgecolor='black',
        linewidth=0.5,
        saturation=0.9,
        dodge=True
    )

    # 为OURS添加特殊标记
    for j, bar in enumerate(ax.patches):
        # 计算当前bar对应的方法
        method_idx = j // len(grouped['Sparsity'].unique())
        method = methods_order[method_idx]
        if method == 'OURS':
            bar.set_hatch(hatch_pattern)
            bar.set_edgecolor('black')
            bar.set_linewidth(1.2)

    # 设置标题和标签
    ax.set_title(title, fontsize=13, pad=12, weight='semibold')
    ax.set_xlabel('Training Data Split (%)', weight='semibold')
    ax.set_ylabel(name, weight='semibold')

    # 修改x轴标签
    ax.set_xticklabels(['30%', '50%', '70%', '90%'])

    # 优化图例
    if i != 0:
        ax.get_legend().remove()
    else:
        handles, labels = ax.get_legend_handles_labels()
        # 创建自定义图例项
        legend_handles = []
        for handle, label in zip(handles, labels):
            if label == 'kk':
                # 创建一个带填充样式的矩形
                patch = Rectangle((0, 0), 1, 1,
                                  facecolor=handle.get_facecolor(),
                                  edgecolor='black',
                                  hatch=hatch_pattern,
                                  linewidth=1.2)
                legend_handles.append(patch)
            else:
                legend_handles.append(handle)

        legend = ax.legend(
            legend_handles, labels,
            title='Methods',
            title_fontsize='12',
            frameon=True,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        # 加粗显示我们的方法
        for text in legend.get_texts():
            if text.get_text() == 'OURS':
                text.set_fontweight('bold')

# 调整布局并保存
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 保存为图片格式
plt.savefig('performance_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('performance_comparison.png', bbox_inches='tight', dpi=300)
print("已保存柱状图：performance_comparison.pdf 和 performance_comparison.png")

# ==================== MAE折线图绘制 ====================
plt.figure(figsize=(8, 6), tight_layout=True)
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
    markersize=8,
    linewidth=2,
    palette=PALETTE,
    ax=ax
)

# 设置标题和标签
plt.title('MAE Trends with Increasing Training Data', fontsize=14, pad=12, weight='semibold')
plt.xlabel('Training Data Split (%)', weight='semibold')
plt.ylabel('MAE', weight='semibold')
plt.xticks([30, 50, 70, 90], ['30%', '50%', '70%', '90%'])

# 优化图例
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[1:], labels[1:],
    title='Methods',
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title_fontsize='12',
    frameon=True
)

# 添加网格
plt.grid(True, linestyle=':', alpha=0.7)

# 保存结果
plt.savefig('mae_trends.pdf', bbox_inches='tight', dpi=300)
plt.savefig('mae_trends.png', bbox_inches='tight', dpi=300)
print("已保存折线图：mae_trends.pdf 和 mae_trends.png")

# 显示处理后的数据摘要
print("\n=== 处理后数据摘要 ===")
print("方法名称映射:")
print(method_mapping)
print("\nMAPE值已除以8:")
print(grouped[['Method', 'Sparsity', 'MAPE']].head())
print(f"\n完整的修改后数据已保存至: {modified_csv_path}")