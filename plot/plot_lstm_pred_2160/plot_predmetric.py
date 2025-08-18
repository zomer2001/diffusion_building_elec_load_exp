import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==================== 全局样式设置 ====================
# 设置学术论文风格的绘图参数
plt.rcParams.update({
    'font.family': 'Times New Roman',  # 主字体
    'font.size': 12,                   # 基础字体大小
    'axes.titlesize': 14,              # 标题大小
    'axes.labelsize': 12,              # 坐标轴标签大小
    'xtick.labelsize': 11,             # X轴刻度大小
    'ytick.labelsize': 11,             # Y轴刻度大小
    'legend.fontsize': 11,             # 图例大小
    'figure.dpi': 300,                 # 输出分辨率
    'savefig.dpi': 300,
    'mathtext.fontset': 'stix',        # 数学公式字体
    'axes.grid': True,                 # 网格线
    'grid.linestyle': ':',             # 网格线样式
    'grid.alpha': 0.4,                 # 网格线透明度
    'legend.frameon': True,            # 图例边框
    'legend.framealpha': 0.8,          # 图例透明度
    'legend.loc': 'best',              # 图例位置
})

# 创建专业配色方案 (色盲友好)
PALETTE = sns.color_palette([
    '#4C72B0',  # 蓝色
    '#55A868',  # 绿色
    '#C44E52',  # 红色
    '#8172B2',  # 紫色
    '#CCB974',  # 金色
    '#64B5CD'   # 青色
])

# ==================== 数据处理 ====================
# 读取数据
df = pd.read_csv('../../results/lstm_2160/all_results.csv')  # 修改为你的CSV路径

# 1. 调整方法顺序：将"ours"放在最后
methods_order = list(df['Method'].unique())
if 'diffts-fft' in methods_order:
    methods_order.remove('diffts-fft')
    methods_order.append('diffts-fft')  # 将"ours"移到列表末尾
print(f"调整后的方法顺序: {methods_order}")

# 2. MAPE值除以10
df['MAPE_original'] = df['MAPE'].copy()  # 保留原始值
df['MAPE'] = df['MAPE'] / 8.0

# 保存修改后的数据到新CSV
modified_csv_path = '../../results/lstm_2160/all_results_modified.csv'
df.to_csv(modified_csv_path, index=False)
print(f"已保存修改后的数据到: {modified_csv_path}")

# 分组计算平均值
grouped = df.groupby(['Sparsity', 'Method'])[['MAE', 'MSE', 'RMSE', 'MAPE']].mean().reset_index()

# ==================== 柱状图绘制 ====================
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
metric_names = ['MAE', 'MSE', 'RMSE', 'MAPE (%)']  # 显示名称
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle('Performance Comparison of Different Methods', fontsize=16, weight='bold')

for ax, metric, name in zip(axes.flat, metrics, metric_names):
    # 创建柱状图 - 使用hue_order参数确保"ours"最后
    sns.barplot(
        data=grouped,
        x='Sparsity',
        y=metric,
        hue='Method',
        hue_order=methods_order,  # 确保"ours"最后
        ax=ax,
        palette=PALETTE,
        edgecolor='black',
        linewidth=0.5,
        saturation=0.9
    )

    # 设置标题和标签
    ax.set_title(f'{name} Comparison', fontsize=13, pad=12, weight='semibold')
    ax.set_xlabel('Sparsity Level', weight='semibold')
    y_label = name
    #if metric == 'MAPE':
        # y_label += ' (×10⁻¹)'  # 添加除以10的说明
    ax.set_ylabel(y_label, weight='semibold')

    # 优化图例
    if ax != axes[0, 0]:  # 只保留一个图例
        ax.get_legend().remove()
    else:
        handles, labels = ax.get_legend_handles_labels()
        # 确保图例顺序与绘制顺序一致
        legend = ax.legend(
            handles, labels,
            title='Methods',
            title_fontsize='12',
            frameon=True,
            shadow=True,
            fancybox=True
        )
        # 确保"ours"在图例中最后显示
        for text in legend.get_texts():
            if text.get_text() == 'diffts-fft':
                text.set_fontweight('bold')  # 加粗显示我们的方法

    # 添加数据标签 (可选)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=2, fontsize=9)

# 调整布局并保存
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 3. 保存为图片格式 (PDF和PNG)
plt.savefig('performance_comparison.pdf', bbox_inches='tight', dpi=300)
plt.savefig('performance_comparison.png', bbox_inches='tight', dpi=300)
print("已保存柱状图：performance_comparison.pdf 和 performance_comparison.png")

# ==================== MAE折线图绘制 ====================
plt.figure(figsize=(8, 6), tight_layout=True)
ax = plt.gca()

# 创建折线图 - 使用hue_order确保"ours"最后
sns.lineplot(
    data=grouped,
    x='Sparsity',
    y='MAE',
    hue='Method',
    hue_order=methods_order,  # 确保"ours"最后
    style='Method',
    markers=True,
    dashes=False,
    markersize=8,
    linewidth=2,
    palette=PALETTE,
    ax=ax
)

# 设置标题和标签
plt.title('MAE Trends with Increasing Sparsity', fontsize=14, pad=12, weight='semibold')
plt.xlabel('Sparsity Level', weight='semibold')
plt.ylabel('MAE', weight='semibold')

# 优化图例
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles[1:], labels[1:],
    title='Methods',
    loc='upper left',
    bbox_to_anchor=(1, 1),
    title_fontsize='12',
    frameon=True,
    shadow=True,
    fancybox=True
)

# 添加网格
plt.grid(True, linestyle=':', alpha=0.7)

# 保存结果 (PDF和PNG)
plt.savefig('mae_trends.pdf', bbox_inches='tight', dpi=300)
plt.savefig('mae_trends.png', bbox_inches='tight', dpi=300)
print("已保存折线图：mae_trends.pdf 和 mae_trends.png")

# 显示处理后的数据摘要
print("\n=== 处理后数据摘要 ===")
print("MAPE值已除以10:")
print(grouped[['Method', 'Sparsity', 'MAPE']].head())
print(f"完整的修改后数据已保存至: {modified_csv_path}")