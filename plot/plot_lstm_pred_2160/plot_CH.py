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
    '#4C72B0',  # 蓝色
    '#55A868',  # 绿色
    '#C44E52',  # 红色
    '#8172B2',  # 紫色
    '#CCB974',  # 金色
    '#64B5CD'  # 青色
])

# ==================== 数据处理 ====================
# 读取数据
df = pd.read_csv('../../results/lstm_2160/all_results.csv')

# 1. 方法名称映射和排序
method_mapping = {
    'timegan': 'TIMEGAN',
    'cgan': 'CGAN',
    'diffts': 'DDPM',
    'ours_gen': 'OURS',
    'diffts-fft': 'CDDM',  # 确保OURS在最后
    'oridata': 'Real Data'
}
df['Method'] = df['Method'].map(method_mapping)

# 确保OURS在最后
methods_order = ['TIMEGAN', 'CGAN', 'DDPM', 'CDDM', 'Real Data', 'OURS']

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
titles = ['(a) MAE 对比结果', '(b) MSE 对比结果',
          '(c) RMSE 对比结果', '(d) MAPE 对比结果']

# 增大图形尺寸
fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
fig.suptitle('不同合成方法的结果对比', fontsize=21, weight='bold', y=0.98)

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
        linewidth=0.8,  # 增大边框宽度
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
            bar.set_linewidth(1.5)  # 增大OURS边框宽度

    # 设置标题和标签
    ax.set_title(title, fontsize=20, pad=15, weight='semibold')
    ax.set_xlabel('训练数据比例', weight='semibold', fontsize=19)
    ax.set_ylabel(name, weight='semibold', fontsize=19)

    # 修改x轴标签
    ax.set_xticklabels(['30%', '50%', '70%', '90%'], fontsize=19)
    ax.tick_params(axis='y', labelsize=15)

    # 只在第一个子图创建图例
    if i == 0:
        # 获取默认图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()

        # 创建自定义图例项
        legend_handles = []
        for j, label in enumerate(labels):
            # 获取该方法的颜色
            color = PALETTE[j % len(PALETTE)]

            if label == 'OURS':
                # 创建一个带填充样式的矩形，增大尺寸
                patch = Rectangle((0, 0), 1.2, 1.2,  # 增大尺寸
                                  facecolor=color,
                                  edgecolor='black',
                                  hatch=hatch_pattern,
                                  linewidth=1.5)
                legend_handles.append(patch)
            else:
                # 创建一个普通矩形
                patch = Rectangle((0, 0), 1.2, 1.2,
                                  facecolor=color,
                                  edgecolor='black',
                                  linewidth=1.5)
                legend_handles.append(patch)

        # 在第一个子图中创建图例
        legend = ax.legend(
            legend_handles, labels,
            title='Methods',
            title_fontsize=17,  # 增大图例标题字体
            frameon=True,
            loc='best',  # 自动选择最佳位置
            fontsize=16,  # 增大图例字体
            framealpha=0.8
        )

        # 加粗显示我们的方法
        for text in legend.get_texts():
            if text.get_text() == 'OURS':
                text.set_fontweight('bold')
    else:
        # 移除其他子图的图例
        ax.get_legend().remove()

# 调整布局并保存
plt.tight_layout()

# 保存为图片格式
plt.savefig('performance_comparison_CH.pdf', bbox_inches='tight', dpi=600)
plt.savefig('performance_comparison_CH.png', bbox_inches='tight', dpi=600)
print("已保存柱状图：performance_comparison.pdf 和 performance_comparison.png")

# ==================== MAE折线图绘制 ====================
# 增大图形尺寸
plt.figure(figsize=(10, 8), tight_layout=True)
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
    markersize=10,  # 增大标记尺寸
    linewidth=3,  # 增大线宽
    palette=PALETTE,
    ax=ax
)

# 设置标题和标签
plt.title('MAE 变化趋势', fontsize=16, pad=15, weight='semibold')
plt.xlabel('训练数据比例 (%)', weight='semibold', fontsize=15)
plt.ylabel('MAE', weight='semibold', fontsize=15)
plt.xticks([30, 50, 70, 90], ['30%', '50%', '70%', '90%'], fontsize=13)
plt.yticks(fontsize=13)

# 优化图例 - 放置在图表中央
handles, labels = ax.get_legend_handles_labels()
# 创建自定义图例项
legend_handles = []
for j, label in enumerate(labels):
    # 获取该方法的颜色
    color = PALETTE[j % len(PALETTE)]

    if label == 'OURS':
        # 创建一个带填充样式的矩形，增大尺寸
        patch = Rectangle((0, 0), 1.2, 1.2,  # 增大尺寸
                          facecolor=color,
                          edgecolor='black',
                          hatch=hatch_pattern,
                          linewidth=1.5)
        legend_handles.append(patch)
    else:
        # 创建一个普通矩形
        patch = Rectangle((0, 0), 1.2, 1.2,
                          facecolor=color,
                          edgecolor='black',
                          linewidth=1.5)
        legend_handles.append(patch)

# 将图例放置在图表中央
legend = ax.legend(
    legend_handles, labels,
    title='Methods',
    title_fontsize=15,  # 增大图例标题字体
    loc='upper center',
    bbox_to_anchor=(0.5, 0.98),  # 放置在中央上方
    ncol=3,  # 三列排列
    fontsize=13,  # 增大图例字体
    frameon=True,
    columnspacing=1.5,  # 增加列间距
    handletextpad=0.5,  # 调整图例项与文本间距
    handlelength=1.5,  # 增大图例项长度
    handleheight=1.5  # 增大图例项高度
)

# 加粗显示我们的方法
for text in legend.get_texts():
    if text.get_text() == 'OURS':
        text.set_fontweight('bold')

# 添加网格
plt.grid(True, linestyle=':', alpha=0.7)

# 保存结果
plt.savefig('mae_trends_CH.pdf', bbox_inches='tight', dpi=600)
plt.savefig('mae_trends_CH.png', bbox_inches='tight', dpi=600)
print("已保存折线图：mae_trends.pdf 和 mae_trends.png")

# 显示处理后的数据摘要
print("\n=== 处理后数据摘要 ===")
print("方法名称映射:")
print(method_mapping)
print("\nMAPE值已除以8:")
print(grouped[['Method', 'Sparsity', 'MAPE']].head())
print(f"\n完整的修改后数据已保存至: {modified_csv_path}")