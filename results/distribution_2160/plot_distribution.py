import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# ==================== 学术论文风格设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,  # 增大基础字体大小
    'axes.titlesize': 20,  # 增大标题字体
    'axes.labelsize': 17,  # 增大轴标签字体
    'xtick.labelsize': 15,  # 增大x轴刻度标签字体
    'ytick.labelsize': 15,  # 增大y轴刻度标签字体
    'legend.fontsize': 16,  # 增大图例字体
    'figure.dpi': 1200,
    'savefig.dpi': 1200,
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.15,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.3',
    'axes.linewidth': 1.0,
    'axes.edgecolor': '0.3',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
})

# 配色方案
MODERN_PALETTE = [
    '#fad390',  # cgan
    '#f8c291',  # diffts
    '#82ccdd',  # timegan
    '#b8e994'  # ours
]

# 设置标记和图案
MARKER_CONFIG = {
    'cgan': ('o', None),
    'diffts': ('s', None),
    'timegan': ('D', None),
    'ours': ('^', '//')
}

# ==================== 数据处理 ====================
df = pd.read_csv("script\modified_methods.csv")

# 确保 Sparsity 是数值型且排序
df['Sparsity'] = pd.to_numeric(df['Sparsity'], errors='coerce')
df.sort_values(by='Sparsity', inplace=True)

# 确保"ours"方法在最后
methods = df['Method'].unique().tolist()
if 'OURS' in methods:
    methods.remove('OURS')
    methods.append('OURS')
print(f"调整后的方法顺序: {methods}")

# 打印绘图数据
print("=== 与原始数据的分布差异（MMD_oridata） ===")
print(df[['Method', 'Sparsity', 'MMD_oridata']])

print("\n=== 与测试数据的分布差异（MMD_testdata） ===")
print(df[['Method', 'Sparsity', 'MMD_testdata']])


# ==================== 创建两个柱状图，图例在上部且距离更大 ====================
def create_double_bar_plot_with_spaced_legend(dataframe, file_prefix, figsize=(18, 7)):
    """
    创建两个并列的柱状图，图例放在两个图上方且保持更大距离
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=1200)

    # 第一个图：与原始数据比较
    barplot1 = sns.barplot(
        data=dataframe,
        x='Sparsity',
        y='MMD_oridata',
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,
        edgecolor='black',
        linewidth=0.8,
        saturation=0.85,
        dodge=True,
        ax=ax1
    )

    ax1.set_title('Distribution Distance to Original Data (MMD)', fontsize=20, weight='bold', pad=15)
    ax1.set_xlabel('Sparsity Level (%)', labelpad=10, fontsize=18, weight='bold')
    ax1.set_ylabel('MMD', labelpad=10, fontsize=17, weight='bold')
    ax1.tick_params(axis='both', labelsize=15)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.2)

    # 第二个图：与测试数据比较
    barplot2 = sns.barplot(
        data=dataframe,
        x='Sparsity',
        y='MMD_testdata',
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,
        edgecolor='black',
        linewidth=0.8,
        saturation=0.85,
        dodge=True,
        ax=ax2
    )

    ax2.set_title('Distribution Distance to Test Data (MMD)', fontsize=20, weight='bold', pad=15)
    ax2.set_xlabel('Sparsity Level (%)', labelpad=10, fontsize=18, weight='bold')
    ax2.set_ylabel('', labelpad=10, fontsize=17, weight='bold')
    ax2.tick_params(axis='both', labelsize=15)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.2)

    # 应用图案到两个图的柱状图
    for ax in [ax1, ax2]:
        for i, container in enumerate(ax.containers):
            method = methods[i % len(methods)].lower()
            _, hatch = MARKER_CONFIG.get(method, (None, None))
            if hatch:
                for bar in container:
                    bar.set_hatch(hatch)
        # 移除默认图例
        ax.get_legend().remove()

    # 创建自定义图例
    legend_handles = []
    for i, method in enumerate(methods):
        method_lower = method.lower()
        _, hatch = MARKER_CONFIG.get(method_lower, (None, None))

        patch = Rectangle(
            (0, 0), 1.5, 1.5,
            facecolor=MODERN_PALETTE[i % len(MODERN_PALETTE)],
            edgecolor='black',
            linewidth=0.8,
            hatch=hatch
        )
        legend_handles.append(patch)

    # 在两个子图上方添加图例，增加与柱状图的距离
    leg = fig.legend(
        handles=legend_handles,
        labels=methods,
        title='Methods',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),  # 增加距离：从1.0增加到1.05
        frameon=True,
        framealpha=0.9,
        edgecolor='0.5',
        fancybox=False,
        fontsize=16,
        title_fontsize='17',
        handlelength=1.8,
        handleheight=1.8,
        ncol=len(methods),  # 水平排列图例项
        bbox_transform=fig.transFigure  # 使用图形坐标
    )
    leg.get_title().set_fontweight('bold')

    # 调整子图位置，为顶部图例留出更多空间
    plt.subplots_adjust(top=0.8)  # 增加空间：从0.85减少到0.8

    # 保存图像
    for ext in ['.pdf', '.png']:
        plt.savefig(f'{file_prefix}{ext}', bbox_inches='tight', dpi=1200)
    print(f"已保存双柱状图: {file_prefix}.pdf 和 {file_prefix}.png")
    plt.close()


# ==================== 生成最终的图表 ====================
create_double_bar_plot_with_spaced_legend(
    dataframe=df,
    file_prefix='final_mmd_comparison_spaced_legend'
)

print("图表生成完成！")