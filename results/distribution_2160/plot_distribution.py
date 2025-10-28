import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# ==================== 学术论文风格设置 ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,  # 增大基础字体大小
    'axes.titlesize': 16,  # 增大标题字体
    'axes.labelsize': 15,  # 增大轴标签字体
    'xtick.labelsize': 13,  # 增大x轴刻度标签字体
    'ytick.labelsize': 13,  # 增大y轴刻度标签字体
    'legend.fontsize': 14,  # 增大图例字体
    'figure.dpi': 1200,  # 提高分辨率
    'savefig.dpi': 1200,  # 提高保存分辨率
    'mathtext.fontset': 'stix',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.15,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.3',
    'axes.linewidth': 1.0,  # 增加线条宽度
    'axes.edgecolor': '0.3',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
})

# 更新更现代、更具审美的配色方案
MODERN_PALETTE = [
    '#fad390',  # 深蓝色 - cgan
    '#f8c291',  # 翡翠绿 - diffts
    '#82ccdd',  # 红宝石色 - timegan
    '#b8e994'  # 紫水晶色 - ours
]

# 设置标记和图案
MARKER_CONFIG = {
    'cgan': ('o', None),  # 圆圈，无填充
    'diffts': ('s', None),  # 方形，无填充
    'timegan': ('D', None),  # 钻石形，无填充
    'diffts-fft': ('^', '//')  # 三角形，斜线图案
}

# ==================== 数据处理 ====================
# 1. 读取CSV文件
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

# 2. 打印绘图数据 (只关注MMD)
print("=== 与原始数据的分布差异（MMD_oridata） ===")
print(df[['Method', 'Sparsity', 'MMD_oridata']])

print("\n=== 与测试数据的分布差异（MMD_testdata） ===")
print(df[['Method', 'Sparsity', 'MMD_testdata']])


# ==================== 学术柱状图绘图函数 ====================
def plot_academic_bar(dataframe, metric, title, file_prefix, figsize=(8, 5)):
    """
    绘制学术风格柱状图 - 已改进配色和图例位置
    """
    # 创建图形和轴
    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()

    # 计算柱状图位置
    num_methods = len(methods)
    num_sparsity = len(dataframe['Sparsity'].unique())

    # 使用seaborn创建基础柱状图
    barplot = sns.barplot(
        data=dataframe,
        x='Sparsity',
        y=metric,
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,  # 使用新的现代配色
        edgecolor='black',
        linewidth=0.8,  # 增加边框宽度
        saturation=0.85,  # 增加饱和度使颜色更鲜明
        dodge=True,
        ax=ax
    )

    # 设置标题和标签
    plt.title(title, fontsize=16, pad=15, weight='bold')  # 增大标题字体
    plt.xlabel('Sparsity Level (%)', labelpad=10, fontsize=15, weight='bold')  # 增大标签字体
    plt.ylabel('MMD', labelpad=10, fontsize=15, weight='bold')  # 增大标签字体

    # 设置刻度标签更清晰
    plt.xticks(rotation=0, fontsize=13)  # 增大x轴标签字体
    plt.yticks(fontsize=13)  # 增大y轴标签字体

    # 设置细化的网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.2)

    # 移除seaborn自带的图例
    ax.get_legend().remove()

    # 应用图案到柱状图
    for i, container in enumerate(ax.containers):
        method = methods[i % num_methods].lower()
        _, hatch = MARKER_CONFIG.get(method, (None, None))
        if hatch:
            for bar in container:
                bar.set_hatch(hatch)

    # 创建自定义学术图例 - 增大图标尺寸
    legend_handles = []
    for i, method in enumerate(methods):
        method_lower = method.lower()
        _, hatch = MARKER_CONFIG.get(method_lower, (None, None))

        # 创建图例项 - 增大尺寸
        patch = Rectangle(
            (0, 0), 1.5, 1.5,  # 增大尺寸 (1.5x1.5)
            facecolor=MODERN_PALETTE[i % len(MODERN_PALETTE)],
            edgecolor='black',
            linewidth=0.8,  # 增加边框宽度
            hatch=hatch
        )
        legend_handles.append(patch)

    # 将图例移到图表右侧外部 - 关键修改
    fig = plt.gcf()
    leg = fig.legend(
        handles=legend_handles,
        labels=methods,
        title='Methods',
        loc='center right',
        bbox_to_anchor=(1.15, 0.5),  # 右侧外部
        frameon=True,
        framealpha=0.9,
        edgecolor='0.5',
        fancybox=False,
        fontsize=14,  # 增大图例字体
        title_fontsize='15',  # 增大图例标题字体
        handlelength=2.0,  # 增大图例项长度
        handleheight=2.0,  # 增大图例项高度
        markerscale=1.5  # 增大图例标记比例
    )

    # 设置图例标题更突出
    leg.get_title().set_fontweight('bold')

    # 调整布局以适应右侧图例
    plt.subplots_adjust(right=0.8)

    # 优化布局
    plt.tight_layout(pad=3.0)  # 增加内边距

    # 保存高质量图像
    for ext in ['.pdf', '.png']:
        plt.savefig(f'{file_prefix}{ext}', bbox_inches='tight', dpi=1200)
    print(f"已保存学术图表: {file_prefix}.pdf 和 {file_prefix}.png")
    plt.close()


# ==================== 创建两个图 ====================

# 图一：与原始数据比较
plot_academic_bar(
    dataframe=df,
    metric='MMD_oridata',
    title='Distribution Distance to Original Data (MMD)',
    file_prefix='academic_original_data_mmd'
)

# 图二：与测试数据比较
plot_academic_bar(
    dataframe=df,
    metric='MMD_testdata',
    title='Distribution Distance to Test Data (MMD)',
    file_prefix='academic_test_data_mmd'
)


# ==================== 面板比较图 ====================
def plot_comparison_panel(dataframe, file_prefix):
    """
    绘制比较面板图 - 包含原始数据和测试数据的比较
    """
    # 创建图形
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), dpi=1200)  # 增大图形尺寸
    fig.suptitle('Distribution Distance Comparison', fontsize=18, y=0.98, weight='bold')  # 增大标题字体

    # 原始数据子图
    ax1 = axs[0]
    sns.barplot(
        data=dataframe,
        x='Sparsity',
        y='MMD_oridata',
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,
        edgecolor='black',
        linewidth=0.8,  # 增加边框宽度
        saturation=0.85,
        dodge=True,
        ax=ax1
    )
    ax1.set_title('Distribution Distance to Original Data (MMD)', fontsize=15, weight='bold')  # 增大子标题字体
    ax1.set_xlabel('Sparsity Level (%)', labelpad=10, fontsize=14, weight='bold')  # 增大标签字体
    ax1.set_ylabel('MMD', labelpad=10, fontsize=14, weight='bold')  # 增大标签字体
    ax1.yaxis.grid(True, linestyle='--', alpha=0.2)
    ax1.tick_params(axis='both', labelsize=13)  # 增大刻度标签字体

    # 测试数据子图
    ax2 = axs[1]
    sns.barplot(
        data=dataframe,
        x='Sparsity',
        y='MMD_testdata',
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,
        edgecolor='black',
        linewidth=0.8,  # 增加边框宽度
        saturation=0.85,
        dodge=True,
        ax=ax2
    )
    ax2.set_title('Distribution Distance to Test Data (MMD)', fontsize=15, weight='bold')  # 增大子标题字体
    ax2.set_xlabel('Sparsity Level (%)', labelpad=10, fontsize=14, weight='bold')  # 增大标签字体
    ax2.set_ylabel('', labelpad=10, fontsize=14, weight='bold')  # 共享y轴标签
    ax2.yaxis.grid(True, linestyle='--', alpha=0.2)
    ax2.tick_params(axis='both', labelsize=13)  # 增大刻度标签字体

    # 应用图案到柱状图
    for ax in axs:
        for i, container in enumerate(ax.containers):
            method = methods[i % len(methods)].lower()
            _, hatch = MARKER_CONFIG.get(method, (None, None))
            if hatch:
                for bar in container:
                    bar.set_hatch(hatch)

        # 移除子图图例
        ax.get_legend().remove()

    # 创建自定义图例 - 增大图标尺寸
    legend_handles = []
    for i, method in enumerate(methods):
        method_lower = method.lower()
        _, hatch = MARKER_CONFIG.get(method_lower, (None, None))

        patch = Rectangle(
            (0, 0), 2, 2,  # 增大尺寸 (1.5x1.5)
            facecolor=MODERN_PALETTE[i % len(MODERN_PALETTE)],
            edgecolor='black',
            linewidth=0.8,  # 增加边框宽度
            hatch=hatch
        )
        legend_handles.append(patch)

    # 添加共享图例 - 增大图标尺寸
    leg = fig.legend(
        handles=legend_handles,
        labels=methods,
        title='Methods',
        loc='upper center',
        bbox_to_anchor=(0.5, 0.96),
        ncol=len(methods),
        frameon=True,
        framealpha=0.9,
        edgecolor='0.5',
        fancybox=False,
        fontsize=14,  # 增大图例字体
        title_fontsize='15',  # 增大图例标题字体
        handlelength=2.0,  # 增大图例项长度
        handleheight=2.0,  # 增大图例项高度
        markerscale=1.5  # 增大图例标记比例
    )
    leg.get_title().set_fontweight('bold')  # 设置图例标题为粗体

    # 调整布局
    plt.tight_layout(pad=4.0, rect=[0, 0, 1, 0.92])  # 增加内边距

    # 保存高质量图像
    for ext in ['.pdf', '.png']:
        plt.savefig(f'{file_prefix}{ext}', bbox_inches='tight', dpi=1200)
    print(f"已保存面板比较图: {file_prefix}.pdf 和 {file_prefix}.png")
    plt.close()


# ==================== 创建面板比较图 ====================
plot_comparison_panel(
    dataframe=df,
    file_prefix='academic_mmd_comparison'
)

print("所有图表生成完成！")