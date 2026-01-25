import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

# ==================== 学术论文风格设置 ====================
plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun', 'Microsoft YaHei'],
    'axes.unicode_minus': False,

    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 17,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'legend.fontsize': 16,

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

# 配色方案 (适配 5 种方法)
MODERN_PALETTE = [
    '#b3cde3',
    '#deebf7',
    '#d1e5f0',
    '#92c5de',
    '#4393c3',
    '#2166ac'
]

# 设置图案标记 (仅 ours 使用斜纹)
MARKER_CONFIG = {
    'ours': '//'
}

# ==================== 数据处理 ====================
df = pd.read_csv('summary_metrics.csv')

df['Method_lower'] = df['Method'].str.lower()
df['Sparsity'] = pd.to_numeric(df['Sparsity'], errors='coerce')
df.sort_values(by='Sparsity', inplace=True)

methods_unique = [m for m in df['Method'].unique() if m.lower() != 'ours']
methods_order = sorted(methods_unique) + ['ours']
methods_lower_order = [m.lower() for m in methods_order]

# ==================== 绘图核心函数 ====================
def create_academic_comparison_plot(dataframe, y_cols, titles, ylabel, file_prefix, figsize=(18, 7)):
    """
    创建两个并列的柱状图，图例放在两个图上方且保持更大距离
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=1200)

    axes = [ax1, ax2]

    for ax, y_col, title in zip(axes, y_cols, titles):
        sns.barplot(
            data=dataframe,
            x='Sparsity',
            y=y_col,
            hue='Method',
            hue_order=methods_order,
            palette=MODERN_PALETTE,
            edgecolor='black',
            linewidth=0.8,
            saturation=0.85,
            dodge=True,
            ax=ax
        )

        ax.set_title(title, fontsize=22, weight='bold', pad=15)
        ax.set_xlabel('数据可用比例（%）', labelpad=10, fontsize=20, weight='bold')
        ax.set_ylabel(ylabel if ax == ax1 else "", labelpad=10, fontsize=17, weight='bold')
        ax.set_xticks(range(len(dataframe['Sparsity'].unique())))
        ax.set_xticklabels(['30%', '50%', '70%', '90%'], fontsize=19)
        ax.tick_params(axis='both', labelsize=19)
        ax.yaxis.grid(True, linestyle='--', alpha=0.2)

        # 应用阴影图案 (仅 OURS)
        for i, container in enumerate(ax.containers):
            method_name = methods_lower_order[i]
            hatch = MARKER_CONFIG.get(method_name)
            if hatch:
                for bar in container:
                    bar.set_hatch(hatch)

        if ax.get_legend():
            ax.get_legend().remove()

    legend_handles = []
    for i, method in enumerate(methods_order):
        hatch = MARKER_CONFIG.get(method.lower())
        patch = Rectangle(
            (0, 0), 1.5, 1.5,
            facecolor=MODERN_PALETTE[i],
            edgecolor='black',
            linewidth=0.8,
            hatch=hatch
        )
        legend_handles.append(patch)

    leg = fig.legend(
        handles=legend_handles,
        labels=[m.upper() for m in methods_order],
        title='合成方法',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
        framealpha=0.9,
        edgecolor='0.5',
        fancybox=False,
        fontsize=16,
        title_fontsize='18',
        handlelength=1.8,
        handleheight=1.8,
        ncol=len(methods_order),
        bbox_transform=fig.transFigure
    )
    leg.get_title().set_fontweight('bold')

    plt.subplots_adjust(top=0.8)

    for ext in ['.pdf', '.png']:
        plt.savefig(f'{file_prefix}{ext}', bbox_inches='tight', dpi=1200)

    print(f"已保存图表: {file_prefix}")
    plt.close()

# ==================== 生成最终的图表 ====================

create_academic_comparison_plot(
    df,
    ['MMD_oridata', 'MMD_testdata'],
    ['与原始数据的分布距离（MMD）', '与测试数据的分布距离（MMD）'],
    'MMD 值',
    'mmd_comparison_spaced_CH'
)

create_academic_comparison_plot(
    df,
    ['KL_oridata', 'KL_testdata'],
    ['与原始数据的分布距离（KL）', '与测试数据的分布距离（KL）'],
    'KL 散度',
    'kl_comparison_spaced'
)

print("所有学术图表生成完成")
