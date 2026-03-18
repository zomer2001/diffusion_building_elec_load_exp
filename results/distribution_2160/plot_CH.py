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
    'grid.alpha': 0.12,

    'legend.frameon': True,
    'legend.framealpha': 0.6,
    'legend.edgecolor': '0.5',

    'axes.linewidth': 1.0,
    'axes.edgecolor': '0.4',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0
})

# ==================== Nature风格浅色配色（全新） ====================
NATURE_PALETTE = [
    '#f7f5c9',
    '#A9C5EB',  # 浅蓝（cgan）
    '#A8D8B9',  # 浅绿（diffts）
    '#F3C7B6',  # 浅橙（timegan）
    '#d7e3f7'   # 稍深蓝（OURS）
]

MODERN_PALETTE = NATURE_PALETTE

# ==================== 标记与填充样式 ====================
MARKER_CONFIG = {
    'cgan': ('o', None),
    'diffts': ('s', None),
    'timegan': ('D', None),
    'wgan': ('P', None),   # 新增
    'ours': ('^', '//')
}

# ==================== 数据处理 ====================
df = pd.read_csv(r"script\modified_methods.csv")

# 🔥 关键修改：vaegan → wgan
# df['Method'] = df['Method'].str.lower().replace({
#     'vaegan': 'wgan'
# })
# df['Method'] = df['Method'].str.upper()

df['Sparsity'] = pd.to_numeric(df['Sparsity'], errors='coerce')
df.sort_values(by='Sparsity', inplace=True)

methods = df['Method'].unique().tolist()

# 保证 OURS 在最后（突出）
if 'OURS' in methods:
    methods.remove('OURS')
    methods.append('OURS')

print(f"调整后的方法顺序: {methods}")

# ==================== 创建双柱状图函数 ====================
def create_double_bar_plot_with_spaced_legend(dataframe, file_prefix, figsize=(18, 7)):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=1200)

    # 左图
    sns.barplot(
        data=dataframe,
        x='Sparsity',
        y='MMD_oridata',
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,
        edgecolor='0.4',
        linewidth=0.8,
        saturation=0.75,
        dodge=True,
        ax=ax1
    )

    #ax1.set_title('与原始数据的分布差异（MMD）', fontsize=22, weight='bold', pad=15)
    ax1.set_xlabel('训练数据可用率（%）', fontsize=20, weight='bold', labelpad=10)
    ax1.set_ylabel('与原始数据的分布差异（MMD）', fontsize=17, weight='bold', labelpad=10)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.2)

    # 右图
    sns.barplot(
        data=dataframe,
        x='Sparsity',
        y='MMD_testdata',
        hue='Method',
        hue_order=methods,
        palette=MODERN_PALETTE,
        edgecolor='0.4',
        linewidth=0.8,
        saturation=0.75,
        dodge=True,
        ax=ax2
    )

    #ax2.set_title('与测试数据的分布差异（MMD）', fontsize=22, weight='bold', pad=15)
    ax2.set_xlabel('训练数据可用率（%）', fontsize=20, weight='bold', labelpad=10)
    ax2.set_ylabel('与测试数据的分布差异（MMD）', fontsize=17, weight='bold', labelpad=10)
    ax2.tick_params(axis='both', labelsize=18)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.2)

    # hatch + 去默认图例
    for ax in [ax1, ax2]:
        for i, container in enumerate(ax.containers):
            method = methods[i % len(methods)].lower()
            _, hatch = MARKER_CONFIG.get(method, (None, None))
            if hatch:
                for bar in container:
                    bar.set_hatch(hatch)
        ax.get_legend().remove()

    # ==================== 自定义图例 ====================
    legend_handles = []
    for i, method in enumerate(methods):
        method_lower = method.lower()
        _, hatch = MARKER_CONFIG.get(method_lower, (None, None))

        patch = Rectangle(
            (0, 0), 1.5, 1.5,
            facecolor=MODERN_PALETTE[i % len(MODERN_PALETTE)],
            edgecolor='0.4',
            linewidth=0.8,
            hatch=hatch
        )
        legend_handles.append(patch)

    leg = fig.legend(
        handles=legend_handles,
        labels=methods,
        title='生成方法',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        frameon=True,
        framealpha=0.6,
        edgecolor='0.5',
        fancybox=False,
        fontsize=15,
        title_fontsize=16,
        handlelength=1.6,
        handleheight=1.6,
        ncol=len(methods),
        bbox_transform=fig.transFigure
    )

    leg.get_title().set_fontweight('bold')

    plt.subplots_adjust(top=0.82)

    for ext in ['.pdf', '.png']:
        plt.savefig(f'{file_prefix}{ext}', bbox_inches='tight', dpi=1200)

    print(f"已保存图: {file_prefix}")
    plt.close()

# ==================== 生成图 ====================
create_double_bar_plot_with_spaced_legend(
    dataframe=df,
    file_prefix='final_mmd_comparison_nature_style'
)

print("图表生成完成")