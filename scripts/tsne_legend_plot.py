import matplotlib.pyplot as plt
import numpy as np

# ==================== 全局样式设置（与原代码完全一致）====================
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

# 原代码的配色和标记方案
COLOR_PALETTE = {
    'Traindata': '#1f77b4',
    'Testdata': '#d62728',
    'CDDM': '#ff7f0e',
    'OURS': '#2ca02c'
}

MARKERS = {
    'Traindata': 'o',
    'Testdata': 's',
    'OURS': 'D',
    'CDDM': '^'
}


# ==================== 生成加长版图例 ====================
def plot_long_legend():
    # 设置画布大小：宽度设为原t-SNE图（8）的2倍，高度适配图例
    fig, ax = plt.subplots(figsize=(16, 2), tight_layout=True)

    # 隐藏坐标轴（只保留图例）
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 绘制占位的散点（仅用于生成图例）
    for data_type in ['Traindata', 'Testdata', 'CDDM', 'OURS']:
        ax.scatter([], [],
                   color=COLOR_PALETTE[data_type],
                   marker=MARKERS[data_type],
                   s=120,  # 与原代码一致的点大小
                   alpha=0.8,
                   label=data_type)

    # 生成加长版图例：横向排列、增大间距、调整样式
    legend = ax.legend(
        title='数据类型',
        title_fontsize=16,  # 标题字体稍大
        fontsize=15,  # 图例内容字体
        loc='center',  # 居中显示
        ncol=4,  # 4列横向排列（加长核心）
        frameon=True,
        framealpha=0.9,
        edgecolor='black',
        columnspacing=3.0,  # 列间距增大，拉长图例
        handletextpad=1.5,  # 标记和文字间距
        markerscale=1.8  # 标记放大，更醒目
    )

    # 保存图例（无多余空白）
    output_path = 'long_legend_tsne.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=600, facecolor='white')
    print(f"加长版图例已保存至: {output_path}")
    plt.close()


# 执行生成
if __name__ == '__main__':
    plot_long_legend()