import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({
    'font.family': 'Arial Narrow',
    'font.size': 16,
    'legend.fontsize': 16,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'axes.grid': False
})

COLOR_PALETTE = {
    'Traindata': '#2ca02c',
    'Testdata': '#d62728',
    'CDDM': '#ff7f0e',
    'OURS': '#1f77b4'
}

markers = {
    'Traindata': 'o',
    'Testdata': 's',
    'OURS': 'D',
    'CDDM': '^'
}

labels = ['Traindata', 'Testdata', 'CDDM', 'OURS']

fig, ax = plt.subplots(figsize=(8, 1.5))
ax.axis('off')

handles = []
for name in labels:
    h = mlines.Line2D(
        [], [],
        color=COLOR_PALETTE[name],
        marker=markers[name],
        linestyle='None',
        markeredgecolor='black',
        markerfacecolor=COLOR_PALETTE[name],  # 实心填充
        markersize=12,
        markeredgewidth=1.2
    )
    handles.append(h)

ax.legend(
    handles=handles,
    labels=labels,
    loc='center',
    ncol=4,
    frameon=True,
    markerscale=1,
    handletextpad=0.8,
    columnspacing=1.5
)

plt.savefig('legend_solid.png', bbox_inches='tight', dpi=600)
plt.close()
print("✅ 已保存：legend_solid.png")