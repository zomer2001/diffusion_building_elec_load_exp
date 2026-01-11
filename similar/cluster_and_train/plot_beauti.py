import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import glob
from scipy.signal import savgol_filter
import seaborn as sns
import numpy as np

# --- 1. 环境配置 ---
try:
    matplotlib.use('MacOSX')
except ImportError:
    matplotlib.use('Agg')

# 学术字体设置
plt.rcParams.update({
    'font.size': 20,  # 进一步调大字号
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5  # 坐标轴线加粗
})

# --- 2. 数据处理 ---
file_path_pattern = '/Users/zomeryang/Documents/MYSOURCE/diffusion_building_elec_load_exp/similar/cluster_and_train/results/lstm/fixed*.csv'
csv_files = glob.glob(file_path_pattern)
if not csv_files:
    print("Error: No CSV files found.")
    exit()

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df_filtered = df.sort_values(by='KL_Similarity')
df_filtered = df_filtered[df_filtered['KL_Similarity'] < 0.35]  # 适当调整截断点
methods = df_filtered['Method'].unique()

# --- 3. 颜色与核心设置 ---
our_method_name = "ours"
boundary_x = 0.17  # 关键分界点：从这里开始体现泛化能力

# 使用专业色板
base_palette = sns.color_palette("muted", n_colors=len(methods))
color_map = {}
idx = 0
for m in methods:
    if m == our_method_name:
        color_map[m] = '#E31A1C'  # 纯正学术红
    else:
        color_map[m] = base_palette[idx] if idx != 3 else base_palette[idx + 1]
        idx += 1

# --- 4. 绘图开始 ---
fig, ax = plt.subplots(figsize=(12, 8))

max_x = 0
y_ours_end = 0
y_baseline_best_end = 1000

for method in methods:
    data = df_filtered[df_filtered['Method'] == method]
    if len(data) < 11: continue

    # 平滑
    x_smooth = savgol_filter(data['KL_Similarity'], 11, 3)
    y_smooth = savgol_filter(data['MAE'], 11, 3)

    is_ours = (method == our_method_name)
    z = 10 if is_ours else 5

    # 绘图
    line, = ax.plot(x_smooth, y_smooth,
                    label=method,
                    color=color_map[method],
                    linewidth=4 if is_ours else 2.5,
                    alpha=1.0 if is_ours else 0.7,
                    zorder=z)

    # 仅在末端添加标记点，保持画面整洁
    ax.scatter(x_smooth[-1], y_smooth[-1],
               color=color_map[method], s=120 if is_ours else 60,
               edgecolors='white', linewidths=1.5, zorder=z + 1)

    # 记录终点坐标
    max_x = max(max_x, x_smooth[-1])
    if is_ours:
        y_ours_end = y_smooth[-1]
    else:
        y_baseline_best_end = min(y_baseline_best_end, y_smooth[-1])

# --- 5. 视觉逻辑润色 (强调泛化) ---

# A. 区域遮罩：左侧淡化，右侧高亮
ax.axvspan(df_filtered['KL_Similarity'].min(), boundary_x, color='white', alpha=0.6, zorder=6)
ax.axvspan(boundary_x, max_x * 1.1, color='#F7F7F7', alpha=0.5, zorder=0)  # 右侧给个极淡的底色

# B. 极简文字标注
ax.text(boundary_x - 0.01, ax.get_ylim()[1] * 0.92, 'In-Distribution',
        ha='right', fontsize=16, color='gray', fontstyle='italic', zorder=8)
ax.text(boundary_x + 0.01, ax.get_ylim()[1] * 0.92, 'OOD Robustness',
        ha='left', fontsize=18, fontweight='bold', color='#333333', zorder=8)

# C. 性能缺口箭头 (Gap)
if y_ours_end < y_baseline_best_end:
    ax.annotate('', xy=(max_x, y_ours_end), xytext=(max_x, y_baseline_best_end),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(max_x + 0.005, (y_ours_end + y_baseline_best_end) / 2, 'Robustness\nGap',
            va='center', fontweight='bold', color=color_map[our_method_name], fontsize=14)

# D. 虚线分界
ax.axvline(x=boundary_x, color='black', linestyle=(0, (5, 5)), linewidth=1, alpha=0.4, zorder=7)

# --- 6. 细节修饰 ---
ax.set_xlabel('Distribution Shift (KL Divergence)', labelpad=12)
ax.set_ylabel('Prediction Error (MAE)', labelpad=12)
ax.set_title('Generalization Performance Comparison', pad=20, fontweight='bold')

# 图例排序：Ours 放在首位
handles, labels = ax.get_legend_handles_labels()
order = [labels.index(our_method_name)] + [i for i, l in enumerate(labels) if l != our_method_name]
ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
          loc='upper left', frameon=True, facecolor='white', framealpha=0.8)

# 去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()

# --- 7. 保存图片 ---
# 保存为 PDF 矢量图（投稿首选）和 高清 PNG
save_path_pdf = "Generalization_Results.pdf"
save_path_png = "Generalization_Results.png"
plt.savefig(save_path_pdf, bbox_inches='tight', dpi=300)
plt.savefig(save_path_png, bbox_inches='tight', dpi=300)

print(f"Success! Figures saved as:\n1. {save_path_pdf}\n2. {save_path_png}")
plt.show()