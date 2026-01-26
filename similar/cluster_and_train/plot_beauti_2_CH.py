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

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'font.family': [
        'Times New Roman',
        'SimSun',
        'Songti SC',
        'Microsoft YaHei',
        'Arial Unicode MS'
    ],

    'mathtext.fontset': 'stix',
    'axes.linewidth': 1.5
})

# --- 2. 数据处理 ---
file_path_pattern = 'results/lstm/fixed*.csv'
csv_files = glob.glob(file_path_pattern)
if not csv_files:
    print("Error: No CSV files found.")
    exit()

df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
df = df.sort_values(by='KL_Similarity')
df = df[df['KL_Similarity'] < 0.35]
methods = df['Method'].unique()

# --- 3. 设置 ---
method_name_map = {
    'ours': 'OURS',
    'timegan': 'TimeGAN',
    'oridata': '真实数据',
    'diffts': 'CDDM',
    'cgan': 'DDPM'
}

our_method_name = 'ours'
boundary_x = 0.17

base_palette = sns.color_palette("muted", n_colors=len(methods))
color_map = {}
idx = 0
for m in methods:
    if m == our_method_name:
        color_map[m] = '#E31A1C'
    else:
        color_map[m] = base_palette[idx] if idx != 3 else base_palette[idx + 1]
        idx += 1

# --- 4. 绘图 ---
fig, ax = plt.subplots(figsize=(12, 8))
max_x = 0

smooth_curves = {}

for method in methods:
    data = df[df['Method'] == method]
    if len(data) < 11:
        continue

    x = savgol_filter(data['KL_Similarity'], 11, 3)
    y = savgol_filter(data['MAE'], 11, 3)

    smooth_curves[method] = (x, y)

    is_ours = method == our_method_name

    ax.plot(
        x, y,
        label=method_name_map.get(method, method),
        color=color_map[method],
        linewidth=4 if is_ours else 2.5,
        alpha=1.0 if is_ours else 0.7,
        zorder=10 if is_ours else 5
    )

    ax.scatter(
        x[-1], y[-1],
        color=color_map[method],
        s=120 if is_ours else 60,
        edgecolors='white',
        linewidths=1.5
    )

    max_x = max(max_x, x[-1])

# --- 5. 分布区域 ---
ax.axvspan(df['KL_Similarity'].min(), boundary_x,
           color='white', alpha=0.6, zorder=6)
ax.axvspan(boundary_x, max_x * 1.1,
           color='#F7F7F7', alpha=0.5, zorder=0)

ax.text(boundary_x - 0.01, ax.get_ylim()[1] * 0.92,
        '分布内',
        ha='right', fontsize=16,
        color='gray', fontstyle='italic', zorder=8)

ax.text(boundary_x + 0.01, ax.get_ylim()[1] * 0.92,
        '分布外',
        ha='left', fontsize=18,
        fontweight='bold', color='#333333', zorder=8)

# --- 6. OOD 持续差值带 ---
if our_method_name in smooth_curves:
    x_ours, y_ours = smooth_curves[our_method_name]

    band_x = []
    band_y_ours = []
    band_y_best = []

    for i, x_val in enumerate(x_ours):
        if x_val < boundary_x:
            continue

        baseline_vals = []
        for m, (x_m, y_m) in smooth_curves.items():
            if m == our_method_name:
                continue
            idx_near = np.argmin(np.abs(x_m - x_val))
            baseline_vals.append(y_m[idx_near])

        if baseline_vals:
            band_x.append(x_val)
            band_y_ours.append(y_ours[i])
            band_y_best.append(min(baseline_vals))

    if band_x:
        ax.fill_between(
            band_x,
            band_y_ours,
            band_y_best,
            where=np.array(band_y_best) > np.array(band_y_ours),
            color='lightgray',
            alpha=0.6,
            zorder=4
        )

# --- 7. 收尾 ---
ax.axvline(boundary_x, color='black',
           linestyle=(0, (5, 5)), alpha=0.4, zorder=7)

ax.set_xlabel('分布偏移程度（MMD）')
ax.set_ylabel('预测误差（MAE）')
ax.set_title('模型泛化性能对比',
             fontweight='bold')

handles, labels = ax.get_legend_handles_labels()
ours_label = method_name_map[our_method_name]
order = [labels.index(ours_label)] + [i for i, l in enumerate(labels) if l != ours_label]

ax.legend([handles[i] for i in order],
          [labels[i] for i in order],
          loc='upper left',
          frameon=True)


ax.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('Generalization_Results_CH.pdf', dpi=300, bbox_inches='tight')
plt.savefig('Generalization_Results_CH.png', dpi=300, bbox_inches='tight')
plt.show()
