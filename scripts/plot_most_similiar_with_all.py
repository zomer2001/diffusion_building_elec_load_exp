import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# =========================
# Backend & Style
# =========================
matplotlib.use('TkAgg')

def set_academic_style():
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'lines.linewidth': 2.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white'
    })


# =========================
# Path Configuration (与 t-SNE 一致)
# =========================
base_dir = '../fakedata'
test_data_folder = '../testdata'
sparsity_rates = [70]


# =========================
# Utility Functions
# =========================
def smooth_curve(x, window=5):
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode='same')


def filter_varying_samples(data_load):
    valid_idx = []
    for i, s in enumerate(data_load):
        if np.all(s == 0):
            continue
        if np.all(s == s[0]):
            continue
        if np.any(np.diff(s) != 0):
            valid_idx.append(i)
    return valid_idx


def find_most_similar_load(target_load, candidates_load, num_samples=5):
    diff = np.abs(candidates_load - target_load)
    total_diff = np.sum(diff, axis=1)
    idx = np.argsort(total_diff)[:num_samples]
    return candidates_load[idx], idx


def find_most_similar_load2(target_loads, candidates_load, num_samples=5):
    diffs = np.abs(candidates_load[:, None] - target_loads)
    total = np.sum(diffs, axis=2)
    idx = np.argsort(np.sum(total, axis=1))[:num_samples]
    return candidates_load[idx], idx


def load_generated_data(folder_path):
    """与 t-SNE 脚本完全一致的生成数据读取方式"""
    data = []
    if os.path.exists(folder_path):
        for sub_folder in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_path):
                for file in os.listdir(sub_path):
                    if file.endswith('.npy'):
                        try:
                            data.append(np.load(os.path.join(sub_path, file)))
                        except Exception as e:
                            print(f'加载失败: {file}, 错误: {e}')
        if data:
            return np.concatenate(data, axis=0)
    return None


# =========================
# Visualization
# =========================
def plot_group(ax, reference, group, title, ref_label, color):
    time_axis = np.arange(len(reference))

    ref_s = smooth_curve(reference)
    group_s = np.array([smooth_curve(g) for g in group])

    # 分布包络
    ax.fill_between(
        time_axis,
        group_s.min(axis=0),
        group_s.max(axis=0),
        color=color,
        alpha=0.18,
        linewidth=0
    )

    # 弱化单条生成曲线
    for g in group_s:
        ax.plot(time_axis, g, color=color, alpha=0.35)

    # 对照曲线
    ax.plot(
        time_axis,
        ref_s,
        color='#2f2f2f',
        linestyle='--',
        linewidth=3.5,
        label=ref_label
    )

    ax.set_title(title)
    ax.set_ylabel('Normalized Load')


def plot_comparison(oridata_sample, ours_similar, diffts_similar, testdata_similar, building_name):
    set_academic_style()

    fig, axes = plt.subplots(4, 2, figsize=(18, 14), sharex=True)

    color_ours = '#2ca02c'     # 与 t-SNE 中 OURS 保持一致
    color_diffts = '#ff7f0e'   # 与 Diff-TS 保持一致

    # Row 1
    plot_group(
        axes[0, 0], oridata_sample, ours_similar,
        f'Ours: Pattern Consistency ({building_name})',
        'Training Load', color_ours
    )
    plot_group(
        axes[0, 1], oridata_sample, diffts_similar,
        f'Diff-TS: Pattern Consistency ({building_name})',
        'Training Load', color_diffts
    )

    # Row 2
    plot_group(
        axes[1, 0], testdata_similar[0], ours_similar,
        'Ours: Similarity to Test Load',
        'Test Load', color_ours
    )
    plot_group(
        axes[1, 1], testdata_similar[0], diffts_similar,
        'Diff-TS: Similarity to Test Load',
        'Test Load', color_diffts
    )

    # Row 3 & 4
    for r, idx in zip([2, 3], [1, 2]):
        plot_group(
            axes[r, 0], testdata_similar[idx], ours_similar,
            f'Ours: Coverage of Closest Test Load {idx}',
            f'Test Load {idx}', color_ours
        )
        plot_group(
            axes[r, 1], testdata_similar[idx], diffts_similar,
            f'Diff-TS: Coverage of Closest Test Load {idx}',
            f'Test Load {idx}', color_diffts
        )

    for ax in axes[-1, :]:
        ax.set_xlabel('Time (hour)')

    plt.tight_layout()
    plt.show()


# =========================
# Main Loop (完全对齐 t-SNE)
# =========================
processed_buildings = set()

for test_folder in os.listdir(test_data_folder):
    parts = test_folder.split('_')
    if len(parts) < 4:
        continue

    building_name = '_'.join(parts[:-1])
    length = int(parts[-2])
    sparsity = int(parts[-1])

    key = f'{building_name}_{sparsity}'
    if key in processed_buildings or length != 2160 or sparsity not in sparsity_rates:
        continue

    processed_buildings.add(key)
    print(f'Processing building: {building_name}, sparsity={sparsity}%')

    # === 训练 & 测试数据 ===
    oridata_file = os.path.join(
        test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_train.npy'
    )
    test_file = os.path.join(
        test_data_folder, test_folder, 'samples', 'energy_norm_truth_24_test.npy'
    )

    if not os.path.exists(oridata_file) or not os.path.exists(test_file):
        print('Missing train/test data, skip.')
        continue

    oridata = np.load(oridata_file)[:, :, 0]
    testdata = np.load(test_file)[:, :, 0]

    # === 生成数据 ===
    diffts_folder = os.path.join(base_dir, 'diffts-fft', str(sparsity), building_name)
    ours_folder = os.path.join(base_dir, 'ours_gen', str(sparsity), building_name)

    diffts_data = load_generated_data(diffts_folder)
    ours_data = load_generated_data(ours_folder)

    if diffts_data is None or ours_data is None:
        print('Missing generated data, skip.')
        continue

    diffts_load = diffts_data[:, :, 0]
    ours_load = ours_data[:, :, 0]

    # === 选取有变化的样本 ===
    valid_indices = filter_varying_samples(oridata)


    # ===== 防御式判断：没有合适 indices 直接跳过 =====
    if len(valid_indices) < 3:
        print(f'No valid varying samples (<3), skip building: {building_name}')
        continue

    selected_indices = np.random.choice(valid_indices, 3, replace=False)
    

    for idx in selected_indices:
        oridata_sample = oridata[idx]

        diffts_similar, _ = find_most_similar_load(oridata_sample, diffts_load, 5)
        ours_similar, _ = find_most_similar_load(oridata_sample, ours_load, 5)

        test_similar, _ = find_most_similar_load(oridata_sample, testdata, 3)
        ours_similar, _ = find_most_similar_load2(test_similar, ours_load, 5)

        plot_comparison(
            oridata_sample,
            ours_similar,
            diffts_similar,
            test_similar,
            building_name
        )

print('All comparison plots finished.')
